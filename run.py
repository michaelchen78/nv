import csv
import sys
import warnings

import numpy as np
from pathlib import Path
import time
import copy
from mpi4py import MPI
from sim import run_ensemble, load_config, get_last_runtime, logger, is_root, COMM, get_seed
from config import RUN_REGISTRY, SPIN_TYPES

from collections import Counter
from functools import lru_cache

from pycce import Pulse, BathArray


# +++++++++++++++++++++++ Norris 2016 helpers +++++++++++++++++++++++ #
@lru_cache(maxsize=None)
def _cdd_cached(cdd_order, total_length):
    """
    Cached internal CDD pulse times in integer δ-units.
    Returns tuple for safe caching.
    """
    cdd_order = int(cdd_order)
    total_length = int(total_length)

    if cdd_order == 0:
        return ()

    if total_length % 2 != 0:
        raise Exception(f"CDD_{cdd_order} over total_length={total_length} does not fit on the grid")

    half = total_length // 2

    prev = _cdd_cached(cdd_order - 1, half)

    pulses = []
    pulses += prev
    pulses += (half,)
    pulses += tuple(half + t for t in prev)
    pulses += (total_length,)

    # X-X at the same time cancels
    counts = Counter(pulses)

    return tuple(sorted(t for t, n in counts.items() if n % 2 == 1))


def cdd(cdd_order, total_length):
    """
    Public wrapper: same behavior as before, returns a list.
    """
    return list(_cdd_cached(cdd_order, total_length))


def random_partition(q, rng, p_cut=0.10):
    """
    Random ordered partition/composition of integer q.
    Lower p_cut gives fewer, larger pieces.
    """
    cuts = np.flatnonzero(rng.random(q - 1) < p_cut) + 1
    return np.diff(np.r_[0, cuts, q]).tolist()


def make_one_sequence(q, delta_ns, tau_ns, max_order, rng,
                      min_pulses=None, max_pulses=None, p_cut=None):
    """
    Make one base sequence of total duration T = q * delta_ns.
    Returns pulse times in ns.

    min_pulses / max_pulses apply to the final pulse list after X-X cancellation.
    """
    orders_by_qi = {
        qi: [
            cdd_order for cdd_order in range(max_order + 1)
            if qi % (2 ** cdd_order) == 0
        ]
        for qi in range(1, q + 1)
    }

    while True:
        parts = random_partition(q, rng, p_cut=p_cut)

        pulse_times = []
        offset = 0

        for qi in parts:
            possible_orders = orders_by_qi[qi]
            cdd_order = rng.choice(possible_orders)

            local_times = cdd(cdd_order, qi)

            pulse_times += [offset + t for t in local_times]
            offset += qi

        counts = Counter(pulse_times)
        pulse_times = sorted(t for t, n in counts.items() if n % 2 == 1)

        num_pulses = len(pulse_times)

        if num_pulses % 2 != 0:  # enforcing even pulse counts
            continue

        if min_pulses is not None and num_pulses < min_pulses:
            continue

        if max_pulses is not None and num_pulses > max_pulses:
            continue

        if num_pulses < 2:
            continue

        pulse_times = np.array(pulse_times, dtype=int)
        spacings_ticks = np.diff(pulse_times)
        # Check the boundary spacing from last pulse in one cycle to first pulse in the next cycle.
        boundary_spacing_ticks = q - pulse_times[-1] + pulse_times[0]
        spacings_ns = np.r_[spacings_ticks, boundary_spacing_ticks] * delta_ns

        if np.all(spacings_ns >= tau_ns):
            return pulse_times * delta_ns


def make_sequences(P, T_ns, delta_ns, tau_ns, min_pulses, max_order=5, seed=0, p_cut=None):
    """
    Generate P base sequences, always including one empty sequence.

    T_ns = q * delta_ns.
    Each returned sequence is an array of pulse times in ns.
    """
    rng = np.random.default_rng(seed)

    if P < 1:
        raise ValueError("P must be at least 1 because one empty sequence is always included")

    if T_ns % delta_ns != 0:
        raise ValueError(f"T_ns={T_ns} must be divisible by delta_ns={delta_ns}")
    q = T_ns // delta_ns
    assert np.isclose(q * delta_ns, T_ns)

    seqs = [np.array([], dtype=int)]

    while len(seqs) < P:
        if is_root():
            print("starting base sequence")
        seq = make_one_sequence(q, delta_ns, tau_ns, max_order, rng, min_pulses=min_pulses, p_cut=p_cut)

        # redundant check that its even pulses
        if len(seq) % 2 != 0:
            raise Exception(f"Generated sequence has odd number of pulses: {seq}")

        # no exact duplicates
        if not any(np.array_equal(seq, old) for old in seqs):
            seqs.append(seq)

        if is_root():
            print("finished generating base sequence: ", seq)

    return seqs


# +++++++++++++++++++++++ Custom run functions +++++++++++++++++++++++ #
def norris2016(config):
    """
    Norris 2016 NG protocol custom run
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    norris_params = config["norris_params"]
    # required
    P = norris_params["P"]  # number of base sequences
    T = norris_params['T']  # duration of base sequences in ms
    delta = norris_params['delta']  # T = q*delta
    M = norris_params['M']  # number of cycles of each base sequence
    tau = norris_params['tau']  # minimum spacing between pulses
    # optional, defaults
    min_pulses = norris_params.get('min_pulses', None)
    p_cut = norris_params.get('p_cut', 0.01)
    # ================= END OF PARAMETERS ================= #

    # ================= MAIN CODE ================= #
    '''get P sequences'''
    # edit parameters
    T_ns = T * 10 ** 6
    delta_ns = delta * 10 ** 6
    tau_ns = tau * 10 ** 6
    for name, val in [("T_ns", T_ns), ("delta_ns", delta_ns), ("tau_ns", tau_ns)]:
        if not np.isclose(val, round(val), rtol=0.0, atol=1e-6):
            raise Exception(f"{name}={val} is not close to an integer")
    T_ns = int(round(T_ns))
    delta_ns = int(round(delta_ns))
    tau_ns = int(round(tau_ns))

    # Build all Norris pulse sequences on rank 0 only.
    full_sequences_ms = None
    base_sequences_ms = None
    if is_root():
        try:
            # list of base sequences, each is a numpy array of pulse times IN NS
            base_sequences = make_sequences(P, T_ns, delta_ns, tau_ns, min_pulses,
                seed=get_seed(config.get("seed_id", None)), p_cut=p_cut)

            # Force the FID / empty sequence to be LAST. so FID index is p=P-1.
            empty_idxs = [idx for idx, seq in enumerate(base_sequences) if len(seq) == 0]
            if len(empty_idxs) != 1:
                raise Exception(f"Expected exactly one empty/FID sequence, found {len(empty_idxs)}")
            fid_sequence = base_sequences.pop(empty_idxs[0])
            base_sequences.append(fid_sequence)
            if len(base_sequences) != P:
                raise Exception(f"Expected {P} base sequences after moving FID, found {len(base_sequences)}")
            if len(base_sequences[-1]) != 0:
                raise Exception("FID sequence was not moved to the last position")

            # base sequences in ms
            base_sequences_ms = [np.asarray(b, dtype=float) * 1.0e-6 for b in base_sequences]

            # list of full sequences, each base sequence repeated M times
            full_sequences_ms = []
            for base_seq in base_sequences:
                if len(base_seq) == 0:
                    full_sequences_ms.append(np.array([], dtype=float))  # FID case
                    continue
                full_seq = np.concatenate([base_seq + m * T_ns for m in range(M)])
                full_sequences_ms.append(full_seq.astype(int) * 1.0e-6)  # convert ns to ms

            print("\nBASE SEQUENCES (ms): ")
            print(base_sequences_ms)
            print("\n")
            print("\nFULL SEQUENCES (ms): ")
            print(full_sequences_ms)
            print("\n")
            print("\nDONE WITH SEQUENCE GENERATION\n\n\n")
        except Exception as exc:
            logger.error(f"Rank 0 failed during Norris sequence construction: {exc}")
            COMM.Abort(1)

    '''Do a single .compute for each of the P sequences'''
    # make the multi run configs only on rank 0
    new_config = None
    if is_root():
        # build new config for each sub run
        new_config = copy.deepcopy(config)
        sub_run_ids = []
        change_params = []
        for idx, sequence in enumerate(full_sequences_ms):
            sub_run_ids.append(f"p-{idx}")
            change_param_dict = {}

            if len(sequence) == 0:
                pulses = 0
            else:
                # redundant checks
                if np.any(np.diff(sequence) <= 0):
                    raise Exception(f"Sequence {idx} is not strictly increasing: {sequence}")
                if sequence[-1] > T * M + 1e-12:
                    raise Exception(f"Sequence {idx} has pulse after final time: last={sequence[-1]}, final={T * M}")

                # build pycce pulses
                delays = np.diff(np.r_[0.0, sequence])
                pulses = [Pulse("x", np.pi, delay=float(delay)) for delay in delays]

            change_param_dict["experiment_params,pulses"] = pulses
            change_param_dict["experiment_params,time_space"] = [T * M]
            change_params.append(change_param_dict)
        # set the params
        new_config["experiment_params"]["custom_time_space"] = True
        new_config["multi_params"]["sub_run_ids"] = sub_run_ids
        new_config["multi_params"]["change_params"] = change_params
    # broadcast to other mpi ranks
    new_config = COMM.bcast(new_config, root=0)
    COMM.Barrier()
    # run the experiment
    multi_run(new_config)

    '''Save pulse sequences'''
    if is_root():
        # Save full sequences
        seq_csv_path = Path(config.get("base_data_dir", "./data")) / config["run_id"] / "sequences.csv"
        seq_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(seq_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence_idx", "pulse_idx", "time_ms"])
            for sequence_idx, sequence in enumerate(full_sequences_ms):
                if len(sequence) == 0:
                    # Explicitly save FID/empty sequence.
                    # Since FID was moved last, this should be sequence_idx=P-1.
                    if sequence_idx != P - 1:
                        raise Exception(
                            f"Empty/FID sequence should be last with index P-1={P - 1}, "
                            f"but got sequence_idx={sequence_idx}")

                    writer.writerow([sequence_idx, -1, np.nan])
                    continue

                for pulse_idx, time_ms in enumerate(sequence):
                    writer.writerow([sequence_idx, pulse_idx, float(time_ms)])

        # Save base pulse sequences
        seq_csv_path = Path(config.get("base_data_dir", "./data")) / config["run_id"] / "base_sequences.csv"
        seq_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(seq_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence_idx", "pulse_idx", "time_ms"])

            for sequence_idx, sequence in enumerate(base_sequences_ms):
                # Explicitly save FID/empty sequence.
                if len(sequence) == 0:
                    # Since FID was moved last, this should be sequence_idx=P-1.
                    if sequence_idx != P - 1:
                        raise Exception(f"Empty/FID sequence should be last with index P-1={P - 1}, "
                                        f"but got sequence_idx={sequence_idx}")
                    writer.writerow([sequence_idx, -1, np.nan])
                    continue
                # save the rest
                for pulse_idx, time_ms in enumerate(sequence):
                    writer.writerow([sequence_idx, pulse_idx, time_ms])


def mean_fid(config):
    """
        Sung/Jin FID noise mean protocol custom run
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    mean_params = config["mean_params"]
    # required
    taus = mean_params["taus"]  # delays in ms
    # optional, defaults
    buffer = mean_params.get('buffer', None)  # buffer in ms
    tau_space = mean_params.get("tau_space", None)
    logspace = mean_params.get("logspace", False)
    # ================= END OF PARAMETERS ================= #

    # ================= MAIN CODE ================= #
    if tau_space is not None:
        if logspace:
            taus = np.logspace(np.log10(tau_space[0]), np.log10(tau_space[1]), (tau_space[2]))
        else:
            taus = np.linspace(tau_space[0], tau_space[1], tau_space[2])
    taus = np.asarray(taus) * 1.0e-6  # convert ns to ms
    if buffer is None:
        buffer = taus / 100.0
    buffer = np.asarray(buffer) * 1.0e-6  # convert ns to ms

    '''manipulate custom cpmg parameters to make a new config'''
    new_config = None
    if is_root():
        new_config = copy.deepcopy(config)
        new_config["experiment_params"]["pulses"] = [('x', np.pi * 0.5, taus)]
        new_config["experiment_params"]["custom_time_space"] = True
        new_config["experiment_params"]["time_space"] = taus + buffer
        if config["supercell_params"].get("custom_bath", False):
            new_config["supercell_params"]["custom_bath"] = True
            atoms = BathArray((1,))
            atoms.add_type(*SPIN_TYPES)
            atoms.N[0] = "13C"
            atoms.xyz[0] = [1.0, 0, 0]
            atoms.from_point_dipole([0, 0, 0])
            # atoms["A"][0] = np.diag([0.0, 0.0, 50000.0])  # kHz, deliberately nonzero
            atoms = BathArray((0,))
            print("custom bath A =", atoms)
            new_config["supercell"] = atoms
            # must do NV manually
            new_config["supercell_params"]["nv_position"] =  [ 0, 0, 0 ]  # central spin position
            new_config["supercell_params"]["alpha"] =[ 0, 0, 1 ]
            new_config["supercell_params"]["beta"] = [ 0, 1, 0 ]
    new_config = MPI.COMM_WORLD.bcast(new_config, root=0)

    '''run the experiment'''
    unit_run(new_config)


def alvarez2011(config):
    """
        Alvarez 2011 DD QNS protocol custom run
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    alvarez_params = config["alvarez_params"]
    # required
    num_cycles = alvarez_params['num_cycles']  # big M
    tau_max = alvarez_params['tau_max']
    num_measurements = alvarez_params['num_measurements']  # little m
    # optional, defaults
    # none
    # ================= END OF PARAMETERS ================= #

    # ================= MAIN CODE ================= #
    '''manipulate custom cpmg parameters to make a new config'''
    new_config = None
    if is_root():
        new_config = copy.deepcopy(config)
        sub_run_ids = []
        change_params = []

        taus = tau_max / np.arange(num_measurements, 0, -1)  # tau_0, tau_1, ... , tau_{m-1}
        pulse_list = 2 * (np.arange(num_cycles) + 1)

        for pulse in pulse_list:
            new_time_space = pulse * taus
            sub_run_ids.append(f"pulses-{pulse}")  # constructing sub_run_ids
            change_param_dict = {}
            change_param_dict["experiment_params,pulses"] = int(pulse)
            change_param_dict["experiment_params,time_space"] = new_time_space
            change_params.append(change_param_dict)

        new_config["experiment_params"]["custom_time_space"] = True
        new_config["multi_params"]["sub_run_ids"] = sub_run_ids
        new_config["multi_params"]["change_params"] = change_params

    new_config = MPI.COMM_WORLD.bcast(new_config, root=0)

    '''run the experiment'''
    multi_run(new_config)


def grid_run(config):
    """
    Utility method to conveniently runs a grid of experiments.
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    grid_params = config["grid_params"]
    # required
    param1 = grid_params['param1']
    param2 = grid_params['param2']
    param1_values = grid_params['param1_values']
    param2_values = grid_params['param2_values']
    # optional, defaults
    # no optional params here
    # ================= END OF PARAMETERS ================= #

    # ================= MAIN CODE ================= #
    '''manipulate custom grid parameters to make a new config'''
    new_config = None
    if is_root():
        new_config = copy.deepcopy(config)
        sub_run_ids = []
        change_params = []

        for i, v1 in enumerate(param1_values):
            for j, v2 in enumerate(param2_values):
                sub_run_ids.append(f"{param1[1]}-{v1}__{param2[1]}-{v2}")  # constructing sub_run_ids
                change_param_dict = {}

                key1 = param1[0] + "," + param1[1]
                key2 = param2[0] + "," + param2[1]
                change_param_dict[key1] = v1
                change_param_dict[key2] = v2

                change_params.append(change_param_dict)

        new_config["multi_params"]["sub_run_ids"] = sub_run_ids
        new_config["multi_params"]["change_params"] = change_params

    new_config = MPI.COMM_WORLD.bcast(new_config, root=0)

    '''run the experiment'''
    multi_run(new_config)


def cpmg(config):
    """
    cpmg custom run
    good example to copy and paste if building a custom run
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    cpmg_params = config["cpmg_params"]
    # required
    n_list = cpmg_params['n_list']
    # optional, defaults
    n_linspace = cpmg_params.get("n_linspace", None)
    # ================= END OF PARAMETERS ================= #

    # ================= MAIN CODE ================= #
    '''manipulate custom cpmg parameters to make a new config'''
    new_config = None
    if is_root():
        new_config = copy.deepcopy(config)
        sub_run_ids = []
        change_params = []

        n_space = n_list
        if n_linspace is not None:
            n_space = np.linspace(*n_linspace).astype(int)

        for n in n_space:
            sub_run_ids.append(f"CPMG-{n}")  # constructing sub_run_ids
            change_param_dict = {}
            change_param_dict["experiment_params,pulses"] = int(n)
            change_params.append(change_param_dict)

        new_config["multi_params"]["sub_run_ids"] = sub_run_ids
        new_config["multi_params"]["change_params"] = change_params

    new_config = MPI.COMM_WORLD.bcast(new_config, root=0)

    '''run the experiment'''
    multi_run(new_config)


# +++++++++++++++++++++++ Primary functions: unit run and multi run +++++++++++++++++++++++ #
def unit_run(config, supercell=None):
    """
    The unit run method, see docs.
    """

    # time space
    time_space = config["experiment_params"]['time_space']
    log_time = config["experiment_params"]["log_time"]
    if not config["experiment_params"].get("custom_time_space", False):
        if len(time_space) != 3:
            raise ValueError("time_space must be [start, stop, num points].")
        if log_time:
            if time_space[0] <= 0 or time_space[1] <= 0:
                raise ValueError("log_time=True requires positive start and stop times.")
            new_time_space = np.logspace(np.log10(time_space[0]), np.log10(time_space[1]), int(time_space[2]))
        else:
            new_time_space = np.linspace(time_space[0], time_space[1], int(time_space[2]))
        config["experiment_params"]['time_space'] = new_time_space
    else:
        if is_root():
            warnings.warn("USING CUSTOM TIME SPACE...")

    # custom bath
    if supercell is not None:
        config["supercell"] = supercell

    # make output directories
    run_id = config["run_id"]  # if single, this is just run_id; if multi it's sub_run_id
    base_out_dir_str = config.get("base_out_dir", "./runs/")  # if single, this is just ~/data/ -->
    base_data_dir_str = config.get("base_data_dir", "./data/")  # if multi it's ~/data/[run_id]/

    out_dir = Path(base_out_dir_str) / run_id
    data_dir = Path(base_data_dir_str) / run_id

    # make directory
    if is_root():
        # check that parent directory is already there
        for base_dir_str in [base_out_dir_str, base_data_dir_str]:
            if not Path(base_dir_str).is_dir():
                msg = f"Base directory {base_dir_str} not found. This directory should already exist."
                logger.error(msg)
                COMM.Abort(1)
        # make the new directory and check that it doesn't already exist
        for new_dir in [out_dir, data_dir]:
            try:
                new_dir.mkdir()
            except FileExistsError:
                msg = f"Directory '{new_dir}' already exists. Delete/rename it or choose a new run_id."
                logger.error(msg)
                COMM.Abort(1)
    COMM.Barrier()

    run_ensemble(config, str(out_dir), str(data_dir))


def multi_run(config):
    """
    The basic multi run method, see docs.
    """

    # have rank 0 initialize the output directories
    multi_run_id = config["run_id"]
    base_out_dir_str = config.get("base_out_dir", "./runs/")
    base_data_dir_str = config.get("base_data_dir", "./data/")
    multi_out_dir = Path(base_out_dir_str) / multi_run_id
    multi_data_dir = Path(base_data_dir_str) / multi_run_id
    multi_out_text = multi_out_dir / "multi_output.txt"

    if is_root():
        # make output directories
        for target_dir, base_dir_str in [(multi_out_dir, base_out_dir_str), (multi_data_dir, base_data_dir_str)]:
            try:
                target_dir.mkdir()
            except FileExistsError:  # error if already exists
                msg = (
                    f"Output directory '{target_dir}' already exists. "
                    "Choose a new run_id or delete/rename the existing folder."
                )
                logger.error(msg)
                MPI.COMM_WORLD.Abort(1)
            except FileNotFoundError:
                msg = f"Base output directory {base_dir_str} not found. This directory should already exist."
                logger.error(msg)
                MPI.COMM_WORLD.Abort(1)
        # make output text file
        with multi_out_text.open("x", encoding="utf-8") as f:  # breaks if it already exists
            f.write("Initialized output text file.\n")
            print_config = copy.deepcopy(config)
            print_config["experiment_params"]["pulses"] = "print too long"
            print_config["multi_params"]["change_params"] = "print too long"
            f.write(f"\nConfig:{print_config}.\n")
    COMM.Barrier()  # ensure all the initializations happen before any rank uses it

    # read in multi-run parameters
    multi_params = config["multi_params"]
    sub_run_ids = multi_params["sub_run_ids"]
    change_params = multi_params["change_params"]

    # do multi-run
    t_start = time.time()
    for sub_run_id, changes in zip(sub_run_ids, change_params):
        '''configure sub run'''
        sub_run_config = copy.deepcopy(config)

        # make the new directory paths
        sub_run_config["run_id"] = sub_run_id
        sub_run_config["base_out_dir"] = str(multi_out_dir)
        sub_run_config["base_data_dir"] = str(multi_data_dir)
        sub_run_config["seed_id"] = config.get("seed_id", multi_run_id)

        # implement changed parameters
        for param, value in changes.items():
            keys = [k.strip() for k in param.split(",")]
            if len(keys) != 2:
                raise ValueError(f"Expected change parameter like 'section,param', got {param!r}")
            section = keys[0]
            name = keys[1]
            if section not in sub_run_config:
                raise KeyError(f"Bad change parameter {param!r}: missing section {section!r}")
            if name not in sub_run_config[section]:
                raise KeyError(
                    f"Bad change parameter {param!r}: section {section!r} has no key {name!r}. "
                    f"Existing keys are: {sorted(sub_run_config[section].keys())}"
                )
            sub_run_config[section][name] = value

        if is_root():
            with multi_out_text.open("a", encoding="utf-8") as f:
                f.write(f"\n\nRank 0 starting sub run: {sub_run_id}")
                # f.write(f"\nChanges: {changes}")
        unit_run(sub_run_config)
        if is_root():
            with multi_out_text.open("a", encoding="utf-8") as f:
                f.write(f"\nRank 0 finished sub run: {sub_run_id}")
                f.write(f"\nRuntime: {get_last_runtime()}s")
    multi_runtime = time.time() - t_start
    if is_root():
        with multi_out_text.open("a", encoding="utf-8") as f:
            f.write(f"\nRank 0 finished multirun. Total rank 0 grid run time: {multi_runtime}s\n")
    COMM.Barrier()  # this may be redundant


# +++++++++++++++++++++++ main method: do not edit +++++++++++++++++++++++ #
def main(config_path):
    config = load_config(config_path)

    # enforce any parameter requirements
    if (str(config["experiment_params"]["cce_type"])[-2:] != "mc"
            and config["simulator_params"].get("polarization", 0) != 0):
        raise Exception("polarization is only supported for mc mode.")
    if ((config["simulator_params"].get("park_2e2n", False) or
         config["simulator_params"].get("generalized_2e2n", False))
         and config["simulator_params"]["order"] != 2):
        raise Exception("order must be 2 for park_2e2n and generalized_2e2n")
    if (config["simulator_params"].get("generalized_2e2n", False)
            and not config["simulator_params"].get("park_2e2n", False)):
        raise Exception("park_2e2n must be on to use generalized 2e_2n")
    if config["run_type"] == "multi":
        if len(config["multi_params"]["sub_run_ids"]) != len(config["multi_params"]["change_params"]):
            raise Exception("sub_run_ids and change_params must have same length")

    run_type = config["run_type"]
    if run_type not in RUN_REGISTRY:
        raise ValueError(f"Unknown run_type {run_type!r}. Valid run_types are: {sorted(RUN_REGISTRY)}")

    method_name = RUN_REGISTRY[run_type]
    method = globals().get(method_name)
    if method is None or not callable(method):
        raise ValueError(f"RUN_REGISTRY maps run_type {run_type!r} to method {method_name!r}, "
            "but that function does not exist in run.py.")

    method(config)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run.py <config.yaml>")
    yaml_path = sys.argv[1]
    main(yaml_path)

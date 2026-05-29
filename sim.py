import sys
import warnings

import numpy as np
import yaml
from pathlib import Path
import time
import copy
import logging
import hashlib
import csv

from pp1 import clean_coherence, average_ensemble, clean_populations

# +++++++++++++++++++++++ Ensemble context +++++++++++++++++++++++ #
_CURRENT_ENSEMBLE = None  # type: int | None


def set_current_ensemble(idx: int | None) -> None:
    """Set the current ensemble index for logging/debug."""
    global _CURRENT_ENSEMBLE
    _CURRENT_ENSEMBLE = idx
    logger.current_ensemble = idx  # also stash on the shared logger so other modules see it (when model.py calls sim)


def get_current_ensemble() -> int | None:  # currently not being used
    """Get the current ensemble index, or None if not set."""
    return _CURRENT_ENSEMBLE


# +++++++++++++++++++++++ Time context +++++++++++++++++++++++ #
_TIME_CONTEXT = {
    "t_start": -1.0,   # seconds
    "t_end": -1.0,     # seconds
    "runtime": -1.0,   # total duration in seconds
}


def _reset_time_context() -> None:
    """Internal helper: clear the time context for a new run."""
    _TIME_CONTEXT["t_start"] = -1.0
    _TIME_CONTEXT["t_end"] = -1.0
    _TIME_CONTEXT["runtime"] = -1.0


def get_time_context():  # currently not used
    """
    Return a shallow copy of timing information for the last run_ensemble call.
    """
    return dict(_TIME_CONTEXT)


def get_last_runtime() -> float:
    """
    Convenience helper: return runtime (seconds) for the last run_ensemble call,
    or -1 if run_ensemble has not successfully completed yet.
    """
    return _TIME_CONTEXT["runtime"]


# +++++++++++++++++++++++ Set up logging +++++++++++++++++++++++ #
LOGGER_NAME = "pycce-logger"
logger = logging.getLogger(LOGGER_NAME)
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1


def is_root():
    return RANK == 0


def setup_run_logger(run_id, run_dir):
    """
    Configure the main logger for this run.

    - Text logs go to <run_dir>/<run_id>.txt (root rank only).
    - logger.save_csv(tag, header, rows) writes CSVs into the same run_dir.
    """
    global logger

    run_dir = Path(run_dir)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if is_root():  # only let rank 0 print out so don't get multiple redundant mpi outputs
        main_log = run_dir / f"output_{run_id}.txt"
        fh = logging.FileHandler(main_log, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())

    # remember context on the logger
    logger.run_id = run_id
    logger.run_dir = str(run_dir)

    def save_csv(tag, header, rows, subdir=None, ignore_mpi=False):
        """
        Write a CSV named <run_id>_<tag>.csv into the run directory.
        Only root rank writes. (unless override by ignore_mpi)
        """
        if not is_root() and not ignore_mpi:
            return

        # choose target directory
        target_dir = run_dir
        if subdir is not None:
            target_dir = run_dir / subdir
            target_dir.mkdir(exist_ok=True)  # safe with MPI, exist_ok=True

        filename = f"{run_id}_{tag}.csv"
        path = target_dir / filename
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            if header is not None:
                writer.writerow(header)
            writer.writerows(rows)

        logger.info("Wrote CSV %s", filename)

    # attach helper method to the logger object
    logger.save_csv = save_csv

    return logger


# +++++++++++++++++++++++ Auxiliary functions +++++++++++++++++++++++ #
def get_seed(seed_string):
    """
    Deterministic 32-bit seed from some string.
    :param seed_string: should be run_id
    :return: a deterministic seed
    """
    h = hashlib.blake2b(seed_string.encode("utf-8"), digest_size=8)  # 64 bits
    return int.from_bytes(h.digest(), "big") & 0xFFFFFFFF


def load_config(path):
    """
    Loads the configuration from a YAML file.
    :param path: string path of the yaml file
    :return: the configuration as a dictionary
    """
    with open(path) as file:
        cfg = yaml.safe_load(file)

    return cfg


# +++++++++++++++++++++++ Primary function: run_ensemble +++++++++++++++++++++++ #
def run_ensemble(config, out_dir_str, data_dir_str):
    """
    Use this method to run experiments. Runs an ensemble of model.run_experiment()'s, cleans and averages the results.
    Given an output `runs' and output `data' directory which already exist, outputs a primary text log to the `runs'
    directory and averaged csv files to `data' directory. Also returns the total_length(t) array. Everything is MPI-compatible.
    :param config: see .yaml file for description
    :param out_dir_str: text output directory
    :param data_dir_str: data output directory
    """
    import model  # local import to avoid circular import with model.py

    # set up configuration
    run_id = config["run_id"]
    supercell_params = config["supercell_params"]
    simulator_params = config["simulator_params"]
    experiment_params = config["experiment_params"]

    ''' set up logging / outputs'''
    out_dir = Path(out_dir_str)
    data_dir = Path(data_dir_str)

    # check that the two directories exist
    if is_root():
        for dir_path, label in [
            (out_dir, "out_dir"),
            (data_dir, "data_dir"),]:
            if not dir_path.exists():
                msg = f"{label} '{dir_path}' not found. This directory should already exist."
                logger.error(msg)
                COMM.Abort(1)
            if not dir_path.is_dir():
                msg = f"{label} '{dir_path}' exists but is not a directory."
                logger.error(msg)
                COMM.Abort(1)
    COMM.Barrier()

    # print stuff into output text file
    setup_run_logger(run_id, out_dir)
    if is_root():
        logger.info("Initialized output for run %s", run_id)
        logger.info("Log directory: %s", out_dir)
        logger.info("Data directory: %s", data_dir)
        print_config = copy.deepcopy(config)
        print_config["experiment_params"]["pulses"] = "print too long"
        if "multi_params" in print_config.keys():
            print_config["multi_params"]["change_params"] = "print too long"
        logger.info("Config: %s", print_config)

    # initialize time context for this run
    _reset_time_context()
    t_start = time.time()
    temp_time = t_start  # so we can record how long EACH ensemble run takes

    '''run ensemble of experiments'''
    time_count = len(experiment_params["time_space"])
    coherence = np.empty((time_count, experiment_params["ensemble_size"]),
        dtype=np.complex128)  # empty coherence array

    base_seed = get_seed(config.get("seed_id", run_id))
    for i in range(experiment_params["ensemble_size"]):
        if is_root():
            logger.info("Starting ensemble experiment %d", i+1)
        set_current_ensemble(i+1)  # or i + 1 if you want 1-based
        new_supercell_params = copy.deepcopy(supercell_params)
        # new seed each loop, deterministic though so separate mpi runs will produce the same supercell
        new_supercell_params["seed"] = (base_seed + i) & 0xFFFFFFFF

        # handle custom bath
        if supercell_params.get("custom_bath", False):
            supercell = config["supercell"]
            nv = model.get_single_nv(supercell_params["nv_position"], supercell_params["alpha"],
                                     supercell_params["beta"])
        else:
            supercell, nv = model.get_supercell(new_supercell_params)
        simulator = model.get_simulator(supercell, nv, simulator_params)

        # run the experiment
        traj = model.run_experiment(simulator, experiment_params)
        if traj.shape != (time_count,):
            raise ValueError(f"Expected traj shape {(time_count,)}, got {traj.shape}")
        coherence[:, i] = traj
        if is_root():
            logger.info("Rank 0 finished ensemble experiment %d in %.2f s", i+1, time.time() - temp_time)
            temp_time = time.time()

    # update time context
    t_end = time.time()
    runtime = t_end - t_start  # seconds
    _TIME_CONTEXT["t_start"] = t_start
    _TIME_CONTEXT["t_end"] = t_end
    _TIME_CONTEXT["runtime"] = runtime
    if is_root():
        logger.info("Rank 0 finished full ensemble run %s in %.2f s", run_id, runtime)

    # Only root rank does postprocessing and filesystem writes
    if is_root():
        if config["experiment_params"].get("populations", False):
            # clean coherence data
            coherence_cleaned, n_removed = clean_populations(coherence)
            logger.info("Number of unstable points was %d", n_removed)
            if n_removed > coherence_cleaned.size * 0.2:
                warnings.warn("Number of unstable points exceed 20\% !")
        else:
            # clean coherence data
            coherence_cleaned, n_removed = clean_coherence(coherence)
            logger.info("Number of unstable points was %d", n_removed)
            if n_removed > coherence_cleaned.size*0.2:
                warnings.warn("Number of unstable points exceed 20\% !")

        # save raw, cleaned csv files
        csv_path = data_dir / f"raw.csv"
        np.savetxt(csv_path, coherence_cleaned, delimiter=",")
        logger.info("Wrote raw (cleaned) CSV to %s", csv_path)

        # average results over the ensemble
        coherence_avg = average_ensemble(coherence_cleaned, avg_method=config["experiment_params"]["avg_method"])

        if experiment_params.get("add_zero", True):
            # add in the point (0.0, 1.0) if it is not there
            time_space = np.asarray(config["experiment_params"]["time_space"], dtype=float)
            coherence_avg = np.asarray(coherence_avg, dtype=np.complex128)
            if not np.any(np.isclose(time_space, 0.0, atol=1e-20, rtol=0.0)):
                time_space = np.insert(time_space, 0, 0.0)
                coherence_avg = np.insert(coherence_avg, 0, 1.0 + 0.0j)
            config["experiment_params"]["time_space"] = time_space  # this assumes all postprocessing done on root

        # split averaged complex coherence into magnitude and phase
        coherence_avg_mag = np.abs(np.asarray(coherence_avg, dtype=np.complex128))
        coherence_avg_phase = np.angle(np.asarray(coherence_avg, dtype=np.complex128))

        # save averaged csv files
        averaged = np.column_stack((config["experiment_params"]["time_space"], coherence_avg))
        csv_path_averaged = data_dir / f"averaged.csv"
        np.savetxt(csv_path_averaged, averaged, delimiter=",", header="t,L_avg,beta", comments="")
        logger.info("Wrote final averaged CSV to %s", csv_path_averaged)

        # same thing for the modulus and phase...
        magnitude = np.column_stack((config["experiment_params"]["time_space"], coherence_avg_mag))
        csv_path_mag = data_dir / f"averaged_mag.csv"
        np.savetxt(csv_path_mag, magnitude, delimiter=",", header="t,modulus(L_avg)", comments="")
        logger.info("Wrote final averaged modulus CSV to %s", csv_path_mag)
        phase = np.column_stack((config["experiment_params"]["time_space"], coherence_avg_phase))
        csv_path_phase = data_dir / f"averaged_phase.csv"
        np.savetxt(csv_path_phase, phase, delimiter=",", header="t,L_avg", comments="")
        logger.info("Wrote final averaged phase CSV to %s", csv_path_phase)

import sys
import numpy as np
import yaml
from pathlib import Path
import time
import copy
import logging
import hashlib
import csv

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

    # fix any non-Pythonic yaml variables
    time_space = cfg["experiment_params"].pop("time_space")
    cfg["experiment_params"]["time_space"] = np.linspace(time_space[0], time_space[1], time_space[2])

    return cfg


# +++++++++++++++++++++++ Primary function: run_ensemble +++++++++++++++++++++++ #
def run_ensemble(config):
    """
    Use this method to run experiments. (Rather than calling anything from model.py directly.) Runs many simulations
    each with a unique bath configuration. A glorified for loop of model.run_experiment. Given a base output directory
    which already exists (should be './runs/'), creates a new directory in the base output directory named run_id.
    Outputs a primary text log, primary csv files matching the return dictionary, and optional checkpoint csv files.
    :param config: see .yaml file for description
    :return: A dictionary where the keys are cce_types (see .yaml) and the values are 2D arrays having shape
    (timesteps, ensemble_size). Each column is one experiment and each rows is a time point. Each element is L(t) from
    0 to 1.
    """
    import model  # local import to avoid circular import with model.py

    # set up configuration
    run_id = config["run_id"]
    base_out_dir_str = config["base_out_dir"]
    ensemble_size = config["ensemble_size"]
    supercell_params = config["supercell_params"]
    simulator_params = config["simulator_params"]
    experiment_params = config["experiment_params"]

    # set up logging / outputs
    run_dir = Path(base_out_dir_str) / run_id
    if is_root():  # only root does the mkdir
        try:
            run_dir.mkdir()  # makes the output directory
        except FileExistsError:  # error if already exists
            msg = (
                f"Run directory '{run_dir}' already exists. "
                "Choose a new run_id or delete/rename the existing folder."
            )
            logger.error(msg)
            MPI.COMM_WORLD.Abort(1)
        except FileNotFoundError:
            msg = f"Base output directory {base_out_dir_str} not found. This directory should already exist."
            logger.error(msg)
            MPI.COMM_WORLD.Abort(1)
    COMM.Barrier()  # ensure run_dir exists before any rank uses it
    run_dir = str(run_dir)
    setup_run_logger(run_id, Path(run_dir))
    if is_root():
        logger.info("Initialized output for run %s in %s", run_id, run_dir)
        logger.info("Config: %s", config)

    # initialize time context for this run
    _reset_time_context()
    t_start = time.time()
    temp_time = t_start  # so we can record how long EACH ensemble run takes

    # run ensemble of experiments
    coherence_dict = {cce_type: np.empty((len(experiment_params["time_space"]), ensemble_size))
                      for cce_type in experiment_params["cce_types"]}
    base_seed = get_seed(run_id)
    for i in range(ensemble_size):
        if is_root():
            logger.info("Starting ensemble experiment %d", i+1)
        set_current_ensemble(i+1)  # or i + 1 if you want 1-based

        new_supercell_params = copy.deepcopy(supercell_params)  #  copy so we can tweak per-run
        new_supercell_params["seed"] = (base_seed + i) & 0xFFFFFFFF  # new seed each loop, deterministic though so
                                                                     # separate mpi runs will produce the same supercell

        supercell = model.get_supercell(new_supercell_params)
        simulator = model.get_simulator(supercell, simulator_params)
        coherence = model.run_experiment(simulator, experiment_params)

        if set(coherence.keys()) != set(coherence_dict.keys()):
            raise RuntimeError(f"Mismatch between ensemble cce_types {set(coherence_dict.keys())} and run_experiment "
                               f"output cce_types {set(coherence.keys())}")
        for cce_type, traj in coherence.items():
            coherence_dict[cce_type][:, i] = traj
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

    # make csv files
    if is_root():
        final_csv_dir = Path(run_dir) / "final_csv_files"
        try:
            final_csv_dir.mkdir()  # error if already exists
        except FileExistsError:
            raise RuntimeError(f"Final CSV directory '{final_csv_dir}' already exists. "
                               "Choose a new run_id or delete/rename the existing folder.")
        for cce_type, arr in coherence_dict.items():
            # each key gives one CSV file: <key>.csv
            csv_path = final_csv_dir / f"{cce_type}.csv"
            np.savetxt(csv_path, arr, delimiter=",")
            logger.info("Wrote final CSV for %s to %s", cce_type, csv_path)

    return coherence_dict


def main(config_path):
    coherence_dict = run_ensemble(load_config(config_path))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python sim.py <config.yaml>")
    yaml_path = sys.argv[1]
    main(yaml_path)

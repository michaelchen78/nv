import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import copy
from mpi4py import MPI
from sim import run_ensemble, load_config, get_last_runtime, logger, is_root, COMM
from config import RUN_REGISTRY


# +++++++++++++++++++++++ Custom run functions +++++++++++++++++++++++ #
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
    T_max = cpmg_params['T_max']
    start = cpmg_params['start']
    delta = cpmg_params['delta']
    n_list = cpmg_params['n_list']
    # optional, defaults
    logspace = cpmg_params.get("logspace", None)
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

            if logspace is not None:
                change_param_dict["experiment_params,time_space"] = np.logspace(np.log10(start), np.log10(T_max),
                                                                                logspace)
            else:
                steps = int(round(T_max/delta)) + 1
                change_param_dict["experiment_params,time_space"] = np.linspace(start, T_max, steps)

            change_param_dict["experiment_params,pulse_id"] = int(n)

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
    if len(time_space) != 3:
        raise ValueError("time_space must be [start, stop, num points].")
    if log_time:
        if time_space[0] <= 0 or time_space[1] <= 0:
            raise ValueError("log_time=True requires positive start and stop times.")
        new_time_space = np.logspace(np.log10(time_space[0]), np.log10(time_space[1]), int(time_space[2]))
    else:
        new_time_space = np.linspace(time_space[0], time_space[1], int(time_space[2]))
    config["experiment_params"]['time_space'] = new_time_space

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
            f.write(f"\nConfig:{config}.\n")
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
            sub_run_config[section][name] = value

        if is_root():
            with multi_out_text.open("a", encoding="utf-8") as f:
                f.write(f"\n\nRank 0 starting sub run: {sub_run_id}")
                f.write(f"\nChanges: {changes}")
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

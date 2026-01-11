import sys

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import time
import copy

from mpi4py import MPI
from numba.cuda.types import grid_group

from preprocessing1 import average_ensemble, fit_curve_basic, plot_coherence_panel
from sim import run_ensemble, load_config, get_last_runtime, logger, is_root, COMM


def run_avg_fit_plot(config, grid_axes=None):
    """
    A utility method to conveniently run sim.py/sim.sh and then also average, fit, and plot the data.
    :param config: the run configuration, see any yaml file. put straight into sim.run_ensemble.
    :param grid_axes: if in grid mode, the axes to plot on
    :return: Nothing, but the method will generate averaged csv files into final_csv_files/, an output text file with
    fit results, and plots. Everything is put into the base output directory which should be ./runs/{run_id}/.
    """
    # run the experiment
    coherence_dict = run_ensemble(config)
    runtime = get_last_runtime()

    # Only root rank does postprocessing, filesystem writes
    if not is_root():
        return

    # just to improve readability...
    time_space = config["experiment_params"]["time_space"]
    run_id = config["run_id"]
    cce_types = config["experiment_params"]["cce_types"]

    # average results over the ensemble
    coherence_dict_avg = average_ensemble(coherence_dict, avg_method=config["avg_method"])
    # save averaged csv files
    final_csv_dir = Path(config["base_out_dir"]) / run_id / "final_csv_files"
    assert final_csv_dir.is_dir(), f"Directory does not exist: {final_csv_dir}"
    for cce_type, arr in coherence_dict_avg.items():
        data = np.column_stack((time_space, arr))
        csv_path = final_csv_dir / f"{cce_type}_averaged.csv"
        np.savetxt(csv_path, data, delimiter=",", header="t, L_avg", comments="")
        logger.info("Wrote final averaged CSV for %s to %s", cce_type, csv_path)

    # fit the curve and write to output
    fit_results = fit_curve_basic(coherence_dict_avg, time_space)
    logger.info("run_avg_fit_plot fit results for run_id=%s:", run_id)
    for cce_type, (T2, p) in fit_results.items():
        logger.info("%s fit results --> T2 (ms), p: (%g, %g)", cce_type, T2, p)

    # plot the results
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_coherence_panel(ax, time_space=time_space, coherence_dict_avg=coherence_dict_avg, config=config,
                         cce_types=cce_types, title=run_id, fit_results=fit_results, runtime=runtime, fontsize=12)
    fig.tight_layout()
    fig.savefig(fname=f"{config['base_out_dir']}/{run_id}/plot.png", dpi=300)
    plt.close(fig)

    if config["is_grid"]:
        plot_coherence_panel(grid_axes, time_space=time_space, coherence_dict_avg=coherence_dict_avg, config=config,
                         cce_types=cce_types, title=run_id, fit_results=fit_results, runtime=runtime, fontsize=12)


def grid_search(base_config, param1, param2, param1_values, param2_values):
    """
    Utility method to conveniently runs a grid of experiments using the method run_avg_fit_plot. Also returns a figure
    with all results. Each experiment has its own output folder, as created by run_avg_fit_plot.
    :param base_config: the base config, see any yaml file
    :param param1: a list of two strings, the first gives the sub-config (e.g., supercell_params or experiment_params)
    and the second gives the specific parameter, for instance size or pulse_id.
    :param param2: same as param1
    :param param1_values: a list, each element is some value of param1
    :param param2_values: same as param1
    :return: nothing
    """
    run_id = base_config["run_id"]
    base_out_dir_str = base_config["base_out_dir"]

    # initialize grid figure
    n_rows = len(param1_values)
    n_cols = len(param2_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:  # normalize axes to a 2D array so the grid figure works
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # have rank 0 initialize the outputs
    grid_out_dir = Path(base_out_dir_str) / run_id
    grid_out_text = None
    if is_root():
        # make output directory
        try:
            grid_out_dir.mkdir()  # makes the output directory
        except FileExistsError:  # error if already exists
            msg = (f"Run directory '{grid_out_dir}' already exists. "
                    "Choose a new run_id or delete/rename the existing folder.")
            logger.error(msg)
            MPI.COMM_WORLD.Abort(1)
        except FileNotFoundError:
            msg = f"Base output directory {base_out_dir_str} not found. This directory should already exist."
            logger.error(msg)
            MPI.COMM_WORLD.Abort(1)

        # make output text file
        grid_out_text = grid_out_dir / "grid-output.txt"
        with grid_out_text.open("x", encoding="utf-8") as f:  # breaks if it already exists
            f.write("Initialized output text file.\n")
            f.write(f"\nBase config:{base_config}.\n")
            f.write(f"\nGrid: param1 is {param1}, param2 is {param2}")
            f.write(f"\nparam1 range is {param1_values}, param2 range is {param2_values}\n\n")
    COMM.Barrier()  # ensure all the initializations happen before any rank uses it

    # do the grid search
    grid_t_start = time.time()
    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            if is_root():
                with grid_out_text.open("a", encoding="utf-8") as f:
                    f.write(f"\nRank 0 starting grid run: {param1[1]}={v1}, {param2[1]}={v2}")

            # first deep copy the base config
            grid_run_config = copy.deepcopy(base_config)

            # then set the two parameter values to their grid values
            grid_run_config[param1[0]][param1[1]] = v1
            grid_run_config[param2[0]][param2[1]] = v2

            # set things up so output files are put in the correct directories
            grid_run_id = f"{param1[1]}-{v1}_{param2[1]}-{v2}"
            grid_run_config["run_id"] = grid_run_id
            grid_run_config["base_out_dir"] = str(grid_out_dir)

            # run the experiment
            ax = axes[i, j]
            run_avg_fit_plot(grid_run_config, grid_axes=ax)
            ax.set_title(f"{param1[1]}={v1}, {param2[1]}={v2}")

            if is_root():
                with grid_out_text.open("a", encoding="utf-8") as f:
                    f.write(f"\nRank 0 finished grid run: {param1[1]}={v1}, {param2[1]}={v2}\n")
                    f.write(f"Rank 0 runtime: {get_last_runtime()}s\n\n")
    grid_runtime = time.time() - grid_t_start
    if is_root():
        with grid_out_text.open("a", encoding="utf-8") as f:
            f.write(f"\nRank 0 finished grid. Total rank 0 grid run time: {grid_runtime}s\n")
    COMM.Barrier()  # this is redundant

    # Only root rank makes the figure
    if not is_root():
        return
    fig.suptitle(f"Coherence: grid over {param1[1]} and {param2[1]}", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(grid_out_dir / f"grid_{param1[1]}_{param2[1]}.png", dpi=300)
    plt.close(fig)


def main(config_path):
    config = load_config(config_path)

    if config['is_grid']:
        grid_dict = config['grid_params']
        grid_search(config, **grid_dict)
    else:
        run_avg_fit_plot(config)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run.py <config.yaml>")
    yaml_path = sys.argv[1]
    main(yaml_path)



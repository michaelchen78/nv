# +++++++++++++++++++++++ Re-plotting from saved data +++++++++++++++++++++++ #
import copy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from sim import load_config, fit_curve_basic, plot_coherence_panel


def replot_grid_conv_only(config_path, grid_dict, save_suffix="_conv-only"):
    """
    [Chat GPT wrote this function.] Rebuild a grid figure using ONLY the conventional CCE averaged results
    from existing CSV files. No simulations are re-run; everything is loaded
    from the *_conv_averaged.csv files produced by grid_search/run_coherence_experiment.

    :param config_path: path to the same YAML file used for the original grid run
    :param grid_dict: dictionary with keys
        "param1", "param2", "param1_values", "param2_values"
        exactly as in the original grid_search call
    :param save_suffix: string appended to the grid png file name so the
        original figure is not overwritten
    :return: nothing
    """
    # load the original base config just to get run_id and other params
    base_config = load_config(config_path)
    base_run_id = base_config["run_id"]

    # where grid_search put everything: ./output/<run_id>/
    # (this matches grid_search: make_output_dir(run_id) with default base="output")
    grid_root = Path("output") / base_run_id
    if not grid_root.is_dir():
        raise RuntimeError(f"Grid root directory '{grid_root}' does not exist.")

    param1 = grid_dict["param1"]
    param2 = grid_dict["param2"]
    param1_values = grid_dict["param1_values"]
    param2_values = grid_dict["param2_values"]

    n_rows = len(param1_values)
    n_cols = len(param2_values)

    # initialize grid figure (same style as grid_search)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=True,
        sharey=True
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            grid_run_id = f"{param1[1]}={v1}--{param2[1]}={v2}"
            run_dir = grid_root / grid_run_id

            if not run_dir.is_dir():
                raise RuntimeError(f"Run directory '{run_dir}' does not exist.")

            # load conventional averaged coherence: t, avg_L
            conv_file = run_dir / f"{grid_run_id}_conv_averaged.csv"
            if not conv_file.is_file():
                raise RuntimeError(f"Cannot find '{conv_file}'.")

            data = np.loadtxt(conv_file, delimiter=",", skiprows=1)
            time_space = data[:, 0]
            conv_L = data[:, 1]

            # shape (num_time_steps, 1) so it still works with fit_curve_basic/plot_coherence_panel
            avg_results = conv_L.reshape(-1, 1)

            # fit T2, p for conventional only
            T2, p = fit_curve_basic(avg_results, time_space, cce_type="conv")

            # try to recover runtime from the text file (optional)
            runtime = None
            info_path = run_dir / f"{grid_run_id}.txt"
            if info_path.is_file():
                with info_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("Runtime:"):
                            # line like: "Runtime:123.45 s"
                            try:
                                runtime_str = line.split("Runtime:")[1].strip().split()[0]
                                runtime = float(runtime_str)
                            except Exception:
                                runtime = None
                            break

            # construct a config for the text overlay in the panel
            panel_config = copy.deepcopy(base_config)
            panel_config[param1[0]][param1[1]] = v1
            panel_config[param2[0]][param2[1]] = v2
            panel_config["compute_params"]["time_space"] = time_space

            ax = axes[i, j]
            plot_coherence_panel(
                ax,
                time_space=time_space,
                result=avg_results,
                cce_types=["conv"],          # <<< ONLY conventional plotted
                title=f"{param1[1]}={v1}, {param2[1]}={v2}",
                T2=T2,
                p=p,
                runtime=runtime,
                config=panel_config,
            )

    fig.suptitle(f"Coherence: grid over {param1[1]} and {param2[1]} (conv only)", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out_path = grid_root / f"new-grid_{param1[1]}-{param2[1]}{save_suffix}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved conventional-only grid figure to: {out_path}")


def main():
    grid_dict = {
        "param1": ["model_params", "size"],
        "param2": ["simulator_params", "r_dipole"],
        "param1_values": [50, 100, 150, 200],
        "param2_values": [4, 6, 10, 20]
    }
    replot_grid_conv_only("config/5.dec.2025_grid-size-r-dipole-0p1.yaml", grid_dict, save_suffix="_conv-only")



if __name__ == "__main__":
    main()
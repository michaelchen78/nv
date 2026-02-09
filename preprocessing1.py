from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import re  # put this near the top of the file with the other imports


def average_ensemble(coherence_dict, avg_method):
    """
    Expects a dictionary where the keys are cce_types (see yaml), and the values are 2D arrays each with shape
    (num time_steps, ensemble_size), as is the output from the function sim.run_ensemble. This will average over the
    ensemble using the method identified. For now there is just a simple mean and median.
    :param coherence_dict: the output of sim.run_ensemble
    :param avg_method: a string identifying the averaging method, straight from the yaml config file
    :return: the averaged ensemble: a dictionary where the keys are cce types and the values are 1D arrays, coherence
    trajectories
    """
    # ------------------------------------------------------------------ #
    # bunch of bs data validation
    # ------------------------------------------------------------------ #
    if not coherence_dict:
        raise ValueError("`results` dictionary is empty; nothing to average.")

    cce_types = list(coherence_dict.keys())

    first_key = cce_types[0]
    first_arr = coherence_dict[first_key]
    if first_arr is None:
        raise ValueError(f"Result for '{first_key}' is None; cannot average.")

    first_arr = np.asarray(first_arr)
    if first_arr.ndim != 2:
        raise ValueError(
            f"Result for '{first_key}' must be 2D (timesteps, ensemble_size); "
            f"got shape {first_arr.shape}."
        )
    t, n = first_arr.shape

    # Check all arrays have the same shape and are non-None
    for k in cce_types[1:]:
        arr = coherence_dict[k]
        if arr is None:
            raise ValueError(f"Result for '{k}' is None; cannot average.")
        arr = np.asarray(arr)
        if arr.shape != (t, n):
            raise ValueError(
                f"Inconsistent shape for key '{k}': expected {(t, n)}, "
                f"got {arr.shape}."
            )

    # ------------------------------------------------------------------ #
    # average the results
    # ------------------------------------------------------------------ #
    if avg_method == "mean":
        averaged = {
            k: np.mean(np.asarray(arr), axis=1)  # average over ensemble axis
            for k, arr in coherence_dict.items()
        }
    elif avg_method == "median":
        averaged = {
            k: np.median(np.asarray(arr), axis=1)  # median over ensemble axis
            for k, arr in coherence_dict.items()
        }
    else:
        raise Exception(f"Unknown averaging method '{avg_method}'")

    return averaged


def fit_curve_basic(coherence_dict_avg, time_space):
    """
    Fit coherence L(t) to the stretched exponential exp(-(t/T2)^p) for every cce_type
    :param coherence_dict_avg: dictionary with keys cce_types and values L(t)
    :param time_space: a numpy time space lining up with results
    :return: dict with keys cce_types and values tuples (T2, p)
    """
    # initial guess and bounds
    p0 = (0.1, 1.5)
    bounds = ([1e-6, 0.5], [1e6, 3.0])

    def stretch(t, T2, p):
        return np.exp(- (t / T2) ** p)

    fit_results = {}
    for cce_type, y in coherence_dict_avg.items():
        y = np.asarray(y)
        if y.shape != time_space.shape:
            raise ValueError(
                f"Shape mismatch for cce_type '{cce_type}': "
                f"y.shape = {y.shape}, time_space.shape = {time_space.shape}"
            )

        # keep only finite points
        mask = np.isfinite(y)
        if mask.sum() < 3:
            # not enough good points to fit – mark and move on
            fit_results[cce_type] = (np.nan, np.nan)
            continue
        try:
            # IMPORTANT: we only pass the finite subset to curve_fit
            res = curve_fit(stretch, time_space[mask], y[mask], p0=p0, bounds=bounds)
            T2, p = res[0]
        except Exception:
            # any error inside curve_fit (including "array must not contain infs or NaNs")
            # is caught here and converted to NaNs
            raise ValueError("This broke on cce_type, run_id" + cce_type)
            T2, p = np.nan, np.nan

        fit_results[cce_type] = (T2, p)

    return fit_results


def plot_coherence_panel(ax, time_space, coherence_dict_avg, cce_types, title=None, fit_results=None, runtime=None,
                         config=None, fontsize=5):
    """
    Draw a coherence plot on the given axes. May plot multiple trajectories.
    :param ax: axes to plot on
    :param time_space: numpy time space lining up with coherence_dict_avg
    :param coherence_dict_avg: a dictionary where the keys are strings (cce_types) and the values are 1D arrays,
    coherence trajectories; this function plots all of these trajectories
    :param cce_types: the CCE methods used
    :param title: title of plot
    :param fit_results: dictionary with keys cce_types and values tuples (T2, p)
    :param runtime: how long it took to run
    :param config: config file, in case want to print more info on the plots
    :return: nothing
    """
    # plot each CCE curve
    for cce_type in cce_types:
        ax.plot(time_space, coherence_dict_avg[cce_type], label=cce_type)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Coherence")

    if title:
        ax.set_title(title)

    # write useful information on the plots
    text_lines = []
    if config is not None:
        text_lines.append(f"size = {config['supercell_params']['size']}, p1_conc = {config['supercell_params']['p1_conc']}")
        text_lines.append(f"order = {config['simulator_params']['order']}, r_bath = "
                          f"{config['simulator_params']['r_bath']}, r_dipole = {config['simulator_params']['r_dipole']}")
        text_lines.append(f"ensemble = {config['experiment_params']['ensemble_size']}, magnetic field = "
                          f"{config['experiment_params']['magnetic_field']}")
    if fit_results is not None:
        for cce_type, (T2, p) in fit_results.items():
            text_lines.append(f"{cce_type} fit results --> T2 (ms), p: ({T2:.3f}, {p:.3f})")
    if runtime is not None:
        text_lines.append(f"rank 0 runtime= {runtime:.2f} s")
    if text_lines:
        ax.text(0.99, 0.85, "\n".join(text_lines), transform=ax.transAxes, va="top", ha="right", fontsize=fontsize)
    ax.legend()


def plot_coherence_csv(csv_path: str | Path):
    """
    csv_path: path to a CSV with 2 columns: time, coherence
    """
    csv_path = Path(csv_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)  # add skiprows=1 if there's a header

    t = data[:, 0]
    L = data[:, 1]

    fig, ax = plt.subplots()
    ax.plot(t, L)
    ax.set_xlabel("Time")
    ax.set_ylabel("Coherence")
    ax.set_title(csv_path.name)
    ax.grid(True)

    plt.show()

    return fig, ax


def plot_cpmg(
    grid_root: str | Path,
    csv_name: str = "gcce_averaged.csv",
    transform: str = "neglog",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    disclaimer- used AI to help write this method, will try to improve it soon
    Build a 2D τ–t_d heatmap from a grid run over many values of N.

    Assumes directory structure like:

        grid_root/
            pulse_id-1_dummy_var-dummy_val/
                final_csv_files/gcce_averaged.csv
            pulse_id-50_dummy_var-dummy_val/
                final_csv_files/gcce_averaged.csv
            ...

    Each gcce_averaged.csv must have two columns:
        τ (ms),  L(τ)

    and the simulation used:
        pulses = N
        time_space = np.linspace(1.5e-3, 16.5e-3, 376) [something like this]

    Parameters
    ----------
    grid_root : str | Path
        Root directory containing all the pulse_id-* subdirectories
        (e.g. "./discovery_runs/2026.1.20_cpmg-grid").
    csv_name : str
        File name inside final_csv_files/ (default "gcce_averaged.csv").
    transform : {"neglog", "abs", "real"}
        How to map coherence -> color:
            "abs"    : |L|
            "real"   : Re(L)
            "neglog" : -log10( max(1 - |L|, 1e-6) )  (nice 0–6 style scale)
    save_path : str | Path | None
        Where to save the figure. If None, saves to
        grid_root / "gcce_2d_diagonal.png".
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig, ax
    """
    grid_root = Path(grid_root)
    if not grid_root.is_dir():
        raise NotADirectoryError(f"{grid_root} is not a directory")

    # ------------------------------------------------------------------ #
    # Find all run dirs and parse N from names "pulse_id-N..."
    # ------------------------------------------------------------------ #
    run_info = []  # list of (N, run_dir, csv_path)
    for d in grid_root.iterdir():
        if not d.is_dir():
            continue
        m = re.search(r"pulse_id-(\d+)", d.name)
        if not m:
            continue
        N = int(m.group(1))
        csv_path = d / "final_csv_files" / csv_name
        if csv_path.is_file():
            run_info.append((N, d, csv_path))
    print("run_info for 2d plot: ", run_info)

    if not run_info:
        raise RuntimeError(
            f"No run directories with pattern 'pulse_id-N' and {csv_name} "
            f"found under {grid_root}"
        )

    # sort by N so y-axis is increasing pulses / t_d
    run_info.sort(key=lambda x: x[0])

    tau_ms_master = None
    Z_rows = []
    Ns = []

    # ------------------------------------------------------------------ #
    # Load each gcce_averaged.csv and build color rows
    # ------------------------------------------------------------------ #
    for N, run_dir, csv_path in run_info:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"CSV {csv_path} must have at least two columns")

        t_ms = data[:, 0]  # this is τ in ms
        L = data[:, 1]

        if tau_ms_master is None:
            tau_ms_master = t_ms
        else:
            if (
                t_ms.shape != tau_ms_master.shape
                or not np.allclose(t_ms, tau_ms_master, rtol=1e-8, atol=1e-12)
            ):
                raise ValueError(
                    f"Time grid mismatch between {csv_path} and previous runs.\n"
                    f"Make sure time_space is the same τ grid for all N."
                )

        # transform coherence -> color
        if transform == "abs":
            z = np.abs(L)
        elif transform == "real":
            z = np.real(L)
        elif transform == "neglog":
            eps = 1e-6
            z = -np.log10(np.clip(1.0 - np.abs(L), eps, 1.0))
            z = -np.log(np.abs(L))
        else:
            raise ValueError(f"Unknown transform '{transform}'")

        Z_rows.append(z)
        Ns.append(N)

    Z = np.vstack(Z_rows)             # shape (n_runs, n_tau)
    Ns = np.asarray(Ns, dtype=float)  # pulses for each row

    # ------------------------------------------------------------------ #
    # Build τ–t_d grids (in µs) and plot
    # ------------------------------------------------------------------ #
    tau_ms = np.asarray(tau_ms_master)
    tau_us = tau_ms * 1e3

    Tau_grid = np.tile(tau_us, (len(Ns), 1))  # same τ along each row
    Td_grid = 2.0 * Ns.reshape(-1, 1) * Tau_grid  # t_d = 2 N τ, in µs

    fig, ax = plt.subplots(figsize=(10, 4))

    pcm = ax.pcolormesh(Tau_grid, Td_grid, Z, shading="auto")
    ax.set_yscale("log")

    ax.set_xlabel(r"$\tau$ ($\mu$s)")
    ax.set_ylabel(r"Time of echo $t_d$ ($\mu$s)")

    if transform == "neglog":
        cbar_label = r"$-\log_{10}(1-|L|)$"
    elif transform == "abs":
        cbar_label = r"$|L|$"
    else:
        cbar_label = r"$L$"

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()

    if save_path is None:
        save_path = grid_root / "gcce_2d_diagonal.png"
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved 2D gcce heatmap to: {save_path}")
    return fig, ax


# def replot_grid_conv_only(config_path, grid_dict, save_suffix="_conv-only"):
#     """
#     [Chat GPT wrote this function.] Rebuild a grid figure using ONLY the conventional CCE averaged results
#     from existing CSV files. No simulations are re-run; everything is loaded
#     from the *_conv_averaged.csv files produced by grid_search/run_coherence_experiment.
#
#     :param config_path: path to the same YAML file used for the original grid run
#     :param grid_dict: dictionary with keys
#         "param1", "param2", "param1_values", "param2_values"
#         exactly as in the original grid_search call
#     :param save_suffix: string appended to the grid png file name so the
#         original figure is not overwritten
#     :return: nothing
#     """
#     # load the original base config just to get run_id and other params
#     base_config = load_config(config_path)
#     base_run_id = base_config["run_id"]
#
#     # where grid_search put everything: ./output/<run_id>/
#     # (this matches grid_search: make_output_dir(run_id) with default base="output")
#     grid_root = Path("./runs") / base_run_id # FICX THISASKJDFNAJDF
#     if not grid_root.is_dir():
#         raise RuntimeError(f"Grid root directory '{grid_root}' does not exist.")
#
#     param1 = grid_dict["param1"]
#     param2 = grid_dict["param2"]
#     param1_values = grid_dict["param1_values"]
#     param2_values = grid_dict["param2_values"]
#
#     n_rows = len(param1_values)
#     n_cols = len(param2_values)
#
#     # initialize grid figure (same style as grid_search)
#     fig, axes = plt.subplots(
#         n_rows, n_cols,
#         figsize=(5 * n_cols, 4 * n_rows),
#         sharex=True,
#         sharey=True
#     )
#     if n_rows == 1 and n_cols == 1:
#         axes = np.array([[axes]])
#     elif n_rows == 1:
#         axes = axes[np.newaxis, :]
#     elif n_cols == 1:
#         axes = axes[:, np.newaxis]
#
#     for i, v1 in enumerate(param1_values):
#         for j, v2 in enumerate(param2_values):
#             grid_run_id = f"{param1[1]}={v1}--{param2[1]}={v2}"
#             run_dir = grid_root / grid_run_id
#
#             if not run_dir.is_dir():
#                 raise RuntimeError(f"Run directory '{run_dir}' does not exist.")
#
#             # load conventional averaged coherence: t, avg_L
#             conv_file = run_dir / f"{grid_run_id}_conv_averaged.csv"
#             if not conv_file.is_file():
#                 raise RuntimeError(f"Cannot find '{conv_file}'.")
#
#             data = np.loadtxt(conv_file, delimiter=",", skiprows=1)
#             time_space = data[:, 0]
#             conv_L = data[:, 1]
#
#             # shape (num_time_steps, 1) so it still works with fit_curve_basic/plot_coherence_panel
#             avg_results = conv_L.reshape(-1, 1)
#
#             # fit T2, p for conventional only
#             T2, p = fit_curve_basic(avg_results, time_space, cce_type="conv")
#
#             # try to recover runtime from the text file (optional)
#             runtime = None
#             info_path = run_dir / f"{grid_run_id}.txt"
#             if info_path.is_file():
#                 with info_path.open("r", encoding="utf-8") as f:
#                     for line in f:
#                         if line.startswith("Runtime:"):
#                             # line like: "Runtime:123.45 s"
#                             try:
#                                 runtime_str = line.split("Runtime:")[1].strip().split()[0]
#                                 runtime = float(runtime_str)
#                             except Exception:
#                                 runtime = None
#                             break
#
#             # construct a config for the text overlay in the panel
#             panel_config = copy.deepcopy(base_config)
#             panel_config[param1[0]][param1[1]] = v1
#             panel_config[param2[0]][param2[1]] = v2
#             panel_config["compute_params"]["time_space"] = time_space
#
#             ax = axes[i, j]
#             plot_coherence_panel(
#                 ax,
#                 time_space=time_space,
#                 result=avg_results,
#                 cce_types=["conv"],          # <<< ONLY conventional plotted
#                 title=f"{param1[1]}={v1}, {param2[1]}={v2}",
#                 T2=T2,
#                 p=p,
#                 runtime=runtime,
#                 config=panel_config,
#             )
#
#     fig.suptitle(f"Coherence: grid over {param1[1]} and {param2[1]} (conv only)", fontsize=16)
#     fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
#
#     out_path = grid_root / f"new-grid_{param1[1]}-{param2[1]}{save_suffix}.png"
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)
#     print(f"Saved conventional-only grid figure to: {out_path}")


def main():
    plot_cpmg("./discovery_runs/2026.1.20_cpmg-long")
    # plot_coherence_csv("./discovery_runs/2025.12.7_template3.0/final_csv_files/2025.12.7_template3.0_cce_averaged.csv")


if __name__ == '__main__':
    main()

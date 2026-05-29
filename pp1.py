import csv
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, nnls
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import re
from scipy.interpolate import PchipInterpolator
from scipy.signal import hilbert
from scipy.interpolate import UnivariateSpline
import pandas as pd
import matplotlib.ticker as mticker
import matplotlib.tri as mtri
import warnings
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter
from scipy.optimize import least_squares
from collections import Counter
from functools import lru_cache
from time import perf_counter
from scipy.integrate import quad
from sympy import false


# +++++++++++++++++++++++ Basic preprocessing and utility methods +++++++++++++++++++++++ #
def clean_populations(coherence):
    """
    reduce impact of unstable points
    """

    n_removed = 0

    y = np.asarray(coherence, dtype=np.complex128)

    if y.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D coherence array got shape {y.shape}")

    # change large-magnitude outliers to 1
    y_clean = y.copy()
    bad_mask = [False]*len(y_clean) # np.abs(y_clean) > 100.0  # THIS IS HOW OUTLIERS ARE DEALT WITH
    n_removed += int(np.count_nonzero(bad_mask))
    y_clean[bad_mask] = np.complex128(0.0)

    return y_clean, n_removed


def clean_coherence(coherence):
    """
    reduce impact of unstable points
    """

    n_removed = 0

    y = np.asarray(coherence, dtype=np.complex128)

    if y.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D coherence array got shape {y.shape}")

    # change large-magnitude outliers to 1
    y_clean = y.copy()
    bad_mask = np.abs(y_clean) > 1.0  # THIS IS HOW OUTLIERS ARE DEALT WITH
    n_removed += int(np.count_nonzero(bad_mask))
    y_clean[bad_mask] = np.complex128(1.0)

    return y_clean, n_removed


def average_ensemble(coherence, avg_method):
    """
    Expects 2D arrays each with shape (num time_steps, ensemble_size). Averages over the ensemble using the specified
    method. For now there is just a simple mean and median.
    :param coherence: 2D array where rows are time steps and columns are individual runs of the ensemble
    :param avg_method: a string identifying the averaging method, straight from the yaml config file
    :return: A 1D array having number of rows timesteps. It is the averaged L(t).
    """

    coherence_arr = np.asarray(coherence, dtype=np.complex128)

    #  validate shape
    if coherence_arr.ndim != 2:
        raise ValueError(
            f"Result  must be 2D (timesteps, ensemble_size); "
            f"got shape {coherence_arr.shape}.")

    # average the results
    if avg_method == "mean":
        averaged = np.mean(np.asarray(coherence_arr, dtype=np.complex128), axis=1)  # average over ensemble axis
    elif avg_method == "median":
        warnings.warn("USING MEDIAN WITH COMPLEX NUMBERS IS NOT CONCEPTUALLY CLEAN!")
        averaged = np.median(np.asarray(coherence_arr, dtype=np.complex128), axis=1)  # median over ensemble axis
    else:
        raise Exception(f"Unknown averaging method '{avg_method}'")

    return averaged


def stretch(t, T2, p):
    return np.exp(- (t / T2) ** p)


def exp(t, R):
    return np.exp(- R * t)


def fourier(csv_path, num_labeled_peaks=8, min_peak_height_ratio=0.1):
    """
    Read a CSV with columns t and L_avg, compute the FFT, and save a new
    Fourier spectrum plot in the same directory as the original CSV.

    The saved plot will be named:
        <original_stem>_fft.png

    Example:
        gcce_averaged.csv -> gcce_averaged_fft.png

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    num_labeled_peaks : int
        Maximum number of peaks to label on the saved FFT plot.
    min_peak_height_ratio : float
        Only label peaks with height at least this fraction of the global max FFT magnitude.

    Returns
    -------
    freqs : np.ndarray
        Positive Fourier frequencies in 1/ms, i.e. kHz.
    fft_vals : np.ndarray
        Complex FFT values.
    fft_mag : np.ndarray
        Magnitude of the FFT.
    plot_path : Path
        Path to the saved FFT plot.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    t = df["t"].to_numpy(dtype=float)
    y = df["L_avg"].to_numpy(dtype=float)

    if len(t) < 2:
        raise ValueError("Need at least 2 time points for a Fourier transform.")

    dt_array = np.diff(t)
    dt = dt_array[0]

    if not np.allclose(dt_array, dt, rtol=1e-6, atol=1e-12):
        raise ValueError("Time grid is not uniformly spaced, so plain FFT is not appropriate.")

    # remove DC offset
    y = y - np.mean(y)

    # FFT
    fft_vals = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=dt)   # units: 1/ms = kHz
    fft_mag = np.abs(fft_vals)

    # find simple local maxima in fft_mag
    peak_indices = []
    if len(fft_mag) >= 3:
        for i in range(1, len(fft_mag) - 1):
            if fft_mag[i] > fft_mag[i - 1] and fft_mag[i] >= fft_mag[i + 1]:
                peak_indices.append(i)

    peak_indices = np.array(peak_indices, dtype=int)

    # keep only sufficiently tall peaks
    if len(peak_indices) > 0:
        height_threshold = min_peak_height_ratio * np.max(fft_mag)
        peak_indices = peak_indices[fft_mag[peak_indices] >= height_threshold]

        # keep only the strongest peaks
        if len(peak_indices) > num_labeled_peaks:
            strongest = np.argsort(fft_mag[peak_indices])[-num_labeled_peaks:]
            peak_indices = peak_indices[strongest]

        # sort labeled peaks left-to-right on x-axis
        peak_indices = peak_indices[np.argsort(freqs[peak_indices])]

    # save plot in same folder as original csv
    plot_path = csv_path.with_name(f"{csv_path.stem}_fft.png")

    plt.figure(figsize=(8, 5))
    plt.plot(freqs, fft_mag)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("FFT Magnitude")
    plt.title(f"Fourier Transform of {csv_path.name}")

    # mark and label peaks on the plot
    if len(peak_indices) > 0:
        plt.plot(freqs[peak_indices], fft_mag[peak_indices], "o")

        for idx in peak_indices:
            plt.annotate(
                f"{freqs[idx]:.12e} kHz",
                xy=(freqs[idx], fft_mag[idx]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45
            )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return freqs, fft_vals, fft_mag, plot_path


# +++++++++++++++++++++++ Fitting and plotting +++++++++++++++++++++++ #
def fit_exp(y, t, t_cutoff=None):
    """
    Fit y to exp(-R t).
    Returns R.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t_cutoff is not None:
        mask = t <= t_cutoff
        t, y = t[mask], y[mask]

    decay_guess = 1 / t[np.argmin(np.abs(y - np.exp(-1)))]

    decay_constant, = curve_fit(exp, t, y, p0=[decay_guess], bounds=(0, np.inf), maxfev=10000, )[0]

    return decay_constant


def fit_stretch(y, t, t_cutoff=None):
    """
    Fit to exp(-(t/T2)^p).
    Returns:
        T2, p, p_cov
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t_cutoff is not None:
        mask = t <= t_cutoff
        t = t[mask]
        y = y[mask]

    # initial guess
    target = np.exp(-1)
    idx = np.argmin(np.abs(y - target))
    t2_guess = t[idx] if t[idx] > 0 else np.median(t[t > 0])
    p_guess = 2.0

    fit_out = curve_fit(stretch, t, y, p0=[t2_guess, p_guess], bounds=([1e-12, 0.1], [np.inf, 10.0]), maxfev=10000)
    p_opt = fit_out[0]
    # p_cov = fit_out[1]

    t2, p = p_opt

    return t2, p


def fit_stretch_peaks(coherence, time_space, min_dense_peak_count=12, min_sparse_peak_count=4, smooth_window=5,
                      refine_radius=2, t_cutoff=None):
    """
    Fit a stretched exponential exp(-(t/T2)^p) for each cce_type.

    Cases handled:
    1. no peaks   -> fit the raw positive data directly
    2. sparse peaks -> fit through refined peak heights if there are at least min_sparse_peak_count
    3. dense peaks  -> fit through refined peak heights if there are at least min_dense_peak_count

    Design choices based on previous failures:
    - NO global prominence threshold based on full signal range
    - NO period estimation / window-per-cycle logic
    - NO upper-envelope hull filtering
    - NO ML / CNN
    - peak detection is done on a lightly smoothed copy, but fitting uses raw values
    - fitting uses robust least-squares instead of ordinary curve_fit
    """
    if t_cutoff is not None:
        print("have not implemented t_cutoff at this time")

    time_space = np.asarray(time_space, dtype=float)

    if time_space.ndim != 1:
        raise ValueError("time_space must be 1D")
    if len(time_space) < 10:
        raise ValueError("need at least 10 time points")
    if not np.all(np.diff(time_space) > 0):
        raise ValueError("time_space must be strictly increasing")
    if min_sparse_peak_count < 1:
        raise ValueError("min_sparse_peak_count must be at least 1")
    if min_dense_peak_count < min_sparse_peak_count:
        raise ValueError("min_dense_peak_count must be >= min_sparse_peak_count")
    if smooth_window < 3:
        raise ValueError("smooth_window must be at least 3")
    if refine_radius < 1:
        raise ValueError("refine_radius must be at least 1")


    def _smooth_trace(y):
        y = np.asarray(y, dtype=float)
        if len(y) < 5:
            return y.copy()

        win = min(smooth_window, len(y) if len(y) % 2 == 1 else len(y) - 1)
        if win < 5:
            return y.copy()
        if win % 2 == 0:
            win -= 1

        return savgol_filter(y, window_length=win, polyorder=2, mode="interp")

    def _robust_sigma(arr):
        arr = np.asarray(arr, dtype=float)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        return 1.4826 * mad

    def _refine_to_raw_max(y, idx, radius):
        """
        Move each candidate index onto the local raw maximum nearby.
        """
        y = np.asarray(y, dtype=float)
        idx = np.asarray(idx, dtype=int)

        if idx.size == 0:
            return np.array([], dtype=int)

        refined = []
        for i in idx:
            lo = max(0, i - radius)
            hi = min(len(y), i + radius + 1)
            j = lo + np.argmax(y[lo:hi])
            if y[j] > 0:
                refined.append(j)

        if len(refined) == 0:
            return np.array([], dtype=int)

        return np.array(sorted(set(refined)), dtype=int)

    def _merge_close_peaks(idx, heights, min_sep):
        """
        If two peaks are too close, keep the taller one.
        """
        idx = np.asarray(idx, dtype=int)
        heights = np.asarray(heights, dtype=float)

        if idx.size == 0:
            return np.array([], dtype=int)

        order = np.argsort(idx)
        idx = idx[order]
        heights = heights[order]

        keep_idx = [idx[0]]
        keep_h = [heights[0]]

        for i in range(1, len(idx)):
            if idx[i] - keep_idx[-1] <= min_sep:
                if heights[i] > keep_h[-1]:
                    keep_idx[-1] = idx[i]
                    keep_h[-1] = heights[i]
            else:
                keep_idx.append(idx[i])
                keep_h.append(heights[i])

        return np.asarray(keep_idx, dtype=int)

    def _direct_fit_data(x, y):
        keep = np.isfinite(x) & np.isfinite(y) & (y > 0)
        return x[keep], y[keep]

    def _fit_stretched_exp(xdata, ydata):
        if xdata.size < 3:
            raise ValueError("not enough fit points")

        below = np.where(ydata <= np.exp(-1))[0]
        T2_guess = xdata[below[0]] if below.size > 0 else xdata[-1]
        T2_guess = max(T2_guess, 1e-12)

        lb = np.array([1e-12, 0.1], dtype=float)
        ub = np.array([max(100.0 * xdata[-1], 1e-6), 5.0], dtype=float)
        x0 = np.array([T2_guess, 1.0], dtype=float)
        x0 = np.clip(x0, lb + 1e-15, ub - 1e-15)

        def residual(theta):
            T2, p = theta
            return stretch(xdata, T2, p) - ydata

        res = least_squares(
            residual,
            x0=x0,
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=0.03,
            max_nfev=20000,
        )

        if not res.success:
            raise RuntimeError(res.message)

        return tuple(res.x)


    fit_results = None

    y = np.asarray(coherence, dtype=float)

    if y.shape != time_space.shape:
        raise ValueError(
            f"Shape mismatch for y.shape = {y.shape}, time_space.shape = {time_space.shape}"
        )

    good = np.isfinite(time_space) & np.isfinite(y)
    x = time_space[good]
    y = y[good]

    if len(x) < 10:
        raise Exception(f"Not enough valid points")

    # -------------------------------------------------------------- #
    # Step 1: candidate peaks from a lightly smoothed copy
    # -------------------------------------------------------------- #
    y_smooth = _smooth_trace(y)
    cand_idx, _ = find_peaks(y_smooth)

    if cand_idx.size == 0:
        xdata, ydata = _direct_fit_data(x, y)
    else:
        prom = peak_prominences(y_smooth, cand_idx)[0]
        widths = peak_widths(y_smooth, cand_idx, rel_height=0.5)[0]

        # residual noise estimate after light smoothing
        sigma = _robust_sigma(y - y_smooth)
        sigma = max(float(sigma), 1e-12)

        # ---------------------------------------------------------- #
        # Step 2: decide whether this trace has real peaks at all
        # ---------------------------------------------------------- #
        max_prom = float(np.max(prom)) if prom.size > 0 else 0.0
        existence_floor = max(4.0 * sigma, 0.05 * max(np.max(y_smooth), 1e-12))

        if max_prom <= existence_floor:
            # no real peaks: fit raw data directly
            xdata, ydata = _direct_fit_data(x, y)
        else:
            # ------------------------------------------------------ #
            # Step 3: keep only peaks that stand out above noise
            # ------------------------------------------------------ #
            strong_prom = prom[prom >= np.quantile(prom, 0.75)]
            if strong_prom.size == 0:
                strong_prom = prom

            prom_floor = max(2.0 * sigma, 0.15 * np.median(strong_prom), 1e-12)

            keep = (
                (y_smooth[cand_idx] > 0) &
                np.isfinite(prom) &
                np.isfinite(widths) &
                (prom >= prom_floor) &
                (widths >= 1.0)
            )

            peak_idx = cand_idx[keep]

            if peak_idx.size == 0:
                xdata, ydata = _direct_fit_data(x, y)
            else:
                # -------------------------------------------------- #
                # Step 4: move selected peaks back onto raw maxima
                # -------------------------------------------------- #
                peak_idx = _refine_to_raw_max(y, peak_idx, radius=refine_radius)

                if peak_idx.size == 0:
                    xdata, ydata = _direct_fit_data(x, y)
                else:
                    peak_idx = _merge_close_peaks(
                        peak_idx,
                        y[peak_idx],
                        min_sep=max(1, refine_radius),
                    )

                    peak_idx = peak_idx[y[peak_idx] > 0]

                    # -------------------------------------------------- #
                    # Step 5: classify sparse vs dense by surviving peak count
                    # -------------------------------------------------- #
                    if peak_idx.size < min_sparse_peak_count:
                        # too few real peaks: treat as no-peaks
                        xdata, ydata = _direct_fit_data(x, y)
                    else:
                        required_peak_count = (
                            min_dense_peak_count
                            if peak_idx.size >= min_dense_peak_count
                            else min_sparse_peak_count
                        )

                        if peak_idx.size < required_peak_count:
                            xdata, ydata = _direct_fit_data(x, y)
                        else:
                            xdata = x[peak_idx]
                            ydata = y[peak_idx]

                            # include the initial point as an anchor
                            if x[0] < xdata[0]:
                                xdata = np.concatenate(([x[0]], xdata))
                                ydata = np.concatenate(([max(y[0], 1e-12)], ydata))

                            keep_fit = np.isfinite(xdata) & np.isfinite(ydata) & (ydata > 0)
                            xdata = xdata[keep_fit]
                            ydata = ydata[keep_fit]

    if xdata.size < 3:
        raise Exception(f"Not enough fit points")

    T2, p = _fit_stretched_exp(xdata, ydata)
    fit_results = (T2, p)

    return fit_results


def fit_stretch_linear(y, t, t_cutoff=None):
    """
    Linear fit to log(-log(L)) vs log(t). See https://doi.org/10.1103/PhysRevB.110.205148.
    Returns:
        T2, p
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t_cutoff is not None:
        mask = t <= t_cutoff
        t = t[mask]
        y = y[mask]

    mask = (t > 0) & (y > 0) & (y < 1)
    t = t[mask]
    y = y[mask]

    x = np.log(t)
    z = np.log(-np.log(y))

    p, intercept = np.polyfit(x, z, 1)

    t2 = np.exp(-intercept / p)

    return t2, p


def plot_trajectory(ax, trajectories, scatter=True, x_label="Time (ms)", y_label="L(t)", plot_text=None,
                    just_scatter=False):
    """
    Draws a trajectory on the given axes. May plot multiple trajectories.
    :param just_scatter:
    :param ax: the axes to plot on
    :param trajectories: dictionary where keys are labels and values are 2D arrays 1st column time, 2nd column L(t).
    :param scatter:
    :param x_label:
    :param y_label:
    :param plot_text:
    """

    # plot each trajectory
    for label, trajectory in trajectories.items():
        time_space = trajectory[:,0]
        coherence = trajectory[:,1]
        if just_scatter:
            ax.scatter(time_space, coherence, marker="x", s=30)
            # ax.plot(time_space, [1] * len(time_space), label="1", color="r", alpha=0.3)
        else:
            line, = ax.plot(time_space, coherence, label=label)
            if scatter:
                ax.scatter(time_space, coherence, marker="x", s=30, color=line.get_color())

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if plot_text is not None:
        fig = ax.figure

        # since we are plotting to existing axes this just checks if there already is text and if so write after it
        if not hasattr(fig, "plot_text_lines"):
            fig.plot_text_lines = []
        fig.plot_text_lines.append(plot_text)
        if hasattr(fig, "plot_text_artist"):
            fig.plot_text_artist.remove()

        ncols = 3
        rows = ["     |     ".join(fig.plot_text_lines[i:i + ncols]) for i in range(0, len(fig.plot_text_lines), ncols)]

        fig.plot_text_bottom = 0.08 + 0.035 * (len(rows) - 1)
        fig.plot_text_artist = fig.text(0.5, 0.07, "\n".join(rows), ha="center", va="bottom", fontsize=9,
                                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.legend()


def fit_plot(run_id, csv_name="averaged_mag.csv", data_dir="./data", plot_name="plot.png", log_time=False,
             log_coherence=False, full_path=None, fit_method=fit_stretch, t_cutoff=None, plot_fit=True):
    """

    :param run_id: run_id of the experiment
    :param csv_name:
    :param data_dir:
    :param plot_name:
    :param log_time:
    :param log_coherence:
    :param full_path:
    :param fit_method:
    :param t_cutoff:
    :param plot_fit:
    :return:
    """

    data_dir_path = Path(full_path) if full_path is not None else Path(data_dir) / run_id

    # single run case: data/run_id/averaged_mag.csv does exist
    csv_path = data_dir_path / csv_name
    if csv_path.exists():
        csv_paths = [csv_path]
    else:  # new case: multi run case, so use data/run_id/sub_run/averaged_mag.csv
        csv_paths = sorted(data_dir_path.glob(f"*/{csv_name}"))
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No {csv_name} found in {data_dir_path} or its immediate subdirectories.")

    # prepare the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    n_colors = len(csv_paths) * (2 if plot_fit else 1)
    ax.set_prop_cycle(color=plt.cm.turbo(np.linspace(0.05, 0.95, n_colors)))

    for csv_path in csv_paths:
        sub_run_id = csv_path.parent.name

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        time_space = data[:, 0]
        coherence = data[:, 1]

        # fit the curve and write to output
        t2, p = fit_method(coherence, time_space, t_cutoff=t_cutoff)
        print(f"{sub_run_id} FIT RESULTS: T2: {t2}, p: {p}")
        fit_curve = stretch(np.asarray(time_space), t2, p)
        fit_traj = np.column_stack((time_space, fit_curve))

        # plot the results
        plot_trajectory(ax, {sub_run_id : data}, scatter=True,
                        plot_text=f"{sub_run_id}: T2 = {t2:.7f} ms, p = {p:.7f}")
        if plot_fit:
            plot_trajectory(ax, {f"{sub_run_id} stretched exponential fit" : fit_traj}, scatter=False)

    # style plot
    if log_time:
        ax.set_xscale("log")
    if log_coherence:
        ax.set_yscale("log")

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(fname=f"{str(data_dir_path)}/{plot_name}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# +++++++++++++++++++++++ custom experiments +++++++++++++++++++++++ #
def get_FFs(base_sequences_ms, T_ms):
    """
    Build Fourier-transform functions directly from pulse-time base sequences.

    Returns:
        list of length p, at index p the object is:
        FF(omega) = ∫_0^T dt exp(+ i omega t) y_p(t)
    where y_p(t) starts at +1 and flips sign at every pulse.

    Units:
        base_sequences_ms and T_ms are in ms
        omega must be in rad/ms
    """

    FFs = []

    for idx, base_seq in enumerate(base_sequences_ms):
        # check base sequences are formatted correctly
        pulse_times = np.asarray(base_seq, dtype=float)
        if np.any((pulse_times <= 0.0) | (pulse_times > T_ms)):
            raise Exception(f"Base sequence {idx} has pulse times outside (0, T_ms]: "
                            f"T_ms={T_ms}, pulse_times={pulse_times}")
        if np.any(np.diff(pulse_times) <= 0.0):
            raise Exception(f"Base sequence {idx} is not strictly increasing: {pulse_times}")
        # prevent double counting, note end pulses dont affect the filter function
        new_pulse_times = pulse_times[pulse_times < T_ms]

        # edges is time boundaries of intervals where y(t)mis constant
        edges = np.concatenate(([0.0], new_pulse_times, [T_ms]))
        # value of y(t) on each interval. len(edges) always = len(signs) + 1
        signs = (-1.0) ** np.arange(len(edges) - 1)

        def F(omega, edges=edges, signs=signs):
            scalar_input = np.ndim(omega) == 0
            omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))

            out = np.zeros(omega_arr.shape, dtype=complex)

            # handles omega = nonzero / zero separately
            nonzero = np.abs(omega_arr) > 1.0e-14
            # omega nonzero case
            if np.any(nonzero):
                w = omega_arr[nonzero]

                exp_edges = np.exp(1j * w[:, None] * edges[None, :])

                interval_integrals = (
                    exp_edges[:, 1:] - exp_edges[:, :-1]
                ) / (1j * w[:, None])

                out[nonzero] = interval_integrals @ signs
            # omega zero case
            if np.any(~nonzero):
                durations = edges[1:] - edges[:-1]
                out[~nonzero] = np.sum(signs * durations)

            # return
            if scalar_input:
                return out[0]
            return out

        FFs.append(F)

    return FFs


def norris_bispectrum(run_id, mu_B=0, data_dir="./data", csv_name="raw.csv"):
    """
    Norris 2016 post-processing --> MLE version
    :param mu_B: MUST BE IN RAD/MS
    :param num_harmonics:
    :param run_id:
    :param data_dir:
    :param csv_name:
    :return:
    """

    '''get params'''
    config_path = Path("./config") / (run_id + ".yaml")
    import yaml
    with open(config_path) as file:
        config = yaml.safe_load(file)
    norris_params = config["norris_params"]
    P = norris_params["P"]  # number of base sequences
    T = norris_params['T']  # duration of base sequences in ms
    delta = norris_params['delta']  # T = q*delta
    M = norris_params['M']  # number of cycles of each base sequence
    tau = norris_params['tau']  # minimum spacing between pulses
    min_pulses = norris_params.get('min_pulses', None)

    '''get data'''
    # find csv files for the various sequences indexed by p
    data_dir_path = Path(data_dir) / run_id
    csv_paths = sorted(data_dir_path.glob(f"*/{csv_name}"), key=lambda p: int(p.parent.name.rsplit("-", 1)[1]))
    if len(csv_paths) == 0:
        raise Exception(f"No CSVs found under {data_dir_path} matching */{csv_name}")
    csv_p_idxs = [int(p.parent.name.rsplit("-", 1)[1]) for p in csv_paths]
    if csv_p_idxs != list(range(P)):
        raise Exception(f"Expected CSV subruns p=0 through p={P - 1}, but found {csv_p_idxs}")
    if csv_p_idxs[-1] != P - 1:
        raise Exception(f"FID should be last with p={P - 1}, but last CSV index is p={csv_p_idxs[-1]}")

    # each raw.csv contains a single row with [ensemble_size] columns
    # coherences is a 2D array: each row is an index p and the # of columns is ensemble size
    sample_size = config["experiment_params"]["ensemble_size"]
    coherences = np.zeros(shape=(P, sample_size), dtype=np.complex128)
    for csv_path in csv_paths:
        data = np.loadtxt(csv_path, delimiter=",", ndmin=2, dtype=np.complex128)
        if data.shape[0] != 1:
            raise Exception(f"{csv_path} should have exactly one data row, found {data.shape[0]}")
        if data.shape[1] != sample_size:
            raise Exception(f"{csv_path} should have exactly {sample_size} columns, found {data.shape[1]}")
        p_idx = int(csv_path.parent.name.rsplit("-", 1)[1])
        coherences[p_idx, :] = data[0, :]

    '''get statistics'''
    # ensemble means
    a = coherences.real
    b = coherences.imag
    b = b  # no conjugate for Norris's sigma+ because pycce gcce returns sigma+
    a_bar = np.mean(a, axis=1)
    b_bar = np.mean(b, axis=1)
    da = a - a_bar[:, None]
    db = b - b_bar[:, None]

    # variance and covariance of the ensemble mean
    var_a_bar = np.sum(da * da, axis=1) / (sample_size * (sample_size - 1))
    var_b_bar = np.sum(db * db, axis=1) / (sample_size * (sample_size - 1))
    cov_ab_bar = np.sum(da * db, axis=1) / (sample_size * (sample_size - 1))

    # Sung SI eq 18 analog
    phis = np.unwrap(np.angle(a_bar + 1j*b_bar))
    var_phi = ( a_bar**2 * var_a_bar + b_bar**2 * var_b_bar
                + 2.0 * a_bar * b_bar * cov_ab_bar ) / (a_bar**2 + b_bar**2)**2

    '''get filter functions'''
    # get base sequences from csv
    base_sequences_ms: list = [None] * P
    with open(data_dir_path / "base_sequences.csv", "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence_idx = int(row["sequence_idx"])
            pulse_idx = int(row["pulse_idx"])
            time_ms = float(row["time_ms"])
            if base_sequences_ms[sequence_idx] is None:
                base_sequences_ms[sequence_idx] = []
            # FID sentinel row: sequence_idx=P-1, pulse_idx=-1, time_ms=nan
            if pulse_idx == -1:
                if sequence_idx != P - 1:
                    raise Exception(f"FID row should have sequence_idx=P-1={P - 1}, got {sequence_idx}")
                base_sequences_ms[sequence_idx] = []
                continue
            base_sequences_ms[sequence_idx].append((pulse_idx, time_ms))
    for sequence_idx in range(P):
        if base_sequences_ms[sequence_idx] is None:
            raise Exception(f"Missing sequence_idx={sequence_idx} in sequences.csv")
        if sequence_idx == P - 1:
            base_sequences_ms[sequence_idx] = np.array([], dtype=float)
        else:
            sequence = sorted(base_sequences_ms[sequence_idx], key=lambda x: x[0])
            # additional check
            pulse_idxs = [pulse_idx for pulse_idx, _ in sequence]
            if pulse_idxs != list(range(len(pulse_idxs))):
                raise Exception(f"Bad pulse_idx sequence for sequence {sequence_idx}: {pulse_idxs}")
            base_sequences_ms[sequence_idx] = np.array([time_ms for _, time_ms in sequence], dtype=float)
    if len(base_sequences_ms[-1]) != 0:
        raise Exception(f"FID should be last at sequence_idx=P-1={P - 1}")
    print("\nBASE SEQUENCES:", base_sequences_ms, "\n\n")
    # subtract noise-mean phase using full executed sequences
    full_sequences_ms = []
    F0_full = np.zeros(P, dtype=float)
    for p, base_seq in enumerate(base_sequences_ms):
        M_eff = 1 if p == P - 1 else M  # FID is last and has M=1
        total_time_ms = T * M_eff
        if len(base_seq) == 0:
            full_seq = np.array([], dtype=float)
        else:
            full_seq = np.concatenate([
                np.asarray(base_seq, dtype=float) + m * T
                for m in range(M_eff)
            ])
        full_sequences_ms.append(full_seq)
        # F(0) over the full executed sequence
        pulse_times = full_seq[(full_seq > 0.0) & (full_seq < total_time_ms)]
        edges = np.concatenate(([0.0], pulse_times, [total_time_ms]))
        signs = (-1.0) ** np.arange(len(edges) - 1)
        F0_full[p] = np.sum(signs * np.diff(edges))
    total_phis = phis.copy()
    mean_phase = mu_B * F0_full
    phis = total_phis - mean_phase
    print("\nTOTAL PHASES ϕ:")
    print(total_phis)
    print("\nFULL-SEQUENCE F(0):")
    print(F0_full)
    print("\nMEAN PHASE μ_B F(0):")
    print(mean_phase)
    print("\nNON-GAUSSIAN PHASES φ = ϕ - μ_B F(0):")
    print(phis)
    # get base FFs
    filter_functions = get_FFs(base_sequences_ms, T)
    if len(filter_functions) != len(coherences):
        raise Exception(f"Number of filter functions ({len(filter_functions)}) does not match "
                        f"number of coherences ({len(coherences)})")

    '''set up Sung Eq. 7 linear system'''
    # make k omega_h
    omega_h = 2 * np.pi / T  # rad/ms
    # Sung test set
    k_pairs = [
        (0, 0),
        (1, 0),
        (2, 0),
        (1, 1),
        (3, 0),
        (2, 1),
        (4, 0),
        (3, 1),
        (2, 2),
        (4, 1),
        (3, 2),
    ]
    N = len(k_pairs)
    def bispectrum_multiplicity(k1, k2):
        if k1 == 0 and k2 == 0:
            return 1
        if k2 == 0 or k1 == k2:
            return 6
        return 12

    # system of linear equations
    matrix = np.zeros((P, N), dtype=float)
    for p, F_p in enumerate(filter_functions):
        M_eff = 1 if p == P - 1 else M  # FID is last
        for n, (k1, k2) in enumerate(k_pairs):
            w1 = k1 * omega_h
            w2 = k2 * omega_h
            G = F_p(w1) * F_p(w2) * F_p(-(w1 + w2))
            m = bispectrum_multiplicity(k1, k2)
            matrix[p, n] = -(M_eff / (6.0 * T ** 2)) * m * np.real(G)
    print("\nK_PAIRS:")
    print(k_pairs)
    print("\nBISPECTRUM MATRIX A:")
    print(matrix)

    '''solve linear system'''
    sigma_phi = np.sqrt(np.maximum(var_phi, 1e-30))
    matrix_w = matrix / sigma_phi[:, None]
    phi_w = phis / sigma_phi
    S2_est, residuals, rank, singular_values = np.linalg.lstsq(matrix_w, phi_w, rcond=None)
    print("\nS2_EST_WEIGHTED_MLE:")
    print(S2_est)
    print("\nLSTSQ RESIDUALS:")
    print(residuals)
    print("\nRANK:")
    print(rank)
    print("\nSINGULAR VALUES:")
    print(singular_values)
    print("\nCONDITION NUMBER:")
    print(singular_values[0] / singular_values[-1])
    print("\nSTANDARDIZED RESIDUALS:")
    print((matrix @ S2_est - phis) / sigma_phi)
    if rank < N:
        raise Exception(f"Rank deficient: rank={rank}, unknowns={N}")

    '''Plot reconstructed bispectrum'''
    def bispectrum_orbit(k1, k2):
        pts = [
            (k1, k2),
            (k2, k1),
            (-k2, k1 + k2),
            (-k1, k1 + k2),
            (-(k1 + k2), k1),
            (-(k1 + k2), k2),
            (-k1, -k2),
            (-k2, -k1),
            (k2, -(k1 + k2)),
            (k1, -(k1 + k2)),
            (k1 + k2, -k1),
            (k1 + k2, -k2),
        ]
        # remove duplicates (important on boundaries like (w,0), (w,w), (0,0))
        return list(dict.fromkeys(pts))
    S2_plot = S2_est * 1.0e3  # convert to 1/s
    full_vals = {}  # extend principal-domain values to the full plane
    for (k1, k2), val in zip(k_pairs, S2_plot):
        for pt in bispectrum_orbit(k1, k2):
            full_vals[pt] = val
    # build rectangular harmonic grid
    all_k1 = [pt[0] for pt in full_vals]
    all_k2 = [pt[1] for pt in full_vals]
    k1_min, k1_max = min(all_k1), max(all_k1)
    k2_min, k2_max = min(all_k2), max(all_k2)
    k1_grid = np.arange(k1_min, k1_max + 1)
    k2_grid = np.arange(k2_min, k2_max + 1)
    K1, K2 = np.meshgrid(k1_grid, k2_grid)
    Z = np.zeros_like(K1, dtype=float)
    for i in range(K1.shape[0]):
        for j in range(K1.shape[1]):
            Z[i, j] = full_vals.get((K1[i, j], K2[i, j]), 0.0)

    # convert harmonic indices to omega/2pi in MHz
    F1 = ((omega_h * K1) / (2 * np.pi)) * 1.0e-3
    F2 = ((omega_h * K2) / (2 * np.pi)) * 1.0e-3

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        F1, F2, Z,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.35,
        antialiased=True
    )
    ax.set_xlabel(r'$\omega_1 / 2\pi$ (MHz)')
    ax.set_ylabel(r'$\omega_2 / 2\pi$ (MHz)')
    ax.set_zlabel(r'$S_2(\omega_1,\omega_2)$ $(1/s)$')
    ax.set_title(r'Reconstructed bispectrum')
    ax.view_init(elev=28, azim=-58)
    fig.savefig(data_dir_path / "S2_bispectrum_3d.png", dpi=300)
    plt.close(fig)


def norris_psd_mle(run_id, num_harmonics, data_dir="./data", csv_name="raw.csv"):
    """
    Norris 2016 post-processing --> MLE version
    :param num_harmonics:
    :param run_id:
    :param data_dir:
    :param csv_name:
    :return:
    """

    '''get params'''
    config_path = Path("./config") / (run_id + ".yaml")
    import yaml
    with open(config_path) as file:
        config = yaml.safe_load(file)
    norris_params = config["norris_params"]
    P = norris_params["P"]  # number of base sequences
    T = norris_params['T']  # duration of base sequences in ms
    delta = norris_params['delta']  # T = q*delta
    M = norris_params['M']  # number of cycles of each base sequence
    tau = norris_params['tau']  # minimum spacing between pulses
    min_pulses = norris_params.get('min_pulses', None)

    '''get data'''
    # find csv files for the various sequences indexed by p
    data_dir_path = Path(data_dir) / run_id
    csv_paths = sorted(data_dir_path.glob(f"*/{csv_name}"), key=lambda p: int(p.parent.name.rsplit("-", 1)[1]))
    if len(csv_paths) == 0:
        raise Exception(f"No CSVs found under {data_dir_path} matching */{csv_name}")
    csv_p_idxs = [int(p.parent.name.rsplit("-", 1)[1]) for p in csv_paths]
    if csv_p_idxs != list(range(P)):
        raise Exception(f"Expected CSV subruns p=0 through p={P - 1}, but found {csv_p_idxs}")
    if csv_p_idxs[-1] != P - 1:
        raise Exception(f"FID should be last with p={P - 1}, but last CSV index is p={csv_p_idxs[-1]}")

    # each raw.csv contains a single row with [ensemble_size] columns
    # coherences is a 2D array: each row is an index p and the # of columns is ensemble size
    sample_size = config["experiment_params"]["ensemble_size"]
    coherences = np.zeros(shape=(P, sample_size), dtype=np.complex128)
    for csv_path in csv_paths:
        data = np.loadtxt(csv_path, delimiter=",", ndmin=2, dtype=np.complex128)
        if data.shape[0] != 1:
            raise Exception(f"{csv_path} should have exactly one data row, found {data.shape[0]}")
        if data.shape[1] != sample_size:
            raise Exception(f"{csv_path} should have exactly {sample_size} columns, found {data.shape[1]}")
        p_idx = int(csv_path.parent.name.rsplit("-", 1)[1])
        coherences[p_idx, :] = data[0, :]

    # '''get statistics'''
    # # ensemble means
    # a = coherences.real
    # b = coherences.imag
    # b = -b  # because we take the conjugate for Norris's \sigma_+
    # a_bar = np.mean(a, axis=1)
    # b_bar = np.mean(b, axis=1)
    # da = a - a_bar[:, None]
    # db = b - b_bar[:, None]
    #
    # # variance and covariance of the ensemble mean
    # var_a_bar = np.sum(da * da, axis=1) / (sample_size * (sample_size - 1))
    # var_b_bar = np.sum(db * db, axis=1) / (sample_size * (sample_size - 1))
    # cov_ab_bar = np.sum(da * db, axis=1) / (sample_size * (sample_size - 1))
    #
    # # Sung SI eq 18 analog
    # chis = -0.5 * np.log(a_bar ** 2 + b_bar ** 2)
    # var_chi = ( a_bar**2 * var_a_bar + b_bar**2 * var_b_bar
    #             + 2.0 * a_bar * b_bar * cov_ab_bar ) / (a_bar**2 + b_bar**2)**2

    '''get statistics'''
    # For PSD, compute chi per ensemble member first.
    # This avoids artificial decay from phase cancellation between different supercells.
    abs_L = np.abs(coherences)

    if np.any(~np.isfinite(abs_L)) or np.any(abs_L <= 0.0):
        raise Exception(f"Bad coherence magnitudes in raw data: min={np.nanmin(abs_L)}, max={np.nanmax(abs_L)}")

    if np.any(abs_L > 1.0 + 1e-12):
        raise Exception(f"Coherence magnitudes exceed 1 after cleaning: max={np.max(abs_L)}")

    chi_samples = -np.log(abs_L)

    chis = np.mean(chi_samples, axis=1)
    var_chi = np.var(chi_samples, axis=1, ddof=1) / sample_size

    # diagnostics: compare against old complex-mean chi
    L_bar = np.mean(coherences, axis=1)
    chis_complex_mean = -np.log(np.abs(L_bar))

    print("\nCHIS FROM MEAN OF PER-ENSEMBLE CHI:")
    print(chis)
    print("\nCHIS FROM -log(abs(mean complex L)) OLD METHOD:")
    print(chis_complex_mean)
    print("\nDIFFERENCE, OLD - NEW:")
    print(chis_complex_mean - chis)

    '''get filter functions'''
    # get base sequences from csv
    base_sequences_ms: list = [None] * P
    with open(data_dir_path / "base_sequences.csv", "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence_idx = int(row["sequence_idx"])
            pulse_idx = int(row["pulse_idx"])
            time_ms = float(row["time_ms"])
            if base_sequences_ms[sequence_idx] is None:
                base_sequences_ms[sequence_idx] = []
            # FID sentinel row: sequence_idx=P-1, pulse_idx=-1, time_ms=nan
            if pulse_idx == -1:
                if sequence_idx != P - 1:
                    raise Exception(f"FID row should have sequence_idx=P-1={P - 1}, got {sequence_idx}")
                base_sequences_ms[sequence_idx] = []
                continue
            base_sequences_ms[sequence_idx].append((pulse_idx, time_ms))
    for sequence_idx in range(P):
        if base_sequences_ms[sequence_idx] is None:
            raise Exception(f"Missing sequence_idx={sequence_idx} in sequences.csv")
        if sequence_idx == P - 1:
            base_sequences_ms[sequence_idx] = np.array([], dtype=float)
        else:
            sequence = sorted(base_sequences_ms[sequence_idx], key=lambda x: x[0])
            # additional check
            pulse_idxs = [pulse_idx for pulse_idx, _ in sequence]
            if pulse_idxs != list(range(len(pulse_idxs))):
                raise Exception(f"Bad pulse_idx sequence for sequence {sequence_idx}: {pulse_idxs}")
            base_sequences_ms[sequence_idx] = np.array([time_ms for _, time_ms in sequence], dtype=float)
    if len(base_sequences_ms[-1]) != 0:
        raise Exception(f"FID should be last at sequence_idx=P-1={P - 1}")
    print("\nBASE SEQUENCES:", base_sequences_ms, "\n\n")
    # get FFs
    filter_functions = get_FFs(base_sequences_ms, T)
    if len(filter_functions) != len(coherences):
        raise Exception(f"Number of filter functions ({len(filter_functions)}) does not match "
                        f"number of coherences ({len(coherences)})")

    '''set up Sung Eq. 7 linear system'''
    # make k omega_h
    omega_h = 2 * np.pi / T  # rad/ms
    harmonics = harmonics = np.asarray(np.arange(num_harmonics), dtype=int)  # np.asarray([5, 6, 7, 8], dtype=int)

    # system of linear equations
    matrix = np.zeros((P, len(harmonics)), dtype=float)
    for p, F_p in enumerate(filter_functions):
        M_eff = 1 if p == P - 1 else M  # FID is last
        for k, harmonic_k in enumerate(harmonics):
            zero_factor = (2.0 - int(harmonic_k == 0)) / 2.0
            matrix[p, k] = (M_eff / T) * zero_factor * np.abs(F_p(harmonic_k * omega_h))**2
    print("\nHARMONIC OMEGAS rad/ms:")
    print(harmonics * omega_h)
    print("\nEQ 7 MATRIX:")
    print(matrix)

    '''solve linear system'''
    # Diagnostic only: unconstrained least squares
    S_lstsq, residuals, rank, singular_values = np.linalg.lstsq(matrix, chis, rcond=None)
    print("\nS_LSTSQ:")
    print(S_lstsq)
    print("\nLSTSQ RESIDUALS:")
    print(residuals)
    print("\nRANK:")
    print(rank)
    print("\nSINGULAR VALUES:")
    print(singular_values)
    print("\nLSTSQ RESIDUAL NORM:")
    print(np.linalg.norm(matrix @ S_lstsq - chis))
    cond = singular_values[0] / singular_values[-1]
    print("\nCONDITION NUMBER:", cond)
    if rank < len(harmonics):
        raise Exception(f"Rank deficient: rank={rank}, unknowns={len(harmonics)}")

    # Actual physical estimate: nonnegative weighted least squares
    sigma_chi = np.sqrt(np.maximum(var_chi, 1e-30))
    sigma_floor = 0.10 * np.median(sigma_chi[np.isfinite(sigma_chi)])
    sigma_chi = np.maximum(sigma_chi, sigma_floor)
    # matrix_w = matrix / sigma_chi[:, None]
    # chis_w = chis / sigma_chi
    # S_est, residual_nnls = nnls(matrix_w, chis_w)

    fit_mask = np.ones(P, dtype=bool)
    fit_mask[P - 1] = False  # drop FID row from PSD fit
    matrix_fit = matrix[fit_mask, :]
    chis_fit = chis[fit_mask]
    sigma_fit = sigma_chi[fit_mask]
    matrix_w = matrix_fit / sigma_fit[:, None]
    chis_w = chis_fit / sigma_fit
    S_est, residual_nnls = nnls(matrix_w, chis_w)
    print("\nSTANDARDIZED RESIDUALS, NON-FID ONLY:")
    print((matrix_fit @ S_est - chis_fit) / sigma_fit)

    print("\nFID STANDARDIZED RESIDUAL, DIAGNOSTIC ONLY:")
    print((matrix[P - 1, :] @ S_est - chis[P - 1]) / sigma_chi[P - 1])

    print("\nS_WEIGHTED_NNLS / S_MLE:")
    print(S_est)
    print("\nWEIGHTED NNLS RESIDUAL:")
    print(residual_nnls)
    print("\nWEIGHTED NNLS RESIDUAL NORM:")
    print(np.linalg.norm(matrix_w @ S_est - chis_w))
    print("\nUNWEIGHTED RESIDUAL NORM OF WEIGHTED SOLUTION:")
    print(np.linalg.norm(matrix @ S_est - chis))
    print("\nCHIS:")
    print(chis)
    print("\nVAR_CHI:")
    print(var_chi)
    print("\nSIGMA_CHI:")
    print(sigma_chi)
    print("\nSTANDARDIZED RESIDUALS:")
    print((matrix @ S_est - chis) / sigma_chi)
    print("\nUNWEIGHTED NNLS COMPARISON:")
    S_unweighted_nnls, residual_unweighted_nnls = nnls(matrix, chis)
    print(S_unweighted_nnls)
    print("unweighted nnls residual:", residual_unweighted_nnls)
    print("weighted objective for unweighted nnls:",
          np.linalg.norm((matrix @ S_unweighted_nnls - chis) / sigma_chi))

    '''Plot reconstructed S(omega) versus omega / 2pi in MHz'''
    omega_plotting = ((omega_h*harmonics) / (2 * np.pi)) * 1.0e-3
    fig, ax = plt.subplots(figsize=(8, 5))
    S_est_per_s = S_est * 1.0e+3
    ax.plot(omega_plotting, S_est_per_s, marker="o", linewidth=1.5)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    for k, freq, S_val in zip(harmonics, omega_plotting, S_est_per_s):
        ax.annotate(f"k={k}", xy=(freq, S_val), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel(r"$\omega/2\pi$ (MHz)")
    ax.set_ylabel(r"$S(\omega) (1/s)$")
    ax.set_title(r"Reconstructed $S(\omega)$")
    fig.tight_layout()
    fig.savefig(data_dir_path / "S_omega-mle.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def norris_psd_least_sq(run_id, num_harmonics, data_dir="./data", csv_name="averaged.csv"):
    """
    Norris 2016 post-processing
    :param num_harmonics:
    :param run_id:
    :param data_dir:
    :param csv_name:
    :return:
    """

    '''get params'''
    config_path = Path("./config") / (run_id + ".yaml")
    import yaml
    with open(config_path) as file:
        config = yaml.safe_load(file)
    norris_params = config["norris_params"]
    P = norris_params["P"]  # number of base sequences
    T = norris_params['T']  # duration of base sequences in ms
    delta = norris_params['delta']  # T = q*delta
    M = norris_params['M']  # number of cycles of each base sequence
    tau = norris_params['tau']  # minimum spacing between pulses
    min_pulses = norris_params.get('min_pulses', None)

    '''get data'''
    # find csv files for the various sequences indexed by p
    data_dir_path = Path(data_dir) / run_id
    csv_paths = sorted(data_dir_path.glob(f"*/{csv_name}"), key=lambda p: int(p.parent.name.rsplit("-", 1)[1]))
    if len(csv_paths) == 0:
        raise Exception(f"No CSVs found under {data_dir_path} matching */{csv_name}")
    csv_p_idxs = [int(p.parent.name.rsplit("-", 1)[1]) for p in csv_paths]
    if csv_p_idxs != list(range(P)):
        raise Exception(f"Expected CSV subruns p=0 through p={P - 1}, but found {csv_p_idxs}")
    if csv_p_idxs[-1] != P - 1:
        raise Exception(f"FID should be last with p={P - 1}, but last CSV index is p={csv_p_idxs[-1]}")
    # pull coherences from csv files
    times = []
    coherences = []
    for csv_path in csv_paths:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2, dtype=np.complex128)
        if data.shape[0] != 1:
            raise Exception(f"{csv_path} should have exactly one data row, found {data.shape[0]}")
        if data.shape[1] != 2:
            raise Exception(f"{csv_path} should have exactly two columns, found {data.shape[1]}")
        time = data[0, 0]
        coherence = data[0, 1]
        times.append(time)
        coherences.append(coherence)
    # data processing on times
    times = np.asarray(times)
    if not np.iscomplexobj(times):
        raise Exception(f"times should be complex before dropping imaginary part: dtype={times.dtype}")
    if not np.allclose(times.imag, 0.0, rtol=0.0, atol=1e-12):
        raise Exception(f"Some time entries have nonzero imaginary parts: {times}")
    times = times.real.astype(float)
    if np.any(times <= 0):
        raise Exception(f"All final times should be positive: {times}")
    if not np.allclose(times, times[0], rtol=0.0, atol=1e-9):
        raise Exception(f"Not all CSV time entries are the same: {times}")
    if not np.isclose(times[0], T * M, rtol=0.0, atol=1e-9):
        raise Exception(f"CSV time={times[0]} is not close to T*M={T * M}")
    # data processing on coherence
    coherences = np.asarray(coherences, dtype=np.complex128)
    if len(coherences) != P:
        raise Exception(f"Number of coherences ({len(coherences)}) does not match P={P}")

    # final 1D array
    coherences = np.asarray(coherences)
    print("\nCOHERENCES: ", coherences, "\n\n")

    '''get filter functions'''
    # get base sequences from csv
    base_sequences_ms: list = [None] * P
    with open(data_dir_path / "base_sequences.csv", "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence_idx = int(row["sequence_idx"])
            pulse_idx = int(row["pulse_idx"])
            time_ms = float(row["time_ms"])
            if base_sequences_ms[sequence_idx] is None:
                base_sequences_ms[sequence_idx] = []
            # FID sentinel row: sequence_idx=P-1, pulse_idx=-1, time_ms=nan
            if pulse_idx == -1:
                if sequence_idx != P - 1:
                    raise Exception(f"FID row should have sequence_idx=P-1={P - 1}, got {sequence_idx}")
                base_sequences_ms[sequence_idx] = []
                continue
            base_sequences_ms[sequence_idx].append((pulse_idx, time_ms))
    for sequence_idx in range(P):
        if base_sequences_ms[sequence_idx] is None:
            raise Exception(f"Missing sequence_idx={sequence_idx} in sequences.csv")
        if sequence_idx == P - 1:
            base_sequences_ms[sequence_idx] = np.array([], dtype=float)
        else:
            sequence = sorted(base_sequences_ms[sequence_idx], key=lambda x: x[0])
            # additional check
            pulse_idxs = [pulse_idx for pulse_idx, _ in sequence]
            if pulse_idxs != list(range(len(pulse_idxs))):
                raise Exception(f"Bad pulse_idx sequence for sequence {sequence_idx}: {pulse_idxs}")
            base_sequences_ms[sequence_idx] = np.array([time_ms for _, time_ms in sequence], dtype=float)
    if len(base_sequences_ms[-1]) != 0:
        raise Exception(f"FID should be last at sequence_idx=P-1={P - 1}")
    print("\nBASE SEQUENCES:", base_sequences_ms, "\n\n")
    # get FFs
    filter_functions = get_FFs(base_sequences_ms, T)
    if len(filter_functions) != len(coherences):
        raise Exception(f"Number of filter functions ({len(filter_functions)}) does not match "
                        f"number of coherences ({len(coherences)})")

    '''set up Sung Eq. 7 linear system'''
    # make k omega_h
    omega_h = 2 * np.pi / T  # rad/ms
    harmonics = np.asarray(np.arange(num_harmonics), dtype=int)

    # make chi
    coherence_abs = np.abs(coherences)
    if np.any(~np.isfinite(coherence_abs)) or np.any(coherence_abs <= 0):
        raise Exception(f"Cannot take log of bad coherence magnitudes: {coherence_abs}")
    if np.any(coherence_abs > 1.0 + 1e-12):
        raise Exception(f"Coherence magnitudes exceed 1: {coherence_abs}")
    chis = -np.log(coherence_abs)  # p-long vector

    # system of linear equations
    matrix = np.zeros((P, len(harmonics)), dtype=float)
    for p, F_p in enumerate(filter_functions):
        M_eff = 1 if p == P - 1 else M  # FID is last
        for k, harmonic_k in enumerate(harmonics):
            zero_factor = (2.0 - int(harmonic_k == 0)) / 2.0
            matrix[p, k] = (M_eff / T) * zero_factor * np.abs(F_p(harmonic_k * omega_h))**2
    print("\nCHI:")
    print(chis)
    print("\nHARMONIC OMEGAS rad/ms:")
    print(harmonics * omega_h)
    print("\nEQ 7 MATRIX:")
    print(matrix)

    '''solve linear system'''
    # Diagnostic only: unconstrained least squares
    S_lstsq, residuals, rank, singular_values = np.linalg.lstsq(matrix, chis, rcond=None)
    print("\nS_LSTSQ:")
    print(S_lstsq)
    print("\nLSTSQ RESIDUALS:")
    print(residuals)
    print("\nRANK:")
    print(rank)
    print("\nSINGULAR VALUES:")
    print(singular_values)
    print("\nLSTSQ RESIDUAL NORM:")
    print(np.linalg.norm(matrix @ S_lstsq - chis))
    cond = singular_values[0] / singular_values[-1]
    print("\nCONDITION NUMBER:", cond)
    if rank < len(harmonics):
        raise Exception(f"Rank deficient: rank={rank}, unknowns={len(harmonics)}")

    # Actual physical estimate: nonnegative least squares
    S_est, residual_nnls = nnls(matrix, chis)
    print("\nS_NNLS:")
    print(S_est)
    print("\nNNLS RESIDUAL:")
    print(residual_nnls)
    print("\nNNLS RESIDUAL NORM:")
    print(np.linalg.norm(matrix @ S_est - chis))

    '''Plot reconstructed S(omega) versus omega / 2pi in MHz'''
    omega_plotting = ((omega_h*harmonics) / (2 * np.pi)) * 1.0e-3

    fig, ax = plt.subplots(figsize=(8, 5))
    S_est_per_s = S_est * 1.0e+3
    ax.plot(omega_plotting, S_est_per_s, marker="o", linewidth=1.5)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    for k, freq, S_val in zip(harmonics, omega_plotting, S_est_per_s):
        ax.annotate(f"k={k}", xy=(freq, S_val), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel(r"$\omega/2\pi$ (MHz)")
    ax.set_ylabel(r"$S(\omega) (1/s)$")
    ax.set_title(r"Reconstructed $S(\omega)$")
    fig.tight_layout()
    fig.savefig(data_dir_path / "S_omega.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mean_fid(run_id, data_dir="./data", csv_name="averaged.csv"):
    """
    Sung/Jin style FID protocol to measure the noise mean
    :param run_id:
    :param data_dir:
    :param csv_name:
    :return:
    """

    '''get data'''
    data_dir_path = Path(data_dir) / run_id
    csv_path = data_dir_path / csv_name
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2, dtype=np.complex128)

    '''data processing'''
    # make it real
    b = np.imag(data[:, 1])
    frac_bad = np.count_nonzero(np.abs(b) > 1.0e-10) / data[:, 1].size
    if frac_bad > 0.1:
        raise Exception(f"More than 10% of points have nonzero imaginary part: {frac_bad:.2%}")
    data = np.real(data)
    # remove outliers
    finite = np.isfinite(data[:,1])
    # cutoff = np.nanpercentile(y, 99)
    keep = finite # & (y < cutoff)
    data = data[keep]
    print("number of non-finite points: ", np.count_nonzero(~finite) ,"\n\n")
    # change large-magnitude outliers to 0
    n_removed = 0
    y_clean = data[:,1].copy()
    bad_mask = ~((-1.0 <= y_clean) & (y_clean <= 1.0))
    n_removed += int(np.count_nonzero(bad_mask))
    y_clean[bad_mask] = 0.0
    # data[:,1] = y_clean
    data = data[~bad_mask]
    print("\nNUMBER OF UNSTABLE POINTS: ", n_removed, "out of a total ", len(data[:, 1]), " points" ,"\n\n")
    np.savetxt(data_dir_path / "final.csv", data, header="t,L_avg", delimiter=",", comments="")
    ''' take sigma_z / tau '''
    beta_vals = data[:,1] / data[:,0]

    '''mean and error bars'''
    beta_mean = np.mean(beta_vals)
    beta_std = np.std(beta_vals, ddof=1)
    beta_sem = beta_std / np.sqrt(beta_vals.size)
    print(f"beta mean = {beta_mean:.6e}")
    print(f"beta std  = {beta_std:.6e}")
    print(f"beta SEM  = {beta_sem:.6e}")

    '''plot the inset'''
    fig, ax = plt.subplots(figsize=(20, 8))

    trajectories = {"beta": np.column_stack((data[:, 0], data[:,1]))}
    plot_trajectory(
        ax,
        trajectories,
        scatter=True,
        y_label="rk",
        just_scatter=True,
    )

    # ax.axhline(beta_mean, linestyle="--", linewidth=2, label=f"mean = {beta_mean:.3e}")
    # ax.fill_between(
    #     data[:, 0],
    #     beta_mean - beta_sem,
    #     beta_mean + beta_sem,
    #     alpha=0.2,
    #     label=f"SEM = {beta_sem:.3e}",
    # )

    ax.legend()
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(fname=f"{str(data_dir_path)}/mean.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def alvarez_pp(run_id, data_dir="./data", csv_name="averaged_mag.csv"):
    """
    alvarez and suter post processing
    :param num_cycles:
    :param run_id:
    :param data_dir:
    :param csv_name:
    :return:
    """

    '''get params'''
    config_path = Path("./config") / (run_id + ".yaml")
    import yaml
    with open(config_path) as file:
        config = yaml.safe_load(file)
    alvarez_params = config["alvarez_params"]
    num_cycles = alvarez_params['num_cycles']  # big M
    tau_max = alvarez_params['tau_max']
    num_measurements = alvarez_params['num_measurements']  # little m

    '''get data'''
    data_dir_path = Path(data_dir) / run_id
    csv_paths = sorted(data_dir_path.glob(f"*/{csv_name}"), key=lambda p: int(p.parent.name.rsplit("-", 1)[1]))

    taus = tau_max / np.arange(num_measurements, 0, -1)  # tau_0, tau_1, ... , tau_{m-1}
    cycles = np.arange(1, num_cycles + 1)  # 1, 2, 3, ..., M
    coh = np.array([np.loadtxt(p, delimiter=",", skiprows=1)[:, 1] for p in csv_paths]).T
    times = 2 * np.outer(taus, cycles)

    # list of 2D arrays, first col is timespace second col is coherence
    # each 2D array is a stroboscopic CPMG run with tau = tau_idx
    runs = list(np.dstack((times, coh)))

    # add 0.0, 1.0 to all the runs
    for idx, run in enumerate(runs):
        runs[idx] = np.vstack(([0.0, 1.0], run))

    '''plot the inset'''
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_idxs = [0, 1, 2]
    trajectories = {}
    fits = {}
    for plot_idx in plot_idxs:
        data = runs[plot_idx]
        print(plot_idx, "plot idx", data)
        trajectories[f"tau-{taus[plot_idx]*10**3:.8f} (mu s)"] = data

        # fit the curve
        decay_parameter = fit_exp(data[:,1], data[:,0])
        fit_curve = exp(np.asarray(data[:,0]), decay_parameter)
        fit_traj = np.column_stack((data[:,0], fit_curve))
        fits[f"tau-{taus[plot_idx]*10**3:.2f} (mu s)"] = fit_traj
    plot_trajectory(ax, trajectories, scatter=True)  # plot points
    # plot_trajectory(ax, fits, scatter=False)  # plot fits
    ax.set_yscale("log")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(fname=f"{str(data_dir_path)}/fig1-inset", dpi=300, bbox_inches="tight")
    plt.close(fig)

    '''plot fig 1'''
    decay_parameters = []
    for idx in np.arange(len(runs)):
        data = np.asarray(runs[idx])
        decay_parameter = fit_exp(data[:,1], data[:,0])
        decay_parameters.append(decay_parameter)
    relaxations = 1.0 / np.asarray(decay_parameters)
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_trajectory(ax, {"R^-1": np.column_stack((taus, relaxations))}, scatter=True,
                    y_label="Relaxation time (ms)", x_label="Pulse delay tau (ms)")  # plot fits
    ax.set_yscale("log")
    ax.set_xscale("log")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(fname=f"{str(data_dir_path)}/fig1", dpi=300, bbox_inches="tight")
    plt.close(fig)

    '''make reconstruction matrix U'''
    decay_parameters = np.asarray(decay_parameters, dtype=float)
    assert num_measurements == len(decay_parameters)
    decay_parameters = decay_parameters[::-1]

    # U[n-1, j-1] = sum_k A_k^2 delta_{j,nk}
    U = np.zeros((num_measurements, num_measurements), dtype=float)
    for n in range(1, num_measurements + 1):  # n = 1, ..., m
        for k in range(1, num_measurements // n + 1):  # k = 1, ..., floor(m/n)
            j = n * k  # j = nk
            if k % 2 == 1:
                A_k_squared = 1.0 / k ** 2
            else:
                A_k_squared = 0.0
            U[n - 1, j - 1] = A_k_squared
    # Solve U S = R, equivalent to S = U^{-1} R
    spectral_density_values = np.linalg.solve(U, decay_parameters)

    '''Plot reconstructed S(omega) versus omega / 2pi in MHz'''
    omega_min = np.pi / tau_max  # rad/ms
    j_values = np.arange(1, num_measurements + 1)  # j = 1, ..., m
    omega_values = j_values * omega_min  # rad/ms
    omega_plotting = (omega_values / (2.0 * np.pi)) * 1.0e-3  # MHz
    S_est_per_s = spectral_density_values * 1.0e3  # 1/ms -> 1/s
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omega_plotting, S_est_per_s, marker="o", linewidth=1.5)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    for j, freq, S_val in zip(j_values, omega_plotting, S_est_per_s):
        ax.annotate(
            f"j={j}",
            xy=(freq, S_val),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel(r"$\omega/2\pi$ (MHz)")
    ax.set_ylabel(r"$S(\omega)$ (1/s)")
    ax.set_title(r"Reconstructed $S(\omega)$")
    fig.tight_layout()
    fig.savefig(data_dir_path / "S_omega_alvarez.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# +++++++++++++++++++++++ Specialized plots +++++++++++++++++++++++ #
def plot_cpmg_ethan(run_path, csv_name="averaged_mag.csv", save_path=None, logtime=False,
                    T_max=None, tau_max=None, noise_floor=None, T_min=None, tau_min=None, y_lim=None, x_lim=None):
    """
    Build a 2D τ–t_d heatmap for many cpmg runs as in Ethan's thesis.

    Assumes directory structure like:
        run_path/
            CPMG-1/csv_name.csv
            CPMPG-5/csv_name.csv
            ...

    Each csv file should have two columns:
        T (ms),  total_length(T)
    """

    '''get the data'''
    run_path = Path(run_path)
    if not run_path.is_dir():
        raise NotADirectoryError(f"{run_path} is not a directory")

    # Find all run dirs and parse N from names "CPMG-N..."
    run_info = []  # list of (N, run_dir, csv_path)
    for d in run_path.iterdir():
        if not d.is_dir():
            continue
        m = re.search(r"CPMG-(\d+)", d.name)
        if not m:
            continue
        N = int(m.group(1))
        csv_path = d / csv_name
        if csv_path.is_file():
            run_info.append((N, d, csv_path))
    print(f"Found {len(run_info)} CPMG runs for 2D plot")
    if not run_info:
        raise RuntimeError(
            f"No run directories with pattern 'pulse_id-N' and {csv_name} "
            f"found under {run_path}"
        )

    '''manipulate the data'''
    tau_all = []
    T_all = []
    L_all = []
    for N, run_dir, csv_path in run_info:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"CSV {csv_path} must have at least two columns")

        T = data[:, 0]  # ms
        L = data[:, 1]
        tau = T / N  # ms

        tau_all.append(tau)
        T_all.append(T)
        L_all.append(L)
    tau_all = np.concatenate(tau_all)
    T_all = np.concatenate(T_all)
    L_all = np.concatenate(L_all)
    original_num_points = L_all.size

    # # noise floor - clip
    # n_removed_noise = 0
    # L_abs = np.abs(L_all)
    # if noise_floor is not None:
    #     below_noise = L_abs < noise_floor
    #     n_removed_noise = np.sum(below_noise)
    #     L_abs = np.maximum(L_abs, noise_floor)
    # chi_all = -np.log10(L_abs)

    # noise floor - remove
    n_removed_noise = 0
    if noise_floor is not None:
        above_noise = np.abs(L_all) > noise_floor
        n_removed_noise = np.sum(~above_noise)

        tau_all = tau_all[above_noise]
        T_all = T_all[above_noise]
        L_all = L_all[above_noise]

    chi_all = -np.log10(np.abs(L_all))

    # populate grid using points above / below
    tau_axis = np.unique(tau_all)
    T_axis = np.unique(T_all)
    Tau_grid, T_grid = np.meshgrid(tau_axis, T_axis)
    chi_grid = np.empty_like(Tau_grid)
    Ns = np.array([N for N, run_dir, csv_path in run_info], dtype=float)

    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            tau_target = Tau_grid[i, j]
            T_target = T_grid[i, j]

            T_lines = Ns * tau_target
            best_N = Ns[np.argmin(np.abs(T_lines - T_target))]

            same_line = np.isclose(T_all, best_N * tau_all, rtol=1e-10, atol=1e-14)
            line_tau = tau_all[same_line]
            line_chi = chi_all[same_line]

            chi_grid[i, j] = line_chi[np.argmin(np.abs(line_tau - tau_target))]

    # manual cutoffs
    point_mask = np.ones_like(chi_all, dtype=bool)
    grid_mask = np.ones_like(chi_grid, dtype=bool)
    if tau_max is not None:
        point_mask &= tau_all <= tau_max
        grid_mask &= Tau_grid <= tau_max
    if T_max is not None:
        point_mask &= T_all <= T_max
        grid_mask &= T_grid <= T_max
    if tau_min is not None:
        point_mask &= tau_all >= tau_min
        grid_mask &= Tau_grid >= tau_min
    if T_min is not None:
        point_mask &= T_all >= T_min
        grid_mask &= T_grid >= T_min
    tau_plot = tau_all[point_mask]
    T_plot = T_all[point_mask]
    chi_plot = chi_all[point_mask]
    chi_grid_plot = chi_grid.copy()
    chi_grid_plot[~grid_mask] = np.nan

    # new colormap
    from matplotlib.colors import LinearSegmentedColormap
    parula_colors = [
        "#F8F855", "#F6F255", "#F4ED55", "#F2E154", "#F2DA54", "#F4D654", "#F6CF54", "#F5C456",
        "#ECBD55", "#E5BC50", "#DDBE4C", "#CCC049", "#C2C34C", "#B8C44F", "#A4C85B", "#99CA62",
        "#86CB72", "#7ECB7B", "#76CA83", "#6FC98C", "#66C59B", "#60C0A9", "#59BCB6", "#54B7C1",
        "#52B2CE", "#51AAD8", "#50A3DE", "#4D99E2", "#4B8FE8", "#4985EF", "#467AF4", "#4A6EF6",
        "#4C63F4", "#4B58EF", "#494DE7", "#4642DE", "#4339CE", "#3F2FB9", "#39279E", "#332487"
    ]
    parula_r = LinearSegmentedColormap.from_list("parula_r", parula_colors, N=256)

    '''plot the data'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sc1 = ax1.scatter(tau_plot, T_plot, c=chi_plot, s=12, cmap=parula_r)
    sc2 = ax2.pcolormesh(Tau_grid, T_grid, chi_grid_plot, shading="auto", cmap=parula_r)

    # logarithmic time
    if logtime:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")

    # plotting settings
    ax1.set_xlim(left=np.min(tau_plot), right=np.max(tau_plot))
    ax2.set_xlim(left=np.min(tau_plot), right=np.max(tau_plot))
    ax1.set_ylim(bottom=np.min(T_plot), top=np.max(T_plot))
    ax2.set_ylim(bottom=np.min(T_plot), top=np.max(T_plot))
    if y_lim is not None:
        ax1.set_ylim(*y_lim)
        ax2.set_ylim(*y_lim)
    if x_lim is not None:
        ax1.set_xlim(*x_lim)
        ax2.set_xlim(*x_lim)
    ax1.set_xlabel(r"$\tau$ (ms)")
    ax1.set_ylabel(r"$T$ (ms)")
    ax2.set_xlabel(r"$\tau$ (ms)")
    ax2.set_ylabel(r"$T$ (ms)")
    ax1.set_title("Scatter")
    ax2.set_title("Nearest-neighbor populated grid")
    if noise_floor is not None:
        fig.text( 0.5, 0.01,
            f"Removed {n_removed_noise} pts below noise floor: {noise_floor}, out of {original_num_points} pts total"
                  , ha="center")
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar1.set_label(r"$\chi$")
    cbar2.set_label(r"$\chi$")
    fig.tight_layout()

    # save the plot
    if save_path is None:
        save_path = run_path / "ethan_plot.png"
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved 2D heatmap to: {save_path}")

    return fig, (ax1, ax2)


def _parse_cpmg_n(path: Path):
    """
    Extracts n from folder names like CPMG-1, CPMG-2, CPMG-10.
    Returns infinity if parsing fails so weird folders sort last.
    """
    match = re.fullmatch(r"CPMG-(\d+)", path.name)
    if match is None:
        return float("inf")
    return int(match.group(1))


def plot_park_s13(cpmg_run_dir, averaged_name="gcce_averaged_modulus", output_name=None, logtime=False,
                  bar_filename=None, right_lim=None):
    """
    After a cpmg() run has finished, plot every averaged CSV from the
    CPMG-* subfolders on one plot.

    If log_time=True, the x-axis is plotted on a log scale like
    Evolution time with ticks 10^1, 10^2, 10^3, etc.
    """
    cpmg_run_dir = Path(cpmg_run_dir)

    if output_name is None:
        output_name = f"park_s13.png"

    if not averaged_name.endswith(".csv"):
        averaged_csv_name = f"{averaged_name}.csv"
    else:
        averaged_csv_name = averaged_name
        averaged_name = averaged_name[:-4]

    if not cpmg_run_dir.is_dir():
        raise FileNotFoundError(f"CPMG run directory does not exist: {cpmg_run_dir}")

    # Bar-Gill 2012 data
    bar_data = None
    if bar_filename is not None:
        bar_path = Path("ref") / bar_filename
        if not bar_path.is_file():
            raise FileNotFoundError(f"Bar-Gill dataset does not exist: {bar_path}")
        bar_data = np.genfromtxt(bar_path, delimiter=",", names=True, dtype=None, encoding=None)

    cpmg_dirs = sorted(
        [p for p in cpmg_run_dir.iterdir() if p.is_dir() and p.name.startswith("CPMG-")],
        key=_parse_cpmg_n,
    )

    if not cpmg_dirs:
        raise FileNotFoundError(f"No CPMG-* subdirectories found in: {cpmg_run_dir}")

    fig, ax = plt.subplots(figsize=(12, 8))

    n_plotted = 0
    bar_color_by_n = {}
    bar_marker_by_n = {1: "o", 4: "s", 12: "o", 64: "^", 128: "d"}
    legend_handles = []

    for subdir in cpmg_dirs:
        csv_path = subdir / "final_csv_files" / averaged_csv_name

        if not csv_path.is_file():
            print(f"Skipping {subdir.name}: missing {csv_path}")
            continue

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        if data.ndim != 2 or data.shape[1] < 2:
            print(f"Skipping {subdir.name}: bad CSV shape {data.shape}")
            continue

        t = 1000.0 * data[:, 0]  # ms -> us
        L_avg = data[:, 1]

        if logtime:
            mask = t > 0
            if not np.any(mask):
                print(f"Skipping {subdir.name}: no positive time values for log x-axis")
                continue

            if np.any(~mask):
                print(f"Skipping nonpositive time values in {subdir.name} for log x-axis")

            t = t[mask]
            L_avg = L_avg[mask]

        n = _parse_cpmg_n(subdir)
        label = f"n = {int(n)}" if n != float("inf") else subdir.name

        line, = ax.plot(t, L_avg, label="_nolegend_", marker="o", markersize=2.2, linewidth=1.7, alpha=0.45)
        if n != float("inf"):
            bar_color_by_n[int(n)] = line.get_color()
            legend_handles.append(Line2D([], [], color=line.get_color(), linewidth=1.7, alpha=0.45, label=label))
        n_plotted += 1

    # add bar-gill data
    if bar_data is not None:
        for bar_n in np.unique(bar_data["pulse_n"]):
            bar_n = int(bar_n)
            if bar_n not in bar_color_by_n:
                continue

            bar_mask = bar_data["pulse_n"] == bar_n
            ax.scatter(bar_data["time_us"][bar_mask], bar_data["coherence"][bar_mask],
                       color=bar_color_by_n[bar_n], s=55, marker=bar_marker_by_n.get(bar_n, "o"),
                       edgecolors="none", label="_nolegend_")

        bar_legend_handle = None
        if bar_data is not None:
            bar_legend_handle = Line2D([], [], linestyle="None", marker="o",
                                       color=bar_color_by_n.get(1, "k"), markersize=7,
                                       label="N. Bar-Gill et al., (2012)")

    if n_plotted == 0:
        plt.close(fig)
        raise RuntimeError(
            f"Found CPMG folders, but no {averaged_csv_name} files were plotted in {cpmg_run_dir}"
        )

    if logtime:
        ax.set_xscale("log")

        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))

        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        ax.tick_params(axis="x", which="major", length=7)
        ax.tick_params(axis="x", which="minor", length=3)

    # ax.set_title(f"{averaged_name} coherence across CPMG runs")
    ax.set_xlim(left=10, right=right_lim)
    ax.set_xlabel(r"Evolution time ($\mu$s)")
    ax.set_ylabel("Coherence")
    ax.grid(False)
    if bar_legend_handle is not None:
        ax.legend(handles=legend_handles + [bar_legend_handle], frameon=False)
    else:
        ax.legend(handles=legend_handles, frameon=False)

    fig.tight_layout()

    output_path = cpmg_run_dir / output_name
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_park_fig2(cpmg_run_dir, averaged_csv_name="averaged_mag.csv", output_name="park_fig2.png", logtime=False,
                   noise_floor=None, y_lim=None, x_lim=None):
    """
    Park AOM 2026 Figure 2 style plots.

    After a cpmg run has finished, plot every final CSV from the CPMG-* subfolders on one plot.

    If log_time=True, the x-axis is plotted on a log scale like
    Evolution time with ticks 10^1, 10^2, 10^3, etc.
    """

    '''set up'''
    cpmg_run_dir = "data" / Path(cpmg_run_dir)
    if not cpmg_run_dir.is_dir():
        raise FileNotFoundError(f"CPMG run directory does not exist: {cpmg_run_dir}")
    cpmg_dirs = sorted([p for p in cpmg_run_dir.iterdir() if p.is_dir() and p.name.startswith("CPMG-")],
                       key=_parse_cpmg_n,)
    if not cpmg_dirs:
        raise FileNotFoundError(f"No CPMG-* subdirectories found in: {cpmg_run_dir}")

    '''make plot'''
    fig, ax = plt.subplots(figsize=(6, 5.5))
    n_plotted = 0
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(cpmg_dirs)))

    # plot data
    original_num_points_total = 0
    n_removed_noise_total = 0
    for color, subdir in zip(colors, cpmg_dirs):
        n = _parse_cpmg_n(subdir)
        # load data
        csv_path = subdir / averaged_csv_name
        if not csv_path.is_file():
            print(f"Skipping {subdir.name}: missing {csv_path}")
            continue
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"Skipping {subdir.name}: bad CSV shape {data.shape}")
            continue
        t = data[:, 0]
        L_avg = data[:, 1]
        original_num_points = len(L_avg)
        original_num_points_total += original_num_points
        # implement noise floor
        if noise_floor is not None:
            above_noise = L_avg > noise_floor
            n_removed_noise = np.sum(~above_noise)
            n_removed_noise_total += n_removed_noise
            t = t[above_noise]
            L_avg = L_avg[above_noise]
        ln_L = -np.log(L_avg)
        # implement logtime
        if logtime:
            mask = t > 0
            if not np.any(mask):
                print(f"Skipping {subdir.name}: no positive time values for log x-axis")
                continue
            if np.any(~mask):
                print(f"Skipping nonpositive time values in {subdir.name} for log x-axis")
            t = t[mask]
            ln_L = ln_L[mask]
            if np.any(ln_L <= 0):
                raise ValueError(f"Cannot use log y-axis for {subdir.name} because -ln(L) contains values <= 0.")
        # plot
        line, = ax.plot(t, ln_L, label=rf"$n={int(n)}$" if n != float("inf") else subdir.name,
                        color=color, linewidth=2.0)
        ax.scatter(t, ln_L, marker="x", color=color, s=45, linewidths=1.5)
        n_plotted += 1

    # plot formatting
    if logtime:
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=10))
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=100))
        ax.yaxis.set_major_formatter(
            mticker.LogFormatterMathtext(base=10.0, labelOnlyBase=False, minor_thresholds=(100, 100))
        )
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))

        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        ax.tick_params(axis="x", which="major", length=7)
        ax.tick_params(axis="x", which="minor", length=3)
    if noise_floor is not None:
        fig.text(0.5, 0.01,
            f"Removed {n_removed_noise_total} pts below noise floor: {noise_floor}, out of {original_num_points_total} pts total",
            ha="center")
    ax.set_xlabel(r"Evolution time (ms)", fontsize=24)
    ax.set_ylabel(r"$-\ln |L(t)|$", fontsize=24)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="major", labelsize=18, length=7, width=1.2)
    ax.tick_params(axis="both", which="minor", length=4, width=1.0)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=16, loc="lower right")
    fig.tight_layout()
    output_path = cpmg_run_dir / output_name
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_overlay(data_dir, output_name="overlay.png", logtime=False, xlim=None):
    """
    Plot every CSV file in data_dir on one plot.
    """
    data_dir = Path("data") / data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")

    fig, ax = plt.subplots(figsize=(12, 8))

    n_plotted = 0
    legend_handles = []

    for csv_path in sorted(data_dir.glob("*.csv")):
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        if data.ndim != 2 or data.shape[1] < 2:
            print(f"Skipping {csv_path.name}: bad CSV shape {data.shape}")
            continue

        t = 1000.0 * data[:, 0]  # ms -> us
        L_avg = data[:, 1]

        if logtime:
            mask = t > 0
            if not np.any(mask):
                print(f"Skipping {csv_path.name}: no positive time values for log x-axis")
                continue

            if np.any(~mask):
                print(f"Skipping nonpositive time values in {csv_path.name} for log x-axis")

            t = t[mask]
            L_avg = L_avg[mask]

        label = csv_path.stem

        line, = ax.plot(t, L_avg, label="_nolegend_", marker="o", markersize=2.2, linewidth=1.7, alpha=0.45)
        legend_handles.append(Line2D([], [], color=line.get_color(), linewidth=1.7, alpha=0.45, label=label))
        n_plotted += 1

    if logtime:
        ax.set_xscale("log")

        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))

        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        ax.tick_params(axis="x", which="major", length=7)
        ax.tick_params(axis="x", which="minor", length=3)

    if xlim is not None:
        ax.set_xlim(**xlim)
    ax.set_xlabel(r"Evolution time ($\mu$s)")
    ax.set_ylabel("Coherence")
    ax.grid(False)
    ax.legend(handles=legend_handles, frameon=False)

    fig.tight_layout()

    output_path = data_dir / output_name
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def main():
    # mean_fid("2026.5.29_mean-nobath")
    # fourier("data/2026.5.26_mean/final.csv")
    # alvarez_pp("2026.5.26_alvarez")
    norris_psd_least_sq("2026.5.28_psd-norris-500", 15)
    norris_psd_mle("2026.5.28_psd-norris-500", 15)
    # norris_bispectrum("2026.5.29_bispectrum-norris-polarized", mu_B=172)
    # plot_park_fig2("2026.5.27_park-100-1",
    #                  logtime=True,
    # #                noise_floor=1.0e-2,
    #                  x_lim = [7.35e-4, 1.84e-1],
    #                  y_lim = [1.00e-1, 3.00]
    #                )
    # fit_plot("2026.5.29_bispectrum",
    #          plot_fit=false,
    #          # data_dir="data/2026.5.21_park-100/CPMG-64",
    #          log_time=False,
    #          csv_name="averaged_phase.csv",
    #          plot_name="phase.png",
    #          # t_cutoff=0.004,
    #        )
    # fourier("data/2026.5.29_bispectrum/final.csv")
    # plot_cpmg_ethan("data/2026.5.26_ethan",
    #                 logtime=True,
    #                 tau_min=1.4e-3,
    #                 tau_max=0.56,
    #                 T_max=3,)


if __name__ == '__main__':
    main()

from __future__ import annotations
import os
import numpy as np
from math import ceil
from typing import Tuple
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter,lfilter 
from src.classes.constants import time_to_seconds
from src.classes.physics.transition_process import EVENT_CODES, EVENT_NAMES

mpl.rcParams['font.family']='DejaVu Sans'
plt.rcParams['font.size']=18
plt.rcParams['axes.linewidth']=2



colors = ['g','b','k','c','m','y','r']
lines  = ['-','--','-.',':']

LOCALIZED_CODES   = np.array([EVENT_CODES["GS_tun_recom"],
                              EVENT_CODES["ES_tun_recom"]])
DELOCALIZED_CODES = np.array([EVENT_CODES["GS_CB_recom"],
                              EVENT_CODES["ES_CB_recom"]])
READOUT_CODES     = np.concatenate([LOCALIZED_CODES, DELOCALIZED_CODES])


def plot_crystal(trap_coords: np.ndarray, hole_location: np.ndarray, nearest: np.ndarray, N: int, nearest_no=3):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(trap_coords[:,0], trap_coords[:,1], trap_coords[:,2], c='blue', label='Traps')
    ax.scatter(hole_location[:,0], hole_location[:,1], hole_location[:,2], c='red', label='List2')
    
    for i in range(N):
        hole = hole_location[nearest[i,0]]
        if np.all(hole == trap_coords[i,:]):
            ax.scatter(trap_coords[i,0], trap_coords[i,1], trap_coords[i,2], c='green', label='Traps')
            continue
        xs = [trap_coords[i,0], hole[0]]
        ys = [trap_coords[i,1], hole[1]]
        zs = [trap_coords[i,2], hole[2]]
        ax.plot(xs, ys, zs, 'k--')
    
    if nearest_no > 1:
        for i in range(N):
            hole = hole_location[nearest[i,1]]
            xs = [trap_coords[i,0], hole[0]]
            ys = [trap_coords[i,1], hole[1]]
            zs = [trap_coords[i,2], hole[2]]
            ax.plot(xs, ys, zs, ls='--',color="green") 
    
    if nearest_no > 2:
        for i in range(N):
            hole = hole_location[nearest[i,2]]
            xs = [trap_coords[i,0], hole[0]]
            ys = [trap_coords[i,1], hole[1]]
            zs = [trap_coords[i,2], hole[2]]
            ax.plot(xs, ys, zs, ls='--',color="pink")

    plt.show()

def closest_divisor(N: int) -> int:
    """
    Finds the closest divisor to split the available MC
    runs into.
    """
    if N == 0: 
        return 0 
    
    if N <101:
        di = 10 
    else: 
        di = ceil(N/10)
    
    candidates = range (1, N+1) 
    divisors = [x for x in candidates if N % x == 0]

    x_best = min(divisors, key=lambda x: abs(x-di))
    return x_best

def time_sequence(input_temps, unit):
    if unit != 's':
        if unit == 'Ma':
            temp = input_temps/time_to_seconds[unit]
            last = temp[-1]
            return (((temp-last)*-1))
        else:
            return (input_temps/time_to_seconds[unit])
    else: 
        return input_temps

def _std_from_moments(total: np.ndarray, total_sq: np.ndarray, n: int) -> np.ndarray:

    if n < 2:
        return np.zeros_like(total)
    var = (total_sq - total * total / n) / (n - 1)
    return np.sqrt(np.clip(var, 0.0, None))

def _tl_window(temperature: np.ndarray) -> np.ndarray:

    mask = np.zeros(temperature.size, dtype=bool)
    heating = np.diff(temperature) > 1e-9             # heating[k]: bin k -> k+1 rises
    if not heating.any():
        return mask
    end   = int(np.flatnonzero(heating)[-1])
    start = end
    while start > 0 and heating[start - 1]:           # walk back over the ramp
        start -= 1
    mask[start:end + 2] = True
    return mask


def _glow_bin_count(t_end: float, bin_width: float) -> int:
    return int(np.ceil((t_end + bin_width) / bin_width)) - 1


def _heating_spans(crystal, t_end: float, bin_width: float, n_full: int,
                   pad: int, margin: int) -> list[tuple[int, int, int, int]]:

    times = np.atleast_1d(np.asarray([] if crystal.times is None else crystal.times, dtype=float))
    knots = np.unique(np.concatenate(([0.0], times[(times > 0) & (times < t_end)], [t_end])))
    seg_start, seg_end = knots[:-1], knots[1:]
    eps     = 1e-6 * (seg_end - seg_start)
    heating = crystal.Tat(seg_end - eps) - crystal.Tat(seg_start + eps) > 1e-9

    spans = []
    for lo, hi in zip(seg_start[heating], seg_end[heating]):
        k0 = max(int(lo // bin_width) - margin, 0)
        k1 = min(int(np.ceil(hi / bin_width)) + margin, n_full)
        if spans and k0 <= spans[-1][3]:
            k0 = spans.pop()[2]
        spans.append((max(k0 - pad, 0), min(k1 + pad, n_full), k0, k1))
    return spans


def _glow_spans(crystal, t_end: float, bin_width: float, savgol_window: int,
                max_bins: int, fallback_bins: int) -> tuple[float, int, list[tuple[int, int, int, int]]]:

    n_full = _glow_bin_count(t_end, bin_width)
    win    = min(savgol_window, n_full if n_full % 2 == 1 else n_full - 1)
    if n_full <= max_bins:
        return bin_width, win, [(0, n_full, 0, n_full)]

    spans = _heating_spans(crystal, t_end, bin_width, n_full, win // 2 + 1, savgol_window)
    if spans and sum(c1 - c0 for c0, c1, _, _ in spans) <= max_bins:
        return bin_width, win, spans

    bin_width = t_end / fallback_bins
    n_full    = _glow_bin_count(t_end, bin_width)
    win       = min(savgol_window, n_full if n_full % 2 == 1 else n_full - 1)
    return bin_width, win, [(0, n_full, 0, n_full)]


def _plot_glow_curve(filename: str, x: np.ndarray, glow_rate: np.ndarray,
                     glow_smoothed: np.ndarray | None = None,
                     xlabel: str = "Temperature (°C)",
                     glow_std: np.ndarray | None = None) -> None:

    fig, ax = plt.subplots(figsize=(8, 5))
    if glow_smoothed is not None:
        ax.plot(x, glow_rate, color="firebrick", linewidth=0.8,
                ls="--", alpha=0.4, label="raw")
        if glow_std is not None:
            ax.fill_between(x, glow_smoothed - glow_std, glow_smoothed + glow_std,
                            color="firebrick", alpha=0.18, linewidth=0,
                            label="±1 s.d.")
        ax.plot(x, glow_smoothed, color="firebrick", linewidth=1.5,
                label="smoothed")
        ax.legend(loc="best", fontsize=10)
    else:
        if glow_std is not None:
            ax.fill_between(x, glow_rate - glow_std, glow_rate + glow_std,
                            color="firebrick", alpha=0.18, linewidth=0)
        ax.plot(x, glow_rate, color="firebrick", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Glow intensity")
    ax.grid(True, color="grey", alpha=0.4, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_ratio_bands(filename: str, x: np.ndarray, xlabel: str,
                      mean_curve: np.ndarray, ratio_min: np.ndarray,
                      ratio_max: np.ndarray, ratio_std: np.ndarray | None = None) -> None:

    fig, ax = plt.subplots(figsize=(8, 5))
    if ratio_std is not None:
        ax.fill_between(x, mean_curve - ratio_std, mean_curve + ratio_std,
                        color="steelblue", alpha=0.2, linewidth=0,
                        label="±1 s.d.")
    ax.plot(x, mean_curve, color="steelblue", linewidth=1.5, label="n/N mean")
    ax.plot(x, ratio_min, color="steelblue", ls="--",
            linewidth=1.0, alpha=0.5, label="min / max")
    ax.plot(x, ratio_max, color="steelblue", ls="--",
            linewidth=1.0, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("n/N Trap ratio")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, color="grey", alpha=0.4, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_photon_history(results, output_file_name, crystal) -> str:

    S, C, L = results.shape

    assert C == 4

    rep_file = output_file_name + "_glow_rep"

    valid_mask = ~np.isnan(results[:, 0, :])
    lengths    = valid_mask.sum(axis=1)

    columns = []
    header  = []
    for i in range(S):
        t_i   = results[i, 0, :lengths[i]].astype(np.float64)
        lum_i = results[i, 2, :lengths[i]]

        lum_mask = lum_i >= 1
        t_ph     = np.repeat(t_i[lum_mask], lum_i[lum_mask].astype(int))
        columns.append([f"{t:.10g}" for t in t_ph])
        columns.append([f"{T:.4f}" for T in crystal.Tat(t_ph) - 273.15])
        header += [f"Time (rep {i+1})", f"Temp (rep {i+1})"]

    n_rows = max((len(c) for c in columns), default=0)

    if os.path.exists(f"{rep_file}.csv"):
        os.remove(f"{rep_file}.csv")
    with open(f"{rep_file}.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for j in range(n_rows):
            f.write(",".join(c[j] if j < len(c) else "" for c in columns) + "\n")
    return rep_file


def clean_up_results(results, output_file_name, crystal,
                     plot_error_bands: bool = True,
                     plot_glow_curve: bool = True,
                     glow_bin_width: float = 2,
                     savgol_window: int = 9,
                     savgol_poly: int = 2,
                     max_glow_bins: int = 2_000_000,
                     fallback_glow_bins: int = 20000):
    S, C, L = results.shape

    assert C == 4

    ratio_file = output_file_name+"_ratio"
    lum_file   = output_file_name+"_lum"

    valid_mask = ~np.isnan(results[:, 0, :])
    lengths    = valid_mask.sum(axis=1)

    time_max    = min(max(results[i, 0, lengths[i]-1] for i in range(S)), crystal.duration)
    steady_time = np.linspace(0, time_max, 20000)

    ratio_results = np.zeros((6, steady_time.size))
    ratio_results[0, :] = steady_time

    lum_results = np.zeros((6, steady_time.size))
    lum_results[0, :] = steady_time

    n_codes      = max(EVENT_NAMES.keys()) + 1
    event_counts = np.zeros((steady_time.size, n_codes), dtype=np.int64)

    header     = ["Time (s)", "Temperature (C)",
                  f"n/N mean ({S} reps)", f"n/N std ({S} reps)",
                  "n/N min (all reps)", "n/N max (all reps)"]
    lum_header = ["Time (s)", "Temperature (C)", f"Lum ({S} reps)",
                  f"photon_loc ({S} reps)", f"photon_deloc ({S} reps)",
                  f"photon_fill ({S} reps)"]

    sumed     = np.zeros(steady_time.size)
    sumed_sq  = np.zeros(steady_time.size)
    ratio_min = np.full(steady_time.size, np.inf)
    ratio_max = np.full(steady_time.size, -np.inf)

    for i in range(S):
        t_i   = results[i, 0, :lengths[i]]
        y_i   = results[i, 1, :lengths[i]]
        lum_i = results[i, 2, :lengths[i]]
        c_i   = results[i, 3, :lengths[i]].astype(int)

        interp_i  = np.interp(steady_time, t_i, y_i)
        sumed    += interp_i
        sumed_sq += interp_i * interp_i
        np.minimum(ratio_min, interp_i, out=ratio_min)
        np.maximum(ratio_max, interp_i, out=ratio_max)

        ev_mask = c_i >= 1
        t_ev    = t_i[ev_mask]
        c_ev    = c_i[ev_mask]
        if t_ev.size > 0:
            bin_idx  = np.searchsorted(steady_time, t_ev, side='right') - 1
            bin_idx  = np.clip(bin_idx, 0, steady_time.size - 1)
            flat_idx = bin_idx * n_codes + c_ev
            counts   = np.bincount(flat_idx, minlength=steady_time.size * n_codes)
            event_counts += counts.reshape(steady_time.size, n_codes)

        lum_mask   = lum_i >= 1
        lum_times  = t_i[lum_mask]
        if lum_times.size > 0:
            bin_idx = np.searchsorted(steady_time, lum_times, side='right') - 1
            bin_idx = np.clip(bin_idx, 0, steady_time.size - 1)
            photons   = lum_i[lum_mask]
            lum_codes = c_i[lum_mask]
            np.add.at(lum_results[2], bin_idx, photons)
            loc = np.isin(lum_codes, LOCALIZED_CODES)
            np.add.at(lum_results[3], bin_idx[loc], photons[loc])
            deloc = np.isin(lum_codes, DELOCALIZED_CODES)
            np.add.at(lum_results[4], bin_idx[deloc], photons[deloc])
            fill = ~np.isin(lum_codes, READOUT_CODES)
            np.add.at(lum_results[5], bin_idx[fill], photons[fill])

    ratio_results[2, :] = sumed / S
    ratio_results[3, :] = _std_from_moments(sumed, sumed_sq, S)
    ratio_results[4, :] = ratio_min
    ratio_results[5, :] = ratio_max

    ratio_results[1, :] = crystal.Tat(ratio_results[0, :]) - 273.15

    dominant_codes = event_counts.argmax(axis=1)
    dominant_names = [EVENT_NAMES.get(int(c), EVENT_NAMES[0])
                      for c in dominant_codes]
    header.append("dominant_event_type")

    if os.path.exists(f"{ratio_file}.csv"):
        os.remove(f"{ratio_file}.csv")
    with open(f"{ratio_file}.csv", "w") as f:
        f.write("# " + ",".join(header) + "\n")
        for j in range(steady_time.size):
            numeric_cols = ",".join(f"{ratio_results[r, j]}" for r in range(ratio_results.shape[0]))
            f.write(f"{numeric_cols},{dominant_names[j]}\n")

    if plot_error_bands:
        temperature_plot = ratio_results[1, :]
        time_plot        = ratio_results[0, :]
        mean_curve       = ratio_results[2, :]
        std_curve        = ratio_results[3, :]

        _plot_ratio_bands(f"{output_file_name}_error_bands.png",
                          temperature_plot, "Temperature (°C)",
                          mean_curve, ratio_min, ratio_max, std_curve)
        _plot_ratio_bands(f"{output_file_name}_error_bands_time.png",
                          time_plot, "Time (s)",
                          mean_curve, ratio_min, ratio_max, std_curve)

    del ratio_results

    lum_results[1, :] = crystal.Tat(lum_results[0, :]) - 273.15

    if os.path.exists(f"{lum_file}.csv"):
        os.remove(f"{lum_file}.csv")

    
    lum_fmt = ["%.10g", "%.4f"] + ["%d"] * (lum_results.shape[0] - 2)
    np.savetxt(f"{lum_file}.csv", lum_results.T, delimiter=",",
               fmt=lum_fmt, header=",".join(lum_header))
    del lum_results

    if plot_glow_curve:

        glow_bin_width, win, spans = _glow_spans(crystal, steady_time[-1], glow_bin_width,
                                                 savgol_window, max_glow_bins, fallback_glow_bins)
        span_edges = [np.arange(c0, c1 + 1) * glow_bin_width for c0, c1, _, _ in spans]
        span_keep  = [slice(k0 - c0, k1 - c0) for c0, _, k0, k1 in spans]
        offsets    = np.cumsum([0] + [k1 - k0 for _, _, k0, k1 in spans])
        span_out   = [slice(offsets[j], offsets[j + 1]) for j in range(len(spans))]
        bin_centers = np.concatenate([(0.5 * (e[:-1] + e[1:]))[keep]
                                      for e, keep in zip(span_edges, span_keep)])
        n_bins      = bin_centers.size

    if plot_glow_curve and n_bins >= 2:

        smooth = win > savgol_poly

        raw_sum    = np.zeros(n_bins)
        raw_sumsq  = np.zeros(n_bins)
        sm_sum     = np.zeros(n_bins)
        sm_sumsq   = np.zeros(n_bins)
        sm_loc     = np.zeros(n_bins)
        sm_deloc   = np.zeros(n_bins)
        sm_fill    = np.zeros(n_bins)

        for i in range(S):
            t_i   = results[i, 0, :lengths[i]]
            lum_i = np.clip(results[i, 2, :lengths[i]], 0, None)
            c_i   = results[i, 3, :lengths[i]].astype(int)

            channels = [(np.where(mask, lum_i, 0.0), accum)
                        for mask, accum in ((np.isin(c_i, LOCALIZED_CODES), sm_loc),
                                            (np.isin(c_i, DELOCALIZED_CODES), sm_deloc),
                                            (~np.isin(c_i, READOUT_CODES), sm_fill))]

            for edges, keep, out in zip(span_edges, span_keep, span_out):
                counts, _ = np.histogram(t_i, bins=edges, weights=lum_i)
                raw_i     = counts / glow_bin_width
                sm_i      = savgol_filter(raw_i, win, savgol_poly) if smooth else raw_i
                raw_i, sm_i = raw_i[keep], sm_i[keep]
                raw_sum[out]   += raw_i
                raw_sumsq[out] += raw_i * raw_i
                sm_sum[out]    += sm_i
                sm_sumsq[out]  += sm_i * sm_i

                for w, accum in channels:
                    r = np.histogram(t_i, bins=edges, weights=w)[0] / glow_bin_width
                    accum[out] += (savgol_filter(r, win, savgol_poly) if smooth else r)[keep]

        glow_raw          = raw_sum / S
        glow_raw_std      = _std_from_moments(raw_sum, raw_sumsq, S)
        glow_smoothed     = np.clip(sm_sum / S, 0, None)
        glow_smoothed_std = _std_from_moments(sm_sum, sm_sumsq, S)
        glow_loc          = np.clip(sm_loc / S, 0, None)
        glow_deloc        = np.clip(sm_deloc / S, 0, None)
        glow_fill         = np.clip(sm_fill / S, 0, None)

        temperature_glow = crystal.Tat(bin_centers) - 273.15

        glow_file = output_file_name + "_glow"
        if os.path.exists(f"{glow_file}.csv"):
            os.remove(f"{glow_file}.csv")
        np.savetxt(f"{glow_file}.csv",
                   np.column_stack([bin_centers, temperature_glow,
                                    glow_raw, glow_raw_std,
                                    glow_smoothed, glow_smoothed_std,
                                    glow_loc, glow_deloc, glow_fill]),
                   delimiter=",",
                   header="Time (s),Temperature (C),"
                          "d(N_holes)/dt (holes/s),d(N_holes)/dt std,"
                          "d(N_holes)/dt smoothed,d(N_holes)/dt smoothed std,"
                          "d(N_holes)/dt loc smoothed,d(N_holes)/dt deloc smoothed,"
                          "d(N_holes)/dt fill smoothed")


        tl = _tl_window(temperature_glow)
        if not tl.any():
            tl = np.ones(n_bins, dtype=bool)

        _plot_glow_curve(f"{output_file_name}_glow_curve.png", temperature_glow[tl],
                         glow_raw[tl], glow_smoothed[tl], xlabel="Temperature (°C)")
        _plot_glow_curve(f"{output_file_name}_glow_curve_time.png", bin_centers[tl],
                         glow_raw[tl], glow_smoothed[tl], xlabel="Time (s)")

    return ratio_file, lum_file


def plot_command(x: np.ndarray, y: np.ndarray, ax, colour: str = "black", label: str | None = None) -> None:
    ax.plot(x,y, color=colour, label = label)

def plot_time_label(ax, unit: str = "s") -> None:
    if unit == 'm':
       ax.set_xlabel("Time (min)")
    elif unit == 'h':
         ax.set_xlabel("Time (hour)")
    elif unit == 'd':
        ax.set_xlabel("Time (day)")
    elif unit == 'y':
         ax.set_xlabel("Time (year)")
    elif unit == 'Ma':
        ax.set_xlabel("Time (Ma)")
        ax.invert_xaxis()
    else:
        ax.set_xlabel("Time (s)")
 
def ratio_vs(x: np.ndarray, ratio: np.ndarray, ax, colour: str = "black", 
             line: str = 'solid',label: str | None = None, alpha_ES:float=1.0):
    ax.plot(x,ratio, color=colour, label = label, ls = line, alpha=alpha_ES)


def plot_forward_ratio(file_name:str, data: np.ndarray, times: np.ndarray, T_unit:str, headers = None):
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))
    for i in range(data.shape[1]):
        if headers is not None:
            h = headers[i]
        else:
            h = None
        ratio_vs(times,data[:,i],ax,colors[i%7],lines[i%4],h)
    
    plot_time_label(ax, T_unit)
    ax.set_ylabel("Trapped electron ratio n/N")
    if headers is not None:
        ax.legend()
    plt.savefig(file_name,dpi=300, transparent=False,bbox_inches='tight')
    plt.close()

def plot_forward_ratio_T(file_name:str, data: np.ndarray, temp: np.ndarray, headers = None):
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))
    for i in range(data.shape[1]):
        if headers is not None:
            h = headers[i]
        else:
            h = None
        ratio_vs(temp,data[:,i],ax,colors[i%7],lines[i%4],h)

    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Trapped electron ratio n/N")
    ax.legend()
    plt.savefig(file_name,dpi=300, transparent=False,bbox_inches='tight')
    plt.close()

def plot_T_profile(file_name:str, temp: np.ndarray, times: np.ndarray, T_unit:str): 
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))
    ax.plot(times,temp, color='black')
    plot_time_label(ax, T_unit)
    ax.set_ylabel("Temperature (C)")
    plt.savefig(file_name,dpi=300, transparent=False,bbox_inches='tight')
    plt.close()

def _load_ratio_csv(path: str):
    """Load the ratio CSV, skipping the trailing string column."""
    with open(path) as f:
        header_line = f.readline()
    with open(path) as f:
        f.readline()  # skip header
        first_data = f.readline().strip()
    parts = first_data.split(",")
    ncols = 0
    for p in parts:
        try:
            float(p)
            ncols += 1
        except ValueError:
            break
    data = np.loadtxt(path, delimiter=",", usecols=range(ncols))
    return data, header_line

def _mean_ratio_columns(header_line: str, ncols: int) -> Tuple[list[int], list[str]]:
    """
    Column indices (and names) of the mean n/N curve(s) in a ratio CSV,
    skipping the std, min/max and dominant-event columns.
    """
    names = [h.strip().lstrip('#').strip() for h in header_line.split(',')]
    idx, labels = [], []
    for i, name in enumerate(names[:ncols]):
        low = name.lower()
        if 'mean' in low and 'std' not in low:
            idx.append(i)
            labels.append(name)
    return idx, labels

def plot_forward_results(ratio_file: str, lum_file:str, T_unit: str = 's', T_type: str = 'constant') -> None:

    data, header = _load_ratio_csv(f"{ratio_file}.csv")
    times = time_sequence(data[:,0], T_unit)

    cols, header_names = _mean_ratio_columns(header, data.shape[1])

    plot_forward_ratio(f"Time_filling_{ratio_file}.png",data[:,cols],times, T_unit,header_names)
    plot_T_profile("Temperature_Profile.png",data[:,1],times,T_unit)

    if T_type != 'constant':
        plot_forward_ratio_T(f"Time_filling_{ratio_file}_T.png",data[:,cols],data[:,1],header_names)

def plot_forward_multi_experiment(ratio_files: list[str],file_name: str,T_unit: str = 's') -> None:
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))
    j=0
    for i in range(len(ratio_files)):
        if( i > 0 and i%7 == 0):
            j = ((j+1)%4)

        file = ratio_files[i]
        data, header = _load_ratio_csv(file)
        times = time_sequence(data[:,0], T_unit)
        cols, _ = _mean_ratio_columns(header, data.shape[1])
        ratio_vs(times,data[:,cols[-1]],ax,colors[i%7],lines[j],f"Experiment no.{i+1}")


    plot_time_label(ax, T_unit)
    ax.set_ylabel("Trapped electron ratio n/N") 
  
    ax.legend()
    plt.savefig(file_name,dpi=300, transparent=False,bbox_inches='tight')
    plt.close()


def plot_analytical_results(analytic_file: str, T_unit: str = 's'): 
    
    data = np.loadtxt(f"{analytic_file}.csv", delimiter=",")
    times = time_sequence(data[:,0], T_unit)
  
    plot_forward_ratio(f"{analytic_file}.png",data[:,1:],times,T_unit)

def plot_forward_multi_analytical_experiment(ratio_files: list[str],file_name: str,T_unit: str = 's') -> None:
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))
    j=0
    for i in range(len(ratio_files)):
        if( i > 0 and i%7 == 0):
            j = ((j+1)%4) 
      
        file = ratio_files[i]
        data = np.loadtxt(file, delimiter=",")
        times = time_sequence(data[:,0], T_unit)
        ratio_vs(times,data[:,1:],ax,colors[i%7],lines[j],f"Experiment no.{i+1}")


    plot_time_label(ax, T_unit)
    ax.set_ylabel("Trapped electron ratio n/N") 
  
    ax.legend()
    plt.savefig(file_name,dpi=300, transparent=False,bbox_inches='tight')
    plt.close()


def plot_analytic_comp_MC_single(analytic_file: str, MC_file: str, T_unit: str = 's'):

    A_data = np.loadtxt(analytic_file, delimiter=",")
    MC_data = np.loadtxt(MC_file, delimiter=",")

    a_times = time_sequence(A_data[:,0], T_unit)
    mc_times = time_sequence(MC_data[:,0], T_unit)

    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))

    ratio_vs(a_times,A_data[:,-1],ax,colors[0],lines[0],"Analytic",alpha_ES=0.25)
    ratio_vs(mc_times,MC_data[:,-1],ax,colors[1],lines[0],"MC")
    plot_time_label(ax, T_unit)
    ax.set_ylabel("Trapped electron ratio n/N")
    ax.legend()
    plt.savefig("MC_vs_Analytic_result.png",dpi=300, transparent=False,bbox_inches='tight')
    plt.close()

def plot_analytic_comp_MC_multi(analytic_file: list[str], MC_file: list[str], T_unit: str = 's'):
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))
    
    j=0
    for i in range(len(MC_file)):
        if( i > 0 and i%7 == 0):
            j = ((j+1)%4) 
      
        file = MC_file[i]
        data = np.loadtxt(file, delimiter=",")
        times = time_sequence(data[:,0], T_unit)
        ratio_vs(times,data[:,-1],ax,colors[i%7],lines[j],f"MC Experiment no.{i+1}")

    j=2
    for i in range(len(analytic_file)):
        if( i > 0 and i%7 == 0):
            j = ((j+1)%4) 
      
        file = analytic_file[i]
        data = np.loadtxt(file, delimiter=",")
        times = time_sequence(data[:,0], T_unit)
        ratio_vs(times,data[:,1:],ax,colors[i%7],lines[j],f"Analytic Experiment no.{i+1}",alpha_ES=0.5)

    plot_time_label(ax, T_unit)
    ax.set_ylabel("Trapped electron ratio n/N")
    ax.legend()
    plt.savefig("MC_vs_Analytic_result.png",dpi=300, transparent=False,bbox_inches='tight')
    plt.close()


def plot_analytic_comp_MC(run_num: int, analytic_file: str|list[str], MC_file: str|list[str], T_unit: str = 's'):

    if run_num == 1 and isinstance(analytic_file,str) and isinstance(MC_file,str):
        plot_analytic_comp_MC_single(analytic_file,MC_file,T_unit)
    elif isinstance(analytic_file,list) and isinstance(MC_file,list):
        plot_analytic_comp_MC_multi(analytic_file,MC_file,T_unit)

def running_mean(y: np.ndarray, k: int = 5, x: np.ndarray | None = None) -> np.ndarray | Tuple[np.ndarray,np.ndarray]:
    ret = np.cumsum(y, dtype=float)
    ret[k:] = ret[k:] - ret[:-k]
    y_smooth = ret[k-1:] /k 
    if x is None:
        return y_smooth

    x_smooth = (x[:len(x)-k+1] + x[k-1:]) / 2 

    return y_smooth, x_smooth


def smoothed_with_running_mean(ax, x: np.ndarray ,y: np.ndarray , k: int = 5):
    """Adds raw data and Running mean smoothing to a plot"""
   
    rmy, rmx = running_mean(y, k, x)

    ratio_vs(rmx,rmy,ax,colors[2],lines[0],f"Running Mean, k={k}")
    ratio_vs(x,y,ax,colors[0],lines[0],"Raw data",alpha_ES=0.5)

    
def smoothed_with_savgol(ax, x: np.ndarray ,y: np.ndarray , win: int = 50, pol: int =3):
    """Adds raw data and Savitzky-Golay Filter smoothing to a plot"""
   
    sg = savgol_filter(y, win, pol)

    ratio_vs(x,sg,ax,colors[2],lines[0],f" Savitzky-Golay Filter, window={win}, order={pol}")
    ratio_vs(x,y,ax,colors[0],lines[0],"Raw data",alpha_ES=0.5)


def smoothed_with_lfilter(ax, x: np.ndarray ,y: np.ndarray, fs: float = 1000.0, fc: float = 30.0, order: int = 4):
    """Adds raw data and Savitzky-Golay Filter smoothing to a plot"""
    # from scipy.signal import butter, sosfilt, sosfiltfilt
    # sos = butter(order, fc, btype="low", fs=fs, output="sos")
    # y_causal = sosfilt(sos, x)
    # # Zero-phase (forward/backward) filtering — no phase shift, similar to filtfilt
    # y_zerophase = sosfiltfilt(sos, x)


    b = [1.0/fc]*int(fc)
    y_lf = lfilter(b,order,y)
    ratio_vs(x,y_lf,ax,colors[4],lines[0],"lf")
    ratio_vs(x,y,ax,colors[0],lines[0],"Raw data",alpha_ES=0.5)


def base_smoothing(filename:str, x: np.ndarray ,y: np.ndarray, S_type:str, T_unit:str = "s", y_label:str ="n/N Trap ratio", 
                   k:int=5, win:int=50, pol:int=3, fs: float = 1000.0, fc: float = 30.0, order: int = 4): 
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))

    if S_type == 'rm':
        smoothed_with_running_mean(ax, x, y, k)
    elif S_type == 'sg':
        smoothed_with_savgol(ax, x, y, win, pol)
    elif S_type == 'lf':
        smoothed_with_lfilter(ax, x, y, fs, fc, order)
    
    plot_time_label(ax,T_unit)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.savefig(filename,dpi=300, transparent=False,bbox_inches='tight')
    plt.close()


def comp_smooth(MC_file: str, T_unit: str = 's'):

    MC_data = np.loadtxt(MC_file, delimiter=",")
    time = time_sequence(MC_data[:,0], T_unit)
    data = MC_data[:,-1]

    base_smoothing("Running_mean_smooth2000.png",time,data, "rm", T_unit=T_unit,k=2000)
    base_smoothing("lfilter1000_1.png",time,data, "lf", T_unit=T_unit, fc=1000, order=1)
    base_smoothing("lfilter2000_1.png",time,data, "lf", T_unit=T_unit, fc=2000, order=1)

 
def plot_analytic_comp_MC_smoothed(analytic_file: str, MC_file: str, T_unit: str = 's'):
    

    A_data = np.loadtxt(analytic_file, delimiter=",")
    MC_data = np.loadtxt(MC_file, delimiter=",")
    
    a_times = time_sequence(A_data[:,0], T_unit)
    mc_times = time_sequence(MC_data[:,0], T_unit)

    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))

    smoothed_with_running_mean(ax, mc_times, MC_data[:,-1], 2000)
    ratio_vs(a_times,A_data[:,-1],ax,"m",lines[1],"Analytic",alpha_ES=0.25)

   
    plot_time_label(ax, T_unit)
    ax.set_ylabel("Trapped electron ratio n/N")
   
    ax.legend()
    plt.savefig("MC_vs_Analytic_smooth_result.png",dpi=300, transparent=False,bbox_inches='tight')
    plt.close()



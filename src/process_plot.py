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
from src.classes.physics.transition_process import EVENT_NAMES

mpl.rcParams['font.family']='DejaVu Sans'
plt.rcParams['font.size']=18
plt.rcParams['axes.linewidth']=2



colors = ['g','b','k','c','m','y','r']
lines  = ['-','--','-.',':']


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

def _plot_glow_curve(output_file_name: str,
                     temperature: np.ndarray,
                     glow_rate: np.ndarray,
                     glow_smoothed: np.ndarray | None = None) -> None:
    """
    Plot the TL glow curve: d(N_holes)/dt vs Temperature.

    If glow_smoothed is provided the raw curve is drawn as a faint dashed
    line and the smoothed curve as a solid line.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if glow_smoothed is not None:
        ax.plot(temperature, glow_rate, color="firebrick", linewidth=0.8,
                ls="--", alpha=0.4, label="raw")
        ax.plot(temperature, glow_smoothed, color="firebrick", linewidth=1.5,
                label="smoothed")
        ax.legend(loc="best", fontsize=10)
    else:
        ax.plot(temperature, glow_rate, color="firebrick", linewidth=1.5)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Glow intensity")
    ax.grid(True, color="grey", alpha=0.4, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(f"{output_file_name}_glow_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def clean_up_results(results, output_file_name, crystal,
                     plot_error_bands: bool = True,
                     plot_glow_curve: bool = True,
                     glow_bin_width: float = 1,
                     savgol_window: int = 15,
                     savgol_poly: int = 2):
    S, C, L = results.shape

    assert C == 4

    ratio_file = output_file_name+"_ratio"
    lum_file   = output_file_name+"_lum"

    valid_mask = ~np.isnan(results[:, 0, :])
    lengths    = valid_mask.sum(axis=1)

    time_max    = max(results[i, 0, lengths[i]-1] for i in range(S))
    steady_time = np.linspace(0, time_max, 20000)
    divisor     = closest_divisor(S)
    additional  = S // divisor - 1

    ratio_results = np.zeros((5 + additional, steady_time.size))
    ratio_results[0, :] = steady_time

    lum_results = np.zeros((3 + additional, steady_time.size))
    lum_results[0, :] = steady_time

    n_codes      = max(EVENT_NAMES.keys()) + 1
    event_counts = np.zeros((steady_time.size, n_codes), dtype=np.int64)

    cnt       = 0
    cnt_lum   = 0
    header     = ["Time (s)", "Temperature (C)"]
    lum_header = ["Time (s)", "Temperature (C)"]

    sumed     = np.zeros(steady_time.size)
    ratio_min = np.full(steady_time.size, np.inf)
    ratio_max = np.full(steady_time.size, -np.inf)

    for i in range(S):
        t_i   = results[i, 0, :lengths[i]]
        y_i   = results[i, 1, :lengths[i]]
        lum_i = results[i, 2, :lengths[i]]
        c_i   = results[i, 3, :lengths[i]].astype(int)

        interp_i = np.interp(steady_time, t_i, y_i)
        sumed   += interp_i
        np.minimum(ratio_min, interp_i, out=ratio_min)
        np.maximum(ratio_max, interp_i, out=ratio_max)

        if ((i + 1) % divisor == 0 and i != 0) or (S == 1):
            ratio_results[2 + cnt, :] = sumed / (i + 1)
            cnt += 1
            header.append(f"n/N (avg {i+1} reps)")

        ev_mask = c_i >= 1
        t_ev    = t_i[ev_mask]
        c_ev    = c_i[ev_mask]
        if t_ev.size > 0:
            bin_idx  = np.searchsorted(steady_time, t_ev, side='right') - 1
            bin_idx  = np.clip(bin_idx, 0, steady_time.size - 1)
            flat_idx = bin_idx * n_codes + c_ev
            counts   = np.bincount(flat_idx, minlength=steady_time.size * n_codes)
            event_counts += counts.reshape(steady_time.size, n_codes)

        lum_times = t_i[lum_i == 1]
        if lum_times.size > 0:
            bin_idx = np.searchsorted(steady_time, lum_times, side='right') - 1
            bin_idx = np.clip(bin_idx, 0, steady_time.size - 1)
            np.add.at(lum_results[2 + cnt_lum], bin_idx, 1)
        if (i + 1) % divisor == 0 and i != 0:
            if i < S - 1:
                lum_results[2 + cnt_lum + 1] += lum_results[2 + cnt_lum]
                lum_header.append(f"Lum (avg {i+1} reps)")
            cnt_lum += 1
        elif S == 1:
            lum_header.append(f"Lum (avg {i+1} reps)")

    ratio_results[3 + additional, :] = ratio_min
    ratio_results[4 + additional, :] = ratio_max
    header.append("n/N min (all reps)")
    header.append("n/N max (all reps)")

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
        mean_curve       = ratio_results[2 + additional, :]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(temperature_plot, mean_curve, color="steelblue",
                linewidth=1.5, label="n/N mean")
        ax.plot(temperature_plot, ratio_min, color="steelblue", ls="--",
                linewidth=1.0, alpha=0.5, label="min / max")
        ax.plot(temperature_plot, ratio_max, color="steelblue", ls="--",
                linewidth=1.0, alpha=0.5)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("n/N Trap ratio")
        ax.set_ylim(0, 1.1)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, color="grey", alpha=0.4, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(f"{output_file_name}_error_bands.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    del ratio_results

    lum_results[1, :] = crystal.Tat(lum_results[0, :]) - 273.15

    if os.path.exists(f"{lum_file}.csv"):
        os.remove(f"{lum_file}.csv")
    np.savetxt(f"{lum_file}.csv", lum_results.T, delimiter=",",
               header=",".join(lum_header))
    del lum_results

    if plot_glow_curve:

        all_lum_times = []
        for i in range(S):
            t_i   = results[i, 0, :lengths[i]]
            lum_i = results[i, 2, :lengths[i]]
            all_lum_times.append(t_i[lum_i == 1])
        all_lum_times = np.concatenate(all_lum_times)

        bin_edges   = np.arange(0, steady_time[-1] + glow_bin_width, glow_bin_width)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        counts, _   = np.histogram(all_lum_times, bins=bin_edges)
        cum_events  = np.cumsum(counts / S)          # average over reps
        glow_raw    = np.gradient(cum_events, bin_centers)

        win = min(savgol_window, len(glow_raw) if len(glow_raw) % 2 == 1 else len(glow_raw) - 1)
        glow_smoothed = np.clip(savgol_filter(glow_raw, win, savgol_poly), 0, None)

        temperature_glow = crystal.Tat(bin_centers) - 273.15

        glow_file = output_file_name + "_glow"
        if os.path.exists(f"{glow_file}.csv"):
            os.remove(f"{glow_file}.csv")
        np.savetxt(f"{glow_file}.csv",
                   np.column_stack([bin_centers, temperature_glow, glow_raw]),
                   delimiter=",",
                   header="Time (s),Temperature (C),"
                          "d(N_holes)/dt (holes/s),d(N_holes)/dt smoothed")

        _plot_glow_curve(output_file_name, temperature_glow,
                         glow_raw, glow_smoothed)

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

def plot_forward_results(ratio_file: str, lum_file:str, T_unit: str = 's', T_type: str = 'constant') -> None:

    data, header = _load_ratio_csv(f"{ratio_file}.csv")
    times = time_sequence(data[:,0], T_unit)

    header_names = header.split(',')[2:]
    
    header_names = [h for h in header_names
                    if 'event' not in h.lower() and 'min' not in h.lower()
                    and 'max' not in h.lower()]

    plot_forward_ratio(f"Time_filling_{ratio_file}.png",data[:,2:-2],times, T_unit,header_names)
    plot_T_profile("Temperature_Profile.png",data[:,1],times,T_unit)

    if T_type != 'constant':
        plot_forward_ratio_T(f"Time_filling_{ratio_file}_T.png",data[:,2:-2],data[:,1],header_names)

def plot_forward_multi_experiment(ratio_files: list[str],file_name: str,T_unit: str = 's') -> None:
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))
    j=0
    for i in range(len(ratio_files)):
        if( i > 0 and i%7 == 0):
            j = ((j+1)%4)

        file = ratio_files[i]
        data, _ = _load_ratio_csv(file)
        times = time_sequence(data[:,0], T_unit)
        ratio_vs(times,data[:,-1],ax,colors[i%7],lines[j],f"Experiment no.{i+1}")


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



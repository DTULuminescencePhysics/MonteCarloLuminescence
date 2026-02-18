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
    """Finds the closest divisor to split the available MC
    runs into"""
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

def clean_up_results(results, output_file_name, crystal):
    S, C, L = results.shape

    assert C == 4

    ratio_file = output_file_name+"_ratio"
    lum_file = output_file_name+"_lum"

    valid_mask = ~np.isnan(results[:, 0, :])
    lengths = valid_mask.sum(axis=1)

    times_list = [results[i, 0, :lengths[i]] for i in range(S)]
    time_union = np.unique(np.concatenate(times_list))

    del times_list
    steady_time = np.linspace(0,time_union[-1],10000)
    divisor = closest_divisor(S)
    additional = S/divisor -1

    ratio_results = np.zeros((int(3+additional), steady_time.size))
    ratio_results[0,:] = steady_time

    cnt = 0
    header = ["Time (s)", "Temperature (C)"]

    sumed = np.zeros(time_union.size)
    for i in range(S):
        sumed += np.interp(time_union, results[i,0, :lengths[i]], results[i,1,:lengths[i]])

        if(((i+1) % divisor == 0 and i !=0) or (S == 1)):

            ratio_results[2+cnt,:] = (np.interp(steady_time,time_union,sumed))/(i+1)

            cnt+=1
            header.append(f"n/N (avg {i+1} reps)")

    ratio_results[1,:] = crystal.Tat(ratio_results[0,:])
    ratio_results[1,:] -= 273.15

    # ── Dominant event type per steady_time bin ──────────────────────
    all_codes = []
    all_times = []
    for i in range(S):
        t = results[i, 0, :lengths[i]]
        c = results[i, 3, :lengths[i]]
        event_mask = c >= 1   # only transition events (exclude no_event)
        all_codes.append(c[event_mask])
        all_times.append(t[event_mask])
    all_codes = np.concatenate(all_codes)
    all_times = np.concatenate(all_times)

    bins = np.digitize(all_times, steady_time)
    dominant_names = []
    for b in range(steady_time.size):
        codes_in_bin = all_codes[bins == b].astype(int)
        if codes_in_bin.size > 0:
            dominant_names.append(EVENT_NAMES[int(np.bincount(codes_in_bin).argmax())])
        else:
            dominant_names.append(EVENT_NAMES[0])
    header.append("dominant_event_type")

    # ── Write ratio CSV with mixed numeric + string columns ─────────
    if os.path.exists(f"{ratio_file}.csv"):
        os.remove(f"{ratio_file}.csv")
    with open(f"{ratio_file}.csv", "w") as f:
        f.write("# " + ",".join(header) + "\n")
        for j in range(steady_time.size):
            numeric_cols = ",".join(f"{ratio_results[r, j]}" for r in range(ratio_results.shape[0]))
            f.write(f"{numeric_cols},{dominant_names[j]}\n")

    del ratio_results

    # ── Luminescence binning (unchanged logic, uses channel 2) ──────
    cnt = 0
    lum_results = np.zeros((int(3+additional), time_union.size))
    lum_results[0,:] = time_union

    lum_header = ["Time (s)", "Temperature (C)"]
    for i in range(S):
        zeros = np.zeros(time_union.size)
        temp = results[i,0,:lengths[i]]
        mask = np.isin(time_union, temp[results[i,2,:lengths[i]]==1])
        zeros[mask] = 1
        lum_results[2+cnt,:]+= zeros
        if((i+1) % divisor == 0 and i !=0):
            if i < S-1:
                lum_results[2+cnt+1,:]+= lum_results[2+cnt,:]
                lum_header.append(f"Lum (avg {i+1} reps)")
            cnt+=1
        elif (S == 1):
            lum_header.append(f"Lum (avg {i+1} reps)")

    lum_results[1,:] = crystal.Tat(lum_results[0,:])
    lum_results[1,:] -= 273.15

    if os.path.exists(f"{lum_file}.csv"):
        os.remove(f"{lum_file}.csv")
    np.savetxt(f"{lum_file}.csv", lum_results.T, delimiter=",", header=",".join(lum_header))
    del lum_results
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
             line: str = 'solid',label: str | None = None, alpha:float=1.0):
    ax.plot(x,ratio, color=colour, label = label, ls = line, alpha=alpha)


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
    ax.set_ylabel("n/N Trap ratio")
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
    ax.set_ylabel("n/N Trap ratio")
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
    # Count numeric columns by reading first data line
    with open(path) as f:
        f.readline()  # skip header
        first_data = f.readline().strip()
    parts = first_data.split(",")
    # Find how many leading columns are numeric
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
    # Remove the event_type column name from header_names for plotting
    header_names = [h for h in header_names if 'event' not in h.lower()]

    plot_forward_ratio(f"Time_filling_{ratio_file}.png",data[:,2:],times, T_unit,header_names)
    plot_T_profile("Temperature_Profile.png",data[:,1],times,T_unit)

    if T_type != 'constant':
        plot_forward_ratio_T(f"Time_filling_{ratio_file}_T.png",data[:,2:],data[:,1],header_names)

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
    ax.set_ylabel("n/N Trap ratio") 
  
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
    ax.set_ylabel("n/N Trap ratio") 
  
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

    ratio_vs(a_times,A_data[:,-1],ax,colors[0],lines[0],"Analytic",alpha=0.25)
    ratio_vs(mc_times,MC_data[:,-1],ax,colors[1],lines[0],"MC")
    plot_time_label(ax, T_unit)
    ax.set_ylabel("n/N Trap ratio")
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
        ratio_vs(times,data[:,1:],ax,colors[i%7],lines[j],f"Analytic Experiment no.{i+1}",alpha=0.5)

    plot_time_label(ax, T_unit)
    ax.set_ylabel("n/N Trap ratio")
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
    ratio_vs(x,y,ax,colors[0],lines[0],"Raw data",alpha=0.5)

    
def smoothed_with_savgol(ax, x: np.ndarray ,y: np.ndarray , win: int = 50, pol: int =3):
    """Adds raw data and Savitzky-Golay Filter smoothing to a plot"""
   
    sg = savgol_filter(y, win, pol)

    ratio_vs(x,sg,ax,colors[2],lines[0],f" Savitzky-Golay Filter, window={win}, order={pol}")
    ratio_vs(x,y,ax,colors[0],lines[0],"Raw data",alpha=0.5)


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
    ratio_vs(x,y,ax,colors[0],lines[0],"Raw data",alpha=0.5)


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
    ratio_vs(a_times,A_data[:,-1],ax,"m",lines[1],"Analytic",alpha=0.25)

   
    plot_time_label(ax, T_unit)
    ax.set_ylabel("n/N Trap ratio")
   
    ax.legend()
    plt.savefig("MC_vs_Analytic_smooth_result.png",dpi=300, transparent=False,bbox_inches='tight')
    plt.close()



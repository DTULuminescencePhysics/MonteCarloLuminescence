from __future__ import annotations
import numpy as np
from math import ceil
import matplotlib as mpl
from matplotlib import pyplot as plt
from src.classes.constants import time_to_seconds
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.classes.output.results_file import output_file

mpl.rcParams['font.family']='DejaVu Sans'
plt.rcParams['font.size']=18
plt.rcParams['axes.linewidth']=2

colors = ['g','b','k','c','m','y','r']
lines  = ['-','--','-.',':']

def get_timesteps_from_nan(results) -> np.ndarray:
    """Takes intermediate results data and finds when each 
    repetition ends i.e. time inputs are nan not floats"""
    times = results[:, 0, :]
    return np.sum(~np.isnan(times), axis=1)

def compute_mean_fill_from_results(results, quantiles=(0.1, 0.5, 0.9)):

    reps, _, _ = results.shape
    lengths = get_timesteps_from_nan(results)
    
    timeSteps = np.unique(np.concatenate([results[r,0,:lengths[r]]for r in range(reps) if lengths[r]>0]))
    timeSteps.sort()
    n_bins = timeSteps.size-1
    codes = np.array((0,1,2,3,4,5,6,7,8,9,10))
    counts = np.zeros((n_bins,codes.size),dtype=float)
    lumin = np.zeros((n_bins,2),dtype=float)
    q_lists = {q: [] for q in quantiles}
    fill_mean = np.zeros(n_bins)
    fill_std = np.zeros(n_bins)
    i = 0
    fill_vals = np.zeros(reps)
    for a in timeSteps[:-1]:
        j= -1
        for r in range(reps):
            L = lengths[r]
            if L <= 0:
                continue

            run_end = results[r, 0, L-1]
            if not (a < run_end):
                continue

            idx = np.searchsorted(results[r, 0, :L], a, side="right") - 1
            if idx >= 0:
                j+=1
                fill_vals[j] = results[r, 1, idx]
                counts[i, int(results[r, 3, idx])] += 1.0
               
        if j> 0:
            fill_mean[i] = np.mean(fill_vals[:j+1])
            fill_std[i] = np.std(fill_vals[:j+1], ddof=1)
            counts[i, :] /= (j+1)
            for q in quantiles:
                q_lists[q].append(np.quantile(fill_vals[:j+1], q))
        i+=1
    
    lumin[:,0] = counts[:, (np.arange(counts.shape[1]) > 1) & (np.arange(counts.shape[1]) % 2 == 0)].sum(axis=1)
    lumin[:,1] = counts[:,1]

    results = {
        "timeSteps": timeSteps,
        "meanTrapRatio": fill_mean,
        "stdTrapRatio": fill_std,
        "quantiles": {q: np.asarray(v) for q, v in q_lists.items()},
        "luminescence": lumin[:,0],
        "filling": lumin[:,1],
        "events": counts
    }

    return results


def clean_up_results(results, crystal, resultsFile: output_file):

    cleaned_results = compute_mean_fill_from_results(results)
    cleaned_results["temperature"] = crystal.Tat(cleaned_results["timeSteps"])
    resultsFile.output_data_build(cleaned_results)
   

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

# def clean_up_results(results, output_file_name, crystal):
#     S, C, L = results.shape

#     assert C == 4

#     ratio_file = output_file_name+"_ratio"
#     lum_file = output_file_name+"_lum"

#     valid_mask = ~np.isnan(results[:, 0, :])
#     lengths = valid_mask.sum(axis=1)

#     times_list = [results[i, 0, :lengths[i]] for i in range(S)]
#     time_union = np.unique(np.concatenate(times_list))

#     del times_list
#     steady_time = np.linspace(0,time_union[-1],10000)
#     divisor = closest_divisor(S)
#     additional = S/divisor -1

#     ratio_results = np.zeros((int(3+additional), steady_time.size))
#     ratio_results[0,:] = steady_time

#     cnt = 0
#     header = ["Time (s)", "Temperature (C)"]

#     sumed = np.zeros(time_union.size)
#     for i in range(S):
#         sumed += np.interp(time_union, results[i,0, :lengths[i]], results[i,1,:lengths[i]])

#         if(((i+1) % divisor == 0 and i !=0) or (S == 1)):

#             ratio_results[2+cnt,:] = (np.interp(steady_time,time_union,sumed))/(i+1)

#             cnt+=1
#             header.append(f"n/N (avg {i+1} reps)")

#     ratio_results[1,:] = crystal.Tat(ratio_results[0,:])
#     ratio_results[1,:] -= 273.15

#     all_codes = []
#     all_times = []
#     for i in range(S):
#         t = results[i, 0, :lengths[i]]
#         c = results[i, 3, :lengths[i]]
#         event_mask = c >= 1   # only transition events (exclude no_event)
#         all_codes.append(c[event_mask])
#         all_times.append(t[event_mask])
#     all_codes = np.concatenate(all_codes)
#     all_times = np.concatenate(all_times)

#     bins = np.digitize(all_times, steady_time)
#     dominant_names = []
#     for b in range(steady_time.size):
#         codes_in_bin = all_codes[bins == b].astype(int)
#         if codes_in_bin.size > 0:
#             print(int(np.bincount(codes_in_bin).argmax()))
#             dominant_names.append(int(np.bincount(codes_in_bin).argmax()))
#             # dominant_names.append(EVENT_NAMES[int(np.bincount(codes_in_bin).argmax())])
#         else:
#             dominant_names.append(0)
#             # dominant_names.append(EVENT_NAMES[0])
#     header.append("dominant_event_type")

#     if os.path.exists(f"{ratio_file}.csv"):
#         os.remove(f"{ratio_file}.csv")
#     with open(f"{ratio_file}.csv", "w") as f:
#         f.write("# " + ",".join(header) + "\n")
#         for j in range(steady_time.size):
#             numeric_cols = ",".join(f"{ratio_results[r, j]}" for r in range(ratio_results.shape[0]))
#             f.write(f"{numeric_cols},{dominant_names[j]}\n")

#     del ratio_results

#     cnt = 0
#     lum_results = np.zeros((int(3+additional), time_union.size))
#     lum_results[0,:] = time_union

#     lum_header = ["Time (s)", "Temperature (C)"]
#     for i in range(S):
#         zeros = np.zeros(time_union.size)
#         temp = results[i,0,:lengths[i]]
#         mask = np.isin(time_union, temp[results[i,2,:lengths[i]]==1])
#         zeros[mask] = 1
#         lum_results[2+cnt,:]+= zeros
#         if((i+1) % divisor == 0 and i !=0):
#             if i < S-1:
#                 lum_results[2+cnt+1,:]+= lum_results[2+cnt,:]
#                 lum_header.append(f"Lum (avg {i+1} reps)")
#             cnt+=1
#         elif (S == 1):
#             lum_header.append(f"Lum (avg {i+1} reps)")

#     lum_results[1,:] = crystal.Tat(lum_results[0,:])
#     lum_results[1,:] -= 273.15

#     if os.path.exists(f"{lum_file}.csv"):
#         os.remove(f"{lum_file}.csv")
#     np.savetxt(f"{lum_file}.csv", lum_results.T, delimiter=",", header=",".join(lum_header))
#     del lum_results
#     return ratio_file, lum_file














# def running_mean(y: np.ndarray, k: int = 5, x: np.ndarray | None = None) -> np.ndarray | Tuple[np.ndarray,np.ndarray]:
#     """Calcualtes running mean"""
#     ret = np.cumsum(y, dtype=float)
#     ret[k:] = ret[k:] - ret[:-k]
#     y_smooth = ret[k-1:] /k 
#     if x is None:
#         return y_smooth

#     x_smooth = (x[:len(x)-k+1] + x[k-1:]) / 2 

#     return y_smooth, x_smooth


 
# def smoothed_with_savgol(ax, x: np.ndarray ,y: np.ndarray , win: int = 50, pol: int =3):
#     """Adds raw data and Savitzky-Golay Filter smoothing to a plot"""
   
#     sg = savgol_filter(y, win, pol)
#     return sg 



# def smoothed_with_lfilter(ax, x: np.ndarray ,y: np.ndarray, fs: float = 1000.0, fc: float = 30.0, order: int = 4):
#     """Adds raw data and Savitzky-Golay Filter smoothing to a plot"""
#     # from scipy.signal import butter, sosfilt, sosfiltfilt
#     # sos = butter(order, fc, btype="low", fs=fs, output="sos")
#     # y_causal = sosfilt(sos, x)
#     # # Zero-phase (forward/backward) filtering — no phase shift, similar to filtfilt
#     # y_zerophase = sosfiltfilt(sos, x)


#     b = [1.0/fc]*int(fc)
#     y_lf = lfilter(b,order,y)
#     return y_lf


# def base_smoothing(filename:str, x: np.ndarray ,y: np.ndarray, S_type:str, T_unit:str = "s", y_label:str ="n/N Trap ratio", 
#                    k:int=5, win:int=50, pol:int=3, fs: float = 1000.0, fc: float = 30.0, order: int = 4): 
#     fig=plt.figure(figsize=(3.37,5.055))
#     ax=fig.add_axes((0.,0.,2.,1.))

#     if S_type == 'rm':
#         smoothed_with_running_mean(ax, x, y, k)
#     elif S_type == 'sg':
#         smoothed_with_savgol(ax, x, y, win, pol)
#     elif S_type == 'lf':
#         smoothed_with_lfilter(ax, x, y, fs, fc, order)
    
#     plot_time_label(ax,T_unit)
#     ax.set_ylabel(y_label)
#     ax.legend()
#     plt.savefig(filename,dpi=300, transparent=False,bbox_inches='tight')
#     plt.close()


# def comp_smooth(MC_file: str, T_unit: str = 's'):

#     MC_data = np.loadtxt(MC_file, delimiter=",")
#     time = time_sequence(MC_data[:,0], T_unit)
#     data = MC_data[:,-1]

#     base_smoothing("Running_mean_smooth2000.png",time,data, "rm", T_unit=T_unit,k=2000)
#     base_smoothing("lfilter1000_1.png",time,data, "lf", T_unit=T_unit, fc=1000, order=1)
#     base_smoothing("lfilter2000_1.png",time,data, "lf", T_unit=T_unit, fc=2000, order=1)

 




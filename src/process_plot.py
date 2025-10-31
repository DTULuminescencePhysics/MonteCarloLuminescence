from __future__ import annotations
import pandas as pd 
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from src.classes.constants import time_to_seconds
import numpy as np
import glob
import os, csv

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

def clean_up_results(results, output_file_name) -> np.memmap:
    S, C, L = results.shape
    assert C == 3

    valid_mask = ~np.isnan(results[:, 0, :])
    lengths = valid_mask.sum(axis=1)

    times_list = [results[i, 0, :lengths[i]] for i in range(S)]
    time_union = np.unique(np.concatenate(times_list))

    del times_list
 
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
    # time, temp, ratio, ng, ne, lum, lums.....
    out = np.memmap(output_file_name, dtype=np.float32, mode='w+', shape=(S+6, time_union.size))
    out[:, :] = 0
    out[0, :] = time_union
    out.flush()
    for i in range(S):
        out[2, :] += np.interp(time_union, results[i,0, :lengths[i]], results[i,1,:lengths[i]])
        temp = results[i,0,:lengths[i]]
        mask = np.isin(time_union, temp[results[i,2,:lengths[i]]==1])
        out[6+i,mask] = 1
        out[5,:]+= out[6+i,:]
        out.flush()
    
    out[2, :] /= S
    out[5, :] /= S
    out.flush()
    return out

def save_data(out: np.memmap, file_name="MC_results.csv") -> None: 
    header = ["Time", "Temperature", "n/N", f"n$_g$", f"n$_e$", "Lum"]
    # for i in range(6,out.shape[0]):
    #     header.append(f"Lum_{i-5}")
    np.savetxt(file_name, out[0:6,:].T, delimiter=",", header=",".join(header))

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
 
def ratio_vs(ratio: np.ndarray, temp: np.ndarray, ax, colour: str = "black"):
    ax.plot(ratio,temp, color=colour)

def time_sequence(input_temps, unit):
    if unit != 's':
        if unit == 'Ma':
            temp = input_temps/time_to_seconds[unit]
            last = temp[-1]
            return (temp-last)*-1
        else:
            return (input_temps/time_to_seconds[unit])
    else: 
        return input_temps
    

def plot_forward_results(input: str |  np.memmap, Time: str = 's', T_type: str = 'constant') -> None:
    if isinstance(input, str): 
        data = np.loadtxt(input, delimiter=",", skiprows=1)
    else: 
        data = input

    times = time_sequence(data[:,0], Time)
    mpl.rcParams['font.family']='DejaVu Sans'
    plt.rcParams['font.size']=18
    plt.rcParams['axes.linewidth']=2
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes([0.,0.,2.,1.])
    plot_command(times, data[:,1], ax, "black")
    plot_time_label(ax, Time)
    ax.set_ylabel("Temperature (C)")
    plt.savefig("Temperature_Profile.png",dpi=300, transparent=False,bbox_inches='tight')
    plt.close()
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes([0.,0.,2.,1.])
    plot_command(times,data[:,2],ax,"black")
    plot_time_label(ax, Time)
    ax.set_ylabel("n/N Trap ratio")
    # ax.set_ylim(0,1)
    plt.savefig("Time_filling_ratio.png",dpi=300, transparent=False,bbox_inches='tight')
    plt.close()

    if T_type != 'constant':
        fig=plt.figure(figsize=(3.37,5.055))
        ax=fig.add_axes([0.,0.,2.,1.])
        plot_command(data[:,1],data[:,2],ax,"black")
        ax.set_xlabel("Temperature (C)")
        ax.set_ylabel("n/N Trap ratio")
        plt.savefig("Temp_filling_ratio.png",dpi=300, transparent=False,bbox_inches='tight')
        plt.close()




# def build_step_series(time_file,elec_file, trp_file, t_grid):
#         p = np.searchsorted(t_grid,time_file)
#         # if not np.all(t_grid[p] == t_file):
#     #     raise ValueError("File times must exactly match a subset of all_time.")
#         m=t_grid.size
#         starts = np.concatenate(([0],p[1:],[m]))
#         counts = np.diff(starts)
#         el = np.repeat(elec_file,counts)
#         tr = np.repeat(trp_file,counts)
#         lum_b = np.concatenate(([False],el[1:]<el[:-1]))
#         lum = lum_b.astype(int)
#         # lum = np.ones(m,dtype=int)
#         # lum[p] = 0
#         return el,tr, lum

# def process_data(MonteCarlo, float_dtype=np.float32, processed_csv="processed.csv",
#                  averaged_csv="Averaged_data.csv", memmap_file="all_data.npy"):
#     """
#     Minimizes RAM usage by:
#       - building a disk-backed memmap for per-file series
#       - computing aggregates in one pass with running accumulators
#       - streaming CSV writes without creating large DataFrames in memory
#     """

#     # -------------------------
#     # 1) Discover inputs + union time grid
#     # -------------------------
#     # raw_list = sorted(glob.glob("*.npy"))
#     raw_list = sorted(glob.glob("*.bin"))
#     if not raw_list:
#         raise FileNotFoundError("No .npy files found in current directory.")

#     # Read first row (time) from each file via mmap (cheap) and union them
#     raw_time = []
#     for fp in raw_list:
#         # a = np.memmap(fp, dtype=np.float64, mode="r")
#         a = np.memmap(fp, dtype=np.float64, mode="r").reshape(-1, 3).T
#         # a = np.load(fp, mmap_mode="r")
#         if a.ndim != 2 or a.shape[0] != 3:
#             raise ValueError(f"{fp} expected shape (3, n); got {a.shape}")
#         raw_time.append(np.asarray(a[0], dtype=float_dtype))

#     all_time = np.unique(np.concatenate(raw_time))
#     all_time = np.asarray(all_time, dtype=float_dtype)

#     # Temperature from MonteCarlo (vectorized), as float_dtype (doesn't need double)
#     temperature = np.asarray(MonteCarlo.T_init + all_time * MonteCarlo.dT, dtype=float_dtype)

#     # -------------------------
#     # 2) Create an on-disk memmap for per-file series
#     #    Layout matches your original all_data:
#     #    rows = 2 + 3*len(files)  (Time, Temperature, then triplets per file)
#     #    cols = len(all_time)
#     # -------------------------
#     n_files = len(raw_list)
#     n_rows = 2 + 3 * n_files
#     n_cols = all_time.shape[0]

#     # 'w+' creates/overwrites the file
#     all_data = np.lib.format.open_memmap(memmap_file, mode="w+", dtype=float_dtype, shape=(n_rows, n_cols))
#     all_data[0, :] = all_time
#     all_data[1, :] = temperature

#     # Running accumulators for averages/sums (avoid storing all columns in RAM)
#     elec_sum = np.zeros(n_cols, dtype=np.float64)  # keep sum at higher precision
#     trap_sum = np.zeros(n_cols, dtype=np.float64)
#     lum_sum  = np.zeros(n_cols, dtype=np.float64)

#     j = 2
#     for fp in raw_list:
#         # a = np.load(fp, mmap_mode="r")
#         # a = np.memmap(fp, dtype=np.float64, mode="r")
#         a = np.memmap(fp, dtype=np.float64, mode="r").reshape(-1, 3).T
#         t_file = np.asarray(a[0], dtype=float_dtype)
#         el_file = np.asarray(a[1], dtype=float_dtype)
#         tr_file = np.asarray(a[2], dtype=float_dtype)

#         el_series, tr_series, lum_series = build_step_series(t_file, el_file, tr_file, all_time)

#         # Write columns for this file directly to disk-backed memmap
#         all_data[j,   :] = el_series
#         all_data[j+1, :] = tr_series
#         all_data[j+2, :] = lum_series

#         # Update running sums for later averages (double precision to reduce drift)
#         elec_sum += el_series.astype(np.float64)
#         trap_sum += tr_series.astype(np.float64)
#         lum_sum  += lum_series.astype(np.float64)

#         j += 3

#     header = ["Time", "Temperature"]
#     for i in range(n_files):
#         header += [f"Electrons_{i+1}", f"Traps_{i+1}", f"Lum_{i+1}"]

#     with open(processed_csv, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(header)
#         for i in range(n_cols):
#             row = [all_data[0, i], all_data[1, i]]
#             for base in range(2, n_rows, 3):
#                 row.append(all_data[base,   i])  # Electrons_k
#                 row.append(all_data[base+1, i])  # Traps_k
#                 row.append(all_data[base+2, i])  # Lum_k
#             w.writerow(row)

   
#     p_vec = MonteCarlo.p_array(temperature)

#     # Averages/sums across files
#     # n = float(n_files)
#     # electrons_avg = (elec_sum / n).astype(float_dtype)
#     # traps_avg     = (trap_sum / n).astype(float_dtype)
#     electrons_avg = elec_sum.astype(float_dtype)
#     traps_avg     = trap_sum.astype(float_dtype)
#     lum_total     = lum_sum.astype(float_dtype)

#     # Derived quantities
#     # n_g = <Electrons_Avg> / (p + 1)
#     # n_e = <Electrons_Avg> * p / (p + 1)
#     denom = (p_vec + 1.0).astype(np.float64)
#     n_g = (electrons_avg.astype(np.float64) / denom).astype(float_dtype)
#     n_e = (electrons_avg.astype(np.float64) * (p_vec.astype(np.float64) / denom)).astype(float_dtype)

#     # Temperature to celsius for output
#     temp_c = (temperature - np.array(273.15, dtype=float_dtype)).astype(float_dtype)

#     # Write averaged CSV streamingly
#     with open(averaged_csv, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["Time", "Temperature", "p", "Electrons_Avg", "Traps_Avg", "Lum_sum", "n$_g$", "n$_e$"])
#         for i in range(n_cols):
#             w.writerow([
#                 all_time[i],
#                 temp_c[i],
#                 p_vec[i],
#                 electrons_avg[i],
#                 traps_avg[i],
#                 lum_total[i],
#                 n_g[i],
#                 n_e[i],
#             ])

#     # -------------------------
#     # 6) Optional cleanup of raw .npy files
#     # -------------------------
#     for fp in raw_list:
#         try:
#             os.remove(fp)
#         except OSError:
#             pass

#     # Return paths for convenience
#     # return {"processed_csv": processed_csv, "averaged_csv": averaged_csv, "memmap_file": memmap_file}


# # def process_data(MonteCarlo):
   
# #     raw_list = glob.glob("*.npy")
# #     raw_time = []
# #     float_dtype = np.float64
# #     for file in raw_list:
# #         data = np.load(file,mmap_mode="r")
# #         raw_time.append(np.asarray(a[0], dtype=float_dtype))

# #     all_time = np.unique(np.concatenate([t for t in raw_time]))
# #     all_data = np.zeros(((len(raw_list)*3)+2,len(all_time)))
# #     all_data[0,:] = all_time
# #     j = 2
    
    
# #     for fp in raw_list:
# #         a = np.load(fp, mmap_mode="r")  
  
# #         if a.shape[0] != 3:
# #             raise ValueError(f"{fp} expected shape (3, n); got {a.shape}")

# #         t_file = np.asarray(a[0], dtype=float)
# #         el_file = np.asarray(a[1], dtype=float)
# #         tr_file = np.asarray(a[2], dtype=float)

# #         el_series, tr_series, lum_series = build_step_series(t_file, el_file, tr_file, all_time)

# #         all_data[j, :] = el_series
# #         all_data[j + 1, :] = tr_series
# #         all_data[j + 2, :] = lum_series

# #         j += 3


# #     all_data[1,:] = MonteCarlo.T_init + all_data[0,:]*MonteCarlo.dT
    
# #     df = pd.DataFrame(all_data.T)
# #     columns = ["Time","Temperature"]
# #     for i in range(len(raw_list)):
# #         columns.append(f"Electrons_{i+1}")
# #         columns.append(f"Traps_{i+1}")
# #         columns.append(f"Lum_{i+1}")

# #     df.columns = columns
# #     df.to_csv("processed.csv")
# #     for file in raw_list:
# #         os.remove(file)
# #     av_df = df[["Time","Temperature"]].copy()
# #     av_df["p"] = MonteCarlo.p_array(av_df["Temperature"])

# #     elec_cols = df.filter(like="Electrons_")
# #     av_df["Electrons_Avg"] = elec_cols.mean(axis=1)
# #     tr_cols = df.filter(like="Traps_")
# #     av_df["Traps_Avg"] = tr_cols.mean(axis=1)
# #     lum_cols = df.filter(like="Lum_")
# #     av_df["Lum_sum"] = lum_cols.sum(axis=1)
# #     av_df["n$_g$"] = av_df["Electrons_Avg"]/(av_df["p"]+1)
# #     av_df["n$_e$"] = av_df["Electrons_Avg"]*av_df["p"]/(av_df["p"]+1)
# #     av_df["Temperature"] -= 273.15

# #     av_df.to_csv("Averaged_data.csv")
# #     return 

# def load_for_populations(path,temp_time):
#     use = [temp_time, "n$_g$", "n$_e$"]
#     return pd.read_csv(
#         path, usecols=use,
#         dtype={c: "float32" for c in use},
#         engine="c", memory_map=True
#     )
# def load_for_totals(path,temp_time):
#     use = [temp_time, "Electrons_Avg", "Traps_Avg"]
#     return pd.read_csv(
#         path, usecols=use,
#         dtype={c: "float32" for c in use},
#         engine="c", memory_map=True
#     )
# def load_for_lum(path):
#     use = ["Temperature", "Lum_sum"]
#     return pd.read_csv(
#         path, usecols=use,
#         dtype={c: "float32" for c in use},
#         engine="c", memory_map=True
#     )
# def running_mean(a: np.ndarray, k: int = 5) -> np.ndarray:
#         kernel = np.ones(k) / k
#         return np.convolve(a, kernel, "valid")

# def hist_and_smooth(t_axis, events, bin_width=1.0, win_deg=50.0):
#     # 1) histogram into integer‑°C bins
#     bins  = np.arange(0, t_axis.max() + bin_width, bin_width)
#     hist, _ = np.histogram(t_axis, bins=bins, weights=events)
#     # 2) convert to intensity per °C
#     hist = hist / bin_width
#     # 3) boxcar smooth over *win_deg* °C
#     k = max(1, int(win_deg / bin_width))
#     return running_mean(hist, k=k)

# def plot_populations():
#     print("here")
#     data_pop = load_for_populations("Averaged_data.csv","Temperature")
#     plt.plot(data_pop["Temperature"], data_pop["n$_g$"], color="black", lw=2, label="n$_g$")
#     plt.plot(data_pop["Temperature"], data_pop["n$_e$"], color="black", lw=2, label="n$_e$")
#     plt.plot(data_pop["Temperature"],data_pop["n$_e$"]/data_pop["n$_g$"])
#     plt.xlabel("Temperature")
#     plt.ylabel("Electrons")
#     plt.legend()
#     plt.savefig("populations.png")
#     # plt.show()

# def plot_ratios():
#     data_pop = load_for_totals("Averaged_data.csv","Time")
#     # plt.plot(data_pop["Time"], data_pop["Electrons_Avg"], color="black", lw=2, label="n$_g$")
#     # plt.plot(data_pop["Time"], data_pop["Traps_Avg"], color="blue", lw=2, label="n$_e$")
#     data_pop["ratio"] = (data_pop["Traps_Avg"]-data_pop["Electrons_Avg"])/data_pop["Traps_Avg"]
   
#     data_pop["ratio"] = data_pop["ratio"].replace([np.inf, -np.inf], np.nan)  # protect against 0 traps
#     data_pop["ratio_ma"] = data_pop["ratio"].rolling(window=100, min_periods=1, center=True).mean() 
#     plt.plot(data_pop["Time"],data_pop["ratio"],color="green",label="ratio")
#     plt.plot(data_pop["Time"],data_pop["ratio_ma"],color="black",label="smooth")

#     # for i in range(1,5):
#     #     smooth = hist_and_smooth(data_pop["Time"],ratio,bin_width=0.5,win_deg=1)
#     #     plt.plot(np.arange(len(smooth)), smooth,label=f"{i}")
#     plt.xlabel("Time")
#     plt.ylabel("Electrons")
#     plt.legend()
#     plt.savefig("ratio.png")
#     # plt.show()



# def plot_smooth(data):
#     for i in range(25,125,25):
#         smooth = hist_and_smooth(data["Temperature"], data["Lum_sum"],win_deg=i)
#         plt.plot(np.arange(len(smooth)), smooth,label=f"{i}")
#     # plt.legend()
#     # plt.show()

# def plot_running_mean(data,window=10):
#     rm = data["Lum_sum"].rolling(window=window, center=True).mean()
#     plt.plot(data["Temperature"], data["Lum_sum"], 'o', color="red", lw=2, label="Lum")
#     plt.plot(data["Temperature"],rm, 'o', color="black", lw=2, label="RM")

# def window_smoothing(data,window=50,):
#     half_window = window / 2
#     r_mean = []
#     temperature = data["Temperature"]
#     for t in data["Temperature"]:
#         mask = (temperature >= t - half_window) & (temperature <= t + half_window)
#         r_mean.append(np.mean(data["Lum_sum"][mask]))

#     r_mean = np.array(r_mean)
#     plt.plot(data["Temperature"], r_mean, color="black", lw=2, label="window")

# def norm_smoothing(data,bandwidth=25.0):
#     temperature = data["Temperature"]
#      # # Define a grid of temperatures for smoothing
#     temp_grid = np.linspace(min(data["Temperature"]), max(data["Temperature"]), 1601)
#     # Bandwidth (controls smoothness, like "window size")
#     from scipy.stats import norm 
#     smoothed = []
#     for t in temp_grid:
#         weights = norm.pdf(temperature, loc=t, scale=bandwidth)
#         smoothed.append(np.sum(weights * data["Lum_sum"]) / np.sum(weights))
#     smoothed = np.array(smoothed)
#     plt.plot(temp_grid, smoothed, lw=2, label="norm")

# def parametric_smoothing(data):
#     import statsmodels.api as sm 
#     frac = 0.1
#     for i in range(1,3):
#         frac = i*0.1
#         lowess = sm.nonparametric.lowess
#         smoothed_loess = lowess(data["Lum_sum"], data["Temperature"], frac=frac)
#         plt.plot(smoothed_loess[:,0], smoothed_loess[:,1], label=f"LOESS (frac={frac})")

# def plot_ratio(out):

#     plt.plot(out[0,:],out[2,:])
#     # L = out[3:,:].mean(axis=0)
#     # smooth = hist_and_smooth(out[1,:],L)
#     # plt.plot(np.arange(len(smooth)),smooth)
#     # summed = np.sum(out[2:,:],axis=0)
#     # plt.plot(out[0,:],summed)
#     # for i in range(2,out.shape[0]): 
#     #     plt.plot(out[0,:],out[i,:],label=f"run no. {i-1}")
#     # plt.legend()
#     plt.show()


# def plot_data():

#     # data = pd.read_csv("Averaged_data.csv")

#     # plot_populations()
#     plot_ratios()
#     # plot_smooth(data)
#     # plt.close()
#     # plot_running_mean(data)
#     data = load_for_lum("Averaged_data.csv")
#     plot_smooth(data)
#     # window_smoothing(data)
#     # norm_smoothing(data)
#     # parametric_smoothing(data)


#     # plt.xlabel("Temperature")
#     # plt.ylabel("Luminesence")
#     # plt.legend()
#     # plt.show()
#     # plt.close()
   

#     plt.xlabel("Temperature")
#     plt.ylabel("Luminesence")
#     plt.legend()
#     plt.savefig("example.png")
#     # plt.show()



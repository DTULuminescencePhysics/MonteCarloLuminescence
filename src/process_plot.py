import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import glob

def build_step_series(time_file,elec_file, trp_file, t_grid):
        p = np.searchsorted(t_grid,time_file)
        # if not np.all(t_grid[p] == t_file):
    #     raise ValueError("File times must exactly match a subset of all_time.")
        m=t_grid.size
        starts = np.concatenate(([0],p[1:],[m]))
        counts = np.diff(starts)
        el = np.repeat(elec_file,counts)
        tr = np.repeat(trp_file,counts)
        lum_b = np.concatenate(([False],el[1:]<el[:-1]))
        lum = lum_b.astype(int)
        # lum = np.ones(m,dtype=int)
        # lum[p] = 0
        return el,tr, lum

def process_data(MonteCarlo):
   
    raw_list = glob.glob("*.npy")
    raw_time = []

    for file in raw_list:
        data = np.load(file)
        first = data[0,:]
        raw_time.append(first)

    all_time = np.unique(np.concatenate([t for t in raw_time]))
    all_data = np.zeros(((len(raw_list)*3)+2,len(all_time)))
    all_data[0,:] = all_time
    j = 2
    
    
    for fp in raw_list:
        a = np.load(fp, mmap_mode="r")  
  
        if a.shape[0] != 3:
            raise ValueError(f"{fp} expected shape (3, n); got {a.shape}")

        t_file = np.asarray(a[0], dtype=float)
        el_file = np.asarray(a[1], dtype=float)
        tr_file = np.asarray(a[2], dtype=float)

        el_series, tr_series, lum_series = build_step_series(t_file, el_file, tr_file, all_time)

        all_data[j, :] = el_series
        all_data[j + 1, :] = tr_series
        all_data[j + 2, :] = lum_series

        j += 3


    all_data[1,:] = MonteCarlo.T_init + all_data[0,:]*MonteCarlo.dT
    
    df = pd.DataFrame(all_data.T)
    columns = ["Time","Temperature"]
    for i in range(len(raw_list)):
        columns.append(f"Electrons_{i+1}")
        columns.append(f"Traps_{i+1}")
        columns.append(f"Lum_{i+1}")

    df.columns = columns
    df.to_csv("processed.csv")

    av_df = df[["Time","Temperature"]].copy()
    av_df["p"] = MonteCarlo.p_array(av_df["Temperature"])

    elec_cols = df.filter(like="Electrons_")
    av_df["Electrons_Avg"] = elec_cols.mean(axis=1)
    tr_cols = df.filter(like="Traps_")
    av_df["Traps_Avg"] = tr_cols.mean(axis=1)
    lum_cols = df.filter(like="Lum_")
    av_df["Lum_sum"] = lum_cols.sum(axis=1)
    av_df["n$_g$"] = av_df["Electrons_Avg"]/(av_df["p"]+1)
    av_df["n$_e$"] = av_df["Electrons_Avg"]*av_df["p"]/(av_df["p"]+1)
    av_df["Temperature"] -= 273.15

    av_df.to_csv("Averaged_data.csv")
    return 


def running_mean(a: np.ndarray, k: int = 5) -> np.ndarray:
        kernel = np.ones(k) / k
        return np.convolve(a, kernel, "valid")

def hist_and_smooth(t_axis, events, bin_width=1.0, win_deg=50.0):
    # 1) histogram into integer‑°C bins
    bins  = np.arange(0, t_axis.max() + bin_width, bin_width)
    hist, _ = np.histogram(t_axis, bins=bins, weights=events)
    # 2) convert to intensity per °C
    hist = hist / bin_width
    # 3) boxcar smooth over *win_deg* °C
    k = max(1, int(win_deg / bin_width))
    return running_mean(hist, k=k)

def plot_populations(data):
    plt.close()
    plt.plot(data["Temperature"], data["n$_g$"], color="black", lw=2, label="n$_g$")
    plt.plot(data["Temperature"], data["n$_e$"], color="black", lw=2, label="n$_e$")
    plt.xlabel("Temperature")
    plt.ylabel("Electrons")
    plt.legend()
    plt.show()

def plot_smooth(data):
    for i in range(25,125,25):
        smooth = hist_and_smooth(data["Temperature"], data["Lum_sum"],win_deg=i)
        plt.plot(np.arange(len(smooth)), smooth,label=f"{i}")
    plt.legend()
    plt.show()

def plot_running_mean(data,window=10):
    rm = data["Lum_sum"].rolling(window=window, center=True).mean()
    plt.plot(data["Temperature"], data["Lum_sum"], 'o', color="red", lw=2, label="Lum")
    plt.plot(data["Temperature"],rm, 'o', color="black", lw=2, label="RM")

def window_smoothing(data,window=50,):
    half_window = window / 2
    r_mean = []
    temperature = data["Temperature"]
    for t in data["Temperature"]:
        mask = (temperature >= t - half_window) & (temperature <= t + half_window)
        r_mean.append(np.mean(data["Lum_sum"][mask]))

    r_mean = np.array(r_mean)
    plt.plot(data["Temperature"], r_mean, color="black", lw=2, label="window")

def norm_smoothing(data,bandwidth=25.0):
    temperature = data["Temperature"]
     # # Define a grid of temperatures for smoothing
    temp_grid = np.linspace(min(data["Temperature"]), max(data["Temperature"]), 1601)
    # Bandwidth (controls smoothness, like "window size")
    from scipy.stats import norm 
    smoothed = []
    for t in temp_grid:
        weights = norm.pdf(temperature, loc=t, scale=bandwidth)
        smoothed.append(np.sum(weights * data["Lum_sum"]) / np.sum(weights))
    smoothed = np.array(smoothed)
    plt.plot(temp_grid, smoothed, lw=2, label="norm")

def parametric_smoothing(data):
    import statsmodels.api as sm 
    frac = 0.1
    for i in range(1,6):
        frac = i*0.1
        lowess = sm.nonparametric.lowess
        smoothed_loess = lowess(data["Lum_sum"], data["Temperature"], frac=frac)
        plt.plot(smoothed_loess[:,0], smoothed_loess[:,1], label=f"LOESS (frac={frac})")

def plot_data():

    data = pd.read_csv("Averaged_data.csv")

    # plot_populations(data)
    # plot_smooth(data)
    # plt.close()
    # plot_running_mean(data)

    window_smoothing(data)
    norm_smoothing(data)
    parametric_smoothing(data)


    # plt.xlabel("Temperature")
    # plt.ylabel("Luminesence")
    # plt.legend()
    # plt.show()
    # plt.close()
   

    plt.xlabel("Temperature")
    plt.ylabel("Luminesence")
    plt.legend()
    plt.show()



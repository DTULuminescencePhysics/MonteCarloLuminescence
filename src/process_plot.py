import pandas as pd 
import matplotlib.pyplot as plt


   for i in range(reps):
        
    all_times = np.unique(np.concatenate([df["Time"].values for df in dfs]))
    temperature = MonteCarlo.phys.T(all_times)
    p = MonteCarlo.phys.calc_p(temperature)
    temperature = temperature -273.15
    combined_parts = []

    for i, df in enumerate(dfs,start=1):
        reindex = df.set_index("Time").reindex(all_times)
        reindex = reindex.interpolate(method="index")
        reindex = reindex.round().astype(int)
        de = reindex["Electrons"].diff()
        reindex["Lum"] = ((de == -1).astype(int))
        reindex = reindex.add_suffix(f"_{i}")
        combined_parts.append(reindex)

    combined = pd.concat(combined_parts,axis=1)
    combined.index.name = "Time"
    combined = combined.reset_index()
    combined.insert(1,"Temperature",temperature)
    combined.insert(2,"p",p)
    i = 3
    for var in ["Electrons", "Traps"]:#, "Lum"]:
        cols = [c for c in combined.columns if c.startswith(var+"_")]
        combined.insert(i,f"{var}_avg",combined[cols].mean(axis=1))
        i+=1
    cols = [c for c in combined.columns if c.startswith("Lum_")]
    combined.insert(i,"Lum_avg",combined[cols].sum(axis=1))

    ground_state = combined["Electrons_avg"]/(combined["p"]+1)
    excited_state = combined["Electrons_avg"]*combined["p"]/(combined["p"]+1)
    combined.insert(3,"n$_g$",ground_state)
    combined.insert(4,"n$_e$",excited_state)

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

    # for i in range(25,125,25):
    #     smooth = hist_and_smooth(combined["Temperature"], combined["Lum_avg"],win_deg=i)
    #     plt.plot(np.arange(len(smooth)), smooth,label=f"{i}")

    # plt.plot(combined["Temperature"], combined["n$_g$"], color="black", lw=2, label="n$_g$")
    # plt.plot(combined["Temperature"], combined["n$_e$"], color="black", lw=2, label="n$_e$")
    # plt.plot(combined["Temperature"], combined["Lum_avg"], 'o', color="black", lw=2, label="Lum")
    # combined["RunningMean"] = combined["Lum_avg"].rolling(window=10, center=True).mean()
    # plt.plot(combined["Temperature"], combined["RunningMean"], 'o', color="black", lw=2, label="Lum")
    
    window = 50  # temperature window size (e.g. ±2.5°C)
    half_window = window / 2
    r_mean = []
    for t in combined["Temperature"]:
        mask = (temperature >= t - half_window) & (temperature <= t + half_window)
        r_mean.append(np.mean(combined["Lum_avg"][mask]))

    r_mean = np.array(r_mean)
    plt.plot(combined["Temperature"], r_mean, color="black", lw=2, label="window")

    # Define a grid of temperatures for smoothing
    temp_grid = np.linspace(min(combined["Temperature"]), max(combined["Temperature"]), 1601)
    # Bandwidth (controls smoothness, like "window size")
    bandwidth = 10.0  
    from scipy.stats import norm 
    smoothed = []
    for t in temp_grid:
        weights = norm.pdf(temperature, loc=t, scale=bandwidth)
        smoothed.append(np.sum(weights * combined["Lum_avg"]) / np.sum(weights))

    smoothed = np.array(smoothed)
    plt.plot(temp_grid, smoothed, lw=2, label="norm")

    # import statsmodels.api as sm 
    # frac = 0.01

    # lowess = sm.nonparametric.lowess
    # smoothed_loess = lowess(combined["Lum_avg"], combined["Temperature"], frac=frac)
    # plt.plot(smoothed_loess[:,0], smoothed_loess[:,1], "g-", linewidth=2, label=f"LOESS (frac={frac})")


    plt.xlabel("Temperature")
    plt.ylabel("Electrons")
    plt.legend()
    plt.show()

    # with pd.ExcelWriter("test.xlsx")as writer:
    #     combined.to_excel(writer, index=False)

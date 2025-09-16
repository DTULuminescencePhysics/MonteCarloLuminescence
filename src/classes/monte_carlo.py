import sys
from dataclasses import dataclass, field
from classes.physics import system
from classes.crystal import box
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
class MC:

    def __init__(self,E_loc,b,s,alpha=None,E_cb=None,
                 T_init=273.15,dT=5, duration=160,
                 D0=None,D_dot=None,
                 rho = None, urho = None,
                 tot = None, h=None, w=None, l=None):
        """Sets up universal parameters used for all 
        iterations of the Monte Carlo simulation"""
        
        self.phys = system(E_loc=E_loc,alpha=alpha,b=b,s=s,
                           E_cb=E_cb,D0=D0,D_dot=D_dot,
                           T_init=T_init,dT=dT)
        
        if tot is not None and urho is not None:
            self.tot = tot 
            self.phys.set_urho(urho)
            side_len = np.cbrt(self.tot/self.phys.rho)
            h=w=l=side_len
            self.crystal = box(h,w,l,1.5)
        else:
            if h is None or w is None or l is None:
                self.crystal = box(boundary_factor=1.5)
            else:
                self.crystal = box(h,w,l,1.5)
            if rho is None and urho is not None:
                self.phys.set_urho(urho)
                self.tot = int(self.phys.rho*self.crystal.volume)
            elif urho is None and rho is not None:
                self.tot = int(rho*self.crystal.volume)
                self.phys.set_rho(rho)
            elif rho is None and urho is None and tot is not None:
                self.tot = tot
                rho = tot/self.crystal.volume
                self.phys.set_rho(rho)
            else: 
                sys.exit("Need to define either a density or a number electrons")

       

        if D0 is not None and D_dot is not None:
            self.filling = True 
        else:
            self.filling = False

        self.duration = duration 
        self.dt_cap = 1
        
    def find_next_recombination(self,T):
        """Function that takes the list of nearest neighbours
        and then computes lifetimes and then draws the recombination
        times"""

        self._lifetime = self.phys.fading_rate(self.crystal.r,T)
        self._recomb_wait = np.random.exponential(self._lifetime)
        return np.min(self._recomb_wait), np.argmin(self._recomb_wait)

    def find_next_filling(self,ne,nt):
        if self.filling and (ne != nt):
            filltime = self.phys.filling_rate(ne,nt)
            return  np.random.exponential(filltime)
        else:
            return 1e20

    def run_simulation(self):
        """Function that runs a single iteration of the 
        Monte Carlo simulation"""

        self.crystal.initialise_el_tr(self.tot,self.tot)
        t_cur = 0.0 
        Temp = self.phys.T(t_cur)
        max_steps = int(self.duration/self.dt_cap +1)
        time = np.zeros(max_steps)
        nel_store = np.zeros(max_steps)
        tr_store = np.zeros(max_steps)
        nel_store[0] = self.crystal.n_el
        tr_store[0] = self.crystal.n_trap
        time[0] = 0
        i = 1
        self.re_calc = True
        while t_cur < self.duration: 
            if self.crystal.n_el > 0:
                recomb_tim, recomb_index = self.find_next_recombination(Temp)
            else:
                recomb_tim = 1e20
            fill_time = self.find_next_filling(self.crystal.n_el,self.crystal.n_trap)
            
            dt = min(recomb_tim,fill_time,self.dt_cap)
            if dt == recomb_tim:
                self.crystal.remove_electron(recomb_index)
                event = 1
                self._lifetime = np.delete(self._lifetime,recomb_index)
                self._recomb_wait = np.delete(self._recomb_wait,recomb_index)
            elif dt == fill_time:
                self.crystal.add_electron()
                event = 0
                self.re_calc = True
            elif dt == self.dt_cap:
                event = 0 
            
            t_cur += dt
            Temp = self.phys.T(t_cur)
            if (max_steps - i < 2):
                to_append = np.zeros(max_steps)
                time = np.append(time,to_append)
                nel_store = np.append(nel_store,to_append)
                tr_store = np.append(tr_store,to_append)
            time[i] = t_cur
            nel_store[i] = self.crystal.n_el
            tr_store[i] = self.crystal.n_trap
            i +=1
        
        df = pd.DataFrame({
            "Time": time[0:i],
            "Electrons": nel_store[0:i],
            "Traps": tr_store[0:i]
        }) 

        return df
    

def run_monte_carlo_simulation():
    input1 = {"E_loc": 0.8, 
             "s":1e10, 
             "rho":8e-4, 
             "factor":1e5, 
             "z":1.8, 
             "b":1e10,
             "c1":"black",
             "c2":"orange"
             }
    input2 = {"E_loc": 0.8, 
             "s":1e10, 
             "rho":3e-4, 
             "factor":1e4, 
             "z":1.8, 
             "b":1e10,
             "c1":"blue",
             "c2":"pink"}
    input3 = {"E_loc": 1.2, 
             "s":1e12, 
             "rho":3e-4, 
             "factor":2e6, 
             "z":1.8, 
             "b":1e12,
             "c1":"green",
             "c2":"grey"}

    inputs = input1
    MonteCarlo = MC(inputs["E_loc"],inputs["b"],inputs["s"],alpha=None,E_cb=None,
                 T_init=273.15,dT=5, duration=160,
                 D0=None,D_dot=None,
                 rho = None, urho = inputs["rho"],
                 tot = 1e2, h=None, w=None, l=None)
    
   
   
    reps = 1000
    
    from joblib import Parallel, delayed
    dfs = Parallel(n_jobs=-1)(delayed(MonteCarlo.run_simulation)() for _ in range(reps))
   
    # dfs =[]
    # for i in range(reps):
    #     df = MonteCarlo.run_simulation()
    #     dfs.append(df)

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


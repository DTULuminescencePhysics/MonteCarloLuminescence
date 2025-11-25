from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from src.classes.monte_carlo import MCBase
from omegaconf import DictConfig
from src.errors import ErrorOutputHandler
from typing import Tuple
from src.classes.constants import time_to_seconds
@dataclass
class InverseMC:
    obs: np.ndarray
    sigma: np.ndarray
    iters: int
    T0_min: float
    T0_max: float
    T_target: float
    duration: float
    dT_min: float = field(default=0)
    dT_max: float = field(default=0)
    tolerance: float = field(default=5)
    n_steps_min: int = field(default=0)
    n_steps_max: int = field(default=0)
    rng: np.random.Generator = field(init=False)
    T_profiles: list = field(init=False)
    MC_crystal: MCBase | list[MCBase] = field(init=False)
    t_edges: np.ndarray = field(init=False)
    T_edges: np.ndarray = field(init=False)
    grid: np.ndarray = field(init=False)
    t_samples: np.ndarray = field(init=False)
    true_T: np.ndarray = field(init=False)
    
    def set_random_generator(self):
        self.rng = np.random.default_rng()


    def check_inputs(self, err: ErrorOutputHandler | None):
        if self.duration <= 0:
            if err is not None:
                err.error("ValueError",additional="self.duration must be positive",fatal=True)
            else:
                raise ValueError("self.duration must be positive")

        if self.n_steps_min < 0 or self.n_steps_max < self.n_steps_min:
            if err is not None:
                err.error("ValueError",additional="Invalid step count bounds",fatal=True)
            else:
                raise ValueError("Invalid step count bounds")

        T0_low = max(self.T0_min, self.T_target - self.tolerance)
        if self.T0_max < T0_low:
            if err is not None:
                err.error("ValueError",
                          additional="No feasible T0: self.T0_max is lower than self.T_target - self.tolerance.",
                          fatal=True)
            else:
                raise ValueError(
                    "No feasible T0: self.T0_max is lower than self.T_target - self.tolerance.")


    def generate_random_profile(self, unit: str, celsius: bool = True,
                                T_min:float | None = None, T_max:float | None = None) -> dict:
        """
        Generate (T0, times, dT_step, dT) suitable for _build_linear_steps so that:
        - T0 is in [self.T0_min, T0_max]
        - temperature is non-increasing
        - final temperature at self.duration is within `tolerance` of T_target.
        """
       
        T0_low = max(self.T0_min, self.T_target - self.tolerance)
        T0 = self.rng.uniform(T0_low, self.T0_max)
        if T_max is not None:
            if T0 > T_max:
                T_max = T0 
        n_steps = self.rng.integers(self.n_steps_min, self.n_steps_max + 1)

        if n_steps > 0:
            while True:
                times = np.sort(self.rng.uniform(0.0, self.duration, size=n_steps))
                if self.duration - times[-1] > 1e-9:
                    break
        else:
            times = np.array([], dtype=float)

        max_step_total = max(T0 - (self.T_target - self.tolerance), 0.0)

        if n_steps > 0 and max_step_total > 0:
            total_step_drop = max_step_total * self.rng.random()
            weights = self.rng.random(n_steps)
            weights /= weights.sum()
            dT_steps = total_step_drop * weights 
        else:
            dT_steps = np.zeros(n_steps, dtype=float)
            total_step_drop = 0.0

        feasible_max_final = T0 - total_step_drop
        low = self.T_target - self.tolerance
        high = self.T_target + self.tolerance

        if feasible_max_final < low:
            feasible_high = feasible_max_final
            feasible_low = feasible_high
        else:
            feasible_low = low
            feasible_high = min(high, feasible_max_final)

        T_final = self.rng.uniform(feasible_low, feasible_high)
        if T_min is not None:
            if T_final < T_min:
                T_min = T_final

        linear_drop_total = T0 - total_step_drop - T_final

        if self.dT_max != 0:
            if n_steps == 0:
                dT = np.array([linear_drop_total / self.duration], dtype=float)
           
            else:
                seg_times = np.concatenate(([0.0], times))
                seg_ends = np.concatenate((times, [self.duration]))
                durations = seg_ends - seg_times 
                if linear_drop_total > 0:
                    w = self.rng.uniform(self.dT_min,(self.dT_max+1e-9),(n_steps+1))
                    ran = self.rng.random(n_steps + 1)
                    w[ran>0.75] = 0 
                    # w = self.rng.random(n_steps + 1)
                    
                    w /= w.sum()
                    drops = w * linear_drop_total
                    dT = drops / durations 
                else:
                    dT = np.zeros(n_steps + 1, dtype=float)
        else:
            dT = np.zeros(n_steps + 1, dtype=float)

        crrnt_T={'unit': unit, 
              'celsius': celsius, 
              'kind': 'linearsteps', 
              'T0': T0, 
              'duration': self.duration, 
              'times': times, 
              'dT_step': dT_steps, 
              'dT': dT}
        
        return crrnt_T

    def intialise_temp_profiles(self,unit: str, celsius: bool = True) -> Tuple[float,float] : 
        self.T_profiles = []
        T_min = self.T_target
        T_max = self.T0_max
        for _ in range(self.iters): 
            self.T_profiles.append(self.generate_random_profile(unit,celsius,T_min,T_max))

        return T_min, T_max

    def likeliness_score(self,res):
        
        L = np.exp(-0.5*np.sum((np.square(res-self.obs))/(np.square(self.sigma))))
        if L > self.rng.random():
            return True 
        else:
            return False 
        
    
    def make_time_temp_grid(self, t_min: float, t_max: float, T_min: float,
        T_max: float, n_bins: int = 50, n_samples: int = 10000, unit:str = 's') -> None:
        """
        Create a regular time-temperature grid.
        """
        self.t_edges = np.linspace(t_min, t_max, n_bins + 1)
        self.T_edges = np.linspace(T_min, T_max, n_bins + 1)
        self.grid = np.zeros((n_bins, n_bins), dtype=int)
        self.t_samples = np.linspace(t_min, t_max*time_to_seconds[unit], n_samples)

    
    def accumulate_profile_on_grid(self, MC_profile: MCBase) -> None:
        """
        Sample a time-temperature profile and increment grid squares that the
        profile passes through (once per profile per square).
        """
        T_samples = MC_profile.crystal.Tat(self.t_samples)
        if MC_profile.crystal.celsius:
            T_samples -= 273.15
        t_bins = np.digitize((self.t_samples/time_to_seconds[MC_profile.crystal.unit]), self.t_edges) - 1
        T_bins = np.digitize(T_samples, self.T_edges) - 1
       
        n_bins = self.grid.shape[0]

        t_bins = np.clip(t_bins, 0, n_bins - 1)
        T_bins = np.clip(T_bins, 0, n_bins - 1)

        pairs = np.stack([T_bins, t_bins], axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        
        for T_idx, t_idx in unique_pairs:
            self.grid[T_idx, t_idx] += 1
        

    def intialise_run(self,cfg: DictConfig | list[DictConfig], experiments:int, err: ErrorOutputHandler):
        """Function that sets up the inverse Monte Carlo ready to run the simulation.
        The random generator is initialised. Values checked. Monte Carlo crystals set"""
        
        self.check_inputs(err)
        err.checkpoint()
        self.set_random_generator()
        
        if experiments == 1 and isinstance(cfg, DictConfig):
            err.output("Setting up temperature profiles")
            unit = cfg.temp.unit
            T_min, T_max = self.intialise_temp_profiles(unit=unit,celsius=cfg.temp.celsius)
            err.output("Temperature profile setup complete")
            err.output("Setting up simulation crystal...")
            self.MC_crystal = MCBase.from_config(cfg)
            self.MC_crystal.RJMCMC_initialise()
            err.output("Crystal setup complete.")
        else:
            err.output("Setting up temperature profiles")
            unit = cfg[0].temp.unit
            T_min, T_max = self.intialise_temp_profiles(unit=unit,celsius=cfg[0].temp.celsius)
            err.output("Temperature profile setup complete")
            self.MC_crystal = []
            for i in range(experiments):    
                err.output("Setting up simulation crystal...")
                self.MC_crystal.append(MCBase.from_config(cfg[i]))
                self.MC_crystal[i].result_csv_path += f"_{i+1}"
                self.MC_crystal[i].RJMCMC_initialise()
                err.output("Crystal setup complete.")
   
        self.make_time_temp_grid(0,self.duration,T_min, T_max, unit = unit)
        if isinstance(self.MC_crystal, MCBase):
            self.true_T = self.MC_crystal.crystal.Tat(self.t_samples)
            if self.MC_crystal.crystal.celsius:
                self.true_T -= 273.15
        else:
            self.true_T = self.MC_crystal[0].crystal.Tat(self.t_samples)
            if self.MC_crystal[0].crystal.celsius:
                self.true_T -= 273.15

        
    
    def run_back_simulation(self):
        count = 0
        if isinstance(self.MC_crystal,MCBase):
            ratio = np.zeros(1)
            for prof in self.T_profiles:
                self.MC_crystal.crystal.set_temperature_profile("linearsteps",prof)
                ratio[0] = self.MC_crystal.inverse_modeling_simulation()
                if self.likeliness_score(ratio):
                    self.accumulate_profile_on_grid(self.MC_crystal)
                    count += 1
                print(f"Number of accepted profiles is {count}")
            self.MC_crystal.clean_up()
        else:
            ratio = np.zeros(len(self.MC_crystal))
            for prof in self.T_profiles:
                for i in range(len(self.MC_crystal)):
                    self.MC_crystal[i].crystal.set_temperature_profile("linearsteps",prof)
                    ratio[i] = self.MC_crystal[i].inverse_modeling_simulation()
                # print(ratio)
                # print(self.obs)
                if self.likeliness_score(ratio):
                    self.accumulate_profile_on_grid(self.MC_crystal[0])
                    count += 1
                print(f"Number of accepted profiles is {count}")
            for i in range(len(self.MC_crystal)):
                self.MC_crystal[i].clean_up()
        
        self.plot_probability_density_grid(count)



    def plot_probability_density_grid(self,count):
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(3.37,5.055))
        ax=fig.add_axes((0.,0.,2.,1.))

        im = ax.imshow(
            (self.grid/count),
            origin="lower",     
            aspect="auto",
            extent=(self.t_edges[0], self.t_edges[-1], self.T_edges[0], self.T_edges[-1])
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Probability")

        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature")

        counts = self.grid.astype(float)    
        n_t = counts.shape[1]

        t_mids = 0.5 * (self.t_edges[:-1] + self.t_edges[1:])
        T_mids = 0.5 * (self.T_edges[:-1] + self.T_edges[1:])

        col_sums = counts.sum(axis=0)  

        cdf = np.cumsum(counts, axis=0) 
        with np.errstate(invalid="ignore", divide="ignore"):
            cdf = cdf / col_sums[None, :]

        probs=(0.6, 0.9)
        paths = {}
        valid = col_sums > 0
        for c in probs:
            p_low = (1.0 - c) / 2.0
            p_high = 1.0 - p_low
            idx_low = np.argmax(cdf >= p_low, axis=0) 
            idx_high = np.argmax(cdf >= p_high, axis=0)
            lower = np.full(n_t, np.nan, dtype=float)
            upper = np.full(n_t, np.nan, dtype=float)
            lower[valid] = T_mids[idx_low[valid]]
            upper[valid] = T_mids[idx_high[valid]]
            paths[c] = {"lower": lower, "upper": upper}

    
        idx = np.argmax(cdf >= 0.5, axis=0) 

        median = np.full(n_t, np.nan, dtype=float)  
        median[valid] = T_mids[idx[valid]]

        b60 = paths[0.6]
        ax.plot(t_mids, b60["lower"], label="60%", linestyle=":",color="green")
        ax.plot(t_mids, b60["upper"],linestyle=":",color="green")
        b90 = paths[0.9]
        ax.plot(t_mids, b90["lower"], label="90%", linestyle=":",color="black")
        ax.plot(t_mids, b90["upper"], linestyle=":",color="black")

        ax.plot(t_mids, median, color="red",label="median")
        ax.plot(self.t_samples/time_to_seconds['Ma'],self.true_T,linewidth=2,linestyle="--",color="white")
        # ax.legend()
        plt.savefig("weights.png",dpi=300, transparent=False,bbox_inches='tight')
        plt.close()
        
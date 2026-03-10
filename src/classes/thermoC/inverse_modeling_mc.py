from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from src.classes.monte_carlo import MCBase
from omegaconf import DictConfig
from src.errors import ErrorOutputHandler
from src.classes.output.graph import chronologyPlot
@dataclass
class InverseMC:
    obs: np.ndarray
    sigma: np.ndarray
    iters: int
    T0_min: float
    T0_max: float
    T_target: float
    duration: float
    seed: int
    tolerance: float = field(default=5)
    n_steps_min: int = field(default=0)
    n_steps_max: int = field(default=0)
    trend: str = field(default='either')
    rng: np.random.Generator = field(init=False)
    Temp_points: np.memmap = field(init=False)
    time_points: np.memmap = field(init=False)
    accepted: np.memmap = field(init=False)
    MC_crystal: MCBase | list[MCBase] = field(init=False)
    pl : chronologyPlot = field(init=False)
    
    def set_random_generator(self):
        self.rng = np.random.default_rng(self.seed)


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
        
        if self.trend == "increasing":
            if self.T0_min > (self.T_target + self.tolerance):
                if err is not None:
                    err.error("ValueError",
                          additional="To be strictly increasing, no feasible T0: T0_min is higher than T_target + tolerance.",
                          fatal=True)
                else:
                    raise ValueError(
                        "No feasible T0: self.T0_min is higher than self.T_target + self.tolerance.") 
        elif self.trend == "decreasing":
            if self.T0_max < (self.T_target + self.tolerance):
                if err is not None:
                    err.error("ValueError",
                          additional="To be strictly decreasing, no feasible T0: T0_max is less than T_target - tolerance.",
                          fatal=True)
                else:
                    raise ValueError("To be strictly decreasing, no feasible T0: T0_max is less than T_target - tolerance.")

    
    def generate_random_profile(self) -> None:
        """
        Generate (T0, times, dT_step, dT) suitable for _build_linear_steps so that:
        - T0 is in [self.T0_min, T0_max]
        - temperature is non-increasing
        - final temperature at self.duration is within `tolerance` of T_target.
        """

        TT_low = self.T_target - self.tolerance
        TT_high = self.T_target + self.tolerance
        max_T_fixed = max(TT_low,TT_high,self.T0_max,self.T0_min)
        min_T_fixed = min(TT_low,TT_high,self.T0_max,self.T0_min)
        
        for i in range(self.iters):
            T0 = self.rng.uniform(self.T0_min, self.T0_max)
            T_final = self.rng.uniform(TT_low, TT_high)
      
            n_steps = self.rng.integers(self.n_steps_min, self.n_steps_max + 1)

            if n_steps > 0:
                t = np.sort(self.rng.uniform(0.0, self.duration, size=n_steps))
                if self.trend == "increasing":
                    min_T = T0 
                    max_T = T_final
                elif self.trend == "decreasing":
                    min_T = T_final
                    max_T = T0
                else:
                    min_T = min_T_fixed
                    max_T = max_T_fixed

                Tem = self.rng.uniform(min_T, max_T, size=n_steps) 
            else:
                t = np.array([], dtype=float)
                Tem = np.array([], dtype=float)
            
            times = np.concatenate(([0.0], t, [self.duration]))

            if self.trend != "either": 
                Tem = np.sort(Tem)
                if self.trend == "decreasing":
                    Tem = np.flip(Tem)
            
            
            temps = np.concatenate(([T0], Tem, [T_final]))
           
           
            self.time_points[i,0:times.size] = times
            self.Temp_points[i,0:temps.size] = temps
            
       
            
    def intialise_temp_profiles(self) -> None : 


        self.Temp_points = np.memmap("T_points.dat", dtype=np.float32, mode='w+', shape=(self.iters+1,self.n_steps_max+2))
        self.time_points = np.memmap("time_points.dat", dtype=np.float32, mode='w+', shape=(self.iters+1,self.n_steps_max+2))
        self.accepted = np.memmap("accepted.dat",dtype=np.bool_,mode='w+',shape=(self.iters+1))
        self.Temp_points[:,:] = -1
        self.time_points[:,:] = -1
        self.accepted[:] = False
        self.generate_random_profile()
        self.Temp_points.flush()
        self.time_points.flush()
        self.accepted.flush()
        


    def likeliness_score(self,res):
        
        L = np.exp(-0.5*np.sum((np.square(res-self.obs))/(np.square(self.sigma))))
        if L > self.rng.random():
            return True 
        else:
            return False 
        

    def intialise_run(self,cfg: DictConfig | list[DictConfig], experiments:int, err: ErrorOutputHandler):
        """Function that sets up the inverse Monte Carlo ready to run the simulation.
        The random generator is initialised. Values checked. Monte Carlo crystals set"""
        
        self.check_inputs(err)
        err.checkpoint()
        self.set_random_generator()
        
        if experiments == 1 and isinstance(cfg, DictConfig):
            err.output("Setting up temperature profiles")
            unit = cfg.temp.unit
            celsius = celsius=cfg.temp.celsius
            self.intialise_temp_profiles()
            to_save = np.array(cfg.temp.temps)
            self.Temp_points[-1,0:to_save.size] = to_save
            to_save = np.array(cfg.temp.times)
            self.time_points[-1,0:to_save.size] = to_save
            del(to_save)
            err.output("Temperature profile setup complete")
            err.output("Setting up simulation crystal...")
            self.MC_crystal = MCBase.from_config(cfg)
            self.MC_crystal.thermochron_initialise()
            err.output("Crystal setup complete.")
        else:
            err.output("Setting up temperature profiles")
            unit = cfg[0].temp.unit
            celsius=cfg[0].temp.celsius
            self.intialise_temp_profiles()
            to_save = np.array(cfg[0].temp.temps)
            self.Temp_points[-1,0:to_save.size] = to_save
            to_save = np.array(cfg[0].temp.times)
            self.time_points[-1,0:to_save.size] = to_save
            del(to_save)
            err.output("Temperature profile setup complete")
            self.MC_crystal = []
            for i in range(experiments):    
                err.output("Setting up simulation crystal...")
                self.MC_crystal.append(MCBase.from_config(cfg[i]))
                self.MC_crystal[i].result_csv_path += f"_{i+1}"
                self.MC_crystal[i].thermochron_initialise()
                err.output("Crystal setup complete.")

        self.pl = chronologyPlot(duration=self.duration,unit=unit,celsius=celsius)
        self.pl.set_profile_files("T_points.dat","time_points.dat","accepted.dat",(self.iters+1,self.n_steps_max+2))
        if isinstance(self.MC_crystal, MCBase):
            self.pl.setup_simulation_tracker(self.MC_crystal.crystal.Tat(self.pl.t_common_unit))
        else:
            self.pl.setup_simulation_tracker(self.MC_crystal[0].crystal.Tat(self.pl.t_common_unit))

    def run_back_simulation(self):
        count = 0
        if isinstance(self.MC_crystal,MCBase):
            ratio = np.zeros(1)
            for i in range(self.iters):
                self.MC_crystal.crystal.set_temperature_profile("linearsteps",self.time_points[i][self.time_points[i]>=0], 
                                                                self.Temp_points[i][self.time_points[i]>=0])
                ratio[0] = self.MC_crystal.inverse_modeling_simulation()
                if self.likeliness_score(ratio):
                    self.accepted[i] = True
                    self.accepted.flush()
                    count += 1
                    self.pl.add_result(self.MC_crystal.crystal.Tat(self.pl.t_common_unit),count)

                print(f"Number of accepted profiles is {count} after {i+1} trials out of {self.iters}") 
            self.MC_crystal.clean_up()
        else:
            ratio = np.zeros(len(self.MC_crystal))
            for i in range(self.iters):
                for j in range(len(self.MC_crystal)):
                    self.MC_crystal[j].crystal.set_temperature_profile("linearsteps",self.time_points[i][self.time_points[i]>=0], 
                                                                self.Temp_points[i][self.time_points[i]>=0])
                    ratio[j] = self.MC_crystal[j].inverse_modeling_simulation()

                if self.likeliness_score(ratio):
                    self.accepted[i] = True
                    count += 1
                    self.pl.add_result(self.MC_crystal[0].crystal.Tat(self.pl.t_common_unit),count)
                print(f"Number of accepted profiles is {count} after {i+1} trials out of {self.iters}") 
            
            for j in range(len(self.MC_crystal)):
                self.MC_crystal[j].clean_up()
        
        # self.Temp_points.flush()
        # self.time_points.flush()
        # self.accepted.flush()
        self.pl.save_close_tracker()
        
        self.pl.make_weighting_plot(n_bins=50,n_samples=10000)
        


  
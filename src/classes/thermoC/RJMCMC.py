from __future__ import annotations
import numpy as np
from omegaconf import DictConfig
from dataclasses import dataclass, field
from src.classes.monte_carlo import MCBase
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import warnings

warnings.filterwarnings('error')
@dataclass
class ReverseJmpMCMC:
    obs: np.ndarray
    iters: int 
    reps: int = field(default=10)
    MC: MCBase | None = field(default=None)
    t_common: np.ndarray = field(init=False)
    crrnt_T: dict  = field(init=False)
    prop_T: dict  = field(init=False)
    k: int  = field(init=False)
    prop_k: int = field(init=False)
    ratio_crrnt: np.ndarray = field(init=False)
    ratio_prop: np.ndarray = field(init=False)
    rng: np.random.Generator = field(init=False)
    
    min_gap: float = 0.001 
    non_increasing: bool = True 
    p_geom: float = 0.4 
    k_max: int = 5

    init_step_mean: float = 30
    init_step_sd:float = 20.0
    init_dT_mean:float = 400
    init_dT_sd: float = 350
    init_T0_mean: float = 200.0
    init_T0_sd: float = 100.0

    time_sd: float = 0.1 
    step_sd: float = 15.0 
    dT_sd: float = 100
    T0_sd: float = 20.0

    birth_prob: float = 0.25 
    death_prob: float = 0.25
    new_step_mean: float = 10 
    new_step_sd:float  = 5
    new_dT_mean: float = 350
    new_dT_sd: float = 200

    prior_dT_sd: float = 100.0
    prior_step_sd: float = 100.0 
    # prior_time_sd: float = 10  
    prior_T0_sd: float = 100.0


    @classmethod
    def from_config(cls, obs: np.ndarray, iters:int, cfg: DictConfig, reps: int = 1) -> "ReverseJmpMCMC":
        """Intialise RJMCMC from input dictionary"""
        obj = cls(obs, iters, reps)
        seed = cfg.setup.seed+cfg.mc.reps
        obj.set_random_generator(seed)
        obj.k = obj.intialise_k()
        obj.crrnt_T = obj.initialise_temperature_profile(obj.k, cfg.temp.unit,cfg.temp.celsius,cfg.temp.duration)
      
        obj.prop_T = obj.crrnt_T.copy()
        obj.MC = MCBase.RJMCMC_setup(cfg, reps, seed+1)

        obj.t_common = np.linspace(0,obj.MC.crystal.duration,num=10000)
        obj.obs = obj.process_ratio(obj.obs[:,1],obj.obs[:,0])
        return obj 
    
    def process_ratio(self, ratio: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Takes a time amnd ratio output and extrapolates it to fit a common set of 
        time values allowing easy comparison between results"""
        interp_obs  = interp1d(time,  ratio,  kind='linear', fill_value='extrapolate')
        return interp_obs(self.t_common)
        

    def intialise_values(self, MC: MCBase, seed):
        """Intialise RJMCMC class from an already exisiting 
        Monte Carlo Class"""
        self.MC = MC
        self.k = self.intialise_k()
        self.crrnt_T = self.initialise_temperature_profile(self.k, self.MC.crystal.unit, self.MC.crystal.celsius, 
                                                           self.MC.crystal.duration)
        self.MC.crystal.set_temperature_profile("linearsteps",self.crrnt_T)
        self.set_random_generator(seed)
        self.MC.seed = seed+1
        self.MC.repetion = self.reps

    def set_random_generator(self, seed: int | None = None) -> None:
        """Set the random number genreator using the seed. The seed 
        will be a user set value plus the current simulation number"""
        self.rng = np.random.default_rng(seed=seed)

    #########################################################################################################################
    #                                        Intialise Temperature Profile                                                  #
    #########################################################################################################################
    def intialise_k(self) -> int:
        """Intialises k which is the number of step points but could also be seen as the point the current 
        temperature behaviour changes. """
        k = 0
        while k < self.k_max and self.rng.random() > self.p_geom:
            k += 1
        return k
    
    def initialise_times(self, k: int, duration: float) -> np.ndarray:
        """
        Sample K ordered internal breakpoints in (0, t_end) with minimal spacing.
        Returns times with shape (K): [t1, ..., tk].
        """
        if k == 0:
            return np.array([])
        # if k == 1:
        #     return np.array(self.rng.uniform(self.min_gap,(duration-self.min_gap+1e-10)))
        
        min_required_gap = (k+1)*self.min_gap

        slack = duration - min_required_gap
        gaps = self.rng.random(k+1)
        gaps /= np.sum(gaps)

        gaps = self.min_gap + (gaps*slack)
        times = np.cumsum(gaps)
        return times[0:k]
    
    def initialise_dT_step(self, k: int) -> np.ndarray:
        """
        Initialize the K step drops at each breakpoint. Negative = step down.
        """
        if k == 0:
           return np.array([])
        
        steps = self.rng.normal(loc=self.init_step_mean, scale=self.init_step_sd, size=k)
        if self.non_increasing:
            steps = np.maximum(steps, 0.0)
        return steps

    def initialise_dT(self, k: int) -> np.ndarray:
        """
        Initialize slopes for the K+1 sections. If non_increasing, clip to ≤ 0.
        """
        g = self.rng.normal(loc=self.init_dT_mean, scale=self.init_dT_sd, size=k+1)
        if self.non_increasing:
            g = np.maximum(g, 0.0)
        return g

    def initialise_T0(self) -> float:
        """Starting temperature before first interval."""
        return float(self.rng.normal(self.init_T0_mean, self.init_T0_sd))
    
    def initialise_temperature_profile(self, k: int, unit: str, celsius: bool, duration: float) -> dict:
        """Function that sets intial temperature profile parameters"""
        crrnt_T={'unit': unit, 
              'celsius': celsius, 
              'kind': 'linearsteps', 
              'T0': self.initialise_T0(), 
              'duration': duration, 
              'times': self.initialise_times(k,duration), 
              'dT_step': self.initialise_dT_step(k), 
              'dT': self.initialise_dT(k)}
        # crrnt_T['dT'][:]=0
     
        return crrnt_T

    #########################################################################################################################
    #                                        Uni-dimensional profile change                                                 #
    #########################################################################################################################
    
    def propose_new_time(self) -> None:
        """Randomly jitter one internal breakpoint, keeping order and min gap."""
        if self.prop_k == 0:
            return
        times = np.concatenate(([0.0], self.prop_T["times"], [self.prop_T["duration"]]))
        idx = self.rng.integers(1,  self.prop_k, endpoint=True)
        lo = times[idx-1] + self.min_gap
        hi = times[idx+1] - self.min_gap
        if hi <= lo:
            return 
        
        prop = np.clip(self.prop_T["times"][idx-1] + self.rng.normal(0, self.time_sd), lo, hi)
        self.prop_T["times"][idx-1] = prop

    def propose_new_dT_size(self) -> None:
        """Perturb one step drop; keep it ≤ 0 for 'down' steps."""
        if self.prop_k == 0:
            return 
    
        j = self.rng.integers(0, self.prop_k)
        val = self.crrnt_T['dT_step'][j] + self.rng.normal(0, self.step_sd)
        
        if self.non_increasing:
            val = np.maximum(0.0, val)

        self.prop_T['dT_step'][j] = val

        total_drop = self.prop_T["T0"] - np.sum(self.prop_T['dT_step'])
        # while total_drop < -5: 
        #     val = self.crrnt_T['dT_step'][j] + self.rng.normal(0, self.step_sd)
        
        #     if self.non_increasing:
        #         val = np.maximum(0.0, val)

        #     self.prop_T['dT_step'][j] = val

        # self.prop_T['dT_step'][j] = val
      
    def propose_new_dT(self) -> None:
        """Perturb one interval slope; optionally enforce ≤ 0."""
        # if self.prop_k != 0:
        #     j = self.rng.integers(0,self.prop_k, endpoint=True)
        # else: 
        #     j = 0 
        j = self.rng.integers(0,self.prop_k, endpoint=True)
        val = self.prop_T["dT"][j] + self.rng.normal(0, self.dT_sd)
        if self.non_increasing:
            val = np.maximum(val, 0.0)
        self.prop_T["dT"][j] = val
        
    def propose_new_T0(self) -> None:
        """Random-walk on starting temperature."""
        temp = self.prop_T['T0'] + self.rng.normal(0, self.T0_sd)
        while temp > 200:
            temp = self.prop_T['T0'] + self.rng.normal(0, self.T0_sd)

        self.prop_T['T0'] = temp 
        # += self.rng.normal(0, self.T0_sd)


    def uni_dimensional_change(self, sigma:float, acc_within: int) -> bool:
        """Selects one of the available changes to the temperature profile that retains the dimension of the model
        i.e k stays constant. """
        # move = self.rng.choice(["time", "step", "dT", "T0"])
        move = self.rng.choice(["time", "step",  "T0"])
        if move == "time":
            self.propose_new_time()
        elif move == "step":
            self.propose_new_dT_size()
        elif move == "dT":
            self.propose_new_dT()
        else:
            self.propose_new_T0()

        if self.prop_T is not self.crrnt_T: 
            self.MC.crystal.set_temperature_profile(self.prop_T["kind"],self.prop_T)
            # if self.MC.crystal.Tat(self.t_common[-1]) < 260:
            #     return False
            self.ratio_prop = self.new_observation_calculation()
            accept, la = self.accept(sigma)
            if accept:
                self.update_profiles(True, acc_within)
                return True
            
        return False
        
    #########################################################################################################################
    #                                        Trans-dimensional profile change                                               #
    #########################################################################################################################              
    def admissible_length(self, lo: float, hi: float) -> float:
        """
        Length of the uniform region for tau_new inside (lo+min_gap, hi-min_gap).
        Returns 0 if invalid.
        """
        L = (hi - self.min_gap) - (lo + self.min_gap)
        if self.non_increasing:
            return max(0.0, L)
        else: 
            return L 

    def propose_birth_step(self) -> int:
        """
        Insert a new step: choose a segment to split, sample a breakpoint inside it,
        add a new step drop and a new gradient (we create one extra interval).
        """
        if self.prop_k == self.k_max:
            return -1

        times = np.concatenate(([0.0], self.prop_T["times"], [self.prop_T["duration"]]))

        s = self.rng.integers(1,times.size)

        # s = int(self.rng.integers(self.prop_k))
        lo = times[s-1]
        hi = times[s]
        
        if hi - lo <= 2 * self.min_gap:
            return -1
        
        s -= 1
        new_time = self.rng.uniform(lo + self.min_gap, hi - self.min_gap)
        self.prop_T["times"] = np.insert(self.prop_T["times"], s, new_time)

        step_dt_step = max(0.0, self.rng.normal(self.new_step_mean, self.new_step_sd))
        self.prop_T["dT_step"] = np.insert(self.prop_T["dT_step"], s, step_dt_step)

        # total_drop = self.prop_T["T0"] - np.sum(self.prop_T['dT_step'])
        # while total_drop < -5: 
        #     step_dt_step = max(0.0, self.rng.normal(self.new_step_mean/2, self.new_step_sd/2))
            
        #     self.prop_T['dT_step'][s] = step_dt_step

        dT_new = self.rng.normal(self.new_dT_mean, self.new_dT_sd)
        if self.non_increasing:
            dT_new = np.maximum(dT_new, 0.0)
        self.prop_T["dT"] = np.insert(self.prop_T["dT"], s + 1, dT_new)
        self.prop_k += 1
        
        return int(s)
        
    def log_q_birth_forward(self, s_chosen: int) -> float:
        """Calcualtes the log proposal of adding a step"""

        times = np.concatenate(([0.0], self.prop_T["times"], [self.prop_T["duration"]]))
        s = s_chosen + 1 
        lo, hi =  times[s-1],  times[s+1]
        L = self.admissible_length(lo,hi)

        # if L <= 0:
        #     return -np.inf
        # if not (lo + self.min_gap < times[s] < hi - self.min_gap): 
        #     return -np.inf
        # if self.non_increasing and (self.prop_T["dT_step"][s_chosen] < 0 or self.prop_T["dT"][s_chosen+1] < 0):
        #     return -np.inf

        lq = np.log(self.birth_prob)
        lq += -np.log(self.k+1)
        lq += -np.log(L)
        lq += self.log_truncnorm_pdf(self.prop_T["dT_step"][s_chosen], self.new_step_mean, self.new_step_sd, lower=0.0)
        lq += self.log_truncnorm_pdf(self.prop_T["dT"][s_chosen+1], self.new_dT_mean, self.new_dT_sd, lower=0.0)

        return float(lq)

    def log_q_birth_reverse(self):
        """Calcualtes the log proposals of removing the additional step"""
        if self.prop_k <= 0:
            return -np.inf

        return np.log(self.death_prob) - np.log(self.prop_k)

    def propose_death_step(self) ->  int:
        """Remove a randomly chosen step (and its breakpoint and one gradient)."""

        if self.prop_k < 1:
            return -1
       
        j = int(self.rng.integers(self.prop_k))
        self.prop_T["times"] = np.delete(self.prop_T["times"], j)
        self.prop_T["dT_step"] = np.delete(self.prop_T["dT_step"], j)
        
        self.prop_T["dT"][j] = (self.prop_T["dT"][j]+self.prop_T["dT"][j+1])/2
        self.prop_T["dT"] = np.delete(self.prop_T["dT"], j + 1)
        self.prop_k -= 1
        
        return j 

    def log_q_death_forward(self, s_chosen: int) -> float:
        """Calcualtes the log proposals of removing the a step"""

        if not (0 <= s_chosen < self.k):
            return -np.inf
        return np.log(self.death_prob)-np.log(self.k)
    
    def log_q_death_reverse(self, s_chosen: int): 
        """Calcualtes the log proposals of adding the removed step back in"""
        
        times = np.concatenate(([0.0], self.crrnt_T["times"], [self.crrnt_T["duration"]]))
        s = s_chosen +1 
        lo, hi = times[s-1], times[s+1]
      
        L = self.admissible_length(lo, hi)
        # if L <= 0:
        #     return -np.inf
        # if not (lo + self.min_gap < times[s]  < hi - self.min_gap):
        #     return -np.inf
        # if self.crrnt_T["dT_step"][s_chosen] < 0 or self.crrnt_T["dT"][s_chosen+1] < 0:
        #     return -np.inf
    
        lq = np.log(self.birth_prob)
        lq += -np.log(self.prop_k + 1)     
        lq += -np.log(L)             
        lq += self.log_truncnorm_pdf(self.crrnt_T["dT_step"][s_chosen], self.new_step_mean, self.new_step_sd, lower=0.0)
        lq += self.log_truncnorm_pdf(self.crrnt_T["dT"][s_chosen+1], self.new_dT_mean, self.new_dT_sd, lower=0.0)
        return lq
    
    def trans_dimension_accept(self, s_chosen: int, sigma: float, birth: bool = True): 
        """Function that calcualtes the relevant forward and backwards log proposals 
        and then accepts or rejects the result"""
        if birth:
            lq_fwd = self.log_q_birth_forward(s_chosen)
            lq_rev = self.log_q_birth_reverse()
        else: 
            lq_fwd = self.log_q_death_forward(s_chosen)
            lq_rev = self.log_q_death_reverse(s_chosen) 

        accept, la = self.accept(sigma,lq_fwd,lq_rev, 0.0)
        
        return accept, la 


    def trans_dimensional_change(self, sigma:float, acc_birth: int, acc_death: int) -> bool:
        """Either adds or removes a temperature step i.e k increases or decreases by 1. """

        u = self.rng.random()
        if u < self.birth_prob: 
            birth = True 
            s_chosen = self.propose_birth_step()
        elif u < self.birth_prob + self.death_prob and self.k > 0:
            birth = False
            s_chosen = self.propose_death_step()
        else:
            return False
        
        # self.prop_T['dT'][:]=0
        if (s_chosen > -1) and (self.prop_T is not self.crrnt_T):
            self.MC.crystal.set_temperature_profile(self.prop_T["kind"],self.prop_T)
            # if self.MC.crystal.Tat(self.t_common[-1]) < 260:
            #     return False
            self.ratio_prop = self.new_observation_calculation()
            accept, la = self.trans_dimension_accept(s_chosen,sigma,birth)
            if accept: 
                if birth:
                    self.update_profiles(True, acc_birth)
                    return True 
                else: 
                    self.update_profiles(True, acc_death)
                    return True
      
        return False 
    
    #########################################################################################################################
    #                                        log prior calculation functions                                                #
    ######################################################################################################################### 

    def log_li_k(self, ratio: np.ndarray, sigma: float) -> float:
        """Log likelihood between observation and observation from proposed profile"""
        res = (self.obs - ratio) # / sigma
        return  -0.5*np.sum(np.square(res)) - self.obs.size*np.log(sigma)
    
    def log_prior_k(self, k: int) -> float: 
        """Log of priror k"""
        return k*np.log(1-self.p_geom) + np.log(self.p_geom)
    
    def log_prior_profile(self, k: int, prof: dict) -> float:
        """Log of the prior temperature profile"""
        lp = 0.0
      
        if k > 0: 
            # if self.non_increasing and (np.any(prof["dT"] < 0) or np.any(prof["dT_step"] < 0)):
            #     return -np.inf
    
            # lp += -0.5*np.sum(np.square((prof["dT"] - self.init_dT_mean)/self.prior_dT_sd))
            lp += -0.5*np.sum(np.square((prof["dT_step"] - self.init_step_mean)/self.prior_step_sd))
            lp += -0.5*np.minimum(0,(0-np.sum(prof["dT_step"])))
            gaps = np.diff(np.concatenate(([0.0], prof["times"], [prof["duration"]])))
            # if np.any(gaps <= self.min_gap) :
            #     return -np.inf
            lp += -0.5*np.sum(gaps)
      
       
        lp += -0.5*((prof["T0"] - self.init_T0_mean)/self.prior_T0_sd)**2
        return lp
    
    def log_posterior(self, k, ratio, sigma, prof):
        """Function that calcualtes the total log posterior for a observation and its 
        relevant profile"""
        return (self.log_li_k(ratio, sigma) + 
                self.log_prior_k(k) + 
                self.log_prior_profile(k, prof))

    def log_prior_sigma(self, sigma: float) -> float:
        if sigma <= 0: return -np.inf
        return -np.log(1.0 + (sigma/10.0)**2)

    def log_truncnorm_pdf(self, x, mean, sd, lower=0.0, higher = None):
        """Calcualtes the lof of the probability density function"""
        a = (lower - mean) / sd
        if higher is None:
            b = np.inf
        else: 
            b = (lower - mean) / sd
        return truncnorm.logpdf(x, a, b, loc=mean, scale=sd)

   
    #########################################################################################################################
    #                                        Utilities                                                                      #
    ######################################################################################################################### 

    def new_observation_calculation(self, t: float = 0.0, t_pcnt: float | None = None, h_pcnt:float | None = None) -> np.ndarray:
        if self.MC is not None:
            # self.MC.crystal.set_temperature_profile(self.prop_T["kind"],self.prop_T)
            self.MC.seed +=self.MC.repetion
            time, ratio = self.MC.RJMCMC_simulation(t, t_pcnt, h_pcnt)
            result = self.process_ratio(ratio,time)
          
            return result
        else:
            return np.empty(0) 

    def get_low(self,times: np.ndarray, s_chosen: int):
        if s_chosen == 0:
            return 0.0 
        else:
            return times[s_chosen]

    def get_heigh(self,times: np.ndarray, s_chosen: int, duration: float): 
        if s_chosen == times.size:
            return duration 
        else: 
            return times[s_chosen+1]

    def accept(self, sigma, log_q_fwd=0.0, log_q_rev=0.0, logJ=0.0):
        """
        Fixed-dimension move: prop_k == k
        """
        if self.k == self.prop_k: 
            log_q_fwd=log_q_rev=logJ=0.0

        lp_cur = self.log_posterior(self.k, self.ratio_crrnt, sigma, self.crrnt_T)
        lp_new = self.log_posterior(self.prop_k, self.ratio_prop, sigma, self.prop_T)
       
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=RuntimeWarning)
    
            try:
                la = (lp_new - lp_cur) + (log_q_rev - log_q_fwd) +logJ
            except Exception as e:
                # print(f"Exception in la calculation: {e}")
                return False, -np.inf

            if w:
                for warn in w:
                    print(f"Warning caught: {warn.message}")
                return False, -np.inf
       
        return (np.log(self.rng.random()) < la), la

    def update_profiles(self, acc: bool = False, acc_type: int | None = None):
        """Updates the temperature profile and current observation
        or reverts bakc to the original if the proposal is rejected"""
        if acc: 
            self.crrnt_T = self.prop_T.copy()
            self.ratio_crrnt = self.ratio_prop
            self.k = self.prop_k 
            if acc_type is not None:    
                acc_type += 1     
        else: 
            self.prop_T = self.crrnt_T.copy()
            self.prop_k = self.k 
    
    def sigma_update(self,sigma):
        """Update sigma"""
     
        sigma_p = sigma * np.exp(self.rng.normal(0, 0.05))
        la = (self.log_li_k(self.ratio_crrnt, sigma_p) + self.log_prior_sigma(sigma_p)
                - self.log_li_k(self.ratio_crrnt, sigma) - self.log_prior_sigma(sigma))
        if np.log(self.rng.random()) < la:
            return sigma_p 
        else: 
            return sigma

    def add_result(self, x, y, it):
        # fade existing lines
        for line in self.lines:
            current_alpha = line.get_alpha()
            new_alpha = max(current_alpha * 0.5, 0.1)
            line.set_alpha(0.1)
            line.set_color("red")
            line.set_label(None)
        
        # self.lines[0].set_alpha(1)
        # self.lines[0].set_color("black")
        # add new line
        line, = self.ax.plot(x, y, color="blue", alpha=1.0, label = it)
        self.lines.append(line)
        self.ax.legend()
        self.fig.canvas.draw_idle()
        plt.pause(0.1) 
 
   
    def rjmcmc_temperature(self,err):
        """Main RJMCMC function that proposes a new temperature profile"""

        plt.close()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Temperature")
        self.lines = [] 
        self.ax.plot(self.t_common,self.MC.crystal.Tat(self.t_common), color="black", alpha=1.0, label="Actual")
        # self.lines.append(line)
        plt.show(block=False)

        # line, = self.ax.plot(self.t_common,self.obs, color="black", alpha=1.0)
        # self.lines.append(line)
        
        # plt.show(block=False)

        self.MC.RJMCMC_initialise()
       
        temp_prof = np.memmap("temp_profiles.dat", dtype=np.float32, mode='w+', shape=(self.t_common.size, self.iters*2))
        count = 0 
        sigma = float(np.std(self.obs) + 1e-6)
        self.prop_k = self.k

        self.MC.crystal.set_temperature_profile(self.prop_T["kind"],self.prop_T)
        self.ratio_crrnt = self.new_observation_calculation()
       
        temp_prof[:,count]=self.MC.crystal.Tat(self.t_common)
        self.add_result(self.t_common, temp_prof[:,count],0)
     
        count+=1
        samples = []
        acc_within = acc_birth = acc_death = 0
        # err.output(self.MC.crystal.__repr__())
        for it in range(self.iters):
            print(it)
            accept = self.uni_dimensional_change(sigma, acc_within)
            if accept: 
                temp_prof[:,count]=self.MC.crystal.Tat(self.t_common)
                self.add_result(self.t_common, temp_prof[:,count],f"{it}_1")
                # err.output(self.MC.crystal.__repr__())
                count+=1
            else: 
                self.update_profiles()

            accept = self.trans_dimensional_change(sigma, acc_birth, acc_death)
            if accept: 
                temp_prof[:,count]=self.MC.crystal.Tat(self.t_common)
                self.add_result(self.t_common, temp_prof[:,count],f"{it}_2")
                count+=1
            else: 
                self.update_profiles()
               
            # sigma = self.sigma_update(sigma)
            

            samples.append(float(sigma))



        stats = {
            "acc_within": acc_within / self.iters,
            "acc_birth": acc_birth / max(1, int(self.iters*self.birth_prob)),
            "acc_death": acc_death / max(1, int(self.iters*self.death_prob)),
        }

        self.MC.RJMCMC_cleanup()
        plt.savefig("Temp_profile_Change.png",dpi=300, transparent=False,bbox_inches='tight')

        return samples, stats



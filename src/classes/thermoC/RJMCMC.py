from __future__ import annotations
import numpy as np
from omegaconf import DictConfig
from dataclasses import dataclass, field
from src.classes.monte_carlo import MCBase
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import warnings
from src.errors import ErrorOutputHandler
from src.classes.constants import time_to_seconds

warnings.filterwarnings('error')

@dataclass 
class TProfile:
    k: int  
    T0: float
    times: np.ndarray
    dT_step: np.ndarray
    dT: np.ndarray
    t_end: float
    T_final: float = field(init=False) 
    tau: np.ndarray = field(init=False)
    output_dict: dict = field(init=False) 

    def __post_init__(self):
        self.update_tau()
        self.compute_end_T()

    def __eq__(self, other):
        if not isinstance(other, TProfile):
            return NotImplemented
        if self.k != other.k:
            return False
        if self.T0 != other.T0: 
            return False
        if not (self.times == other.times).all():
            return False 
        if not (self.dT_step == other.dT_step).all():
            return False 
        if not (self.dT == other.dT).all():
            return False 
        return True
    
    def __ne__(self, other):
        return  not self.__eq__(other)
    
    @classmethod
    def initialise(cls, rng:np.random.Generator, 
                   k_max: int, p_geom: float, 
                   init_T0_mean: float, init_T0_sd:float,
                   min_gap: float, duration: float,
                   init_step_mean: float, init_step_sd: float,
                   init_dT_mean:float, init_dT_sd: float,
                   T_min:float, unit: str,
                   non_increasing: bool = True, celsius: bool = True):
        
        k = 0
        while k < k_max and rng.random() > p_geom:
            k += 1
        
        T0 = float(rng.normal(init_T0_mean, init_T0_sd))
        T_gap_max = T0 - T_min
        if k == 0:
            times = np.array([])
            dT_step = np.array([])
        else:
            min_required_gap = (k+1)*min_gap
            slack = duration - min_required_gap
            gaps = rng.random(k+1)
            gaps /= np.sum(gaps)

            gaps = min_gap + (gaps*slack)
            times = np.cumsum(gaps)
            times = times[0:k]
            max_temp_drop = T_gap_max * rng.random()

            limit = (max_temp_drop-(k*init_step_mean))/init_step_sd 
            x = rng.normal(0,1,k)
            S = np.sum(x)
            if S > limit: 
                x *= (limit / S)
            
            dT_step = init_step_mean + init_step_sd*x
       
       
        if non_increasing:
            dT_step = np.maximum(dT_step, 0.0)


        max_temp_drop = T_gap_max - np.sum(dT_step)
        dt = np.diff(np.concatenate(([0.0], times, [duration])))

        limit = (max_temp_drop- np.sum(dt*init_dT_mean))/init_dT_sd
        final = T_min - 10 
        # i=1
        dT = np.zeros((k+1))
        while final < T_min:

            x = rng.normal(0,1,(k+1))
   
            if non_increasing:
                x = np.maximum(x, -x)
     
            S = np.dot(x,dt)
      
            if S > limit: 
                x *= (limit / S) 
       
            dT = init_dT_mean + init_dT_sd*x
            if non_increasing:
                dT = np.maximum(dT, 0.0)

            final = T0 - np.sum(dT_step) - np.dot(dT,dt)
          
            
       
        obj = cls(k,T0,times,dT_step,dT,duration)
        obj.set_dictionary(unit, celsius)
      
        return obj
    
    @classmethod
    def make_copy(cls, profile: TProfile):
        """Intialises a copy of the profile using an existing one"""
        obj = cls(profile.k, profile.T0, profile.times.copy(), 
                  profile.dT_step.copy(), profile.dT.copy(), profile.t_end)
        obj.set_dictionary(profile.output['unit'],profile.output['celsius'])
        return obj
      
    def compute_end_T(self):
        """Computes the current end temperature"""
        dt = np.diff(self.tau)
        ld = np.sum(self.dT*dt)
        sd = np.sum(self.dT_step)
        self.T_final = self.T0 - ld - sd

    def set_dictionary(self, unit: str, celsius: bool = True):
        """Sets the dictionary values that are passed to the 
        MC crystals"""
        self.output={'unit': unit, 
              'celsius': celsius, 
              'kind': 'linearsteps', 
              'T0': self.T0, 
              'duration': self.t_end, 
              'times': self.times, 
              'dT_step': self.dT_step, 
              'dT': self.dT}

    def update_tau(self):
        """Sets the array of start, drop times and end time"""
        self.tau = np.concatenate(([0.0],self.times,[self.t_end]))

    def update_k(self, k: int):
        "updates the value of k"
        self.k = k 

    def update_T0(self, T0: float | None = None): 
        """Updates the value of T0"""
        if T0 is not None:
            self.T0 = T0 
     
        self.output['T0'] = self.T0

    def update_times(self, times: np.ndarray| None = None):
        """Updates the times of the drops"""
        if times is not None:
            self.times = times
      
        self.output['times'] = self.times  

    def update_dT_step(self, dT_step: np.ndarray| None = None):
        """Updates the size of the steps (down)""" 
        if dT_step is not None:
            self.dT_step = dT_step 
      
        self.output['dT_step'] = self.dT_step  

    def update_dT(self, dT: np.ndarray| None = None):
        """Updates the gradients of sections between drops"""
        if dT is not None:
            self.dT = dT 
        
        self.output['dT'] = self.dT  
    
    def update_controller(self, val: str): 
        if val == 'T0':
            self.update_T0()
        elif val == 'time':
            self.update_times()
            self.update_tau()
        elif val == 'step': 
            self.update_dT_step()
        elif val == 'dT':
            self.update_dT()
        elif val == 'da': 
            self.update_times()
            self.update_tau()
            self.update_dT_step()
            self.update_dT()
        else:
            self.update_T0()
            self.update_times()
            self.update_tau()
            self.update_dT_step()
            self.update_dT()

        self.compute_end_T() 

    def update_profile(self, profile: TProfile, val: str | None = None): 
        """Updates an entire profile or a specific value in the profile 
        from an existing profile"""
        self.k = profile.k 
        if val == 'T0':
            self.update_T0(profile.T0)
        elif val == 'time':
            self.update_times(profile.times.copy())
            self.update_tau()
        elif val == 'step': 
            self.update_dT_step(profile.dT_step.copy())
        elif val == 'dT':
            self.update_dT(profile.dT.copy())
        elif val == 'da': 
            self.update_times(profile.times.copy())
            self.update_tau()
            self.update_dT_step(profile.dT_step.copy())
            self.update_dT(profile.dT.copy())
        else:
            self.update_T0(profile.T0)
            self.update_times(profile.times.copy())
            self.update_tau()
            self.update_dT_step(profile.dT_step.copy())
            self.update_dT(profile.dT.copy())
        
        self.compute_end_T() 

    def get_T0_lim(self,T0_max: float, T0_min: float, min_T:float):
        """Returns the maximum and minimum values T) can be altered by 
        to remain within the specified limits""" 
        upper = T0_max - self.T0
        lower = max((T0_min - self.T0), (min_T - self.T_final))
        return upper, lower

    def get_time_lim(self, j: int, min_T:float, min_gap): 
        """Sets the maximum changes that can be made to 
        time, j to stay within min_T"""
        lo = (self.tau[j]  + min_gap) - self.tau[j+1]
        hi = (self.tau[j+2] - min_gap) -self.tau[j+1]

        delta_therm_min = -np.inf
        delta_therm_max = np.inf
        M = self.T_final-min_T
        D = self.dT[j] - self.dT[j+1]
        if D > 0: 
            delta_therm_max = M / D
        elif D < 0:
            delta_therm_min = M / D

        lower = max(lo, delta_therm_min)
        upper = min(hi, delta_therm_max)
      
        if lower > upper:
            return  0.0, 0.0

        return upper, lower
      
    def get_dT_step_lim(self, min_T:float):
        """Sets the maximum change that can be made to a step 
        to stay within the specified limit"""
        return (self.T_final - min_T)

    def get_dT_lim(self, j: int, min_T: float):
        """Sets the maximum change that can be made to dT to
        stay within the specified limit""" 
        gaps = np.diff(self.tau)
        dt = gaps[j]
        return ((self.T_final - min_T)/dt)
    
    def propose_new_T0(self, T0_max: float, T0_min: float, min_T:float,
                       std: float, ran: np.random.Generator):
        upper, lower = self.get_T0_lim(T0_max, T0_min, min_T)
        if upper != lower and upper > lower:
            self.T0 += truncnorm.rvs(lower/std,upper/std,loc=0, scale = std, random_state = ran)
            self.update_controller('T0')

    def propose_new_time(self, j: int, min_T:float, min_gap:float,
                       std: float, ran: np.random.Generator):
        
        upper, lower = self.get_time_lim(j, min_T, min_gap)
        if upper == lower:
            return
        prop = truncnorm.rvs(lower/std,upper/std,loc=0, scale = std, random_state = ran)
        self.times[j] += prop
        self.update_controller('time')
       
           
    def propose_new_dT_step(self, j: int, min_T:float, std: float, ran: np.random.Generator, increasing: bool = True):
        
        lower = self.get_dT_step_lim(min_T)
        self.dT_step[j] -= truncnorm.rvs(lower/std,np.inf/std,loc=0, scale = std, random_state = ran)
        
        if increasing: 
            self.dT_step[j] =np.maximum(0.0,self.dT_step[j])
       
        self.update_controller('step')
       

    def propose_new_dT(self, j: int, min_T:float,
                       std: float, ran: np.random.Generator,  increasing: bool = True):
        lower = self.get_dT_lim(j, min_T)
        self.dT[j] -= truncnorm.rvs(lower/std,np.inf/std,loc=0, scale = std, random_state = ran)
        if increasing: 
            self.dT[j] =np.maximum(0.0,self.dT[j])
        
        self.update_controller('dT')

    def propose_birth_step(self, j: int, min_T:float, min_gap: float, mean_s: float ,
                           std_s: float, mean_t: float ,
                           std_t: float, ran: np.random.Generator, increasing: bool = True):
        
        lo = self.tau[j]
        hi = self.tau[j+1]
        if hi - lo <= 2 * min_gap:
            return -1
        
        new_time = ran.uniform(lo + min_gap, hi - min_gap)
        self.times =  np.insert(self.times, j, new_time)
        self.dT = np.insert(self.dT, j+1, 0)
        self.dT_step = np.insert(self.dT_step, j, 0)
        self.update_controller('da')
        final = min_T - 100
        i=1
        while final < min_T: 
            self.dT[j+1] = np.maximum(ran.normal(mean_t,std_t),0.0)
            self.dT_step[j] = np.maximum(ran.normal(mean_s,std_s),0.0)
            self.update_controller('da') 
            final = self.T_final
            i+=1
            if i >100: 
                self.k += 1
                return -1 

        self.k += 1
    
    def propose_death_step(self, j:int): 
        self.times = np.delete(self.times,j)
        self.dT_step = np.delete(self.dT_step,j)
        dt = (self.dT[j]+self.dT[j+1])/2    
        self.dT = np.delete(self.dT,j+1)
        self.dT[j] = dt 
        self. k -= 1 
        self.update_controller('da')


    def admissible_length(self, s: int, min_gap: float, increasing: bool) -> float:
        """
        Length of the uniform region for tau_new inside (lo+min_gap, hi-min_gap).
        Returns 0 if invalid.
        """
        lo, hi = self.tau[s], self.tau[s+2]

        L = (hi - min_gap) - (lo + min_gap)
        if increasing:
            return max(0.0, L)
        else: 
            return L
        
    def set_random_generator(self, seed:int|None = None) -> None:
        """Set the random number genreator using the seed. The seed 
        will be a user set value plus the current simulation number"""
        self.rng = np.random.default_rng(seed=seed)

    

@dataclass
class ReverseJmpMCMC:
    obs: np.ndarray
    iters: int
    MC_crystal: MCBase | list[MCBase] = field(init=False)
    T_target: float
    duration: float
    seed: int 
    
    tolerance: float = field(default=5)
    min_gap: float = 0.0000001 
    non_increasing: bool = True 
    p_geom: float = 0.6
    k_max: int = 5

    init_step_mean: float = 20
    init_step_sd:float = 10.0
    init_dT_mean:float = 300
    init_dT_sd: float = 100
    init_T0_mean: float = 100.0
    init_T0_sd: float = 20.0
    T0_max: float =  150
    T0_min:float = 50

    time_sd: float = 0.5
    step_sd: float = 20.0 
    dT_sd: float = 150
    T0_sd: float = 15.0

    birth_prob: float = 0.5 
    death_prob: float = 0.5
    new_step_mean: float = 20 
    new_step_sd:float  = 10
    new_dT_mean: float = 300
    new_dT_sd: float = 100

    prior_dT_sd: float = 500.0
    prior_step_sd: float = 200.0 
    prior_T0_sd: float = 25.0

    T_min: float = field(init=False)
    t_common: np.ndarray = field(init=False)
    crrnt_T: TProfile  = field(init=False)
    prop_T: TProfile  = field(init=False)

  
    ratio_crrnt: np.ndarray = field(init=False)
    ratio_prop: np.ndarray = field(init=False)
    rng: np.random.Generator = field(init=False)


    def set_random_generator(self) -> None:
        """Set the random number genreator using the seed. The seed 
        will be a user set value plus the current simulation number"""
        self.rng = np.random.default_rng(seed=self.seed)

    def intialise_run(self,cfg: DictConfig | list[DictConfig], experiments:int, err: ErrorOutputHandler):
        """Function that sets up the inverse Monte Carlo ready to run the simulation.
        The random generator is initialised. Values checked. Monte Carlo crystals set"""
        
        # self.check_inputs(err)
        # err.checkpoint()
        self.set_random_generator()
        self.T_min = self.T_target - self.tolerance
        if experiments == 1 and isinstance(cfg, DictConfig):
            err.output("Setting up simulation crystal...")
            self.MC_crystal = MCBase.from_config(cfg)
            self.MC_crystal.RJMCMC_initialise()
            err.output("Crystal setup complete.")
            unit = cfg.temp.unit
            celsius = cfg.temp.celsius
            self.t_common = np.linspace(0,self.MC_crystal.crystal.duration, 10000)
        else:
            self.MC_crystal = []
            for i in range(experiments):    
                err.output("Setting up simulation crystal...")
                self.MC_crystal.append(MCBase.from_config(cfg[i]))
                self.MC_crystal[i].result_csv_path += f"_{i+1}"
                self.MC_crystal[i].RJMCMC_initialise()
                err.output("Crystal setup complete.")
            self.t_common = np.linspace(0,self.MC_crystal[0].crystal.duration, 10000)
            unit = cfg[0].temp.unit
            celsius = cfg[0].temp.celsius

     
        self.crrnt_T = TProfile.initialise(self.rng, self.k_max,self.p_geom, 
                                           self.init_T0_mean, self.init_T0_sd,
                                           self.min_gap,self.duration,
                                           self.init_step_mean,self.init_step_sd,
                                           self.init_dT_mean, self.init_dT_sd,
                                           self.T_min, unit, self.non_increasing,
                                           celsius)
        
        self.prop_T = TProfile.make_copy(self.crrnt_T)
       
        
    #########################################################################################################################
    #                                        Uni-dimensional profile change                                                 #
    #########################################################################################################################
    
    def propose_new_time(self) -> None:
        """Randomly jitter one internal breakpoint, keeping order and min gap."""
        if self.prop_T.k == 0:
            return
        idx = int(self.rng.integers(0, self.prop_T.k))

        self.prop_T.propose_new_time(idx,self.T_min,self.min_gap, self.time_sd,self.rng)

    def propose_new_dT_size(self) -> None:
        """Perturb one step drop; keep it ≤ 0 for 'down' steps."""
        if self.prop_T.k == 0:
            return 
       
        j = int(self.rng.integers(0, self.prop_T.k))
        self.prop_T.propose_new_dT_step(j,self.T_min,self.step_sd,self.rng, self.non_increasing)
      
      
    def propose_new_dT(self) -> None:
        """Perturb one interval slope; optionally enforce ≤ 0."""
       
        j = int(self.rng.integers(0,self.prop_T.k, endpoint=True))
        self.prop_T.propose_new_dT(j,self.T_min,self.dT_sd,self.rng,self.non_increasing)
      
        
    def propose_new_T0(self) -> None:
        """Random-walk on starting temperature."""

        self.prop_T.propose_new_T0(self.T0_max,self.T0_min,self.T_min,
                                   self.T0_sd,self.rng)
      


    def uni_dimensional_change(self, sigma:float, acc_within: int) -> bool:
        """Selects one of the available changes to the temperature profile that retains the dimension of the model
        i.e k stays constant. """
        move = self.rng.choice(["time", "step", "dT", "T0"])
       
        if move == "time":
            self.propose_new_time()
        elif move == "step":
            self.propose_new_dT_size()
        elif move == "dT":
            self.propose_new_dT()
        else:
            self.propose_new_T0()
      
        accept = False
       
        if self.prop_T != self.crrnt_T:
            if self.prop_T.T_final >= self.T_min: 
                self.set_temp_profile(self.prop_T)
                self.ratio_prop = self.new_observation_calculation()
                accept, la = self.accept(sigma)
        self.update_profiles(move,accept, acc_within)
    
        return accept
        
    #########################################################################################################################
    #                                        Trans-dimensional profile change                                               #
    #########################################################################################################################              
   
    def propose_birth_step(self) -> int:
        """
        Insert a new step: choose a segment to split, sample a breakpoint inside it,
        add a new step drop and a new gradient (we create one extra interval).
        """
        if self.prop_T.k == self.k_max:
            return -1

        s = int(self.rng.integers(0,self.prop_T.k))

        self.prop_T.propose_birth_step(s,self.T_min,self.min_gap,self.new_step_mean,self.new_step_sd,
                                       self.new_dT_mean,self.new_dT_sd,self.rng,self.non_increasing)
    
        return s
        
    def log_q_birth_forward(self, s_chosen: int) -> float:
        """Calcualtes the log proposal of adding a step"""

        L = self.prop_T.admissible_length(s_chosen, self.min_gap, self.non_increasing)
        
        lq = np.log(self.birth_prob)
        lq += -np.log(self.prop_T.k)
        lq += -np.log(L)
        lq += self.log_truncnorm_pdf(self.prop_T.dT_step[s_chosen], self.new_step_mean, self.new_step_sd, lower=0.0)
        lq += self.log_truncnorm_pdf(self.prop_T.dT[s_chosen+1], self.new_dT_mean, self.new_dT_sd, lower=0.0)

        return float(lq)

    def log_q_birth_reverse(self):
        """Calcualtes the log proposals of removing the additional step"""
        if self.prop_T.k <= 0:
            return -np.inf

        return np.log(self.death_prob) - np.log(self.prop_T.k)

    def propose_death_step(self, T_min: float) ->  int:
        """Remove a randomly chosen step (and its breakpoint and one gradient)."""

        if self.prop_T.k < 1:
            return -1
       
        j = int(self.rng.integers(self.prop_T.k))
        self.prop_T.propose_death_step(j)
        if self.prop_T.T_final < T_min:
            return -1 
        return j 

    def log_q_death_forward(self, s_chosen: int) -> float:
        """Calcualtes the log proposals of removing the a step"""

        if not (0 <= s_chosen < self.crrnt_T.k):
            return -np.inf
        return np.log(self.death_prob)-np.log(self.crrnt_T.k)
    
    def log_q_death_reverse(self, s_chosen: int): 
        """Calcualtes the log proposals of adding the removed step back in"""
     
      
        L = self.crrnt_T.admissible_length(s_chosen,self.min_gap,self.non_increasing)
       
        lq = np.log(self.birth_prob)
        lq += -np.log(self.crrnt_T.k)     
        lq += -np.log(L)             
        lq += self.log_truncnorm_pdf(self.crrnt_T.dT_step[s_chosen], self.new_step_mean, self.new_step_sd, lower=0.0)
        lq += self.log_truncnorm_pdf(self.crrnt_T.dT[s_chosen+1], self.new_dT_mean, self.new_dT_sd, lower=0.0)
        
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
        elif u < self.birth_prob + self.death_prob and self.crrnt_T.k > 0:
            birth = False
            s_chosen = self.propose_death_step(self.T_min)
        else:
            return False
      
        accept = False
       
        if (s_chosen > -1): 
            if (self.prop_T != self.crrnt_T):
                if self.prop_T.T_final >= self.T_min: 
                    self.set_temp_profile(self.prop_T)
                    self.ratio_prop = self.new_observation_calculation()
                    accept, la = self.trans_dimension_accept(s_chosen,sigma,birth)
            
        if birth: 
            self.update_profiles('da',accept,acc_birth)
        else: 
            self.update_profiles('da',accept,acc_death)
        return accept 
    
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
    
    def log_prior_profile(self, k: int, prof: TProfile) -> float:
        """Log of the prior temperature profile"""
        lp = 0.0
        # return lp
        if k > 0: 
            mean = np.zeros(k+1)
            mean[prof.dT>10] = self.init_dT_mean
            lp += -0.5*np.sum(np.square((prof.dT - mean)/self.prior_dT_sd))

            # lp += -0.5*np.sum(np.square((prof.dT - self.init_dT_mean)/self.prior_dT_sd))
            # lp += -0.5*np.sum(np.square((prof.dT_step - self.init_step_mean)/self.prior_step_sd))
            mean = np.zeros(k)
            mean[prof.dT_step>1] = self.init_step_mean
            lp += -0.5*np.sum(np.square((prof.dT_step - mean)/self.prior_step_sd))

            # lp += -0.5*np.minimum(0,(0-np.sum(prof.dT_step)))
            # gaps = np.diff(prof.tau)
    
            # lp += -0.5*np.sum(gaps)
      
        lp += -0.5*((prof.T0  - self.init_T0_mean)/self.prior_T0_sd)**2
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
    

    def set_temp_profile(self, prof: TProfile): 
        if isinstance(self.MC_crystal, MCBase): 
            self.MC_crystal.crystal.set_temperature_profile(prof.output["kind"],prof.output)
        else: 
            for crystal in self.MC_crystal:
                crystal.crystal.set_temperature_profile(prof.output["kind"],prof.output)

    def store_profile(self, temp_prof: np.memmap, count: int):
        if isinstance(self.MC_crystal, MCBase): 
            temp_prof[:,count]=self.MC_crystal.crystal.Tat(self.t_common)-273.15
        else:
            temp_prof[:,count]=self.MC_crystal[0].crystal.Tat(self.t_common)-273.15
        

    def new_observation_calculation(self, t: float = 0.0, t_pcnt: float | None = None, h_pcnt:float | None = None) -> np.ndarray:
        
        if isinstance(self.MC_crystal, MCBase):
            result = np.zeros(1)
            self.MC_crystal.seed += self.MC_crystal.repetion
            result[0] =  self.MC_crystal.RJMCMC_simulation(t,t_pcnt,h_pcnt)
        else: 
            result = np.zeros(len(self.MC_crystal))
            i=0
            for crystal in self.MC_crystal:
                crystal.seed += crystal.repetion 
                result[i] = crystal.RJMCMC_simulation(t,t_pcnt,h_pcnt)
                i+=1 

        return result
       

    def accept(self, sigma, log_q_fwd=0.0, log_q_rev=0.0, logJ=0.0):
        """
        Fixed-dimension move: prop_k == k
        """
        if self.crrnt_T.k == self.prop_T.k:
            log_q_fwd=log_q_rev=logJ=0.0

        lp_cur = self.log_posterior(self.crrnt_T.k, self.ratio_crrnt, sigma, self.crrnt_T)
        lp_new = self.log_posterior(self.prop_T.k, self.ratio_prop, sigma, self.prop_T)
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

    def update_profiles(self, code: str, acc: bool = False, acc_type: int | None = None):
        """Updates the temperature profile and current observation
        or reverts bakc to the original if the proposal is rejected"""
        if acc:
            self.crrnt_T.update_profile(self.prop_T,code)
            if acc_type is not None:    
                acc_type += 1     
        else: 
            self.prop_T.update_profile(self.crrnt_T,code)
           
    
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
 
   
    def rjmcmc_temperature(self):
        """Main RJMCMC function that proposes a new temperature profile"""

        plt.close()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Temperature")
        self.lines = []
        if isinstance(self.MC_crystal, MCBase): 
            self.ax.plot(self.t_common,(self.MC_crystal.crystal.Tat(self.t_common)-273.15), color="black", alpha=1.0, label="Actual")
            self.true_T = (self.MC_crystal.crystal.Tat(self.t_common)-273.15)
        else:
            self.ax.plot(self.t_common,self.MC_crystal[0].crystal.Tat(self.t_common), color="black", alpha=1.0, label="Actual")
            self.true_T = (self.MC_crystal[0].crystal.Tat(self.t_common)-273.15)
        # self.lines.append(line)
        plt.show(block=False)

        # line, = self.ax.plot(self.t_common,self.obs, color="black", alpha=1.0)
        # self.lines.append(line)
        
        # plt.show(block=False)
        self.set_temp_profile(self.crrnt_T)

       
        temp_prof = np.memmap("temp_profiles.dat", dtype=np.float32, mode='w+', shape=(self.t_common.size, self.iters*2))
        count = 0 
        sigma = self.obs*0.1# float(np.std(self.obs) + 1e-6)
        
        self.ratio_crrnt = self.new_observation_calculation()
        self.store_profile(temp_prof, count)
        
        self.add_result(self.t_common, temp_prof[:,count],0)
     
        count+=1
        samples = []
        acc_within = acc_birth = acc_death = 0

        for it in range(self.iters):
            accept = self.uni_dimensional_change(sigma, acc_within)
            if accept: 
                self.store_profile(temp_prof, count)
                self.add_result(self.t_common, temp_prof[:,count],f"{it}_1")
                # err.output(self.MC.crystal.__repr__())
                count+=1
           
            accept = self.trans_dimensional_change(sigma, acc_birth, acc_death)
            if accept: 
                self.store_profile(temp_prof, count)

                self.add_result(self.t_common, temp_prof[:,count],f"{it}_2")
                count+=1
           
               
            # sigma = self.sigma_update(sigma)
            

            samples.append(float(sigma))



        stats = {
            "acc_within": acc_within / self.iters,
            "acc_birth": acc_birth / max(1, int(self.iters*self.birth_prob)),
            "acc_death": acc_death / max(1, int(self.iters*self.death_prob)),
        }
        if isinstance(self.MC_crystal, MCBase): 
            self.MC_crystal.RJMCMC_cleanup()
        else: 
            for crystal in self.MC_crystal:
                crystal.RJMCMC_cleanup()

        plt.savefig("Temp_profile_Change.png",dpi=300, transparent=False,bbox_inches='tight')
        self.plot_probability_density_grid(count,temp_prof)
        return samples, stats


    def make_time_temp_grid(self, t_min: float, t_max: float, T_min: float,
        T_max: float, n_bins: int = 50, n_samples: int = 10000, unit:str = 's') -> None:
        """
        Create a regular time-temperature grid.
        """
        self.t_edges = np.linspace(t_min, t_max, n_bins + 1)
        self.T_edges = np.linspace(T_min, T_max, n_bins + 1)
        self.grid = np.zeros((n_bins, n_bins), dtype=int)
        self.t_samples = np.linspace(t_min, self.duration, n_samples)

    def accumulate_profile_on_grid(self,temp_prof, count) -> None:
        """
        Sample a time-temperature profile and increment grid squares that the
        profile passes through (once per profile per square).
        """
       
        t_bins = np.digitize((self.t_samples), self.t_edges) - 1
        n_bins = self.grid.shape[0]
        t_bins = np.clip(t_bins, 0, n_bins - 1)
        for i in range(count): 
            T_samples = temp_prof[:,i]
   
            T_bins = np.digitize(T_samples, self.T_edges) - 1
            T_bins = np.clip(T_bins, 0, n_bins - 1)

            pairs = np.stack([T_bins, t_bins], axis=1)
            unique_pairs = np.unique(pairs, axis=0)
        
            for T_idx, t_idx in unique_pairs:
                self.grid[T_idx, t_idx] += 1

    
    def plot_probability_density_grid(self,count, temp_prof):

        self.make_time_temp_grid(0,self.duration,-5,150)
        self.accumulate_profile_on_grid(temp_prof,count)

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
        ax.plot(self.t_samples,self.true_T,linewidth=2,linestyle="--",color="white")
        # ax.legend()
        plt.savefig("weights.png",dpi=300, transparent=False,bbox_inches='tight')
        plt.close()
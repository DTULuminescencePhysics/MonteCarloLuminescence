from __future__ import annotations
import numpy as np
from math import lgamma, factorial
from dataclasses import dataclass, field
from src.classes.monte_carlo import MCBase
from omegaconf import DictConfig
from src.errors import ErrorOutputHandler
from src.classes.output.graph import chronologyPlot_running
from src.classes.output.temp_results_file import chronology_results
from src.helper_functions import _as_1d_array
from typing import  Literal, Optional, Dict, List, Tuple


MonotonicMode = Literal["free", "increasing", "decreasing"]
LikelihoodMode = Literal["Gaussian", "FullGaussian", "student-t", "L1", "L1L2Hybrid", "Bernoulli"] 

@dataclass
class Bounds:
    """Generic min/max bounds."""
    lo: float
    hi: float

    def contains(self, x: float) -> bool:
        return np.isfinite(x) and (self.lo <= x <= self.hi)
    
    @property
    def midpoint(self):
        return 0.5*(self.lo+self.hi)
    @property    
    def width(self,):
        return self.hi - self.lo
    @property
    def half_width(self,):
        return 0.5*self.width
    
    def _rand_in_bounds(self,rng: np.random.Generator):
        return float(rng.uniform(self.lo, self.hi))

@dataclass
class TemperatureProfile:
    """
    Piecewise-linear time/temperature profile with fixed time domain [0, duration].
    times and temps include endpoints:
      times[0] == 0, times[-1] == duration
      temps[0] == T0, temps[-1] == Tf
    """
    times: np.ndarray  
    temps: np.ndarray 

    def copy(self) -> "TemperatureProfile":
        return TemperatureProfile(self.times.copy(), self.temps.copy())

    @property
    def k_nodes(self) -> int:
        return int(self.times.shape[0])

    @property
    def n_internal(self) -> int:
        return max(0, self.k_nodes - 2)

    def as_points(self) -> List[Tuple[float, float]]:
        return list(zip(self.times.tolist(), self.temps.tolist()))

    def interpolate(self, t: float) -> float:
        """Linear interpolation within existing nodes (expects 0<=t<=duration)."""
        idx = np.searchsorted(self.times, t, side="right") - 1
        idx = int(np.clip(idx, 0, self.k_nodes - 2))
        t0, t1 = self.times[idx], self.times[idx + 1]
        y0, y1 = self.temps[idx], self.temps[idx + 1]
        if t1 == t0:
            return float(y0)
        w = (t - t0) / (t1 - t0)
        return float(y0 + w * (y1 - y0))

@dataclass
class log_likelihood:
    observation: np.ndarray
    sigma: np.ndarray

    def _ll(self, pred:np.ndarray) -> float:
        return (np.sum(np.power(((self.observation-pred)/self.sigma),2))*(-0.5))


class ReverseJumpMCMC:
    """
    Reversible Jump MCMC over temperature profiles (variable number of internal points).

  
    This class handles:
      - Birth/death of internal (time, temp) nodes
      - Local time/temperature perturbations
      - Optional endpoint perturbations (T0 and Tf) within their own bounds
      - Constraints:
          * time domain fixed [0, duration]
          * times strictly increasing
          * all temps within global bounds
          * T0 and Tf each within their respective bounds
          * monotonic mode: increasing / decreasing / free

    Notes on RJ acceptance:
      - Birth: choose an interval uniformly among current segments, sample t_new uniform in that interval,
               sample T_new ~ Normal(interpolated, sigma_birth)
      - Death: choose an internal knot uniformly to remove
      - Jacobian is 1 (identity transform between random u and inserted values)
      - Acceptance ratio includes forward/backward proposal probabilities and densities.

    Parameters
    ----------
    k_init:
        Initial number of internal points (the chain can add/remove later).
    duration:
        End time of profile; time domain is fixed to [0, duration].
    T0_bounds:
        Bounds for the starting temperature (at time 0).
    Tf_bounds:
        Bounds for the final temperature (at time duration).
    T_global_bounds:
        Bounds for ALL temperatures (including endpoints and internal nodes).
    monotonic:
        "increasing", "decreasing", or "free".
    target_log_prob:
        Callable that takes a TemperatureProfile and returns log probability.
    rng:
        Optional numpy Generator.

    Proposal controls (can be tuned)
    --------------------------------
    p_birth, p_death, p_move_time, p_move_temp, p_move_endpoints:
        Move-type probabilities (will be normalized internally).
    sigma_birth:
        Std dev for temperature proposal at a new birth knot (around interpolated temp).
    sigma_temp:
        Std dev for internal temperature random-walk moves.
    sigma_time_frac:
        Time random-walk step size as a fraction of local neighboring interval length.
    sigma_endpoints:
        Std dev for endpoint (T0/Tf) random-walk moves.
    min_internal, max_internal:
        Hard bounds on number of internal points allowed.
    """

    def __init__(self, iters: int, seed: int, timeSpan: Bounds, T0_bounds: Bounds,
                 Tf_bounds: Bounds, T_global_bounds: Bounds, monotonic: MonotonicMode,
                 likelihood_log_prob: log_likelihood,
                 min_internal: int = 0, max_internal: int = 200,
                 p_birth: float = 0.20, p_death: float = 0.20, p_move_time: float = 0.25, 
                 p_move_temp: float = 0.25, p_move_endpoints: float = 0.10,
                 sigma_birth: float = 1.0, sigma_t_birth: float = 1.0, sigma_temp: float = 0.5, sigma_time_frac: float = 0.15,
                 sigma_endpoints: float = 0.5) -> None:
        
        if timeSpan.hi <= 0:
            raise ValueError("end time must be > 0")
        if monotonic not in ("free", "increasing", "decreasing"):
            raise ValueError("monotonic must be 'free', 'increasing', or 'decreasing'")
        if not (T_global_bounds.lo <= T0_bounds.lo <= T0_bounds.hi <= T_global_bounds.hi):
            # Not strictly required, but usually intended; relax by removing this check if you want.
            pass
        if not (T_global_bounds.lo <= Tf_bounds.lo <= Tf_bounds.hi <= T_global_bounds.hi):
            pass
        
        self.iters = iters
        self.seed = seed
        
        self.timeSpan = timeSpan
        self.T0_bounds = T0_bounds
        self.Tf_bounds = Tf_bounds
        self.T_global_bounds = T_global_bounds
        self.monotonic = monotonic

        self.likelihood_log_prob = likelihood_log_prob

        self.sigma_birth = float(sigma_birth)
        self.sigma_t_birth = float(sigma_t_birth)
        self.sigma_temp = float(sigma_temp)
        self.sigma_time_frac = float(sigma_time_frac)
        self.sigma_endpoints = float(sigma_endpoints)

        self.min_internal = int(min_internal)
        self.max_internal = int(max_internal)

        self.move_names = ["birth", "death", "move_time", "move_temp", "move_endpoints"]

        self.p_birth = float(p_birth)
        self.p_death = float(p_death)
        self.p_move_time = float(p_move_time)
        self.p_move_temp = float(p_move_temp)
        self.p_move_endpoints = float(p_move_endpoints)
                                      
        self.move_probs = np.array((self.p_birth,self.p_death,self.p_move_time,self.p_move_temp,self.p_move_endpoints))
        self._normalize_move_weights()
        self.result_store = chronology_results(self.iters,self.max_internal,"w+")
    
    @classmethod
    def from_config(cls, seed: int, obs: np.ndarray, duration: float, cfg: DictConfig, err: ErrorOutputHandler) -> "ReverseJumpMCMC":
     
        timeSpan = Bounds(0,duration)
        T0_bounds = Bounds(cfg.T0_lo,cfg.T0_hi)
        Tf_bounds= Bounds(cfg.T_Target-cfg.T_tolerance,cfg.T_Target+cfg.T_tolerance)
        T_max = max(T0_bounds.hi,Tf_bounds.hi)
        T_min = min(T0_bounds.lo,Tf_bounds.lo)
        T_global_bounds= Bounds(T_min-cfg.T_tolerance,T_max+cfg.T_tolerance)
        likelihood_log_prob= log_likelihood(obs,cfg.rjmcmc.log_likelihood.sigma)
        

        return cls(cfg.iters, seed, timeSpan, T0_bounds, Tf_bounds, T_global_bounds, 
                   cfg.monotonic, likelihood_log_prob, cfg.min_internal,cfg.max_internal, 
                   cfg.rjmcmc.parameters.p_birth, cfg.rjmcmc.parameters.p_death, 
                   cfg.rjmcmc.parameters.p_move_time, cfg.rjmcmc.parameters.p_move_temp, 
                   cfg.rjmcmc.parameters.p_move_endpoints, cfg.rjmcmc.parameters.sigma_birth,
                   cfg.rjmcmc.parameters.sigma_t_birth, 
                   cfg.rjmcmc.parameters.sigma_temp, cfg.rjmcmc.parameters.sigma_time_frac,
                   cfg.rjmcmc.parameters.sigma_endpoints)

    def set_random_generator(self):
        self.rng = np.random.default_rng(self.seed)

    def update_tracker(self, count:int):
        if isinstance(self.MC_crystal, MCBase):
            self.pl.add_result(self.MC_crystal.crystal.Tat(self.pl.t_common_unit),count)
        else:
            self.pl.add_result(self.MC_crystal[0].crystal.Tat(self.pl.t_common_unit),count)

    def store_profile(self, it: int|None, prof:TemperatureProfile, accpt:bool)-> None:
        if it is None:
            return
        self.result_store.write_result(it,prof.times,prof.temps,accpt,self.current_logp)

        if accpt: 
            self.update_tracker(it)

    def initialise_run(self, cfg: DictConfig | list[DictConfig], experiments:int, err: ErrorOutputHandler) -> None:
        self.set_random_generator()
        if experiments == 1 and isinstance(cfg, DictConfig):
            err.output("Setting up temperature profiles")
            unit = cfg.temp.unit
            celsius = celsius=cfg.temp.celsius
            err.output("Temperature profile setup complete")
            err.output("Setting up simulation crystal...")
            self.MC_crystal = MCBase.from_config(cfg)
            self.MC_crystal.thermochron_initialise()
            err.output("Crystal setup complete.")
        else:
            err.output("Setting up temperature profiles")
            unit = cfg[0].temp.unit
            celsius=cfg[0].temp.celsius

            err.output("Temperature profile setup complete")
            self.MC_crystal = []
            for i in range(experiments):    
                err.output("Setting up simulation crystal...")
                self.MC_crystal.append(MCBase.from_config(cfg[i]))
                self.MC_crystal[i].result_csv_path += f"_{i+1}"
                self.MC_crystal[i].thermochron_initialise()
                err.output("Crystal setup complete.")

        self.pl = chronologyPlot_running(duration=self.timeSpan.hi,unit=unit,celsius=celsius)
        self.pl.set_profile_files(self.iters,self.max_internal)
        if isinstance(self.MC_crystal, MCBase):
            self.pl.setup_simulation_tracker(self.MC_crystal.crystal.Tat(self.pl.t_common_unit))
        else:
            self.pl.setup_simulation_tracker(self.MC_crystal[0].crystal.Tat(self.pl.t_common_unit))

        self.current = self._make_initial_profile()
        self.current_logp = self._log_target(self.current)

        if not np.isfinite(self.current_logp):
            raise ValueError("Initial profile has non-finite target_log_prob; check constraints or your target.")
        self._reset_move_stats()

    def step(self, it: int|None = None) -> None:
        move = self.rng.choice(self.move_names, p=self.move_probs)
        self.attempt_success[move]["selected"] += 1

        if move == "birth":
            prop, log_q_fwd, log_q_bwd = self._propose_birth()
        elif move == "death":
            prop, log_q_fwd, log_q_bwd = self._propose_death()
        elif move == "move_time":
            prop, log_q_fwd, log_q_bwd = self._propose_move_time()
        elif move == "move_temp":
            prop, log_q_fwd, log_q_bwd = self._propose_move_temp()
        else: 
            prop, log_q_fwd, log_q_bwd = self._propose_move_endpoints()

        if prop is None:
            return

        self.attempt_success[move]["usable"] += 1
        prop_logp = self._log_target(prop)

        
        if not np.isfinite(prop_logp):
            return
       
        log_alpha = (prop_logp - self.current_logp) + (log_q_bwd - log_q_fwd)
        if np.log(self.rng.random()) < log_alpha:
            self.current = prop
            self.current_logp = prop_logp
            self.attempt_success[move]["accepted"] += 1
            self.store_profile(it,prop,True)
        else:
            self.store_profile(it,prop,False)

        return 

    def run(self) -> None:
    
        for i in range(self.iters):
            self.step(i)
            if i%100 ==0:
                self.result_store.flush() 
         
        self.result_store.flush()
        self.pl.save_close_tracker()
       

    def acceptance_rates(self) -> Dict[str, Dict[str,float]]:
        rates = { 
            name: {
                "usable":   0.0, 
                "accepted": 0.0, 
            }
            for name in self.move_names
        }
       
        for m in self.move_names:
            att = self.attempt_success[m]["selected"] 
            rates[m]["usable"] = self.attempt_success[m]["usable"]/att if att > 0 else 0.0
            rates[m]["accepted"] = self.attempt_success[m]["accepted"]/att if att > 0 else 0.0

        return rates

    def _set_sigmas(self, sigmas:Dict[str, float]) -> None:
        self.sigma_birth = float(sigmas["birth"])
        self.sigma_t_birth = float(sigmas["birth_t"])
        self.sigma_temp = float(sigmas["move_temp"])
        self.sigma_time_frac = float(sigmas["move_time"])
        self.sigma_endpoints = float(sigmas["move_endpoints"])

    def _get_sigmas(self) -> Dict[str, float]:
        return {
            "birth": float(self.sigma_birth),
            "birth_t": float(self.sigma_t_birth),
            "move_temp": float(self.sigma_temp),
            "move_time": float(self.sigma_time_frac),
            "move_endpoints": float(self.sigma_endpoints),
        }
    def _set_probs(self, probs:Dict[str, float]|None = None) -> None: 
        if probs is None:
            self.p_birth = self.move_probs[0]
            self.p_death = self.move_probs[1]
            self.p_move_time = self.move_probs[2]
            self.p_move_temp =  self.move_probs[3]
            self.p_move_endpoints = self.move_probs[4]
        else:
            self.p_birth = float(probs["birth"])
            self.p_death = float(probs["death"])
            self.p_move_time = float(probs["move_time"])
            self.p_move_temp =  float(probs["move_temp"])
            self.p_move_endpoints = float(probs["move_endpoints"])
            self.move_probs[0] = self.p_birth 
            self.move_probs[1] = self.p_death 
            self.move_probs[2] = self.p_move_time 
            self.move_probs[3] = self.p_move_temp 
            self.move_probs[4] = self.p_move_endpoints

    def _get_probs(self) -> Dict[str, float]:
        return {
            "birth": float(self.p_birth),
            "death": float(self.p_death),
            "move_time": float(self.p_move_time),
            "move_temp": float(self.p_move_temp),
            "move_endpoints": float(self.p_move_endpoints),
        }
    
    def burn_in_setup(self, overall_bounds:Tuple[float,float]=(0.15,0.45), 
                      birth_accept_target: Optional[Tuple[float,float]] = None, death_accept_target: Optional[Tuple[float,float]] = None,
                      move_time_accept_target: Optional[Tuple[float,float]] = None, move_temp_accept_target: Optional[Tuple[float,float]] = None,
                      move_endpoints_accept_target: Optional[Tuple[float,float]] = None,
                      sigma_birth_bounds: Optional[Tuple[float,float]] = None, sigma_time_bounds: Optional[Tuple[float,float]] = None,
                      sigma_temp_bounds: Optional[Tuple[float,float]] = None, sigma_endpoints_bound: Optional[Tuple[float,float]] = None, 
                      ) -> Tuple[Bounds,Dict[str,Bounds],Dict[str,Bounds]]:
        
       
        overall_accept_target = Bounds(overall_bounds[0],overall_bounds[1])
       
        per_move_accept_target = {
            "birth": Bounds(birth_accept_target[0],birth_accept_target[1]) if birth_accept_target is not None else overall_accept_target,
            "death": Bounds(death_accept_target[0],death_accept_target[1]) if death_accept_target is not None else overall_accept_target,
            "move_time": Bounds(move_time_accept_target[0],move_time_accept_target[1]) if move_time_accept_target is not None else overall_accept_target, 
            "move_temp": Bounds(move_temp_accept_target[0],move_temp_accept_target[1]) if move_temp_accept_target is not None else overall_accept_target,
            "move_endpoints": Bounds(move_endpoints_accept_target[0],move_endpoints_accept_target[1]) if move_endpoints_accept_target is not None else overall_accept_target
        }

        per_move_sigma_bounds = {
            "birth": Bounds(sigma_birth_bounds[0],sigma_birth_bounds[1]) if sigma_birth_bounds is not None else Bounds(1e-4,1e2),
            "move_time": Bounds(sigma_time_bounds[0],sigma_time_bounds[1]) if sigma_time_bounds is not None else Bounds(1e-4,1e2), 
            "move_temp": Bounds(sigma_temp_bounds[0],sigma_temp_bounds[1]) if sigma_temp_bounds is not None else Bounds(1e-3,1e2),
            "move_endpoints": Bounds(sigma_endpoints_bound[0],sigma_endpoints_bound[1]) if sigma_endpoints_bound is not None else Bounds(1e-4,1e1)
        }

        return overall_accept_target, per_move_accept_target, per_move_sigma_bounds

    # ---------------- Burn-in tuning ----------------
    def burn_in_tune(self, max_steps: int, window: int,
                     eta_sigma0: float, eta_prob0:float, overall_check: bool,
                     individual_check:bool, overall_bounds: Tuple[float,float] = (0.15,0.45),
                     birth_accept_target: Optional[Tuple[float,float]] = None, death_accept_target: Optional[Tuple[float,float]] = None,
                     move_time_accept_target: Optional[Tuple[float,float]] = None, move_temp_accept_target: Optional[Tuple[float,float]] = None,
                     move_endpoints_accept_target: Optional[Tuple[float,float]] = None,
                     sigma_birth_bounds: Optional[Tuple[float,float]] = None, sigma_time_bounds: Optional[Tuple[float,float]] = None,
                     sigma_temp_bounds: Optional[Tuple[float,float]] = None, sigma_endpoints_bound: Optional[Tuple[float,float]] = None,
                     min_move_prob: float = 0.03, max_move_prob:float = 0.8, centre_pull: float = 0.25,
                     adjustment_factor: float = 0.6, patience_windows: int = 2, verbose: bool = True,) -> None:
        """
        Run burn-in and *adapt* move probabilities + proposal scales and stops when 
        criteria is met.

        Adaptation happens ONLY inside this function. After it returns, kernels are fixed.
        Stopping criteria options:
          - overall_accept_target=(lo,hi): uses total accepted / total attempted across all move types
          - per_move_accept_target={move:(lo,hi), ...}: e.g. {"move_temp":(0.2,0.4), "birth":(0.05,0.2)}
          - birth_death_min=x: requires birth and death acceptance each >= x (minimums)

        You can provide any combination; all provided criteria must be satisfied to stop.

        Returns a summary dict.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if window <= 0:
            raise ValueError("window must be > 0")
        
        overall_accept_target, per_move_accept_target, per_move_sigma_bounds = self.burn_in_setup(overall_bounds, birth_accept_target, 
                                                                                                  death_accept_target, move_time_accept_target,
                                                                                                  move_temp_accept_target, move_endpoints_accept_target,
                                                                                                  sigma_birth_bounds, sigma_time_bounds,
                                                                                                  sigma_temp_bounds, sigma_endpoints_bound)
        p_range = Bounds(min_move_prob,max_move_prob)
        move_sigmas = self._get_sigmas()
        move_probs  = self._get_probs()
        self._reset_move_stats()

        consecutive_ok = 0
        steps_done = 0
        n_windows = int(np.ceil(max_steps / window))

        for w in range(n_windows):
            steps_this = min(window, max_steps - steps_done)
            if steps_this <= 0:
                break

            for _ in range(steps_this):
                self.step()
                steps_done += 1

            per_move_rates = self.acceptance_rates()
            overall_rate = self._overall_acceptance_rate()

            ok = self._meets_acceptance_criteria(overall_rate, per_move_rates,
                                                 overall_check, individual_check,
                                                 overall_accept_target, per_move_accept_target)
            
            if ok:
                consecutive_ok += 1
            else:
                consecutive_ok = 0

            eta_sigma = eta_sigma0 / ((w+1) ** adjustment_factor)
            eta_prob = eta_prob0 / ((w+1) ** adjustment_factor)
         
            if consecutive_ok < patience_windows:
                if verbose:
                    msg = (
                    f"[burn_in window {w+1}/{n_windows}] "
                    f"steps={steps_done}/{max_steps} "
                    f"overall_acc={overall_rate:.3f} "
                    f"per_move_acc={{" + ", ".join(f"{k}:{per_move_rates[k]["accepted"]:.3f}" for k in self.move_names) + "}} "
                    f"ok={ok} consec_ok={consecutive_ok}/{patience_windows} "
                    )

                for m in self.move_names:
                    score = 0.5 * self.attempt_success[m]["usable"] + 0.5 * self.attempt_success[m]["accepted"]
                    move_probs[m] = self._adapt_prob_weight(move_probs[m],score,eta_prob,p_range)

                print(per_move_rates["birth"]["accepted"],per_move_rates["death"]["accepted"])
                new_bith = 0.5*(per_move_rates["birth"]["accepted"] + per_move_rates["death"]["accepted"])
                print(new_bith)
                per_move_rates["birth"]["accepted"] = new_bith
                print(per_move_rates["birth"]["accepted"])
                for key in move_sigmas:
                    print(key)
                    # if not per_move_accept_target[key].contains(per_move_rates[key]["accepted"]):
                    move_sigmas[key] = self._tune_sigma(move_sigmas[key], eta_sigma, per_move_rates[key]["accepted"], 
                                                        per_move_accept_target[key], per_move_sigma_bounds[key], centre_pull)

                self._set_sigmas(move_sigmas)

                if self.current.n_internal <= self.min_internal + 1:
                    self.p_birth *= 1.10
                    self.p_death *= 0.90
                elif self.current.n_internal >= self.max_internal - 1:
                    self.p_birth *= 0.90
                    self.p_death *= 1.10       
                
             
                self._set_probs(move_probs)
              
                self._normalize_move_weights()
                move_probs = self._get_probs()

         
            if verbose:
                msg2 = ( f"move_probs={{p_birth:{self.p_birth:.4g}, p_death:{self.p_death:.4g}, "
                    f"p_move_temp:{self.p_move_temp:.4g}, p_move_time:{self.p_move_time:.4g},"
                    f"p_move_endpoints:{self.p_move_endpoints:.4g}}}"
                    f"sigmas={{sigma_birth:{self.sigma_birth:.4g}, sigma_temp:{self.sigma_temp:.4g}, "
                    f"sigma_time_frac:{self.sigma_time_frac:.4g}, sigma_endpoints:{self.sigma_endpoints:.4g}}}"
                )
                
                print(msg+msg2)

            self._reset_move_stats()

            if consecutive_ok >= patience_windows:
                break

    def _overall_acceptance_rate(self) -> float:
        total_att=0
        total_acc=0
        for m in self.move_names:
            total_att += self.attempt_success[m]["selected"]
            total_acc += self.attempt_success[m]["accepted"]
        
        return (total_acc / total_att) if total_att > 0 else 0.0
       

    def _meets_acceptance_criteria(self, overall_rate: float, per_move_rates: Dict[str, Dict[str,float]], 
                                   overall_check: bool, individual_check: bool, overall_accept_target: Bounds, 
                                   per_move_accept_target: Dict[str,Bounds],) -> bool:

        
        if overall_check and overall_accept_target.contains(overall_rate):
            return False
        
        if individual_check:
            for m in self.move_names: 
                if not per_move_accept_target[m].contains(per_move_rates[m]["accepted"]):
                    return False
        
        return True 


    @staticmethod
    def _tune_sigma(sigma: float, eta: float, acc: float, target_range: Bounds, 
                    range_bounds: Bounds, centre_pull: float) -> float:

        if target_range.contains(acc):
            new_sigma = sigma
        # if not np.isfinite(acc) or acc <= 0:
        #     factor = 1.0 / eta
        # elif acc < target_range.lo:
        #     factor = 1.0 / np.sqrt(eta)  # reduce step
        # elif acc > target_range.hi:
        #     factor = np.sqrt(eta)        # increase step
        # else:
        #     factor = 1.0
        # new_sigma = float(max(1e-12, sigma * factor))

        # new_sigma = min(max(new_sigma, range_bounds.lo), range_bounds.hi)
        # print("1",new_sigma)
        else:
            new_sigma = np.exp(np.log(sigma) + eta * (acc-target_range.midpoint))
            new_sigma = min(max(new_sigma, range_bounds.lo), range_bounds.hi)
        # print("3",new_sigma)
        # if range_bounds.contains(new_sigma) and centre_pull > 0.0:
        #     edge_factor = abs(new_sigma - range_bounds.midpoint) / range_bounds.half_width
        #     pull_strength = centre_pull * eta * edge_factor
        #     new_sigma = new_sigma + pull_strength * (range_bounds.midpoint - new_sigma)
        #     # new_sigma = min(max(new_sigma, range_bounds.lo),  range_bounds.hi)
        #     print("4",new_sigma)
        print(sigma,new_sigma)
        return new_sigma

    @staticmethod
    def _adapt_prob_weight(p: float, usable_rate: float, eta: float, range: Bounds):
        logit = np.log(max(p, 1e-12))
        logit_new = logit + eta * (usable_rate - 0.5)
        p_new = np.exp(logit_new)
        return min(max(p_new, range.lo), range.hi)

    def _normalize_move_weights(self) -> None:
        tot = np.sum(self.move_probs)
        self.move_probs = self.move_probs/tot
        self._set_probs()

    def _reset_move_stats(self) -> None:
        self.attempt_success = { 
            name: {
                "selected": 0, 
                "usable":   0, 
                "accepted": 0, 
            }
            for name in self.move_names
        }

        # ---------------- Target / Likelihood / Prior ----------------

    def _predict(self, profile: TemperatureProfile, t: float = 0.0, 
                 t_pcnt: float | None = None, h_pcnt:float | None = None) -> np.ndarray:
        if isinstance(self.MC_crystal, MCBase):
            result = np.zeros(1)
            self.MC_crystal.seed += self.MC_crystal.repetion
            self.MC_crystal.crystal.set_temperature_profile("linearsteps",profile.times.copy(),profile.temps.copy())
            result[0] =  self.MC_crystal.thermochron_simulation(t,t_pcnt,h_pcnt)
        else: 
            result = np.zeros(len(self.MC_crystal))
            i=0
            for crystal in self.MC_crystal:
                crystal.seed += crystal.repetion
                crystal.set_temperature_profile("linearsteps",profile.times.copy(),profile.temps.copy())
                result[i] = crystal.thermochron_simulation(t,t_pcnt,h_pcnt)
                i+=1 
        return result

    def _log_target(self, profile: TemperatureProfile) -> float:
        if not self._is_valid(profile):
            return -np.inf

        y_pred = self._predict(profile)
        ll = self.likelihood_log_prob._ll(y_pred) 
        if not np.isfinite(ll):
            return -np.inf
        
        return ll

    # ---------------- Initialization & Constraints ----------------

    def _make_initial_profile(self,) -> TemperatureProfile:

        k_init = self.rng.integers(self.min_internal,self.max_internal)
        T0 = self.T0_bounds._rand_in_bounds(self.rng)
        Tf = self.Tf_bounds._rand_in_bounds(self.rng)

        if not self.T_global_bounds.contains(T0) or not self.T_global_bounds.contains(Tf):
            raise ValueError("Endpoint bounds must be within global temperature bounds (or adjust checks).")

        if k_init == 0:
            times = np.array([self.timeSpan.lo, self.timeSpan.hi], dtype=float)
            temps = np.array([T0, Tf], dtype=float)
        else:
            internal_times = np.linspace(self.timeSpan.lo, self.timeSpan.hi, num=k_init + 2, dtype=float)[1:-1]
            times = np.concatenate(([self.timeSpan.lo], internal_times, [self.timeSpan.hi])).astype(float)
            
            base = np.linspace(T0, Tf, num=k_init + 2, dtype=float)
            noise = self.rng.normal(0.0, self.sigma_temp, size=k_init + 2)
            noise[0] = 0.0
            noise[-1] = 0.0
            temps = np.clip(base + noise, self.T_global_bounds.lo, self.T_global_bounds.hi)
            temps[0] = T0
            temps[-1] = Tf
            temps = self._project_monotonic(temps)
     
        prfl = TemperatureProfile(times=times, temps=temps)
        
        if not self._is_valid(prfl):
            prfl.temps = np.clip(prfl.temps, self.T_global_bounds.lo, self.T_global_bounds.hi)
            prfl.temps = self._project_monotonic(prfl.temps)
            if not self._is_valid(prfl):
                raise ValueError("Failed to build a valid initial profile. Check bounds/monotonic constraints.")
        
        return prfl

    def _is_valid(self, s: TemperatureProfile) -> bool:
        if s.times.shape != s.temps.shape or s.times.ndim != 1:
            return False
        if s.times[0] != 0.0 or s.times[-1] != self.timeSpan.hi:
            return False
        if np.any(~np.isfinite(s.times)) or np.any(~np.isfinite(s.temps)):
            return False
        if np.any(np.diff(s.times) <= 0):
            return False

        if not (self.min_internal <= s.n_internal <= self.max_internal):
            return False

        if np.any(s.temps < self.T_global_bounds.lo) or np.any(s.temps > self.T_global_bounds.hi):
            return False

        if not self.T0_bounds.contains(float(s.temps[0])):
            return False
        if not self.Tf_bounds.contains(float(s.temps[-1])):
            return False

        if self.monotonic == "increasing":
            if np.any(np.diff(s.temps) < 0):
                return False
        elif self.monotonic == "decreasing":
            if np.any(np.diff(s.temps) > 0):
                return False

        return True

    def _project_monotonic(self, temps: np.ndarray) -> np.ndarray:
        """
        Simple projection for monotonic modes:
        - increasing: enforce non-decreasing by cumulative max
        - decreasing: enforce non-increasing by cumulative min
        - free: no-op

        This is used ONLY for initialization/fallback. Proposals are rejected if invalid,
        preserving detailed balance.
        """
        y = temps.copy()
        if self.monotonic == "increasing":
            y = np.maximum.accumulate(y)
        elif self.monotonic == "decreasing":
            y = np.minimum.accumulate(y)
        return y

    # ---------- Proposals ----------

    def _propose_birth(self) -> Tuple[Optional[TemperatureProfile], float, float]:
        s = self.current
        if s.n_internal >= self.max_internal:
            return None, 0.0, 0.0

        # Choose an interval uniformly among segments (k_nodes-1 segments)
        p=(s.times[1:] - s.times[:-1])/(s.times[-1]-s.times[0])
        n_seg = s.k_nodes - 1
        seg_idx = self.rng.choice(n_seg, p=p)
       

        tL, tR = float(s.times[seg_idx]), float(s.times[seg_idx + 1])
        if not (tR > tL):
            return None, 0.0, 0.0
        u1 = self.rng.uniform(0.0, 1.0)
    
        t_new = tL + self.sigma_t_birth*u1*(tR-tL)
        TL, TR = float(s.temps[seg_idx]), float(s.temps[seg_idx + 1])

        u2= self.rng.uniform(-0.5, 0.5)
        T_new = TL + u1*self.sigma_t_birth*(TR-TL)+self.sigma_birth*u2

        if not self.T_global_bounds.contains(T_new):
            return None, 0.0, 0.0

        times_new = np.insert(s.times, seg_idx + 1, t_new)
        temps_new = np.insert(s.temps, seg_idx + 1, T_new)
        prop = TemperatureProfile(times_new, temps_new)

        if not self._is_valid(prop):
            return None, 0.0, 0.0

        log_q_fwd = (
            np.log(t_new-tL)
            +np.log(tR-t_new)
            +np.log(self.sigma_t_birth*self.sigma_birth)
            +np.log(self.move_probs[1])
        )

        log_q_bwd = (
            np.log(tR-tL)
            +np.log(self.T_global_bounds.width)
            +np.log(self.move_probs[0])
        )

        return prop, float(log_q_fwd), float(log_q_bwd)

    def _propose_death(self) -> Tuple[Optional[TemperatureProfile], float, float]:
        s = self.current
        if s.n_internal <= self.min_internal:
            return None, 0.0, 0.0

      
        internal_indices = np.arange(1, s.k_nodes - 1)
        rm_idx = int(self.rng.choice(internal_indices))

        t_rm = float(s.times[rm_idx])

        times_new = np.delete(s.times, rm_idx)
        temps_new = np.delete(s.temps, rm_idx)
        prop = TemperatureProfile(times_new, temps_new)

        if not self._is_valid(prop):
            return None, 0.0, 0.0

        log_q_fwd = (
            np.log(s.times[rm_idx+1]-s.times[rm_idx-1])
            +np.log(self.T_global_bounds.width)
            +np.log(self.move_probs[0])
        )
        log_q_bwd = (
            np.log(t_rm-s.times[rm_idx-1])
            +np.log(s.times[rm_idx+1]-t_rm)
            +np.log(self.move_probs[1])
            +np.log(self.sigma_t_birth*self.sigma_birth)
        )
       
        return prop, float(log_q_fwd), float(log_q_bwd)

    def _propose_move_time(self) -> Tuple[Optional[TemperatureProfile], float, float]:
        s = self.current
        if s.n_internal == 0:
            return None, 0.0, 0.0

        # Pick an internal knot
        idx = int(self.rng.integers(1, s.k_nodes - 1))
        t_prev, t_cur, t_next = float(s.times[idx - 1]), float(s.times[idx]), float(s.times[idx + 1])

        # Propose within neighbor interval using a symmetric normal step scaled by local interval
        local = min(t_cur - t_prev, t_next - t_cur)
        if local <= 0:
            return None, 0.0, 0.0
        t_new = t_cur + self.sigma_time_frac*self.rng.normal(0.0,1.0)
        
        # Keep strict ordering by enforcing open interval (t_prev, t_next)
        if not (t_prev < t_new < t_next):
            return None, 0.0, 0.0

        times_new = s.times.copy()
        times_new[idx] = t_new
        prop = TemperatureProfile(times_new, s.temps.copy())

        if not self._is_valid(prop):
            return None, 0.0, 0.0

        log_q_fwd = (
            np.log(t_new-t_prev)
            +np.log(t_next-t_new)
        )
        log_q_bwd = (
            np.log(t_cur-t_prev)
            +np.log(t_next-t_cur)
        )

        return prop, log_q_fwd, log_q_bwd 

    def _propose_move_temp(self) -> Tuple[Optional[TemperatureProfile], float, float]:
        s = self.current
        if s.n_internal == 0:
            # still can move nothing; but skip
            return None, 0.0, 0.0

        idx = int(self.rng.integers(1, s.k_nodes - 1))  # internal only
        T_cur = float(s.temps[idx])
        T_new = float(T_cur + self.sigma_temp*self.rng.normal(0.0,1.0))
        if not self.T_global_bounds.contains(T_new):
            return None, 0.0, 0.0

        temps_new = s.temps.copy()
        temps_new[idx] = T_new
        prop = TemperatureProfile(s.times.copy(), temps_new)

        if not self._is_valid(prop):
            return None, 0.0, 0.0

        return prop, 0.0, 0.0

    def _propose_move_endpoints(self) -> Tuple[Optional[TemperatureProfile], float, float]:
        s = self.current
      
        temps_new = s.temps.copy()

        if self.rng.random() < 0.5:
            T0_cur = float(temps_new[0])
            T0_new = float(T0_cur + self.sigma_endpoints*self.rng.normal(0.0,1.0))
            if (not self.T0_bounds.contains(T0_new)) or (not self.T_global_bounds.contains(T0_new)):
                return None, 0.0, 0.0
            temps_new[0] = T0_new
        else:
            Tf_cur = float(temps_new[-1])
            Tf_new = float(Tf_cur + self.sigma_endpoints*self.rng.normal(0.0,1.0))
            if (not self.Tf_bounds.contains(Tf_new)) or (not self.T_global_bounds.contains(Tf_new)):
                return None, 0.0, 0.0
            temps_new[-1] = Tf_new

        prop = TemperatureProfile(s.times.copy(), temps_new)

        if not self._is_valid(prop):
            return None, 0.0, 0.0

        return prop, 0.0, 0.0























from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass, field
from joblib import Parallel, delayed
# from multiprocessing import Pool
from omegaconf import DictConfig
from src.classes.constants import cnst  

from src.errors import ErrorOutputHandler
from src.classes.physics.crystal import Box
from src.process_plot import clean_up_results, save_data
from src.process_plot import plot_forward_results


@dataclass
class MCBase:
    repetion: int 
    seed: int 
    trap_pcnt: float 
    hole_pcnt: float
    crystal: Box
    max_dt: float = field(default=1)
    max_dt_time_chk: float = field(init=False)
    max_dt_cnt: int = 0
    max_length: int = field(default=10000000)
    data_path: str = field(default="sim_prelim_results.dat")
    result_csv_path: str = field(default="MC_results.csv")
    results: np.memmap = field(init=False) 


    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MCBase":
        crs = Box.from_config(cfg)
        # if cfg.mc.max_dt is None:
        #     mdt = 1000
        # else: 
        #     mdt = cfg.mc.max_dt 

        return cls(cfg.mc.reps, cfg.setup.seed, cfg.mc.t_pcnt, 
                   cfg.mc.h_pcnt, crs)
    
   
    @classmethod
    def RJMCMC_setup(cls, cfg: DictConfig, reps: int, seed: int) -> "MCBase":
        crs = Box.from_config(cfg)
      
        return cls(reps, seed, cfg.mc.t_pcnt, cfg.mc.h_pcnt, crs, data_path = "RJMCMC_prelim_results.dat")

    
    def max_dt_setter(self):
        self.max_dt_cnt=0
        if (self.crystal.kind not in {"constant", "linear"} and 
            self.crystal.times.size == 0):
            if self.crystal.dT[0] == 0: 
                self.crystal.kind = "constant"
            else:
                self.crystal.kind = "linear"


        if self.crystal.kind == "constant" :
            self.max_dt =  self.crystal.duration*10
            self.max_dt_time_chk = self.crystal.duration*10
        elif self.crystal.kind in {"step","steps"}:
            self.max_dt = self.crystal.times[0]/100
            self.max_dt_time_chk = self.crystal.times[0]
        elif self.crystal.kind == "linear" :
            self.max_dt = 1/self.crystal.dT
            self.max_dt_time_chk = 1e50
        elif self.crystal.kind == "linearsteps":
            if self.crystal.dT[0] == 0: 
                self.max_dt = self.crystal.times[0]/100
            else:
                self.max_dt = 1/self.crystal.dT[0]
            self.max_dt_time_chk = self.crystal.times[0]
       
    def max_dt_finder(self):
        if self.max_dt_time_chk > self.crystal.duration:
            return
        self.max_dt_cnt+=1
        if self.crystal.kind == "constant" :
            self.max_dt =  self.crystal.duration/100
            self.max_dt_time_chk = self.crystal.duration*10
        elif self.crystal.kind == "step":
            self.max_dt =  self.crystal.duration/100
            self.max_dt_time_chk = self.crystal.duration*10
        elif self.crystal.kind == "steps":
            if self.max_dt_cnt == self.crystal.times.size:
                self.max_dt =  self.crystal.duration/100
                self.max_dt_time_chk = self.crystal.duration*10
            else:
                self.max_dt = (self.crystal.times[self.max_dt_cnt]-self.crystal.times[self.max_dt_cnt-1])/100
                self.max_dt_time_chk = self.crystal.times[self.max_dt_cnt]
        elif self.crystal.kind == "linear" :
            self.max_dt = 1/self.crystal.dT
            self.max_dt_time_chk = 1e50
        elif self.crystal.kind == "linearsteps":
            if self.max_dt_cnt >= self.crystal.times.size:
                self.max_dt_time_chk = self.crystal.duration*10
                max_time = self.crystal.duration
            else: 
                self.max_dt_time_chk = self.crystal.times[self.max_dt_cnt]
                max_time = self.crystal.times[self.max_dt_cnt]
                

            if self.crystal.dT[self.max_dt_cnt] == 0: 
                self.max_dt = (max_time-self.crystal.times[self.max_dt_cnt-1])/100
            else:
                self.max_dt = 1/self.crystal.dT[self.max_dt_cnt]


           
    def single_experiment_run(self, rep: int, t: float = 0.0, 
                              t_pcnt: float | None = None, h_pcnt:float | None = None) -> None:
        
        if t_pcnt is None:
            t_pcnt = self.trap_pcnt 
        if h_pcnt is None:
            h_pcnt = self.hole_pcnt

        self.crystal.lattice_setup((self.seed+rep), t_pcnt, h_pcnt, t=t)
        i=0
        self.results[rep,0,i] = self.crystal.time
        self.results[rep,1,i] = self.crystal.t_cnt
        self.results[rep,2,i] = 0
        i+=1 
        self.max_dt_setter()
        
        while self.crystal.time < self.crystal.duration:
            if self.crystal.time >= self.max_dt_time_chk:
                self.max_dt_finder()
            dt = min(self.crystal.fill,self.crystal.fade,self.max_dt)
        
        
            self.crystal.event_bool = True
            if dt == self.crystal.fill:
                self.crystal.trap_new_electron()
                event = 0
            elif dt == self.crystal.fade:                
                self.crystal.remove_electron()
                event = 1
            else:
                self.crystal.event_bool = False
                event = 0
            self.crystal.timestep(dt)

            if self.crystal.time >= self.crystal.duration:
                self.results[rep,0,i] = self.crystal.duration
                self.results[rep,1,i] = self.results[rep,1,i-1]
                self.results[rep,2,i] = 0
                i+=1
                break 

            self.results[rep,0,i] = self.crystal.time
            self.results[rep,1,i] = self.crystal.t_cnt
            self.results[rep,2,i] = event
            i+=1

        self.results[rep,1,:] /= self.crystal.N
        self.results.flush()

    def monte_carlo_loop(self, t: float = 0.0, t_pcnt: float | None = None, h_pcnt:float | None = None) -> None:
        for i in range(self.repetion):
            self.single_experiment_run(i, t, t_pcnt, h_pcnt)
            # print(f"Rep {i+1} of {self.repetion} completed")

    def full_monte_carlo_simulation(self, err: ErrorOutputHandler):
        """Runs the monte carlo simulation in the forward direction"""
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

        self.results = np.memmap(self.data_path, dtype=np.float32, mode='w+', shape=(self.repetion, 3, self.max_length))
        self.results[:,:,:] = np.nan
        self.results.flush()
        err.output(f"Starting Monte Carlo simulation with {self.repetion} repetitions")

        self.monte_carlo_loop()
        
        err.output("Monte Carlo simulation complete, cleaning up results...")
        err.output("To be placed in file: sim_results.dat")

        data_path = "sim_results.dat"
        out = clean_up_results(self.results, data_path)
        out[1,:] = self.crystal.Tat(out[0,:])
        p = np.exp(-self.crystal.E_loc/(cnst.k_b_ev*out[1,:]))
        out[3, :] = 1/(p+1)
        out[4, :] = p/(p+1)
        if self.crystal.celsius:
            out[1,:] -= 273.15
        out.flush()
        os.remove(data_path)
        del self.results

        save_data(out, file_name=self.result_csv_path)
        err.output("Results cleaned and saved.")
        plot_forward_results(out.T, self.crystal.unit, self.crystal.kind)

    def RJMCMC_initialise(self): 
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

        self.results = np.memmap(self.data_path, dtype=np.float32, mode='w+', shape=(self.repetion, 3, self.max_length))
        self.results[:,:,:] = np.nan
        self.results.flush()

    def RJMCMC_cleanup(self): 
        if os.path.exists(self.data_path):
            os.remove(self.data_path)


    def RJMCMC_simulation(self, t: float = 0.0, t_pcnt: float | None = None, h_pcnt:float | None = None):
        self.results[:,:,:] = np.nan
        self.results.flush()
        self.monte_carlo_loop(t, t_pcnt, h_pcnt)
        self.results.flush()

       
        valid_mask = ~np.isnan(self.results[:, 0, :])
        lengths = valid_mask.sum(axis=1)
        times_list = [self.results[i, 0, :lengths[i]] for i in range(self.repetion)]
        time_union = np.unique(np.concatenate(times_list))
        del times_list

        ratio = np.zeros((time_union.size))
       
        for i in range(self.repetion):
            ratio += np.interp(time_union, self.results[i,0, :lengths[i]], self.results[i,1,:lengths[i]])
            
            
        ratio /= self.repetion
       

        return time_union, ratio
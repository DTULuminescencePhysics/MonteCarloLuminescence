from __future__ import annotations
import os
import copy
import numpy as np
from dataclasses import dataclass, field
from joblib import Parallel, delayed
from omegaconf import DictConfig
from src.classes.constants import cnst

from src.errors import ErrorOutputHandler
from src.classes.physics.crystal import Box
from src.classes.physics.transition_process import is_luminescence


def _max_dt_finder(
    max_dt_cnt: int, crystal: Box, max_dt: float, max_dt_time_chk: float
) -> tuple[int, float, float]:
    """Pure-function equivalent of MCBase.max_dt_finder for worker processes.

    Returns (updated_cnt, updated_max_dt, updated_max_dt_time_chk).
    """

    if max_dt_time_chk > crystal.duration:
        return max_dt_cnt, max_dt, max_dt_time_chk
       
    max_dt_cnt+=1
    max_dt_time_chk = crystal.times[max_dt_cnt+1]

    if crystal.dT[max_dt_cnt] == 0: 
            max_dt =  (max_dt_time_chk - crystal.time)/100
    else:
        max_dt = 1/abs(crystal.dT[max_dt_cnt])
    
    if max_dt_cnt == crystal.times.size:
        max_dt_time_chk = crystal.duration*10

    return max_dt_cnt, max_dt, max_dt_time_chk
  

def _run_single_rep(rep: int, crystal: Box, seed: int, t_pcnt: float,
    h_pcnt: float, t: float, initial_max_dt: float, initial_max_dt_time_chk: float,
    data_path: str, max_length: int, total_reps: int) -> None:
    """Run one Monte Carlo repetition in a worker process.

    Writes results directly to the shared memmap file and flushes.
    Each rep writes only to row *rep*, so concurrent writes are safe.
    """
    results = np.memmap(data_path, dtype=np.float32, mode='r+',
                        shape=(total_reps, 4, max_length))

    crystal.lattice_setup(seed + rep, t_pcnt, h_pcnt, t=t)

    max_dt = initial_max_dt
    max_dt_time_chk = initial_max_dt_time_chk
    max_dt_cnt = 0

    i = 0
    results[rep, 0, i] = crystal.time
    results[rep, 1, i] = crystal.t_cnt
    results[rep, 2, i] = 0
    results[rep, 3, i] = 0
    i += 1

    while crystal.time < crystal.duration:
        if crystal.time >= max_dt_time_chk:
            max_dt_cnt, max_dt, max_dt_time_chk = _max_dt_finder(
                max_dt_cnt, crystal, max_dt, max_dt_time_chk
            )

        # Clear stale event_code so any early-return path is recorded as no_event.
        crystal.event_code = 0

        dt = min(crystal.fill_time, crystal.exec_time, max_dt)

        if dt == crystal.fill_time:
            crystal.trap_new_electron()
            event_code = crystal.event_code
        elif dt == crystal.exec_time:
            crystal.operate_electron()
            event_code = crystal.event_code
        else:
            event_code = 0
        crystal.timestep(dt)

        if crystal.time >= crystal.duration:
            results[rep, 0, i] = crystal.duration
            results[rep, 1, i] = results[rep, 1, i - 1]
            results[rep, 2, i] = 0
            results[rep, 3, i] = 0
            i += 1
            break

        results[rep, 0, i] = crystal.time
        results[rep, 1, i] = crystal.t_cnt
        results[rep, 2, i] = 1 if is_luminescence(event_code) else 0
        results[rep, 3, i] = event_code
        i += 1

    results[rep, 1, :] /= crystal.N
    results.flush()
    print(f"Rep {rep + 1} of {total_reps} completed")



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
    max_length: int = field(default=500000)
    data_path: str = field(default="sim_prelim_results.dat")
    result_csv_path: str = field(default="MC_results")
    n_jobs: int = field(default=1)
    results: np.memmap = field(init=False)


    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MCBase":
        crs = Box.from_config(cfg)
        n_jobs = getattr(cfg.setup, 'n_jobs', 1)
        return cls(cfg.setup.reps, cfg.setup.seed, cfg.setup.t_pcnt,
                   cfg.setup.h_pcnt, crs, n_jobs=n_jobs)


    @classmethod
    def thermochron_setup(cls, cfg: DictConfig, reps: int, seed: int) -> "MCBase":
        crs = Box.from_config(cfg)

        return cls(reps, seed, cfg.setup.t_pcnt, cfg.setup.h_pcnt, crs, data_path = "RJMCMC_prelim_results.dat")

    def max_dt_setter(self):
        self.max_dt_cnt=0
    
        if self.crystal.kind == "Constant" :
            self.max_dt =  self.crystal.duration*10
            self.max_dt_time_chk = self.crystal.duration*10
        elif self.crystal.kind == "Linear" :
            self.max_dt = (1/abs(self.crystal.dT))
            self.max_dt_time_chk = self.crystal.duration*10
        else:
            if self.crystal.dT[0] == 0: 
                self.max_dt = (self.crystal.times[1]-self.crystal.times[0])/100
            else:
                self.max_dt = 1/abs(self.crystal.dT[0])

            self.max_dt_time_chk = self.crystal.times[1]
    

    def max_dt_finder(self): 
        if self.max_dt_time_chk > self.crystal.duration:
            return
       
        self.max_dt_cnt+=1
        self.max_dt_time_chk = self.crystal.times[self.max_dt_cnt+1]

        if self.crystal.dT[self.max_dt_cnt] == 0: 
                self.max_dt =  (self.max_dt_time_chk - self.crystal.time)/100
        else:
            self.max_dt = 1/abs(self.crystal.dT[self.max_dt_cnt])
        
        if self.max_dt_cnt == self.crystal.times.size:
            self.max_dt_time_chk = self.crystal.duration*10
    

    # def single_experiment_run(self, rep: int, t: float = 0.0,
    #                           t_pcnt: float | None = None, h_pcnt:float | None = None) -> None:
    #     if t_pcnt is None:
    #         t_pcnt = self.trap_pcnt
    #     if h_pcnt is None:
    #         h_pcnt = self.hole_pcnt
    #     self.crystal.lattice_setup((self.seed+rep), t_pcnt, h_pcnt, t=t)
    #     i=0
    #     self.results[rep,0,i] = self.crystal.time
    #     self.results[rep,1,i] = self.crystal.t_cnt
    #     self.results[rep,2,i] = 0
    #     i+=1
    #     self.max_dt_setter()
    #     while self.crystal.time < self.crystal.duration:
    #         if self.crystal.time >= self.max_dt_time_chk:
    #             self.max_dt_finder()
    #         dt = min(self.crystal.fill,self.crystal.fade,self.max_dt)
    #         self.crystal.event_bool = True
    #         if dt == self.crystal.fill:
    #             self.crystal.trap_new_electron()
    #             event = 0
    #         elif dt == self.crystal.fade:
    #             self.crystal.remove_electron()
    #             event = 1
    #         else:
    #             self.crystal.event_bool = False
    #             event = 0
    #         self.crystal.timestep(dt)
    #         if self.crystal.time >= self.crystal.duration:
    #             self.results[rep,0,i] = self.crystal.duration
    #             self.results[rep,1,i] = self.results[rep,1,i-1]
    #             self.results[rep,2,i] = 0
    #             i+=1
    #             break
    #         self.results[rep,0,i] = self.crystal.time
    #         self.results[rep,1,i] = self.crystal.t_cnt
    #         self.results[rep,2,i] = event
    #         i+=1
    #     self.results[rep,1,:] /= self.crystal.N
    #     self.results.flush()

    def single_experiment_run_modified(self, rep: int, t: float = 0.0,
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
        self.results[rep,3,i] = 0
        i+=1
        self.max_dt_setter()

        while self.crystal.time < self.crystal.duration:
            if self.crystal.time >= self.max_dt_time_chk:
                self.max_dt_finder()

            # Clear stale event_code so any early-return path is recorded as no_event.
            self.crystal.event_code = 0

            dt = min(self.crystal.fill_time,self.crystal.exec_time,self.max_dt)

            # self.crystal.event_bool = True
            if dt == self.crystal.fill_time:
                self.crystal.trap_new_electron()
                event_code = self.crystal.event_code
            elif dt == self.crystal.exec_time:
                self.crystal.operate_electron()
                event_code = self.crystal.event_code
            else:
                # self.crystal.event_bool = False
                event_code = 0
            self.crystal.timestep(dt)

            if self.crystal.time >= self.crystal.duration:
                self.results[rep,0,i] = self.crystal.duration
                self.results[rep,1,i] = self.results[rep,1,i-1]
                self.results[rep,2,i] = 0
                self.results[rep,3,i] = 0
                i+=1
                break

            self.results[rep,0,i] = self.crystal.time
            self.results[rep,1,i] = self.crystal.t_cnt
            self.results[rep,2,i] = 1 if is_luminescence(event_code) else 0
            self.results[rep,3,i] = event_code
            i+=1

        self.results[rep,1,:] /= self.crystal.N
        self.results.flush()

    def monte_carlo_loop(self, t: float = 0.0, t_pcnt: float | None = None, h_pcnt: float | None = None) -> None:
        if t_pcnt is None:
            t_pcnt = self.trap_pcnt
        if h_pcnt is None:
            h_pcnt = self.hole_pcnt

        if self.n_jobs == 1:
            for i in range(self.repetion):
                self.single_experiment_run_modified(i, t, t_pcnt, h_pcnt)
                print(f"Rep {i+1} of {self.repetion} completed")
            return

        self.max_dt_setter()

        n_workers = min(self.n_jobs, self.repetion, os.cpu_count() or 1)
        crystal_copies = [copy.deepcopy(self.crystal) for _ in range(self.repetion)]

        Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_run_single_rep)(i, crystal_copies[i], self.seed, t_pcnt,
                h_pcnt, t, self.max_dt, self.max_dt_time_chk, self.data_path,
                self.max_length, self.repetion)
            for i in range(self.repetion)
        )

    def full_monte_carlo_simulation(self, err: ErrorOutputHandler):
        """Runs the monte carlo simulation in the forward direction"""
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

        self.results = np.memmap(self.data_path, dtype=np.float32, mode='w+', shape=(self.repetion, 4, self.max_length))
        self.results[:,:,:] = np.nan
        self.results.flush()
        err.output(f"Starting Monte Carlo simulation with {self.repetion} repetitions")

        self.monte_carlo_loop()

        # Re-open as read-only view for post-processing
        self.results = np.memmap(self.data_path, dtype=np.float32, mode='r',
                                 shape=(self.repetion, 4, self.max_length))

        err.output("Monte Carlo simulation complete, cleaning up results...")
        err.output("To be placed in file: sim_results.dat")


    def clean_up(self):
        try:
            del self.results
        except:
            pass

        if os.path.exists(self.data_path):
            os.remove(self.data_path)


    def thermochron_initialise(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

        self.results = np.memmap(self.data_path, dtype=np.float32, mode='w+', shape=(self.repetion, 3, self.max_length))
        self.results[:,:,:] = np.nan
        self.results.flush()

    def thermochron_cleanup(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)


    def thermochron_simulation(self, t: float = 0.0, t_pcnt: float | None = None, h_pcnt:float | None = None):
        def last_non_nan(arr,div):
            valid = np.where(~np.isnan(arr))[0]
            if valid.size > 0:
                return arr[valid[-1]]
            else:
                div -= 1
                return 0

        self.results[:,:,:] = np.nan
        self.results.flush()
        self.monte_carlo_loop(t, t_pcnt, h_pcnt)
        self.results.flush()

        div = self.repetion
        ratio = 0
        for i in range(self.repetion):
            ratio += last_non_nan(self.results[i, 1],div)

        ratio /= div

        return ratio



    def inverse_modeling_simulation(self,):

        def last_non_nan(arr,div):
            valid = np.where(~np.isnan(arr))[0]
            if valid.size > 0:
                return arr[valid[-1]]
            else:
                div -= 1
                return 0

        self.results[:,:,:] = np.nan
        self.results.flush()
        self.monte_carlo_loop()
        self.results.flush()

        div = self.repetion
        ratio = 0
        for i in range(self.repetion):
            ratio += last_non_nan(self.results[i, 1],div)

        ratio /= div

        return ratio

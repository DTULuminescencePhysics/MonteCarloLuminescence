from __future__ import annotations

from dataclasses import dataclass
from omegaconf import DictConfig
from typing import Iterable, Sequence, Dict,TYPE_CHECKING
import numpy as np
import h5py

if TYPE_CHECKING:
    from src.classes.output.temp_results_file import chronology_results

"""
This module will store final data sets for all simulations
The file is split into three sections 
    - Inputs: store of all input parameters needed to repeat an experiments
    - Outputs: Output of forward modelling data
    - Chronology: Thermochronology inputs and predicted history  
  
    Store ragged time/temperature records in HDF5.

Each record contains:
    - flag   : bool
    - times  : 1D array of length Li
    - temps  : 1D array of length Li
    - value  : float

Data is stored as:
    /flags       shape (N,)         bool
    /values      shape (N,)         float64
    /offsets     shape (N+1,)       int64
    /times_flat  shape (sum(Li),)   float64
    /temps_flat  shape (sum(Li),)   float64

Record i is reconstructed with:
    start = offsets[i]
§    stop = offsets[i + 1]
    times = times_flat[start:stop]
    temps = temps_flat[start:stop]
"""


class output_file:

    def __init__(self,filename:str, cfg: DictConfig, compression: str = "gzip",
        compression_level: int = 4,):

        self.name = filename 
        self.ds_kwargs = {}
        if compression is not None:
            self.ds_kwargs["compression"] = compression
            if compression == "gzip":
                self.ds_kwargs["compression_opts"] = compression_level



    def initial_file_build(self,cfg: DictConfig): 

        with h5py.File(self.name,"a") as f:
            inputs = f.require_group("inputs")

            inputs.attrs['Trap_%'] = cfg.setup.t_pcnt
            inputs.attrs['Hole_%'] = cfg.setup.h_pcnt
            inputs.attrs["repetitions"] = cfg.setup.reps


            phys = inputs.require_group("physics")

            phys.attrs["E_loc"]=cfg.phys.E_loc
            phys.attrs["b"]=cfg.phys.b
            phys.attrs["alpha"]=cfg.phys.alpha
            phys.attrs["E_cb"]=cfg.phys.E_cb
            phys.attrs["s"]=cfg.phys.s
            phys.attrs["rho"]=cfg.phys.rho
            phys.attrs["urho"]=cfg.phys.urho
            phys.attrs["D0"]=cfg.phys.D0
            phys.attrs["D_dot"]=cfg.phys.D_dot
            phys.attrs["Dd_unit"]=cfg.phys.Dd_unit

            temp = inputs.require_group("temperature")
            temp.attrs['unit'] = cfg.temp.unit
            temp.attrs['celsius'] = cfg.temp.celsius
            temp.attrs['kind'] = cfg.temp.kind
            temp.attrs['T0'] = cfg.temp.T0
            temp.attrs['duration'] = cfg.temp.duration
            temp.attrs['times'] = cfg.temp.times
            temp.attrs['temps'] = cfg.temp.temps
            temp.attrs['dT'] = cfg.temp.dT 

    def output_data_build(self): 

        with h5py.File(self.name,"a") as f:
            outputs = f.require_group("outputs") 

    def chronology_data_initial_build(self,cfg: DictConfig): 

        with h5py.File(self.name,"a") as f:
            chron = f.require_group("chronology")

            chron.attrs["method"] = cfg.chron.method
            chron.attrs["iterations"] = cfg.chron.iters
            chron.attrs["T_Target"] = cfg.chron.T_Target
            chron.attrs["T0_low"] = cfg.chron.T0_lo
            chron.attrs["T0_high"] = cfg.chron.T0_hi
            chron.attrs["T_tolerance"] = cfg.chron.T_tolerance
            chron.attrs["monotonic"] = cfg.chron.monotonic
            chron.attrs["min_internal"] = cfg.chron.min_internal
            chron.attrs["max_internal"] = cfg.chron.max_internal

            if chron.attrs["method"] == "RJMCMC": 
                rjmcmc = chron.require_group("RJMCMC")

                log_like = rjmcmc.require_group("log_likelihood")
                log_like.attrs["mode"] = cfg.chron.rjmcmc.log_likelihood.mode
                log_like.attrs["sigma"] = cfg.chron.rjmcmc.log_likelihood.sigma
                log_like.attrs["cov"] = cfg.chron.rjmcmc.log_likelihood.cov
                log_like.attrs["nud"] = cfg.chron.rjmcmc.log_likelihood.nud

                log_prior_prob = rjmcmc.require_group("log_prior_prob")
                log_prior_prob.attrs["lam_k"] = cfg.chron.rjmcmc.log_prior_prob.lam_k
                log_prior_prob.attrs["sigma_curv"] = cfg.chron.rjmcmc.log_prior_prob.sigma_curv
                log_prior_prob.attrs["nu_curv"] = cfg.chron.rjmcmc.log_prior_prob.nu_curv
                log_prior_prob.attrs["dt_min"] = cfg.chron.rjmcmc.log_prior_prob.dt_min
                log_prior_prob.attrs["p_creep"] = cfg.chron.rjmcmc.log_prior_prob.p_creep
                log_prior_prob.attrs["sigma_creep"] = cfg.chron.rjmcmc.log_prior_prob.sigma_creep
                log_prior_prob.attrs["nu_step"] = cfg.chron.rjmcmc.log_prior_prob.nu_step
                log_prior_prob.attrs["sigma_step"] = cfg.chron.rjmcmc.log_prior_prob.sigma_step
                log_prior_prob.attrs["dT_min_change"] = cfg.chron.rjmcmc.log_prior_prob.dT_min_change
                log_prior_prob.attrs["hard_reject_short_dt"] = cfg.chron.rjmcmc.log_prior_prob.hard_reject_short_dt
                log_prior_prob.attrs["gap_barrier_alpha"] = cfg.chron.rjmcmc.log_prior_prob.gap_barrier_alpha
                log_prior_prob.attrs["gap_barrier_power"] = cfg.chron.rjmcmc.log_prior_prob.gap_barrier_power
                log_prior_prob.attrs["eps"] = cfg.chron.rjmcmc.log_prior_prob.eps

                parameters = rjmcmc.require_group("parameters")
                parameters.attrs["p_birth"] = cfg.chron.rjmcmc.parameters.p_birth
                parameters.attrs["p_death"] = cfg.chron.rjmcmc.parameters.p_birth
                parameters.attrs["p_move_time"] = cfg.chron.rjmcmc.parameters.p_birth
                parameters.attrs["p_move_temp"] = cfg.chron.rjmcmc.parameters.p_birth
                parameters.attrs["p_move_endpoints"] = cfg.chron.rjmcmc.parameters.p_birth
                parameters.attrs["sigma_birth"] = cfg.chron.rjmcmc.parameters.p_birth
                parameters.attrs["sigma_temp"] = cfg.chron.rjmcmc.parameters.p_birth
                parameters.attrs["sigma_time_frac"] = cfg.chron.rjmcmc.parameters.p_birth
                parameters.attrs["sigma_endpoints"] = cfg.chron.rjmcmc.parameters.p_birth

                if cfg.chron.rjmcmc.burn_in.burn: 
                    burn_in = rjmcmc.require_group("burn_in")
                    burn_in.attrs["max_steps"] = cfg.chron.rjmcmc.burn_in.max_steps
                    burn_in.attrs["window"] = cfg.chron.rjmcmc.burn_in.window
                    burn_in.attrs["overall_accept_target"] = cfg.chron.rjmcmc.burn_in.overall_accept_target
                    burn_in.attrs["per_move_accept_target"] = cfg.chron.rjmcmc.burn_in.per_move_accept_target
                    burn_in.attrs["birth_death_min"] = cfg.chron.rjmcmc.burn_in.birth_death_min
                    burn_in.attrs["min_move_prob"] = cfg.chron.rjmcmc.burn_in.min_move_prob
                    burn_in.attrs["max_adjust_factor"] = cfg.chron.rjmcmc.burn_in.max_adjust_factor
                    burn_in.attrs["target_k_internal"] = cfg.chron.rjmcmc.burn_in.target_k_internal
                    burn_in.attrs["k_window"] = cfg.chron.rjmcmc.burn_in.k_window
                    burn_in.attrs["patience_windows"] = cfg.chron.rjmcmc.burn_in.patience_windows
                    burn_in.attrs["verbose"] = cfg.chron.rjmcmc.burn_in.verbose

    def burn_in_update(self,sigmas:Dict[str, float]):

        with h5py.File(self.name,"a") as f:     
            chron = f.require_group("chronology")
            rjmcmc = chron.require_group("RJMCMC")
            parameters = rjmcmc.require_group("parameters")
            for name, value in sigmas.items():
                parameters.attrs[name] = value

    def chronological_results(self,chronology_results:chronology_results):
        
        with h5py.File(self.name,"a") as f:     
            chron = f.require_group("chronology")
            results = chron.require_group("results")
            all_times = chronology_results.all_times
            results.create_dataset("times", data=all_times, **self.ds_kwargs)
            all_temps = chronology_results.all_temps
            results.create_dataset("temperatures", data=all_temps, **self.ds_kwargs)
            offsets = chronology_results.offset
            results.create_dataset("offsets", data=offsets, **self.ds_kwargs)
            values = chronology_results.values
            results.create_dataset("p_logs", data=values, **self.ds_kwargs)

            






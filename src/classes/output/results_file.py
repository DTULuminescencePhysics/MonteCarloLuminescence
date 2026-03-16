from __future__ import annotations

from dataclasses import dataclass
from omegaconf import DictConfig
from typing import Dict,TYPE_CHECKING
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

        self.initial_file_build(cfg)

    def write_attrs_if_not_none(self, group, attrs: dict) -> None:
        """
        Write key/value pairs to HDF5 attributes, skipping None values.
        """
        for key, value in attrs.items():
            if value is None:
                continue

            # Optional: convert some types explicitly
            if isinstance(value, (list, tuple)):
                value = np.asarray(value)

            group.attrs[key] = value


    def initial_file_build(self,cfg: DictConfig): 
        input_attrs = {
            "Trap_%": cfg.setup.t_pcnt,
            "Hole_%": cfg.setup.h_pcnt,
            "repetitions": cfg.setup.reps,
        }
        phys_attrs = {
            "E_loc": cfg.physics.E_loc,
            "b": cfg.physics.b,
            "alpha": cfg.physics.alpha,
            "E_cb": cfg.physics.E_cb,
            "s": cfg.physics.s,
            "rho": cfg.physics.rho,
            "urho": cfg.physics.urho,
            "D0": cfg.physics.D0,
            "D_dot": cfg.physics.D_dot,
            "Dd_unit": cfg.physics.Dd_unit,
        }
        temp_attrs = {
            "unit": cfg.temp.unit,
            "celsius": cfg.temp.celsius,
            "kind": cfg.temp.kind,
            "T0": cfg.temp.T0,
            "duration": cfg.temp.duration,
            "times": cfg.temp.times,
            "temps": cfg.temp.temps,
            "dT": cfg.temp.dT,
        }
        with h5py.File(self.name,"a") as f:
            inputs = f.require_group("inputs")
            self.write_attrs_if_not_none(inputs, input_attrs)

            phys = inputs.require_group("physics")
            self.write_attrs_if_not_none(phys, phys_attrs)

            temp = inputs.require_group("temperature")
            self.write_attrs_if_not_none(temp, temp_attrs)

    def output_data_build(self): 

        with h5py.File(self.name,"a") as f:
            outputs = f.require_group("outputs") 

    def chronology_data_initial_build(self,cfg: DictConfig): 
        chron_attrs = {
            "method": cfg.chronology.method,
            "iterations": cfg.chronology.iters,
            "T_Target": cfg.chronology.T_Target,
            "T0_low": cfg.chronology.T0_lo,
            "T0_high": cfg.chronology.T0_hi,
            "T_tolerance": cfg.chronology.T_tolerance,
            "monotonic": cfg.chronology.monotonic,
            "min_internal": cfg.chronology.min_internal,
            "max_internal": cfg.chronology.max_internal,
        }
        log_like_attrs = {
            "mode": cfg.chronology.rjmcmc.log_likelihood.mode,
            "sigma": cfg.chronology.rjmcmc.log_likelihood.sigma,
            "cov": cfg.chronology.rjmcmc.log_likelihood.cov,
            "nud": cfg.chronology.rjmcmc.log_likelihood.nud,
        }
        log_prior_prob_attrs = {
            "lam_k": cfg.chronology.rjmcmc.log_prior_prob.lam_k,
            "sigma_curv": cfg.chronology.rjmcmc.log_prior_prob.sigma_curv,
            "nu_curv": cfg.chronology.rjmcmc.log_prior_prob.nu_curv,
            "dt_min": cfg.chronology.rjmcmc.log_prior_prob.dt_min,
            "p_creep": cfg.chronology.rjmcmc.log_prior_prob.p_creep,
            "sigma_creep": cfg.chronology.rjmcmc.log_prior_prob.sigma_creep,
            "nu_step": cfg.chronology.rjmcmc.log_prior_prob.nu_step,
            "sigma_step": cfg.chronology.rjmcmc.log_prior_prob.sigma_step,
            "dT_min_change": cfg.chronology.rjmcmc.log_prior_prob.dT_min_change,
            "hard_reject_short_dt": cfg.chronology.rjmcmc.log_prior_prob.hard_reject_short_dt,
            "gap_barrier_alpha": cfg.chronology.rjmcmc.log_prior_prob.gap_barrier_alpha,
            "gap_barrier_power": cfg.chronology.rjmcmc.log_prior_prob.gap_barrier_power,
            "eps": cfg.chronology.rjmcmc.log_prior_prob.eps,
        }

        parameters_attrs = {
            "p_birth": cfg.chronology.rjmcmc.parameters.p_birth,
            "p_death": cfg.chronology.rjmcmc.parameters.p_death,
            "p_move_time": cfg.chronology.rjmcmc.parameters.p_move_time,
            "p_move_temp": cfg.chronology.rjmcmc.parameters.p_move_temp,
            "p_move_endpoints": cfg.chronology.rjmcmc.parameters.p_move_endpoints,
            "sigma_birth": cfg.chronology.rjmcmc.parameters.sigma_birth,
            "sigma_temp": cfg.chronology.rjmcmc.parameters.sigma_temp,
            "sigma_time_frac": cfg.chronology.rjmcmc.parameters.sigma_time_frac,
            "sigma_endpoints": cfg.chronology.rjmcmc.parameters.sigma_endpoints,
        }
        burn_in_attrs = {
            "max_steps": cfg.chronology.rjmcmc.burn_in.max_steps,
            "window": cfg.chronology.rjmcmc.burn_in.window,
            "overall_accept_target": cfg.chronology.rjmcmc.burn_in.overall_accept_target,
            "per_move_accept_target": cfg.chronology.rjmcmc.burn_in.per_move_accept_target,
            "birth_death_min": cfg.chronology.rjmcmc.burn_in.birth_death_min,
            "min_move_prob": cfg.chronology.rjmcmc.burn_in.min_move_prob,
            "max_adjust_factor": cfg.chronology.rjmcmc.burn_in.max_adjust_factor,
            "target_k_internal": cfg.chronology.rjmcmc.burn_in.target_k_internal,
            "k_window": cfg.chronology.rjmcmc.burn_in.k_window,
            "patience_windows": cfg.chronology.rjmcmc.burn_in.patience_windows,
            "verbose": cfg.chronology.rjmcmc.burn_in.verbose,
        }
        with h5py.File(self.name,"a") as f:
            chron = f.require_group("chronology")
            self.write_attrs_if_not_none(chron, chron_attrs)
           
            if chron.attrs["method"] == "RJMCMC": 
                rjmcmc = chron.require_group("RJMCMC")


                log_like = rjmcmc.require_group("log_likelihood")
                self.write_attrs_if_not_none(log_like, log_like_attrs)

                log_prior_prob = rjmcmc.require_group("log_prior_prob")
                self.write_attrs_if_not_none(log_prior_prob, log_prior_prob_attrs)

                parameters = rjmcmc.require_group("parameters")
                self.write_attrs_if_not_none(parameters, parameters_attrs)

                if cfg.chronology.rjmcmc.burn_in.burn:
                    burn_in = rjmcmc.require_group("burn_in")
                    self.write_attrs_if_not_none(burn_in, burn_in_attrs)

    def burn_in_update(self,sigmas:Dict[str, float]):

        with h5py.File(self.name,"a") as f:     
            chron = f.require_group("chronology")
            rjmcmc = chron.require_group("RJMCMC")
            parameters = rjmcmc.require_group("parameters")
            for name, value in sigmas.items():
                parameters.attrs[name] = value

    def chronological_results(self,chronology_results:chronology_results):
        acc = chronology_results.accepted
        total_accept = acc.sum()
        acc_offsets = np.zeros((total_accept*2))
        offsets = chronology_results.offset
        j=0
        for i in range(acc.size): 
            if acc[i]:
                if i==0:
                    acc_offsets[1]=offsets[i]
                else:
                    acc_offsets[j]=offsets[i-1]
                    acc_offsets[j+1]=offsets[i]
                j+=2


        with h5py.File(self.name,"a") as f:     
            chron = f.require_group("chronology")
            results = chron.require_group("results")
            results.create_dataset("offsets", data=offsets, **self.ds_kwargs)
            del offsets
            results.create_dataset("accepted_offsets",data=acc_offsets,**self.ds_kwargs)
            del acc_offsets
            all_times = chronology_results.all_times
            results.create_dataset("times", data=all_times, **self.ds_kwargs)
            del all_times
            all_temps = chronology_results.all_temps
            results.create_dataset("temperatures", data=all_temps, **self.ds_kwargs)
            del all_temps
            values = chronology_results.values
            results.create_dataset("p_logs", data=values, **self.ds_kwargs)
            del values

            T_min, T_max = chronology_results.T_min_max()
            results.create_dataset("T_minimum", data=T_min)
            
            results.create_dataset("T_maximum", data=T_max)

    def chronological_result_get_accepted_offsets(self): 

        with h5py.File(self.name,"r") as f:
            
           
            acc_offsets = f["chronology/results/accepted_offsets"][:]
        
        return acc_offsets.astype(np.int32)
    
    def chronological_result_get_times(self): 

        with h5py.File(self.name,"r") as f:  
            times = f["chronology/results/times"][:]
        
        return times
    
    def chronological_result_get_temps(self): 

        with h5py.File(self.name,"r") as f:  
            temps = f["chronology/results/temperatures"][:]
        
        return temps
    
    def chronological_result_max_T_min_T(self,): 

        with h5py.File(self.name,"r") as f: 

            T_min = f["chronology/results/T_minimum"][()]
            T_max = f["chronology/results/T_maximum"][()]

        return (T_min), (T_max)





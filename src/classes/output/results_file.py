from __future__ import annotations
from typing import Dict,TYPE_CHECKING
import numpy as np
import h5py

if TYPE_CHECKING:
    from src.classes.output.temp_results_file import chronology_results
    from omegaconf import DictConfig
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

    def __init__(self,filename:str, cfg: DictConfig | None, compression: str = "gzip",
        compression_level: int = 4,):

        self.name = filename 
        self.ds_kwargs = {}
        if compression is not None:
            self.ds_kwargs["compression"] = compression
            if compression == "gzip":
                self.ds_kwargs["compression_opts"] = compression_level
        if cfg is not None:
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
            "Boundary" : cfg.setup.boundary,
        }
        phys_attrs = {
            "uc_h": cfg.physics.uc_h,
            "uc_w": cfg.physics.uc_w,
            "uc_l": cfg.physics.uc_l,
            "dimension": cfg.physics.dimension,
            "rho": cfg.physics.rho,
            "urho": cfg.physics.urho,

            "E_loc": cfg.physics.E_loc,
            "E_loc_sigma":cfg.physics.E_loc_sigma,
            "E_cb": cfg.physics.E_cb,
            "E_cb_sigma": cfg.physics.E_cb_sigma,

            "D0": cfg.physics.D0,
            "D_dot": cfg.physics.D_dot,
            "Dd_unit": cfg.physics.Dd_unit,
            "Combine_when_fill":cfg.physics.combine_when_fill,
            "recom_pre_fill":cfg.physics.recom_pre_fill,

            "b": cfg.physics.b,
            "alpha_ES": cfg.physics.alpha_ES,
            "alpha_GS": cfg.physics.alpha_GS,
            "retrapping_ratio_tun":cfg.physics.R_tun,    
            "VRH":cfg.physics.VRH,
           
            "s": cfg.physics.s,
            "mu":cfg.physics.mu,
            "retrapping_ratio_CB":cfg.physics.R_CB,

            "retrap_mask_factor":cfg.physics.retrap_mask_factor,

            "Band tail to trap ratio": cfg.physics.shallow_deep_ratio,
            "threshold_depth": cfg.physics.threshold_depth,
            "E_u": cfg.physics.E_u,
            "b_BT": cfg.physics.b_BT,
            "alpha_BT": cfg.physics.alpha_BT,
            "init_shallow": cfg.physics.init_shallow,
            "sh_pcnt": cfg.physics.sh_pcnt,
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

    def output_data_build(self, results: dict): 

        exp_cnt = self.output_result_get_number_exp(True)
        with h5py.File(self.name,"a") as f:
            res = f.require_group("outputs")
            outputs = res.require_group(f"experiment_{exp_cnt}")

            outputs.create_dataset("timeSteps",data=results["timeSteps"], **self.ds_kwargs)
            outputs.create_dataset("temperature",data=results["temperature"], **self.ds_kwargs)
            outputs.create_dataset("meanTrapRatio",data=results["meanTrapRatio"], **self.ds_kwargs)
            outputs.create_dataset("stdTrapRatio",data=results["stdTrapRatio"], **self.ds_kwargs)
            
            outputs.create_dataset("q_low",data=results["quantiles"][0.1], **self.ds_kwargs)
            outputs.create_dataset("q_mid",data=results["quantiles"][0.5], **self.ds_kwargs)
            outputs.create_dataset("q_high",data=results["quantiles"][0.9], **self.ds_kwargs)

            outputs.create_dataset("luminescence",data=results["luminescence"], **self.ds_kwargs)
            outputs.create_dataset("filling",data=results["filling"], **self.ds_kwargs)
            outputs.create_dataset("events",data=results["events"], **self.ds_kwargs)

    def output_result_get_number_exp(self,set=False):
        with h5py.File(self.name,"a") as f:
            res = f.require_group("outputs")
            if "experimentCount" in res.attrs:
                if set:
                    res.attrs["experimentCount"] =  + 1
                exp_cnt = res.attrs["experimentCount"]
            else: 
                res.attrs["experimentCount"] = 1
                exp_cnt = 1
        return exp_cnt


    def output_result_get_times(self, exp_cnt:int = 1): 

        with h5py.File(self.name,"r") as f:  
            times = f[f"outputs/experiment_{exp_cnt}/timeSteps"][:]
        
        return times
    
    def output_result_get_all_times(self,exp_cnt:int=1):
        exp = self.output_result_get_times(1)
        times = exp[:,None]
        if exp_cnt > 1:
            for i in range(2,(exp_cnt+1)):
                exp = self.output_result_get_times(i) 
                times = np.column_stack([times,exp]) 
        return times


    def output_result_get_temperature(self, exp_cnt:int = 1): 

        with h5py.File(self.name,"r") as f:  
            temp = f[f"outputs/experiment_{exp_cnt}/temperature"][:]
        
        return temp
    
    def output_result_get_meanTrap(self, exp_cnt:int = 1): 

        with h5py.File(self.name,"r") as f:  
            mean = f[f"outputs/experiment_{exp_cnt}/meanTrapRatio"][:]
        
        return mean
    
    def output_result_get_all_meanTrap(self,exp_cnt:int=1):
        exp = self.output_result_get_meanTrap(1)
        mean = exp[:,None]
        if exp_cnt > 1:
            for i in range(2,(exp_cnt+1)):
                exp = self.output_result_get_meanTrap(i) 
                mean = np.column_stack([mean,exp]) 
        return mean

    def output_result_get_stdTrap(self, exp_cnt:int = 1): 

        with h5py.File(self.name,"r") as f:  
            std = f[f"outputs/experiment_{exp_cnt}/stdTrapRatio"][:]
        
        return std
    
    def output_result_get_all_stdTrap(self,exp_cnt:int=1):
        exp = self.output_result_get_stdTrap(1)
        std = exp[:,None]
        if exp_cnt > 1:
            for i in range(2,(exp_cnt+1)):
                exp = self.output_result_get_stdTrap(i) 
                std = np.column_stack([std,exp]) 
        return std

    def output_result_get_quantiles(self, exp_cnt:int = 1): 

        with h5py.File(self.name,"r") as f:  
            lo = f[f"outputs/experiment_{exp_cnt}/q_low"][:]
            mi = f[f"outputs/experiment_{exp_cnt}/q_mid"][:]
            hi = f[f"outputs/experiment_{exp_cnt}/q_high"][:]

        return lo, mi , hi
    
    def output_result_get_all_quantiles(self,exp_cnt:int=1):
        explo, expmi, exphi = self.output_result_get_quantiles(1)
        lo = explo[:,None]
        mi = expmi[:,None]
        hi = exphi[:,None]

        if exp_cnt > 1:
            for i in range(2,(exp_cnt+1)):
                explo, expmi, exphi = self.output_result_get_quantiles(i) 
                lo = np.column_stack([lo,explo]) 
                mi = np.column_stack([mi,expmi]) 
                hi = np.column_stack([hi,exphi]) 

        return lo, mi , hi

    def output_result_get_all_ratioPlot(self,quant:bool=True,st:bool=True): 
        exp_cnt = self.output_result_get_number_exp()
        times = self.output_result_get_all_times(exp_cnt)
        mean = self.output_result_get_all_meanTrap(exp_cnt)
        if quant: 
            lo, mi, hi = self.output_result_get_all_quantiles(exp_cnt)
        else:
            lo=mi=hi=np.array(None) 
        if st: 
            std= self.output_result_get_all_stdTrap(exp_cnt)
        else:
            std = np.array(None) 

        return exp_cnt, times, mean, lo, mi, hi, std 
    
    def output_result_get_lum(self, exp_cnt:int = 1): 

        with h5py.File(self.name,"r") as f:  
            lum = f[f"outputs/experiment_{exp_cnt}/luminescence"][:]
        
        return lum
    
    def output_result_get_filling(self, exp_cnt:int = 1): 

        with h5py.File(self.name,"r") as f:  
            fill = f[f"outputs/experiment_{exp_cnt}/filling"][:]
        
        return fill
    
    def output_result_get_events(self, exp_cnt:int = 1): 

        with h5py.File(self.name,"r") as f:  
            events = f[f"outputs/experiment_{exp_cnt}/events"][:]
        
        return events    
    
    def output_result_get_nonzero_events(self,exp_cnt:int = 1): 

        events = self.output_result_get_events(exp_cnt)
        nonzero_col_mask = np.any(events != 0, axis=0)
        nonzero_col_mask[0] = False
        event_type = np.where(nonzero_col_mask)[0]
        nonzero_events = events[:, nonzero_col_mask]
        return nonzero_events, event_type
    
    def output_result_get_all_nonzero_events(self, exp_cnt:int = 1):

        exp, exptyp = self.output_result_get_nonzero_events(1)
        nonzero_events = {1: exp}
        event_type = {1: exptyp} 
        if exp_cnt > 1:
            for i in range(2,(exp_cnt+1)):
                exp, exptyp = self.output_result_get_nonzero_events(i)
                nonzero_events[i] = exp
                event_type[i] = exptyp

        return nonzero_events, event_type
    
    def output_result_get_lum_fill(self, exp_cnt:int = 1):
        lum = self.output_result_get_lum(exp_cnt)
        fill = self.output_result_get_filling(exp_cnt)
        return np.column_stack([lum,fill])

    def output_result_get_all_lum_fill(self, exp_cnt:int = 1):
        exp = self.output_result_get_lum_fill(1)
        a = ["Luminescence", "Filling"]
        nonzero_events = {1: exp}
        event_type = {1: a} 
        if exp_cnt > 1:
            for i in range(2,(exp_cnt+1)):
                exp = self.output_result_get_lum_fill(i)
                nonzero_events[i] = exp
                event_type[i] = a

        return nonzero_events, event_type


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
        
        parameters_attrs = {
            "p_birth": cfg.chronology.rjmcmc.parameters.p_birth,
            "p_death": cfg.chronology.rjmcmc.parameters.p_death,
            "p_move_time": cfg.chronology.rjmcmc.parameters.p_move_time,
            "p_move_temp": cfg.chronology.rjmcmc.parameters.p_move_temp,
            "p_move_endpoints": cfg.chronology.rjmcmc.parameters.p_move_endpoints,
            "sigma_birth": cfg.chronology.rjmcmc.parameters.sigma_birth,
            "sigma_t_birth": cfg.chronology.rjmcmc.parameters.sigma_t_birth,
            "sigma_temp": cfg.chronology.rjmcmc.parameters.sigma_temp,
            "sigma_time_frac": cfg.chronology.rjmcmc.parameters.sigma_time_frac,
            "sigma_endpoints": cfg.chronology.rjmcmc.parameters.sigma_endpoints,
        }
        burn_in_attrs = {
            "max_steps": cfg.chronology.rjmcmc.burn_in.max_steps,
            "window": cfg.chronology.rjmcmc.burn_in.window,
            "patience_windows": cfg.chronology.rjmcmc.burn_in.patience_windows,
            "adjustment_factor": cfg.chronology.rjmcmc.burn_in.adjustment_factor,
            "eta_sigma": cfg.chronology.rjmcmc.burn_in.eta_sigma,
            "eta_prob": cfg.chronology.rjmcmc.burn_in.eta_prob,
            "move_bounds": cfg.chronology.rjmcmc.burn_in.move_bounds,
           
            "overall_check": cfg.chronology.rjmcmc.burn_in.overall_check,
            "individual_check": cfg.chronology.rjmcmc.burn_in.individual_check,
            "overall_accept_target": cfg.chronology.rjmcmc.burn_in.overall_accept_target,
            "birth_accept_target": cfg.chronology.rjmcmc.burn_in.birth_accept_target,
            "death_accept_target": cfg.chronology.rjmcmc.burn_in.death_accept_target,
            "move_time_accept_target": cfg.chronology.rjmcmc.burn_in.move_time_accept_target,
            "move_temp_accept_target": cfg.chronology.rjmcmc.burn_in.move_temp_accept_target,
            "move_endpoints_accept_target": cfg.chronology.rjmcmc.burn_in.move_endpoints_accept_target,
            "sigma_birth_bounds": cfg.chronology.rjmcmc.burn_in.sigma_birth_bounds,
            "sigma_birth_t_bounds": cfg.chronology.rjmcmc.burn_in.sigma_birth_t_bounds,
            "sigma_time_bounds": cfg.chronology.rjmcmc.burn_in.sigma_time_bounds,
            "sigma_temp_bounds": cfg.chronology.rjmcmc.burn_in.sigma_temp_bounds,
            "sigma_endpoints_bounds": cfg.chronology.rjmcmc.burn_in.sigma_endpoints_bounds,
            "verbose": cfg.chronology.rjmcmc.burn_in.verbose,
        }
        with h5py.File(self.name,"a") as f:
            chron = f.require_group("chronology")
            self.write_attrs_if_not_none(chron, chron_attrs)
           
            if chron.attrs["method"] == "RJMCMC": 
                rjmcmc = chron.require_group("RJMCMC")
                rjmcmc.attrs["logLikeSigma"]=cfg.chronology.rjmcmc.logLikeSigma

                parameters = rjmcmc.require_group("parameters")
                self.write_attrs_if_not_none(parameters, parameters_attrs)

                if cfg.chronology.rjmcmc.burn_in.burn:
                    burn_in = rjmcmc.require_group("burn_in")
                    self.write_attrs_if_not_none(burn_in, burn_in_attrs)

    def burn_in_update(self,sigmas:Dict[str, float],probs:Dict[str, float]):

        with h5py.File(self.name,"a") as f:     
            chron = f.require_group("chronology")
            rjmcmc = chron.require_group("RJMCMC")
            parameters = rjmcmc.require_group("parameters")
            parameters.attrs["p_birth"] =  probs["birth"]
            parameters.attrs["p_death"] =  probs["death"]
            parameters.attrs["p_move_time"] =  probs["move_time"]
            parameters.attrs["p_move_temp"] =  probs["move_temp"]
            parameters.attrs["p_move_endpoints"] =  probs["move_endpoints"]
            parameters.attrs["sigma_birth"] =  sigmas["birth"]
            parameters.attrs["sigma_t_birth"] =  sigmas["birth_t"]
            parameters.attrs["sigma_temp"] =  sigmas["move_temp"]
            parameters.attrs["sigma_time_frac"] =  sigmas["move_time"]
            parameters.attrs["sigma_endpoints"] =  sigmas["move_endpoints"]

    def get_final_ratios(self, exp_num:int = 1):
        ratios = np.zeros(exp_num)
        for i in range(exp_num): 
            ratios[i] = self.output_result_get_meanTrap((i+1))[-1]

        return ratios
    def get_sigmas(self, exp_num:int = 1):
        stds = np.zeros(exp_num)
        for i in range(exp_num): 
            stds[i] = self.output_result_get_meanTrap((i+1))[-1]

        return stds

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

    def tempertature_profile_get_temps(self,):

        with h5py.File(self.name,"r") as f: 
            temps = f["inputs/temperature"].attrs["temps"] 
        
        return temps
    
    def tempertature_profile_get_times(self,):

        with h5py.File(self.name,"r") as f: 
            times =  f["inputs/temperature"].attrs["times"]
           
        
        return times


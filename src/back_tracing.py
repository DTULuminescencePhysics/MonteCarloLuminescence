from __future__ import annotations
from src.errors import ErrorOutputHandler
from omegaconf import DictConfig
import numpy as np
from src.classes.thermoC.RJMCMC import ReverseJumpMCMC
from src.classes.thermoC.inverse_modeling_mc import InverseMC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.classes.output.results_file import output_file

def RJMCMC_control_functions(output: output_file, obs:np.ndarray,sigma:np.ndarray, chron: DictConfig , cfg: DictConfig | list[DictConfig], 
                         duration: float, seed:int, experiments: int, err: ErrorOutputHandler,):
    """Function that controls the Reverse Jump Markov Chain Monte Carlo method used for 
    thermochronometry"""
   
    # sigma = obs*0.1
    rjmcmc_obj = ReverseJumpMCMC.from_config(seed,obs,duration,chron,sigma)

    rjmcmc_obj.initialise_run(cfg,experiments,err)
    if chron.rjmcmc.burn_in.burn:
        rjmcmc_obj.burn_in_tune(chron.rjmcmc.burn_in.max_steps,chron.rjmcmc.burn_in.window,
                         chron.rjmcmc.burn_in.eta_sigma, chron.rjmcmc.burn_in.eta_prob, chron.rjmcmc.burn_in.overall_check,
                         chron.rjmcmc.burn_in.individual_check, chron.rjmcmc.burn_in.overall_accept_target,
                         chron.rjmcmc.burn_in.birth_accept_target, 
                         chron.rjmcmc.burn_in.death_accept_target, chron.rjmcmc.burn_in.move_time_accept_target,
                         chron.rjmcmc.burn_in.move_temp_accept_target, chron.rjmcmc.burn_in.move_endpoints_accept_target,
                         chron.rjmcmc.burn_in.sigma_birth_bounds, chron.rjmcmc.burn_in.sigma_birth_t_bounds,
                         chron.rjmcmc.burn_in.sigma_time_bounds, chron.rjmcmc.burn_in.sigma_temp_bounds, 
                         chron.rjmcmc.burn_in.sigma_endpoints_bounds, chron.rjmcmc.burn_in.move_bounds,
                       
                            chron.rjmcmc.burn_in.adjustment_factor, 
                         chron.rjmcmc.burn_in.patience_windows, chron.rjmcmc.burn_in.verbose)       

        output.burn_in_update(rjmcmc_obj._get_sigmas(),rjmcmc_obj._get_probs())
    
    rjmcmc_obj.run()
    output.chronological_results(rjmcmc_obj.result_store)
    rjmcmc_obj.result_store.delete_stores()
    

def MC_control_functions(output: output_file, obs:np.ndarray, cfg: DictConfig | list[DictConfig], 
                         duration: float, seed:int, experiments: int, err: ErrorOutputHandler,):
    """Function that controls the monte carlo method used for thermochronometry"""
    sigma = obs*0.1
   
    inverse_obj = InverseMC(obs=obs,sigma=sigma,iters=1000,
                            T0_min=50,T0_max=150,T_target=0,
                            duration=duration,seed=seed,
                            tolerance=5,n_steps_min=0,
                            n_steps_max=10, trend='either')

    inverse_obj.intialise_run(cfg,experiments,err)
    inverse_obj.run_back_simulation()
    output.chronological_results(inverse_obj.result_store)
    inverse_obj.result_store.delete_stores()


def back_tracing_selector(output: output_file, cfg: DictConfig | list[DictConfig], 
                                experiments: int, err: ErrorOutputHandler,):
    
    obs = output.get_final_ratios(experiments)
    sigma = output.get_sigmas(experiments)
    
    if experiments == 1:
        output.chronology_data_initial_build(cfg)
        duration = cfg.temp.duration
        seed = cfg.setup.seed
        chron = cfg.chronology 
    else:
        output.chronology_data_initial_build(cfg[0])
        duration = cfg[0].temp.duration
        seed = cfg[0].setup.seed
        chron = cfg[0].chronology 

    if chron.method == "RJMCMC":
        RJMCMC_control_functions(output,obs,sigma,chron,cfg,duration,seed,experiments,err)
    elif chron.method == "IVMC": 
        MC_control_functions(output,obs,cfg,duration,seed,experiments,err)
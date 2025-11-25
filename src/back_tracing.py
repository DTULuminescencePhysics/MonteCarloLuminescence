from __future__ import annotations
from src.errors import ErrorOutputHandler
from omegaconf import DictConfig
import numpy as np
from src.classes.thermoC.RJMCMC import ReverseJmpMCMC
from src.classes.thermoC.inverse_modeling_mc import InverseMC


def extract_comparison(file_names: str|list[str],experiments: int) -> np.ndarray:
    """Extracts final ratios from either Monte Carlo or Analytical run file(s)"""
    if experiments == 1: 
        data = np.loadtxt(file_names, delimiter=",")
        return np.array(data[-1,-1])
    else: 
        comp = np.zeros(experiments)
        for i in range(experiments):
            data = np.loadtxt(file_names[i], delimiter=",")
            comp[i] = data[-1,-1]
        return comp

def set_observation_values(cfg: DictConfig | list[DictConfig], 
                                experiments: int, err: ErrorOutputHandler, 
                                file_names: str | list[str] | None = None, 
                                extract: bool = True) -> np.ndarray:
    """Sets the observable values either extracting them from monte carlo 
    or analytical run parameters; loading them directly from an input file
    or loading them from the configuration data"""
    if extract:
        if file_names is not None:
            obs = extract_comparison(file_names, experiments)
        else: 
            obs = np.zeros(1)
            err.error("Asked to extract observations from files but files not specified", fatal=True)
    elif file_names is not None:
        try:
            obs = np.loadtxt(file_names, delimiter=",")
        except:
            obs = np.zeros(1)
            err.error(f"Tried to extract end ratios from {file_names} but was not successful", fatal=True)
    else:
        obs = np.zeros(1)
        print("Need to add functionality to add end points to input parameters")

    return obs


def RJMCMC_control_functions(cfg: DictConfig | list[DictConfig], 
                                  experiments: int, err: ErrorOutputHandler, 
                                  file_names: str | list[str] |None = None, extract: bool = True):
    """Function that controls the Reverse Jump Markov Chain Monte Carlo method used for 
    thermochronometry"""
    obs = set_observation_values(cfg, experiments, err, file_names, extract)


def MC_control_functions(cfg: DictConfig | list[DictConfig], 
                                  experiments: int, err: ErrorOutputHandler,
                                file_names: str|None = None, extract: bool = True):
    """Function that controls the monte carlo method used for thermochronometry"""
   
    obs = set_observation_values(cfg, experiments, err, file_names, extract)
    sigma = obs*0.1
    if experiments == 1:
        duration = cfg.temp.duration
    else:
        duration = cfg[0].temp.duration

    inverse_obj = InverseMC(obs,sigma,1000,50,150,0,duration,0,800,n_steps_min=0,n_steps_max=10)

    inverse_obj.intialise_run(cfg,experiments,err)
    inverse_obj.run_back_simulation()
    
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
    sigma = obs*0.1
    if experiments == 1:
        duration = cfg.temp.duration
        seed = cfg.setup.seed
    else:
        duration = cfg[0].temp.duration
        seed = cfg[0].setup.seed

    rjmcmc_obj = ReverseJmpMCMC(obs,iters=1000,T_target=0,
                                duration=duration,seed=seed,
                                tolerance=5,min_gap=0.001,
                                non_increasing=True,p_geom=0.5,
                                k_max=100,
                                init_step_mean=10,
                                init_step_sd=50 ,
                                init_dT_mean=500,
                                init_dT_sd=200, 
                                init_T0_mean=100,
                                init_T0_sd=10, T0_max=150,T0_min=50)

    rjmcmc_obj.intialise_run(cfg,experiments,err)
    rjmcmc_obj.rjmcmc_temperature()

def MC_control_functions(cfg: DictConfig | list[DictConfig], 
                                  experiments: int, err: ErrorOutputHandler,
                                file_names: str|None = None, extract: bool = True):
    """Function that controls the monte carlo method used for thermochronometry"""
   
    obs = set_observation_values(cfg, experiments, err, file_names, extract)
    sigma = obs*0.1
    if experiments == 1:
        duration = cfg.temp.duration
        seed = cfg.setup.seed
    else:
        duration = cfg[0].temp.duration
        seed = cfg[0].setup.seed

    inverse_obj = InverseMC(obs=obs,sigma=sigma,iters=1000,
                            T0_min=50,T0_max=150,T_target=0,
                            duration=duration,seed=seed,
                            dT_min=0,dT_max=1000,tolerance=5,
                            n_steps_min=0,n_steps_max=100)

    inverse_obj.intialise_run(cfg,experiments,err)
    inverse_obj.run_back_simulation()


def repeition_compare(cfg: DictConfig, 
                                  experiments: int, err: ErrorOutputHandler, 
                                  file_names: str | list[str] |None = None, extract: bool = True):
    
    from src.classes.monte_carlo import MCBase
    obs = set_observation_values(cfg, experiments, err, file_names, extract)
    sigma = obs*0.1
    inverse_obj = InverseMC(obs,sigma,1000,50,150,0,cfg.temp.duration,0,800,n_steps_min=0,n_steps_max=10)
    inverse_obj.MC_crystal = MCBase.from_config(cfg)
    inverse_obj.MC_crystal.RJMCMC_initialise()

    reps = 5000
    running_ratio = np.zeros((reps,7))
    prev_N = 0
    N_init = 100
    header =  [f"Repetitions"]
    for j in range(3): 
        inverse_obj.MC_crystal.crystal.set_dimensions(N_init)
        while inverse_obj.MC_crystal.crystal.N <= prev_N:
            N_init += 50
            inverse_obj.MC_crystal.crystal.set_dimensions(N_init)
        prev_N = inverse_obj.MC_crystal.crystal.N
        inverse_obj.MC_crystal.seed = 0
        for i in range(reps):
            inverse_obj.MC_crystal.seed+= 1
            running_ratio[i:,4+j] = running_ratio[i:,4+j] + inverse_obj.MC_crystal.inverse_modeling_simulation()
            running_ratio[i,4+j] /= (i+1)
            running_ratio[i,j+1] =  (np.abs(running_ratio[i,4+j] - obs)/obs)
            running_ratio[i,0] = i+1
        
        N_init += 50
        header.append(f"N: {prev_N}")
        print(prev_N)

    np.savetxt(f"error.csv", running_ratio, delimiter=",",header=",".join(header))
    
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(3.37,5.055))
    ax=fig.add_axes((0.,0.,2.,1.))
    ax.plot(running_ratio[:,0],running_ratio[:,1],label=header[1])
    ax.plot(running_ratio[:,0],running_ratio[:,2],label=header[2])
    ax.plot(running_ratio[:,0],running_ratio[:,3],label=header[3])
    plt.savefig("error.png",dpi=300, transparent=False,bbox_inches='tight')
    plt.legend()
    plt.close()
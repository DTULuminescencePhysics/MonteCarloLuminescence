from __future__ import annotations
from copy import deepcopy
from typing import Callable, Dict, Any, Union, Tuple
from omegaconf import OmegaConf, DictConfig, ListConfig
from numpy import ndarray, atleast_1d, asarray, ndim, concatenate
from src.errors import ErrorOutputHandler
import inspect 


ArrayLike = Union[float, ndarray]
Builder = Callable[..., Callable[[ArrayLike], ArrayLike]]
Builder2 = Callable[..., Callable[[ArrayLike, ArrayLike], ArrayLike]]
Builder3 = Callable[..., Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]]
Builder4 = Callable[..., Callable[[ArrayLike], Tuple[str, ArrayLike]]]


def _filter_kwargs(builder: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs the builder accepts."""
    sig = inspect.signature(builder)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def _as_1d(t: ArrayLike) -> ndarray:
    """Convert any scalar/array input to a 1D float array for internal math."""
    return atleast_1d(asarray(t, dtype=float))

def _return_like_input(t_in: ArrayLike, out_1d: ndarray) -> ArrayLike:
    """Return a scalar if input was scalar; otherwise the 1D array."""
    return out_1d.item() if ndim(t_in) == 0 else out_1d

def _as_1d_array(x: ArrayLike) -> ndarray:
        if isinstance(x, ndarray):
            arr = x.astype(float, copy=False).ravel()
        else:
            arr = asarray([float(x)], dtype=float)
        return arr

def cfg_temperature_check(cfg: DictConfig, err: ErrorOutputHandler): 
    """ Function that checks if the temperature profile information is acceptable and throws and error if not"""
    checker = True
    err.output("Starting check of temperature profile variables")  
    if not isinstance(cfg.temp.T0,(float,int)):
        err.error(f"T0 value in the temperature profile was set as, {cfg.temp.T0}. This is not a float this should be correct", fatal=True)
        checker = False
    if not isinstance(cfg.temp.duration,(float,int)):
        err.error(f"Duration value in the temperature profile was set as, {cfg.temp.duration}. This is not a float this should be correct", fatal=True)
        checker = False 
    
    if cfg.temp.kind == "constant" or not checker:
        err.output("Temperature profile variables check complete")
        return   

    if cfg.temp.kind == "linear":
        if isinstance(cfg.temp.dT,(float,int)):
            pass
        elif (cfg.temp.dT is not None):
            err.error(f"dT value is not None but is also not a float. Value entered: {cfg.temp.dT}.", fatal=False)
            checker = False
        else: 
            checker = False
        if not checker:
            try:
                cfg.temp.dT = (cfg.temp.T0 - cfg.temp.temps[-1] / cfg.temp.duration)
                err.output(f"A starting temperature of {cfg.temp.T0} and a final temperature of {cfg.temp.temps[-1] } \
                           have been identified with a duration of {cfg.temp.duration} giving a linear change of {cfg.temp.dT}.")
                checker = True 
            except: 
                err.error(f"A starting temperature of {cfg.temp.T0} and a final temperature of {cfg.temp.temps[-1] } \
                           have been identified with a duration of {cfg.temp.duration} but it was not possible to calculate a gradient.", fatal=True)           
        
        err.output("Temperature profile variables check complete")
        return   
    elif cfg.temp.kind is None:

        for t in cfg.temp.times:
            if not isinstance(t,(float,int)):
                err.error(f"Value in times is not a float, input is: {t}.",fatal=True)
                checker = False 
        for t in cfg.temp.temps:
            if not isinstance(t,(float,int)):
                err.error(f"Value in temps is not a float, input is {t}.",fatal=True)
                checker = False

        if not cfg.temp.times == sorted(cfg.temp.times):
            err.error("Not all values in times are increasing so profile cannot be generated", fatal=True)
            checker = False

        if not checker:
            err.output("Temperature profile variables check complete")
            return   

        if len(cfg.temp.times) != len(cfg.temp.temps):
            err.error(f"Length of times and temps list must be equal. They were {len(cfg.temp.times)} and {len(cfg.temp.temps)} will attempt to fix.",fatal=False)
            checker = False 
            if (len(cfg.temp.temps) == len(cfg.temp.times)+2):
                if cfg.temp.times[0] != 0 and cfg.temp.times[-1] != cfg.temp.duration:
                    cfg.temp.times.insert(0,0)
                    cfg.temp.times.append(cfg.temp.duration)
                    checker = True 
            elif (len(cfg.temp.temps) == len(cfg.temp.times)+1):
                if cfg.temp.times[0] != 0 and cfg.temp.times[-1] == cfg.temp.duration:
                    cfg.temp.times.insert(0,0)
                    checker = True 
                elif cfg.temp.times[0] == 0 and cfg.temp.times[-1] != cfg.temp.duration:
                    cfg.temp.times.append(cfg.temp.duration)
                    checker = True
            elif((len(cfg.temp.temps)+1  == len(cfg.temp.times)) and (cfg.temp.temps[0] != cfg.temp.T0)):
                cfg.temp.temps.insert(0,cfg.temp.T0)
                checker = True
        
            if checker:
                err.error(f"Success fixing disparity in lengths of temps and times. New entries are {cfg.temp.times} and {cfg.temp.temps}.",fatal=False)
                err.clear_errors()
                err.output(f"Success fixing disparity in lengths of temps and times. New entries are {cfg.temp.times} and {cfg.temp.temps}.")
            else: 
                err.error(f"Unable to fix disparity in length of temps and times.",fatal=True)
        
        if not checker:
            err.output("Temperature profile variables check complete")
            return   

        if cfg.temp.times[-1] != cfg.temp.duration:    
            err.error(f"Final time must match duration. The values {cfg.temp.times[-1]} and {cfg.temp.duration}",fatal=True)

        if cfg.temp.times[-1] != cfg.temp.duration and cfg.temp.temps[0] != cfg.temp.T0:
            err.error(f"Missing starting time and temperature so will attempt to fix",fatal=False)

        
        if cfg.temp.temps[0] != cfg.temp.T0:
            if cfg.temp.times[0] != 0.0: 
                err.error(f"Missing starting time and temperature so will attempt to fix",fatal=False)
                cfg.temp.times.insert(0,0)
                cfg.temp.temps.insert(0,cfg.temp.T0)
                err.error(f"Success fixing initial time and temperature. New entries are {cfg.temp.times} and {cfg.temp.temps}.",fatal=False)
                err.clear_errors()
                err.output(f"Success fixing initial time and temperature. New entries are {cfg.temp.times} and {cfg.temp.temps}.")
            else: 
                err.error(f"Initial temperature must match T0. The values {cfg.temp.temps[0]} and {cfg.temp.T0}",fatal=False)
        
       

    else: 
        err.error(f"The temperature profile kind parameter is {cfg.temp.kind} but should be set to null, constant or linear.")

       
    err.output("Temperature profile variables check complete")
    return 



def cfg_list_check(cfg: DictConfig, err: ErrorOutputHandler) -> Tuple[int, DictConfig] | Tuple[int, list[DictConfig]]:
    """
    Given a DictConfig `cfg`, look at the fields in configs, which is currently just physics.
    If any of them are lists then check if the lists are the same length and if they are seperate them 
    into different input parameters. Returns the number of experiemnts to complete and the configuration 
    or a list of the configurations.
    """


    configs = {"physics"}
    list_vars = []
    lengths = []
    for key, _ in cfg.items():
        if key in configs:
            to_check = OmegaConf.select(cfg,key)
            for key2, value in to_check.items(): 
                if isinstance(value,ListConfig):
                    lengths.append(len(value))
                    list_vars.append(key2)
  
    max_length = max(lengths) if lengths else 1
    if max_length == 1: 
        return 1, cfg
    
    if lengths.count(lengths[0])==len(lengths): 
        err.output("All listed values in the physics input of same length")
    else:
        err.error("The listed physics values need to be of the same length or a single input",fatal=True)
        return 0, cfg 
  
    runs = []
    for i in range(max_length):
        cfg_copy = deepcopy(cfg)
        for key in configs: 
            for key2 in list_vars:
                path = f"{key}.{key2}"
                v =  OmegaConf.select(cfg,path)
                value = v[i]
                OmegaConf.update(cfg_copy,path,v[i])
        runs.append(cfg_copy)

    return max_length, runs



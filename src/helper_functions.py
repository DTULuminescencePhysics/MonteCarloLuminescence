from __future__ import annotations
from copy import deepcopy
from typing import Callable, Dict, Any, Union, Tuple
from omegaconf import OmegaConf, DictConfig, ListConfig
from numpy import ndarray, atleast_1d, asarray, ndim
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



from __future__ import annotations
from typing import Callable, Tuple
import numpy as np 

from src.classes.physics.transitions import Transitions
from src.helper_functions import ArrayLike, _return_like_input
from src.classes.constants import cnst

# Filling Registry
@Transitions.register_fill("dose")
def build_fill_dose(D0: float) -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
    """Dosing equation from A.Larsen et al. Radiation Measurements 44 (2009) 467-471
    which is equivalent to the dosing equation from 
    G.E. King et al.  Quaternary Geochronology 33 (2016) 76-87 if N is fixed"""
    def f(N: ArrayLike, e: ArrayLike, D_dot: ArrayLike) -> ArrayLike:
        diff = N-e
        # ratio = e/N
        if diff <= 0:
            lam= 1e-20
        else: 
            lam = (D_dot*diff/D0)
   
        return _return_like_input(N,np.array(lam))
    return f

@Transitions.register_fill("none")
def build_fill_none() -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
    def f(N: ArrayLike, e: ArrayLike, D_dot) -> ArrayLike:
        return -1e20
    return f

# Fading transition Registry
@Transitions.register_tran("ground_state_tunnel", "_gs_tun")
def build_gs_tunnel(alpha_GS:float, b:float) -> Callable[[ArrayLike], ArrayLike]:
    def f(r:ArrayLike) -> ArrayLike:
        
        lifetime = (b * np.exp(-alpha_GS * r))

        return lifetime
    return f

@Transitions.register_tran("excited_state_tunnel", "_es_tun")
def build_es_tunnel(alpha:float, b:float) -> Callable[[ArrayLike], ArrayLike]:
    def f(r:ArrayLike) -> ArrayLike:
        
        lifetime = (b * np.exp(-alpha * r))

        return lifetime
    return f

@Transitions.register_tran("ground_state_to_cb", "_gs_con")
def build_gs_ex_cb(E_cb:float, s:float) -> Callable[[ArrayLike], ArrayLike]:
    def f(T:ArrayLike) -> ArrayLike:
        
        lifetime = (s * np.exp(-E_cb / (cnst.k_b_ev * T)))

        return lifetime
    return f

@Transitions.register_tran("excited_state_to_cb", "_es_con")
def build_gs_ex_cb(E_loc:float, E_cb:float, s:float) -> Callable[[ArrayLike], ArrayLike]:
    def f(T:ArrayLike) -> ArrayLike:
        
        lifetime = (s * np.exp(-(E_cb - E_loc) / (cnst.k_b_ev * T)))

        return lifetime
    return f

@Transitions.register_tran("cb_mobility", "_cb_mob")
def build_cb_mobility(mu:float) -> Callable[[ArrayLike], ArrayLike]:
    def f(r:ArrayLike) -> ArrayLike:
        
        lifetime = np.exp(r/mu)**2

        return lifetime
    return f
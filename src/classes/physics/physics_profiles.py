from __future__ import annotations
from typing import Callable
import numpy as np 

from src.classes.physics.system_physics import CrystalPhysics
from src.helper_functions import ArrayLike, _return_like_input
from src.classes.constants import cnst


# Filling Registry

@CrystalPhysics.register_fill("dose")
def build_fill_dose(D0: float) -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
    def f(N: ArrayLike, e: ArrayLike, D_dot: ArrayLike) -> ArrayLike:
        diff = N-e
        if diff <= 0:
            out = np.array(1e-20)
        else: 
            out = np.array(D0/(diff*D_dot))
        
        return _return_like_input(N,out)
    return f

@CrystalPhysics.register_fill("none")
def build_fill_none() -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
    def f(N: ArrayLike, e: ArrayLike, D_dot) -> ArrayLike:
        return 1e20
    return f

# Fading registry 

@CrystalPhysics.register_fade("therm_tunnel")
def build_fade_therm_tun(E_loc:float, b: float, alpha: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    Ek = E_loc/cnst.k_b_ev
    def f(T: ArrayLike, r: ArrayLike) -> ArrayLike:
        out = 1/(b*np.exp((-alpha*r)-(Ek/T)))
        return _return_like_input(r,out)
    return f

@CrystalPhysics.register_fade("therm_tunnel_delocaise")
def build_fade_therm_tun_deloc(E_loc:float, b: float, alpha: float, E_cb: float, s: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    Ek = E_loc/cnst.k_b_ev
    Ebk = E_cb/cnst.k_b_ev
    def f(T: ArrayLike, r: ArrayLike) -> ArrayLike:
        out = 1/(b*np.exp((-alpha*r)-(Ek/T)))+(s*np.exp(-(Ebk/T)))
        return _return_like_input(r,out)
    return f

   
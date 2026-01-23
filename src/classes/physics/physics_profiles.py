from __future__ import annotations
from typing import Callable
import numpy as np 

from src.classes.physics.system_physics import CrystalPhysics
from src.helper_functions import ArrayLike, _return_like_input
from src.classes.constants import cnst


# Filling Registry
@CrystalPhysics.register_fill("dose")
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
                   
        out = np.array(1/lam)
   
        return _return_like_input(N,out)
    return f

@CrystalPhysics.register_fill("none")
def build_fill_none() -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
    def f(N: ArrayLike, e: ArrayLike, D_dot) -> ArrayLike:
        return -1e20
    return f

# Fading registry 
@CrystalPhysics.register_fade("therm_tunnel")
def build_fade_therm_tun(E_loc:float, b: float, alpha: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    def f(T: ArrayLike, r: ArrayLike) -> ArrayLike:
        out = 1/(b * np.exp(((-E_loc / (cnst.k_b_ev * T)) -(alpha * r))))
        return _return_like_input(r,out)
    return f

@CrystalPhysics.register_fade("therm_tunnel_delocalise")
def build_fade_therm_tun_deloc(E_loc:float, b: float, alpha: float, E_cb: float, s: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    def f(T: ArrayLike, r: ArrayLike) -> ArrayLike:
        if isinstance(r, np.ndarray):
            if r.size == 0:
                return -1e20 
        else: 
            if r is None or r == 0: 
                return -1e20
                 
        term1 = (b * np.exp(-((E_loc / (cnst.k_b_ev * T)) +(alpha * r))))
        term2 = (s * np.exp(-E_cb / (cnst.k_b_ev * T)))
        out = 1/(term1 + term2)

        return _return_like_input(r,out)
    return f

@CrystalPhysics.register_fade("therm_tunnel_delocalise_GS_tunnel")
def build_fade_therm_tun_deloc_GS_tun(E_loc:float, b: float, alpha: float, alpha_GS: float, E_cb: float, s: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    def f(T: ArrayLike, r: ArrayLike) -> ArrayLike:
        if isinstance(r, np.ndarray):
            if r.size == 0:
                return -1e20 
        else: 
            if r is None or r == 0: 
                return -1e20
                 
        term1 = (b * np.exp(-((E_loc / (cnst.k_b_ev * T)) +(alpha * r))))
        term2 = (s * np.exp(-E_cb / (cnst.k_b_ev * T)))
        term3 = (b * np.exp(-alpha_GS * r))
        out = 1/(term1 + term2 + term3)

        return _return_like_input(r,out)
    return f

# @CrystalPhysics.register_fade("therm_tunnel_delocalise")
# def build_fade_therm_tun_deloc(E_loc:float, b: float, urho: float, E_cb: float, s: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:

#     def f(T: ArrayLike, ur: ArrayLike) -> ArrayLike:
#         if isinstance(ur, np.ndarray):
#             if ur.size == 0:
#                 return -1e20 
#         else: 
#             if ur is None or ur == 0: 
#                 return -1e20

#         term1 = (b * np.exp(-E_loc / (cnst.k_b_ev * T) -((1/np.cbrt(urho)) * ur)))    
#         term2 = (s * np.exp(-E_cb / (cnst.k_b_ev * T)))
#         out = 1/(term1 + term2)

#         return _return_like_input(ur,out)
#     return f

@CrystalPhysics.register_fade("GE_king_2016")
def build_fade_therm_tun_band(E_loc:float, b: float, alpha: float, E_cb: float, s: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    def f(T: ArrayLike, r: ArrayLike) -> ArrayLike:
  
        term1 = (b * np.exp(- alpha * r))
        term2 = (s * np.exp(-(E_loc - E_cb) / (cnst.k_b_ev * T)))
        out = 1/(term1 + term2)

      
        return _return_like_input(r,out)
        
    return f

############################################################################
# Analytic model physics profiles
############################################################################
@CrystalPhysics.register_fill("dose_ratio")
def build_dose_ratio_change() -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
    """Dosing equation from A.Larsen et al. Radiation Measurements 44 (2009) 467-471
    which is equivalent to the dosing equation from 
    G.E. King et al.  Quaternary Geochronology 33 (2016) 76-87 if N is fixed"""
    def f(ratio: ArrayLike, D0: ArrayLike, D_dot: ArrayLike) -> ArrayLike:
        out = np.array((D_dot/D0)*(1-ratio))
        
        return _return_like_input(ratio,out)
    return f


@CrystalPhysics.register_fade("GE_king_2016_unitless")
def build_fade_therm_tun_band_unitless(E_loc:float, b: float, E_cb: float, s: float, urho: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    def f(T: ArrayLike, ur: ArrayLike) -> ArrayLike:
        
        term1 = (b * np.exp(-(1/np.cbrt(urho)) * ur))
        term2 = (s * np.exp(-(E_loc - E_cb) / (cnst.k_b_ev * T)))
        out = (term1 + term2)

      
        return _return_like_input(ur,out)
    return f

@CrystalPhysics.register_fade("GE_king_2016_unit")
def build_fade_therm_tun_band_unit(E_loc:float, b: float, E_cb: float, s: float, alpha: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    def f(T: ArrayLike, r: ArrayLike) -> ArrayLike:
        
        term1 = (b * np.exp(- alpha * r))
        term2 = (s * np.exp(-(E_loc - E_cb) / (cnst.k_b_ev * T)))
        out = (term1 + term2)

      
        return _return_like_input(r,out)
        
    return f

@CrystalPhysics.register_fade("therm_tunnel_delocalise_unit")
def build_fade_therm_tun_deloc_analyic_unit(E_loc:float, b: float, alpha: float, E_cb: float, s: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:

    def f(T: ArrayLike, r: ArrayLike) -> ArrayLike:
        term1 = (b * np.exp(-E_loc / (cnst.k_b_ev * T) - alpha * r))
        term2 = s * np.exp(-E_cb / (cnst.k_b_ev * T))

        out = (term1 + term2)

        return _return_like_input(r,out)
    return f

@CrystalPhysics.register_fade("therm_tunnel_unit")
def build_fade_thermdeloc_analytic_unit(E_loc:float, b: float, alpha: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:

    def f(T: ArrayLike, r: ArrayLike) -> ArrayLike:
        term1 = (b * np.exp(-E_loc / (cnst.k_b_ev * T) - alpha * r))
        out = term1 

        return _return_like_input(r,out)
    return f

@CrystalPhysics.register_fade("therm_tunnel_delocalise_unitless")
def build_fade_therm_tun_deloc_analyic_unitless(E_loc:float, b: float, urho: float, E_cb: float, s: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:

    def f(T: ArrayLike, ur: ArrayLike) -> ArrayLike:
        term1 = (b * np.exp(-E_loc / (cnst.k_b_ev * T) -((1/np.cbrt(urho)) * ur)))
        term2 = s * np.exp(-E_cb / (cnst.k_b_ev * T))

        out = (term1 + term2)

        return _return_like_input(ur,out)
    return f

@CrystalPhysics.register_fade("therm_tunnel_unitless")
def build_fade_thermdeloc_analytic_unitless(E_loc:float, b: float, urho: float)-> Callable[[ArrayLike, ArrayLike], ArrayLike]:

    def f(T: ArrayLike, ur: ArrayLike) -> ArrayLike:
        term1 = (b * np.exp(-E_loc / (cnst.k_b_ev * T) -((1/np.cbrt(urho)) * ur)))
        out = term1 

        return _return_like_input(ur,out)
    return f
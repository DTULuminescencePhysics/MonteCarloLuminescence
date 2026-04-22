from __future__ import annotations
from typing import Callable, ClassVar, Dict, List
from dataclasses import dataclass, field
import numpy as np
from src.helper_functions import _filter_kwargs,ArrayLike, Builder, Builder2, Builder3
from src.classes.constants import cnst
from functools import partial


@dataclass
class CrystalPhysics:
    _fill: Callable[..., ArrayLike] = field(init=False, repr=False)
    _fade: Callable[..., ArrayLike] = field(init=False, repr=False)
    
    _gs_tun: Callable[..., ArrayLike] = field(init=False, repr=False)
    _es_tun: Callable[..., ArrayLike] = field(init=False, repr=False)
    _gs_con: Callable[..., ArrayLike] = field(init=False, repr=False)
    _es_con: Callable[..., ArrayLike] = field(init=False, repr=False)
    _cb_mob: Callable[..., ArrayLike] = field(init=False, repr=False)

    FILL_REGISTRY: ClassVar[Dict[str, Builder3]] = {}
    FADE_REGISTRY: ClassVar[Dict[str, Builder2]] = {}
    TRAN_REGISTRY: ClassVar[Dict[str, Builder]] = {}

    def __init__(self, fill_kind: str, fade_kind: str, tran_kind: Dict[str, str]|None = None, **kwargs):
                 
        fill_kind = fill_kind.lower()
        fade_kind = fade_kind.lower()
        if fill_kind not in self.FILL_REGISTRY:
            raise ValueError(f"Unknown fill kind '{fill_kind}'. Available: {sorted(self.FILL_REGISTRY)}")
        if fade_kind not in self.FADE_REGISTRY:
            raise ValueError(f"Unknown empty kind '{fade_kind}'. Available: {sorted(self.FADE_REGISTRY)}")

        f_builder = self.FILL_REGISTRY[fill_kind]
        e_builder = self.FADE_REGISTRY[fade_kind]

        fi_kwargs = _filter_kwargs(f_builder, kwargs)
        fa_kwargs = _filter_kwargs(e_builder, kwargs)

        object.__setattr__(self, "_fill", f_builder(**fi_kwargs))
        object.__setattr__(self, "_fade", e_builder(**fa_kwargs))

        if tran_kind is not None:
            for tran, attr in tran_kind.items():
                f_builder = self.TRAN_REGISTRY[tran]
                f_kwargs = _filter_kwargs(f_builder, kwargs)
                object.__setattr__(self, attr, f_builder(**f_kwargs))


    # Registration helpers
    @classmethod
    def register_fill(cls, kind: str) -> Callable[[Builder3], Builder3]:
        kind = kind.lower()
        def deco(fn: Builder3) -> Builder3:
            if not callable(fn):
                raise TypeError("fill builder must be callable")
            cls.FILL_REGISTRY[kind] = fn
            return fn
        return deco

    @classmethod
    def register_fade(cls, kind: str) -> Callable[[Builder2], Builder2]:
        kind = kind.lower()
        def deco(fn: Builder2) -> Builder2:
            if not callable(fn):
                raise TypeError("empty builder must be callable")
            cls.FADE_REGISTRY[kind] = fn
            return fn
        return deco
    
    @classmethod
    def register_tran(cls, kind: str) -> Callable[[Builder], Builder]:
        kind = kind.lower()
        def deco(fn: Builder) -> Builder:
            if not callable(fn):
                raise TypeError("empty builder must be callable")
            cls.TRAN_REGISTRY[kind] = fn
            return fn
        return deco

    @classmethod
    def available_physics(cls) -> dict[str, list[str]]:
        return {
            "fill": sorted(cls.FILL_REGISTRY),
            "fade": sorted(cls.FADE_REGISTRY),
        }
    

    def lifetime_dist_independ(E:float, s:float, T:float) -> float:
        return s * np.exp(-E/(cnst.k_b_ev * T))


    def lifetime_dist_depend(alpha_ES:float, b:float, r:ArrayLike) -> ArrayLike:
        if isinstance(r, np.ndarray):
            if r.size == 0:
                return -1.e20
            
        else:
            if r is None or r == 0: 
                return -1.e20

        return b * np.exp(-alpha_ES * r)


@dataclass
class _ThermalParameters(CrystalPhysics):

    E_loc:     float                  # Energy gap between ground and excited state
    b :        float                  # attmpt to tunnel frequency
    alpha_ES:     float | None = field(default=None) # Excited state tunnelling
    alpha_GS:  float | None = field(default=None) # Ground state tunnelling
    E_cb:      float | None = field(default=None) # Conduction band energy
    s :        float | None = field(default=None) # Escape frequency
    rho:       float | None = field(default=None) # Density
    urho:      float | None = field(default=None) # Unitless density
    D0:        float | None = field(default=None) # Characteristic does
    D_dot:     float | None = field(default=None) # Radition per second
    
   
    def __post_init__(self):

        if self.alpha_ES is None:
            self.alpha_ES = self.set_alpha()
        if self.rho is not None and self.urho is None:
            self.urho_from_rho(self.alpha_ES)
        elif self.urho is not None and self.rho is None:
            self.rho_from_urho(self.alpha_ES)

        fill_kind = "dose" if self.D0 is not None else "none"
        fade_kind = "therm_tunnel_delocalise" if self.E_cb is not None else "therm_tunnel"
        
        CrystalPhysics.__init__(self,fill_kind,fade_kind,**vars(self))
    
    def set_alpha(self):
        """Square tunneling potential"""
        alpha_ES = 2*np.sqrt(2*cnst.m_e*self.E_loc*cnst.ev_to_j)/ cnst.h_bar
        return alpha_ES 
    
    def set_rho(self,rho):
        self.rho = rho
        
    def set_urho(self,urho):
        self.urho = urho 

    def urho_from_rho(self,alpha_ES):
        if self.rho is not None:
            self.urho = (4*np.pi* self.rho/3)/np.power(alpha_ES,3)

    def rho_from_urho(self,alpha_ES):
        if self.urho is not None:
            self.rho = self.urho*np.power(alpha_ES,3)*(3/(np.pi*4))

    





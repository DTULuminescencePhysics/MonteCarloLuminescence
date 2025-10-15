from __future__ import annotations
from typing import Callable, ClassVar, Dict,  Optional
from dataclasses import dataclass, field
import numpy as np
from src.helper_functions import _filter_kwargs,ArrayLike, Builder2, Builder3
from src.classes.constants import cnst


@dataclass
class CrystalPhysics:
    _fill:  Callable[..., ArrayLike] = field(init=False, repr=False)
    _fade: Callable[..., ArrayLike] = field(init=False, repr=False)

    FILL_REGISTRY:  ClassVar[Dict[str, Builder3]] = {}
    FADE_REGISTRY: ClassVar[Dict[str, Builder2]] = {}

    def __init__(self, fill_kind: str, fade_kind: str, **kwargs):
                 
        fill_kind  = fill_kind.lower()
        fade_kind = fade_kind.lower()
        if fill_kind not in self.FILL_REGISTRY:
            raise ValueError(f"Unknown fill kind '{fill_kind}'. Available: {sorted(self.FILL_REGISTRY)}")
        if fade_kind not in self.FADE_REGISTRY:
            raise ValueError(f"Unknown empty kind '{fade_kind}'. Available: {sorted(self.FADE_REGISTRY)}")

        f_builder = self.FILL_REGISTRY[fill_kind]
        e_builder = self.FADE_REGISTRY[fade_kind]

        fi_kwargs = _filter_kwargs(f_builder, kwargs)
        fa_kwargs = _filter_kwargs(e_builder, kwargs)

        object.__setattr__(self, "_fill",  f_builder(**fi_kwargs))
        object.__setattr__(self, "_fade", e_builder(**fa_kwargs))

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
    def available_physics(cls) -> dict[str, list[str]]:
        return {
            "fill":  sorted(cls.FILL_REGISTRY),
            "fade": sorted(cls.FADE_REGISTRY),
        }

@dataclass
class _ThermalParameters(CrystalPhysics):

    E_loc: float                  # Energy gap between ground and excited state
    b :    float                  # attmpt to tunnel frequency
    alpha: float | None = field(default=None)
    E_cb:  float | None = field(default=None) # Conduction band energy
    s :    float | None = field(default=None) # Escape frequency
    rho:   float | None = field(default=None) # Density
    urho:  float | None = field(default=None) # Unitless density
    D0:    float | None = field(default=None) # Characteristic does
    D_dot: float | None = field(default=None) # Radition per second
   
    def __post_init__(self):

        if self.alpha is None:
            self.alpha = self.set_alpha()
        if self.rho is not None and self.urho is None:
            self.urho_from_rho(self.alpha)
        elif self.urho is not None and self.rho is None:
            self.rho_from_urho(self.alpha)

        fill_kind = "dose" if self.D0 is not None else "none"
        fade_kind = "therm_tunnel_delocaise" if self.E_cb is not None else "therm_tunnel"
        
        CrystalPhysics.__init__(self,fill_kind,fade_kind,**vars(self))
    
    def set_alpha(self):
        """Square tunneling potential"""
        alpha = 2*np.sqrt(2*cnst.m_e*self.E_loc*cnst.ev_to_j)/ cnst.h_bar
        return alpha 
    
    def set_rho(self,rho):
        self.rho = rho
        
    def set_urho(self,urho):
        self.urho = urho 

    def urho_from_rho(self,alpha):
        if self.rho is not None:
            self.urho = (4*np.pi* self.rho/3)/np.power(alpha,3)

    def rho_from_urho(self,alpha):
        if self.urho is not None:
            self.rho = self.urho*np.power(alpha,3)*(3/(np.pi*4))

    







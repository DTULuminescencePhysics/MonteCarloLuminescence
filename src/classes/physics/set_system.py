from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from src.classes.constants import cnst, time_to_seconds
from src.classes.physics.system_physics import CrystalPhysics
import src.classes.physics.physics_profiles


@dataclass
class _ThermalParameters(CrystalPhysics):

    E_loc:   float                  # Energy gap between ground and excited state
    b :      float                  # attmpt to tunnel frequency
    alpha:   float | None = field(default=None)
    E_cb:    float | None = field(default=None) # Conduction band energy
    s :      float | None = field(default=None) # Escape frequency
    rho:     float | None = field(default=None) # Density
    urho:    float | None = field(default=None) # Unitless density
    D0:      float | None = field(default=None) # Characteristic does
    D_dot:   float | None = field(default=None) # Radition per time unit
    Dd_unit: str = field(default='s') # Units of Radiation s : Gy/s; ka : Gy/Ka etc.
    phys_type: str   | None = field(default=None) # Kind of fading model


    def __post_init__(self):

        if self.alpha is None:
            self.alpha = self.set_alpha()
        if self.rho is not None and self.urho is None:
            self.urho_from_rho(self.alpha)
        elif self.urho is not None and self.rho is None:
            self.rho_from_urho(self.alpha)

        if self.D_dot is not None:
            self.D_dot /= time_to_seconds[self.Dd_unit]

        fill_kind, fade_kind = self.set_fill_and_fade()

        CrystalPhysics.__init__(self,fill_kind,fade_kind,**vars(self))
    
    def set_fill_and_fade(self):
        """Function that returns the fade and fill kinds that can be passed to the CrystalPhysics
        intializer."""
            
        if self.phys_type == "king":
            fade_kind = "GE_king_2016"
        elif self.phys_type == "king_ratio":
            fade_kind = "GE_king_2016_ratio"
        elif self.phys_type == "ratio":
            fade_kind = "therm_tunnel_delocaise_ratio"
        else:
            fade_kind = "therm_tunnel_delocaise" if self.E_cb is not None else "therm_tunnel"

        fill_kind = "dose" if self.D0 is not None else "none"

        return fill_kind, fade_kind
    
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
            self.urho = self.rho*(((4*np.pi)/3)/np.power(alpha,3))

    def rho_from_urho(self,alpha):
        if self.urho is not None:
            self.rho = self.urho*(np.power(alpha,3)*(3/(np.pi*4)))

from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from src.classes.constants import cnst, time_to_seconds
# from src.classes.physics.system_physics import CrystalPhysics
from typing import Dict, List
# import src.classes.physics.physics_profiles
from src.classes.physics.transitions import Transitions
import src.classes.physics.transition_profiles


@dataclass
class _ThermalParameters(Transitions):    # CrystalPhysics

    E_loc:       float                  # Energy gap between ground and excited state
    b :          float                  # attmpt to tunnel frequency
    E_loc_sigma: float = field(default=0.0)   # std dev of Gaussian for E_loc (eV); 0 = fixed
    E_cb_sigma:  float = field(default=0.0)   # std dev of Gaussian for E_cb  (eV); 0 = fixed
    alpha_ES:   float | None = field(default=None)
    alpha_GS:float | None = field(default=None)
    E_cb:    float | None = field(default=None) # Conduction band energy
    s :      float | None = field(default=None) # Escape frequency
    rho:     float | None = field(default=None) # Density
    urho:    float | None = field(default=None) # Unitless density
    D0:      float | None = field(default=None) # Characteristic doses
    D_dot:   float | None = field(default=None) # Radition per time unit
    mu:      float | None = field(default=None) # Electron spread range in CB
    Dd_unit: str = field(default='s')           # Units of Radiation s : Gy/s; ka : Gy/Ka etc.
    phys_type: str   = field(default="")        # Kind of fading model
    VRH:     bool   = field(default=False)      # If True, tunneling retrapping uses Miller-Abrahams VRH
    enable_fill:      bool = field(default=True)
    enable_tunneling: bool = field(default=True)
    enable_cb:        bool = field(default=True)

    # Simulation mode and optical stimulation (OSL)
    mode:      str   = field(default="TL")      # "TL" | "OSL"
    stim_kind: str   = field(default="cw")      # "cw" (constant I) | "lm" (linear I(t))
    
    sigma_IR:  float | None = field(default=None)  # IR optical cross section (m^2)
    I0_IR:     float | None = field(default=None)  # IR CW intensity / LM offset (photons m^-2 s^-1)
    I_rate_IR: float = field(default=0.0)        # IR LM ramp rate (photons m^-2 s^-3)
    
    sigma_blue:  float | None = field(default=None)  # blue/UV cross section (m^2)
    I0_blue:     float | None = field(default=None)  # blue CW intensity / LM offset (photons m^-2 s^-1)
    I_rate_blue: float = field(default=0.0)           # blue LM ramp rate (photons m^-2 s^-3)


    def __post_init__(self):

        self._validate_osl()

        if self.alpha_ES is None:
            self.alpha_ES = self.set_alpha()
        if self.rho is not None and self.urho is None:
            self.urho_from_rho(self.alpha_ES)
        elif self.urho is not None and self.rho is None:
            self.rho_from_urho(self.alpha_ES)

        if self.D_dot is not None:
            self.D_dot /= time_to_seconds[self.Dd_unit]

        # fill_kind, fade_kind = self.set_fill_and_fade()
        fill_kind, tran_kind = self.set_transitions()

        # CrystalPhysics.__init__(self,fill_kind,fade_kind,tran_kind,**vars(self))
        Transitions.__init__(self, fill_kind, tran_kind, **vars(self))

        self._processes = self.build_processes(tran_kind, **vars(self))

    
    def _validate_osl(self):
        if self.mode == "OSL":
            if self.s is None:
                raise ValueError("OSL mode requires the escape frequency 's' to be set.")
            
            # At least one optical channel must be active: IR (GS->ES) and/or blue/UV (GS/ES->CB). 
            # Each channel needs a cross section and a non-zero intensity ('I0_IR' for cw, or 'I_rate_IR' for lm).

            ir_on   = self.sigma_IR is not None and (self.I0_IR or self.I_rate_IR)
            blue_on = self.sigma_blue is not None and (self.I0_blue or self.I_rate_blue)
            if not (ir_on or blue_on):
                raise ValueError(
                    "OSL mode requires at least one optical channel: IR "
                    "('sigma_IR' + 'I0_IR'/'I_rate_IR') and/or blue ('sigma_blue' + "
                    "'I0_blue'/'I_rate_blue').")
            if blue_on and not self.enable_cb:
                raise ValueError(
                    "Blue/UV stimulation drives GS/ES -> conduction band; "
                    "'enable_cb' must be True when the blue channel is active.")

    def set_transitions(self):

        if self.enable_fill and self.D0 is not None:
            fill_kind = "dose"
        else:
            fill_kind = "none"

        trans_kind: list[str] = []
        if self.enable_tunneling:
            trans_kind += ["ground_state_tunnel",
                           "excited_state_tunnel"]
        if self.enable_cb:
            trans_kind += ["ground_state_to_cb",
                           "excited_state_to_cb"]

        return fill_kind, trans_kind
    
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
            self.urho = self.rho*(((4*np.pi)/3)/np.power(alpha_ES,3))

    def rho_from_urho(self,alpha_ES):
        if self.urho is not None:
            self.rho = self.urho*(np.power(alpha_ES,3)*(3/(np.pi*4)))

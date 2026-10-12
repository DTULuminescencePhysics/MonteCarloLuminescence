from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from src.classes.constants import cnst, time_to_seconds
# from src.classes.physics.system_physics import CrystalPhysics
from typing import Dict, List
# import src.classes.physics.physics_profiles
from src.classes.physics.transitions import Transitions
import src.classes.physics.transition_profiles
from src.classes.physics.cross_section import CrossSection, photon_energy_eV


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
    rho_h:   float | None = field(default=None) # Hole density; None -> rho
    urho:    float | None = field(default=None) # Unitless density
    D0:      float | None = field(default=None) # Characteristic doses
    D_dot:   float | None = field(default=None) # Radition per time unit
    dose_time: float | None = field(default=None) # Dose-on duration (temp time unit); None -> duration
    mu:      float | None = field(default=None) # Electron spread range in CB
    mu_fill: float | None = field(default=None) # Carrier spread range when filling; None -> mu
    fill_mode: str = field(default="uniform")   # uniform | spatial
    recom_new_pair: bool = field(default=False) # True: e-h pair born in one event may recombine
    Dd_unit: str = field(default='s')           # Units of Radiation s : Gy/s; ka : Gy/Ka etc.
    phys_type: str   = field(default="")        # Kind of fading model
    VRH:     bool   = field(default=False)      # If True, tunneling retrapping uses Miller-Abrahams VRH
    enable_fill:      bool = field(default=True)
    enable_tunneling: bool = field(default=True)
    enable_cb:        bool = field(default=True)

    # Simulation mode and optical stimulation (OSL)
    mode:      str   = field(default="TL")      # "TL" | "OSL"
    stim_kind: str   = field(default="cw")      # "cw" (constant I) | "lm" (linear I(t))
    
    sigma_IR:  float | None = field(default=None)  # IR peak/amplitude cross section sigma0 (m^2)
    I0_IR:     float | None = field(default=None)  # IR CW intensity / LM offset (photons m^-2 s^-1)
    I_rate_IR: float = field(default=0.0)        # IR LM ramp rate (photons m^-2 s^-3)
    
    sigma_blue:  float | None = field(default=None)  # blue/UV peak/amplitude cross section sigma0 (m^2)
    I0_blue:     float | None = field(default=None)  # blue CW intensity / LM offset (photons m^-2 s^-1)
    I_rate_blue: float = field(default=0.0)           # blue LM ramp rate (photons m^-2 s^-3)

    # Wavelength-dependent photoionisation cross-section sigma(E, T) per channel.
    xsec_model_IR: str = field(default="const")   # const | lucovsky | grimmeiss
    lambda_IR:     float | None = field(default=None)  # IR stimulation wavelength (nm)
    E0_IR:         float | None = field(default=None)  # IR optical threshold / trap depth (eV)
    m_eff_IR:      float = field(default=1.0)      # effective-mass ratio m*/m_c (grimmeiss)

    xsec_model_blue: str = field(default="const")
    lambda_blue:     float | None = field(default=None)  # blue/UV stimulation wavelength (nm)
    E0_blue:         float | None = field(default=None)  # blue/UV optical threshold (eV)
    m_eff_blue:      float = field(default=1.0)

    # Baril & Huntley (2003) IR intra-centre (GS->ES) resonance parameters,
    # used by the IR channel when xsec_model_IR=baril_voigt.
    
    E1_IR:       float = field(default=1.44)        # principal resonance centre (eV)
    sigma_g1_IR: float = field(default=0.136)       # principal Gaussian std dev (eV)
    gamma_l1_IR: float = field(default=0.010)       # principal Lorentzian HWHM (eV)
    A1_IR:       float = field(default=1.0)          # principal relative amplitude
    E2_IR:       float | None = field(default=1.57) # secondary Gaussian centre (eV)
    sigma_g2_IR: float = field(default=0.21)        # secondary Gaussian std dev (eV)
    A2_IR:       float = field(default=0.333)        # secondary relative amplitude
    E3_IR:       float | None = field(default=None) # optional third Gaussian centre (eV)
    sigma_g3_IR: float = field(default=0.0)
    A3_IR:       float = field(default=0.0)


    def __post_init__(self):

        self._validate_osl()
        self._validate_fill()
        self._validate_dose_time()

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

        self._build_cross_sections()

    def _build_cross_sections(self):
        self._sigma_IR_fn = CrossSection.build(
            self.xsec_model_IR, sigma0=self.sigma_IR, E0=self.E0_IR,
            m_eff=self.m_eff_IR,
            E1=self.E1_IR, sigma_g1=self.sigma_g1_IR, gamma_l1=self.gamma_l1_IR,
            A1=self.A1_IR, E2=self.E2_IR, sigma_g2=self.sigma_g2_IR, A2=self.A2_IR,
            E3=self.E3_IR, sigma_g3=self.sigma_g3_IR, A3=self.A3_IR)
        self._sigma_blue_fn = CrossSection.build(
            self.xsec_model_blue, sigma0=self.sigma_blue, E0=self.E0_blue,
            m_eff=self.m_eff_blue)
        self._E_IR = (photon_energy_eV(self.lambda_IR * 1e-9)
                      if self.lambda_IR is not None else None)
        self._E_blue = (photon_energy_eV(self.lambda_blue * 1e-9)
                        if self.lambda_blue is not None else None)

    
    def _validate_fill(self):
        mode = str(self.fill_mode).lower()
        if mode not in ("uniform", "spatial"):
            raise ValueError(
                f"Unknown fill_mode '{self.fill_mode}'. Available: 'uniform', 'spatial'.")
        self.fill_mode = mode

        if self.mu_fill is None:
            self.mu_fill = self.mu
        if mode == "spatial":
            if self.mu_fill is None:
                raise ValueError(
                    "fill_mode='spatial' requires a carrier spread range: set "
                    "'mu_fill' explicitly, or set 'mu' to inherit it.")
            if self.mu_fill <= 0:
                raise ValueError(f"mu_fill must be positive, received {self.mu_fill}.")

    def _validate_dose_time(self):
        # Duration comes from _temp, time unit is determined by temp history

        duration = getattr(self, "duration", None)
        if duration is None:
            return
        if self.dose_time is None:
            self.dose_time = duration
            return
        unit = getattr(self, "unit", "s")
        if unit != "s":
            self.dose_time *= time_to_seconds[unit]   # same unit as temp.duration
        if self.dose_time > duration:
            raise ValueError("Dose duration cannot exceed simulation time!")

    def _validate_osl(self):
        if self.mode == "OSL":
            if self.s is None:
                raise ValueError("OSL mode requires the escape frequency 's' to be set.")
            
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

            if ir_on and self.xsec_model_IR.lower() != "const":
                if self.xsec_model_IR.lower() == "baril_voigt":
                    # Intra-centre resonance: centred on E1_IR, not a threshold E0.
                    if self.lambda_IR is None or self.E1_IR is None:
                        raise ValueError(
                            "IR cross-section model 'baril_voigt' requires "
                            "'lambda_IR' (nm) and 'E1_IR' (eV).")
                elif self.lambda_IR is None or self.E0_IR is None:
                    raise ValueError(
                        f"IR cross-section model '{self.xsec_model_IR}' requires "
                        "'lambda_IR' (nm) and 'E0_IR' (eV).")
            if blue_on and self.xsec_model_blue.lower() != "const":
                if self.lambda_blue is None or self.E0_blue is None:
                    raise ValueError(
                        f"Blue cross-section model '{self.xsec_model_blue}' requires "
                        "'lambda_blue' (nm) and 'E0_blue' (eV).")

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

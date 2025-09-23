import numpy as np
from dataclasses import dataclass, field
from src.classes.electron_traps import _electraps
from src.classes.constants import cnst


@dataclass(kw_only=True)
class _time:
    "Private dataclass holding the time parameter"
    time: float = 0 

    def timestep(self,dt):
        self.time += dt

@dataclass(kw_only=True)
class _bs:
    "Contains the tunnelling and Escape frequency"
    b : float = field(default=1e10) # attmpt to tunnel frequency 
    s : float = field(default=1e10) # Escape frequency
@dataclass(kw_only=True)
class _temp(_time):
    "Private temperature class that is built as a function of time"
    T_init : float = field(default=273.15) #Initial temperature
    dT : float = field(default=5) #Heating rate
    
    @property
    def T(self):
        """Calculates the temperature"""
        return self.T_init + self.time*self.dT
@dataclass(kw_only=True)
class _density:
    "Contains the density parameters"
    rho : float = field(default=None)
    urho : float = field(default=None)

    def set_rho(self,rho):
        self.rho = rho
        
    def set_urho(self,urho):
        self.urho = urho 

    def urho_from_rho(self,alpha):
        self.urho = (4*np.pi* self.rho/3)/np.power(alpha,3)

    def rho_from_urho(self,alpha):
        self.rho = self.urho*np.power(alpha,3)*(3/(np.pi*4))

@dataclass(kw_only=True) 
class _dose(_electraps):
    "Contains "
    D0: float = field(default=None) # Characteristic does
    D_dot : float = field(default=None) # Radition per second

    @property
    def fill(self):
        """Filling rate based on number of electrons and traps"""
        if self.n_trap == self.n_el:
            return 1e20
        else:
            return self.D0/(self.D_dot*(self.n_trap - self.n_el))

@dataclass(kw_only=True)
class _cndctBand(_bs, _temp):
    E_cb: float = field(default=None) # Conduction band energy

    @property
    def deloc_decay(self):
        """Recombination rate with the conduciton band"""
        return self.s * (np.exp(-self.E_cb)-np.exp(cnst.k_b_ev*self.T))

@dataclass(kw_only=True)
class _thermal(_bs,_temp,_density,_electraps):
    E_loc: float # Energy barrier height   
    alpha: float = field(default=None)

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = self.set_alpha()
        if self.rho is not None and self.urho is None:
            self.urho_from_rho(self.alpha)
        elif self.urho is not None and self.rho is None:
            self.rho_from_urho(self.alpha)

    def set_alpha(self):
        """Square tunneling potential"""
        alpha = 2*np.sqrt(2*cnst.m_e*self.E_loc*cnst.ev_to_j)/ cnst.h_bar
        return alpha 
    
    @property
    def p(self):
        return np.exp(-self.E_loc/(cnst.k_b_ev*self.T))
    
    def p_array(self,T):
        return np.exp(-self.E_loc/(cnst.k_b_ev*T))

    
    @property
    def tunn_decay(self):
        """Rate of decay by tunnelling from excited state"""
        return self.b *np.exp(-self.alpha*self.r)*self.p
    
    @property
    def fading_rate(self):
        return 1/self.tunn_decay
    
    
@dataclass(kw_only=True) 
class _thermal_cndct(_thermal,_cndctBand):

    @property
    def fading_rate(self):
        return 1/(self.tunn_decay+self.deloc_decay)

@dataclass(kw_only=True)
class Thermal(_thermal):

    @property
    def fill(self):
        return 1e40 
    
@dataclass(kw_only=True)
class ThermalConductionBand(_thermal_cndct):

    @property
    def fill(self):
        return 1e40 
    
@dataclass(kw_only=True)
class ThermalDose(_thermal,_dose):
    pass
@dataclass(kw_only=True)
class ThermalConductionDose(_thermal_cndct,_dose):
    pass                        



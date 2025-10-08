from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict

from src.classes.electron_traps import _electraps
from src.classes.constants import cnst
from src.classes.temperature.temp_profile_class import TimeTempProfile, ArrayLike
import src.classes.temperature.temp_profiles

@dataclass(kw_only=True)
class _time:
    """Private dataclass holding the time parameter"""
    time: float = 0 

    def timestep(self,dt) -> None:
        """Moves time forward by dt"""
        self.time += dt
@dataclass(kw_only=True)
class _bs:
    """Contains the tunnelling and Escape frequency"""
    b : float = field(default=1e10) # attmpt to tunnel frequency 
    s : float = field(default=1e10) # Escape frequency

@dataclass(kw_only=True)
class _temp(_time, TimeTempProfile):
    """Private temperature class that is built as a function of time"""
    kind: str = field(default="constant")
    TPkwargs: Dict[str,Any] 
    
    def __post_init__(self) -> None:
        """  Intialise the TimeTempProfile """   
        TimeTempProfile.__init__(self,self.kind,**self.TPkwargs)
    
    @property
    def T(self) -> ArrayLike:
        """Calculates the current temperature"""
        return self(self.time)
    
    def Tat(self,time: ArrayLike) -> ArrayLike:
        """Calculates the temperature for a given time(s)"""
        return self(time)

# T0=150
# p1 = TimeTempProfile("constant", T0=10,  times=[0.1,0.3], dT_step=[20,50], dT=[200,50],k=10,T_inf=None)
# p2 = TimeTempProfile("step", T0=T0, times=0.2, dT_step=20,k=10,T_inf=None)
# p3 = TimeTempProfile("linear", T0=T0, dT=300,k=10,T_inf=None)
# p7 = TimeTempProfile("steps", T0=T0, times=[0.01,0.1,0.2,0.3], dT_step=[20,20,20,20])
# p4 = TimeTempProfile("linearsteps", T0=T0, times=[0.01,0.1,0.2,0.3], dT_step=[20,30,20,20], dT=[10,50,60,40,100])
# p5 = TimeTempProfile("exponential", T0=T0, T_inf=10, k=10)
# p6 = TimeTempProfile("lineardrops", T0=T0, times=[0.1,0.3], dT_step=[20,50], dT=[200,50])

# t = np.linspace(0, 0.5, 1001)

# plt.plot(t,p1(t),label='constant')
# plt.plot(t,p2(t),label='step')
# plt.plot(t,p3(t),label='linear')
# plt.plot(t,p4(t),label='linearsteps')
# plt.plot(t,p5(t),label='exponential')
# plt.plot(t,p6(t),label='lineardrops')
# plt.plot(t,p7(t),label='steps')
# plt.legend()
# plt.show()




@dataclass(kw_only=True)
class _density:
    """Contains the density parameters"""
    rho: float | None = field(default=None)
    urho: float | None = field(default=None)

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
    E_loc: float    
    alpha: float | None = field(default=None)

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



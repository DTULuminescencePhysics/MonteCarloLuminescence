from dataclasses import dataclass, field
from classes.constants import cnst
import numpy as np


@dataclass
class system:

    E_loc:float # Energy barrier height 
    b : float # attmpt to tunnel frequency 
    s : float # Escape frequency
    T_init : float = field(default=273.15) #Initial temperature
    dT : float = field(default=5) #Heating rate
    alpha: float = field(default=None) # tunneling rate constant 
    E_cb: float = field(default=None) # Conduction band energy
    D0: float = field(default=None) # Characteristic does
    D_dot : float = field(default=None) # Radition per second
    rho : float = field(init=False)
    urho : float = field(init=False)

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = self.set_alpha()

    def set_alpha(self):
        """Square tunneling potential"""
        alpha = 2*np.sqrt(2*cnst.m_e*self.E_loc*cnst.ev_to_j)/ cnst.h_bar
        return alpha
    
    def T(self,t):
        """Calculates the temperature"""
        return self.T_init + t*self.dT

    def calc_p(self,T):
        return np.exp(-self.E_loc/(cnst.k_b_ev*T))
    
    def tunn_decay(self,r,T):
        """Rate of decay by tunnelling from excited state"""
        return self.b *(np.exp(-self.alpha*r-(self.E_loc/(cnst.k_b_ev*T))))

    def deloc_decay(self,T):
        """Recombination rate with the conduciton band"""
        if self.E_cb is not None:
            return self.s * (np.exp(-self.E_cb)-np.exp(cnst.k_b_ev*T))
        else:
            return 0 
      
    def fading_rate(self,r,T):
        return 1/(self.tunn_decay(r,T) + self.deloc_decay(T)) 
    
    def filling_rate(self,ne,nt):
        """Filling rate based on number of electrons and traps"""
        return self.D0/(self.D_dot*(nt-ne)) 

    def set_rho(self,rho):
        self.rho = rho 
        self.urho = (4*np.pi* self.rho/3)/np.power(self.alpha,3)

    def set_urho(self,urho):
        self.urho = urho 
        self.rho = self.urho*np.power(self.alpha,3)*(3/(np.pi*4))

    def ur(self, r):
        """Convert distance r (in m) to unitless form"""
        return (np.cbrt((4*np.pi*self.rho)/3)*r)
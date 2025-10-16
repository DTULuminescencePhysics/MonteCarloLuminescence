from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
from src.classes.constants import time_to_seconds
from src.helper_functions import ArrayLike
from src.classes.physics.temperature.temp_profile_class import TimeTempProfile
import src.classes.physics.temperature.temp_profiles


@dataclass
class _time:
    """Private dataclass holding the time parameter"""
    time: float = field(default=0)
    unit: str = field(default='s')
    
    def timestep(self,dt) -> None:
        """Moves time forward by dt"""
        self.time += dt


@dataclass
class _temp(_time, TimeTempProfile):
    """Private temperature class that is built as a function of time"""
   
    kind:    str    = field(default='constant') 
    T0:      float  = field(default=0) 
    celsius: bool             = field(default=True) 
    times:   ArrayLike | None = field(default=None) 
    dT_step: ArrayLike | None = field(default=None) 
    dT:      ArrayLike | None = field(default=None) 
    T_inf:   ArrayLike | None = field(default=None) 
    k:       ArrayLike | None = field(default=None) 
    T_chng:  bool                = field(default=True)
    T:       ArrayLike           = field(init=False,default=0.0)

    
    def __post_init__(self) -> None:
        """  Intialise the TimeTempProfile """
        # super().__post_init__()
        self.unit_celcius_checker()
       
        TimeTempProfile.__init__(self,**vars(self))
        self.T = self(self.time)
     
        super().__post_init__()
    
    def unit_celcius_checker(self) -> None:
        if self.unit != 's': 
            if self.times is not None:
                self.times *= time_to_seconds[self.unit]
            if self.dT is not None:
                self.dT /= time_to_seconds[self.unit]
        if self.celsius:
            self.T0 = self.T0+273.15
            if self.dT_step is not None:
                self.dT_step += 273.15
            if self.T_inf is not None:
                self.T_inf += 273.15


    def set_temperature_profile(self, kind: str, **kwargs) -> None:
        """Sets a new TimeTempProfile"""
        self.kind = kind 
        for k, v in kwargs.items():
            if hasattr(self,k):
                setattr(self,k,v)
        
        self.unit_celcius_checker()
        TimeTempProfile.__init__(self,**vars(self))
        self.T = self(self.time)

    
    def timestep(self, dt) -> None:
        """Moves time forward by dt and updates the 
        Temperature if needed"""
        super().timestep(dt)
        ct = self.T
        temp = self(self.time)
        self.T_chng = True if ct != temp else False
        self.T = temp

  
    
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

from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field
from src.classes.constants import time_to_seconds
from src.helper_functions import ArrayLike
from src.classes.physics.temperature.temp_profile_class import TimeTempProfile
import src.classes.physics.temperature.temp_profiles
import numpy as np

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
   
    kind:    str              = field(default="Other") 
    T0:      float            = field(default=0) 
    celsius: bool             = field(default=True)
    duration:float            = field(default=100)
    times:   ArrayLike | None = field(default=None) 
    temps:   ArrayLike | None = field(default=None) 
    dT:      ArrayLike | None = field(default=None)
    T_chng:  bool             = field(default=True)
    T:       ArrayLike        = field(init=False,default=0.0)

    
    def __post_init__(self) -> None:
        """  Intialise the TimeTempProfile """
        if self.times is not None:
            self.times = np.asarray(self.times,dtype=np.float64)
        if self.temps is not None:
            self.temps = np.asarray(self.temps,dtype=np.float64)

       
        self.unit_celsius_checker()
     
        TimeTempProfile.__init__(self,**vars(self))
        self.T = self(self.time)
        super().__post_init__()
    
    def unit_celsius_checker(self) -> None:
        if self.celsius:
            self.T0 = self.T0+273.15
            if self.temps is not None:
                self.temps+=273.15
           

        if self.times is not None and self.temps is not None and self.kind == "Other":
            if self.temps.size == 2 :
                if self.temps[0] == self.temps[1]: 
                    self.kind = "Constant"
                else: 
                    self.kind = "Linear"
                    self.dT = (self.temps[0]-self.temps[1])/self.duration

        if self.unit != 's':
            self.duration = self.duration *time_to_seconds[self.unit]
            if self.times is not None:
                self.times = self.times * time_to_seconds[self.unit]
            
            if self.dT is not None:
                self.dT = self.dT / time_to_seconds[self.unit]

        if self.kind == "Other":
                e = 1e-13
  
                order = np.argsort(self.times, kind="mergesort")
                self.times = self.times[order]
                self.temps = self.temps[order]

                self.dT = np.zeros((self.times.size-1))
                for i in range(1, self.times.size):
                    if  self.times[i] <=  self.times[i - 1]:
                        self.times[i] =  self.times[i - 1] + e

                    self.dT[i-1] = (self.temps[i-1]-self.temps[i])/(self.times[i]-self.times[i-1])
        
      
        
    def set_temperature_profile(self, kind: str, times: np.ndarray, temps: np.ndarray) -> None:
        """Sets a new TimeTempProfile"""
        self.kind = kind 
        self.times = times 
        self.temps = temps
        self.T0 = self.temps[0]
        self.duration = self.times[-1]
        self.unit_celsius_checker()
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
    




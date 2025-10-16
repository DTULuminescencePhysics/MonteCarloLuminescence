from __future__ import annotations
import numpy as np
from src.classes.physics.crystal import Box 


class ReverseJmpMCMC:

    def __init__(self, crystal: Box, obs: np.ndarray):
        self.crystal = crystal 
        self.obs = obs 
        self.prdct = None
        self.crrnt_T = None 
        self.k = None
        self.T0_c = None 
        self.times_c = None
        self.dT_step_c = None
        self.dT_c = None 



        
    
    def initialise_temperature_profile(self, unit: str, celsius: bool, duration: float) -> None:
        """Function that sets intial temperature profile parameters"""
        self.intialise_k()
        self.crrnt_T={'unit': unit, 
              'celsius': celsius, 
              'kind': 'constant', 
              'T0': self.initialise_T0(), 
              'duration': duration, 
              'times': self.initialise_times(), 
              'dT_step': self.initialise_dT_step(), 
              'dT': self.initialise_dT()}

    def intialise_k(self):
        """Intialises k which is the number of step points but could also be seen as the point the current 
        temperature behaviour changes. """
        return 1

    def initialise_T0(self):
        """Intialises the starting temperature"""
        return 1
    
    def initialise_times(self):
        """Intialises the time the k changes occur"""
        return 1
    
    def initialise_dT_step(self):
        """Intialises the size of the k steps down in temperature"""
        return 1
    
    def initialise_dT(self):
        """Intialises the gradient(s) of the k+1 sections of linear change"""
        return 1

    
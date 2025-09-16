from dataclasses import dataclass, field

import numpy as np
from classes.constants import mp
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

@dataclass
class box:
    Height: float = field(default=50*mp.ang)
    Width: float = field(default=50*mp.ang)
    Length: float = field(default=50*mp.ang)
    boundary_factor: float = field(default=1.5)
    n_el : int = field(init=False, repr=False) # Number of electrons
    n_trap : int = field(init=False, repr=False) # Number of electron traps

    electrons : np.ndarray = field(init=False, repr=False) # Store of electron coordinates
    traps     : np.ndarray = field(init=False, repr=False) # Store of trap coordinates
    distances : np.ndarray = field(init=False, repr=False) # Matrix of all distances between electrons and traps
    r         : np.ndarray = field(init=False, repr=False) # Store of the nearest neighbour distances in same order as electrons

    @property
    def dims(self):
        """Returns the dimensions of the crystal"""
        return self.Length, self.Width, self.Height
    
    @property
    def bdims(self):
        """Returns the dimensions of the crystal with boundary padding"""
        return self.Length*self.boundary_factor, self.Width*self.boundary_factor, self.Height*self.boundary_factor
    
    @property 
    def volume(self):
        """Returns the crystal volume"""
        return self.Length*self.Width*self.Height
    
    @property 
    def dvolume(self):
        """Returns the crystal volume"""
        return self.Length*self.Width*self.Height*self.boundary_factor**3

    def initialise_electrons(self,n):
        """Intialises n free electrons in the crystal"""
        self.electrons = np.random.rand(int(n), 3)*self.dims
        self.n_el = n

    def initialise_traps(self,n):
        """Intialises n (altered to ensure similar density at boundaries) electron traps
        in the crystal"""
        tot = int(n * self.boundary_factor**3)
        self.traps = np.random.rand(tot, 3)*self.bdims
        self.n_trap = tot

    def nearest_neighbour(self):
        """Computes the complete distnaces matrix and finds the nearest neighbours r"""
        self.distances = cdist(self.electrons,self.traps,'euclidean')
        self.recalculate_nearest_neighbour()

    def initialise_el_tr(self,ne,nt):
        self.initialise_electrons(ne)
        self.initialise_traps(nt)
        self.nearest_neighbour()

    def add_electron(self):
        """Adds a single electron and hole to the crystal"""
        # Generate new electron and trap
        new_e = np.random.rand(1, 3)*self.dims  
        new_t = np.random.rand(1, 3)*self.bdims
        # Compute the row and columns of the dsiance matrix
        new_row = cdist(new_e,self.traps,'euclidean')
        new_col = cdist(self.electrons,new_t,'euclidean')
        new_elm = cdist(new_e,new_t)
        # Add the new new row and column to the distance matrix
        self.distances = np.hstack((self.distances,new_col))
        full_new_row = np.hstack((new_row,new_elm))
        self.distances = np.vstack((self.distances,full_new_row))
        # Add the electron and trap to the relevant variable
        self.electrons = np.vstack((self.electrons,new_e))
        self.traps = np.vstack((self.traps,new_t))
        self.recalculate_nearest_neighbour()
        # Increase the number of electrons and traps by 1
        self.n_el +=1 
        self.n_trap +=1

    def remove_electron(self,elec_i):
        """Removes a single electron and hole to the crystal"""
        # Get the index of the trap that's nearest neighbour is electron
        # indexed, elec_i    
        trap_i = self.t_index[elec_i]
        # Remove electron and trap indexed at ne and nt respectively 
        self.electrons = np.delete(self.electrons,elec_i, axis=0)
        self.traps = np.delete(self.traps,trap_i, axis=0)
        # Delete row and column from the distances matrix
        self.distances = np.delete(self.distances,elec_i,axis=0)
        self.distances = np.delete(self.distances,trap_i,axis=1)
        # Delete the nearest neighbour distances for ne and nt
        self.r = np.delete(self.r,elec_i)
        self.t_index[self.t_index > elec_i] -= 1
        # self.recalculate_nearest_neighbour()
        self.n_el -=1 
        self.n_trap -=1

    def recalculate_nearest_neighbour(self):
        """Finds a new set of nearest neighbour distances and stores the 
        trap indices"""
        # Get indices of nearest neighbour lengths in the distance matrix
        e_index, t_index = linear_sum_assignment(self.distances)
       
        # Reorder so electron indices run from 0 to n 
        sorted_e_index = np.argsort(e_index)
        ei = e_index[sorted_e_index]
        ti = t_index[sorted_e_index]
        # Form new nearest neighbour matrix
        self.r = self.distances[ei,ti]
        self.t_index = ti
  

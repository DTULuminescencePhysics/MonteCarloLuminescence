from dataclasses import dataclass
from classes.constants import cnst
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

@dataclass(frozen=True)
class system_parameters:

    alpha: float # tunneling rate constant (1/Å)
    b : float # attmpt to tunnel frequency (s⁻¹)
    s : float # Escape frequency (s⁻¹)
    E_cb: float # Conduction band energy (eV)
    E_loc:float # Energy barrier height (eV)
    
    D0: float 
    D_dot : float



@dataclass
class crystal:
    Height: float
    Width: float
    Length: float
    boundary_factor: float
    n_0 : int 
    trap_0 : int


    def initialise_electrons(self):
        self.electrons = np.random.rand(self.n_0, 3)
        self.electrons[:, 0] *= self.Length
        self.electrons[:, 1] *= self.Width
        self.electrons[:, 2] *= self.Height

    def initialise_traps(self):
        self.traps = np.random.rand(self.trap_0, 3)
        self.traps[:, 0] *= self.Length
        self.traps[:, 1] *= self.Width
        self.traps[:, 2] *= self.Height

    def nearest_neighbours(self):
        # tree = cKDTree(self.electrons)
        distance_matrix = cdist(self.traps,self.electrons,'euclidean')
        # distances = np.min(distance_matrix,axis=1)
        trap_indices,tree_indices = linear_sum_assignment(distance_matrix)
        distances = distance_matrix[trap_indices,tree_indices] 
        # return matched_distances
        # distances, indices = tree.query(self.traps, k=1)
        return distances #, matched_distances*1e10
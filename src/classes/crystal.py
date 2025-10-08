from __future__ import annotations
from math import ceil 
import numpy as np
from src.classes.constants import mp

from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from src.classes.electron_traps import _electraps

class box_2:
    """contains the crystal that will perform the Monte Carlo calculation
    
    param self.rng:          Used to seed random number generations
    type  self.rng:          numpy.random.Generator

    param self.h:            number of units cells the crysal is in height
    type  self.h:            int

    param self.w:            Number of units cells the crysal is in width
    type  self.w:            int

    param self.l:            Number of units cells the crysal is in length
    type  self.l:            int

    param self.height:       Height of crystal in m
    type  self.height:       float

    param self.width:        width of crystal in m
    type  self.width:        float

    param self.length:       length of crystal in m
    type  self.length:       float

    param self.cell_dims:    (3 x 1) array containing the height, width and length of a unit cell in m 
    type  self.cell_dims:    numpy.ndarray[float]

    param self.h_pad:        Number of unit cells used to pad one side of crystal height
    type  self.h_pad:        int 

    param self.w_pad:        Number of unit cells used to pad one side of crystal width
    type  self.w_pad:        int

    param self.l_pad:        Number of unit cells used to pad one side of crystal length
    type  self.l_pad:        int

    param self.h_full:       Number of unit cells the full crystal with boundary padding has in height
    type  self.h_full:       int

    param self.w_full:       Number of unit cells the full crystal with boundary padding has in width
    type  self.w_full:       int

    param self.l_full:       Number of unit cells the full crystal with boundary padding has in length
    type  self.l_full:       int

    param self.volume:       volume of crystal in m^3
    type  self.volume:       float
    
    param self.N:            Number of possible trapping sites 
    type  self.N:            int

    param self.HN:           Number of possible electron holes
    type  self.HN:           int

    param self.trap_coords:  (self.N x 1) array containing the flattened index of all trap positions
    type  self.trap_coords:  numpy.ndarray[int]

    param self.hole_coords:  (self.N x 1) array containing the flattened index of all hole positions
    type  self.hole_coords:  numpy.ndarray[int]     

    param self.nearest:      (self.N x 5) array containing the indices of the 5 nearest holes to trap of index i
    type  self.nearest:      numpy.ndarray[int]

    param self.precise_trap: (self.N,3) array of precise location within the unit cell all traps are located, stored as a 32bit integer to reduce memory
    type  self.precise_trap: numpy.ndarray[int]
    
    param self.precise_hole: (self.N,3) array of precise location within the unit cell all holes are located, stored as a 32bit integer to reduce memory
    type  self.precise_hole: numpy.ndarray[int]

    param self.t_cnt:        Number of cuurently trapped electrons
    type  self.t_cnt:        int

    param self.h_cnt:        Number of holes currently available i.e. able to accept a trapped electron
    type  self.h_cnt:        int

    param self.occ_trap:     (self.N x 1) array 1s and 0s used to indicate available traps
    type  self.occ_trap:     numpy.ndarray[uint8]

    param self.occ_hole:     (self.N x 1) array of 1s and 0s used to indicate available holes
    type  self.occ_hole:     numpy.ndarray[uint8]

    param self.dist:         (self.N x self.NH) array of distances between all possible traps and holes
    type  self.dist:         numpy.ndarray[float]

    param self.d:            (self.t_cnt x 1) array of the nearest neighbour distances between available traps and holes
    type  self.d:            numpy.ndarry[float] 


    """
    def __init__(self, unit_h:float, unit_w:float, unit_l: float, 
                 rho:float, dimension: float = (7.5*mp.n)):

        self.rng = np.random.default_rng(12345) # Random number generator
        self.cell_dims = np.array((unit_h,unit_w,unit_l)) #Unit cell of
        self.h = ceil(dimension/unit_h)
        self.height = self.h*unit_h

        self.w = ceil(dimension/unit_w)
        self.width = self.w*unit_w

        self.l = ceil(dimension/unit_l)
        self.length = self.l*unit_l
        self.volume = self.height*self.width*self.length
        self.N = self.NumberOfTraps(rho)
        self.lattice_setup(rho)
        self.occ_trap = np.zeros(self.N,dtype=np.uint8)
        self.occ_hole = np.zeros(self.HN,dtype=np.uint8)
        self.t_cnt = 0 
        self.h_cnt = 0 

    def NumberOfTraps(self,rho: float):
        return round(rho*self.volume)

    
    def topk_manhattan(self,traps, holes, k=20):
        """Finds the k nearest holes that are not necerssarily 
        unique based upon the manhattan distance"""
        idxs = np.empty((self.N, k), dtype=np.uint16)
        for i, p in enumerate(traps):
            d = np.abs(holes - p)*self.cell_dims
            d = d.sum(axis=1)
            part = np.argpartition(d, k-1)[:k]
            order = np.argsort(d[part])
            best = part[order]
            idxs[i] = best
                      
        return idxs
    
    def lattice_setup(self, rho: float):
        """Function that generates the trape and hole locations
        and then creates a distance matrix to store them in."""
        lattice = np.zeros((self.h,self.w,self.l),dtype=np.uint8)
        total = lattice.size
        location = np.random.choice(total,size=self.N,replace=False)
        coords = np.unravel_index(location,lattice.shape)
        lattice[coords] = 1
        
        self.h_pad = ceil((0.5*self.h)/2.)
        self.w_pad = ceil((0.5*self.w)/2.)
        self.l_pad = ceil((0.5*self.l)/2.)
        self.h_full = self.h + 2*self.h_pad
        self.w_full = self.w + 2*self.w_pad 
        self.l_full = self.l + 2*self.l_pad
        pad_lattice = np.pad(lattice,pad_width=((self.h_pad,self.h_pad),(self.w_pad,self.w_pad),(self.l_pad,self.l_pad)),
                             mode='constant',constant_values=0)
        trap_coords = np.argwhere(pad_lattice == 1)
        
        total = pad_lattice.size
        self.HN = self.NumberofHoles(rho)

        location = np.random.choice(total,size=self.HN,replace=False)
        hole_location = np.unravel_index(location,pad_lattice.shape)
        hole_location = np.column_stack(hole_location)

        self.nearest = self.topk_manhattan(trap_coords,hole_location)
        self.set_hole_trap_precise_locations()
        self.create_distance_matrix(trap_coords,hole_location)

        self.trap_coords = self.xyz_coords_to_flat(trap_coords, self.h_full, self.w_full, self.l_full)
        self.hole_coords = self.xyz_coords_to_flat(hole_location, self.h_full, self.w_full, self.l_full)
        self.lattice_dim = np.array((self.h_full,self.w_full,self.l_full))

    def NumberofHoles(self,rho):
        """Returns the number of holes to generate with the boundary padding"""
        return round(self.h_full*self.w_full*self.l_full*np.prod(self.cell_dims)*rho)
                     
    def xyz_coords_to_flat(self, arr: np.ndarray, h: int, w: int, l: int):
        """Flattens array of xyz coordiantes into single list""" 
        flat_coords = np.ravel_multi_index(arr.T, (h, w, l))
        return flat_coords.astype(np.uint32)
       
    def flat_to_xyz_coords(self, flat_coords: np.ndarray, h: int, w: int, l: int):
        """Takes flattened coordinate list and converts them back to 
        xyz coordinates"""
        coords = np.array(np.unravel_index(flat_coords, (h,w,l))).T 
        return coords

    def flat_i_to_xyz(self,flat_coord: np.ndarray, h: int, w: int, l: int):
        """Takes single flat coordinate and converts it back to xyz"""
        return np.unravel_index(flat_coord, (h,w,l))

    def xyz_to_flat(self, arr: np.ndarray):
        """Turns an array into arry of flat bits"""
        bits = np.packbits(arr.reshape(-1)).astype(np.uint8)
        return bits

    def flat_to_xyz(self, bits: np.ndarray, h: int, w: int, l: int):
        """Recovers binary 3D arry from flattened bits"""
        flat = np.unpackbits(bits)[:h*w*l]
        arr = flat.reshape(h, w, l).astype(bool)
        return arr

    def flat_to_xyz_occupied(self, bits: np.ndarray, h: int, w: int, l: int):
        """Returns the xyz coordinates of non-zero elements of a 3D array 
        that has been flattened into a bit representation"""
        flat = np.unpackbits(bits)[:h*w*l]
        flat_indices = np.flatnonzero(flat)
        coords = np.array(np.unravel_index(flat_indices,(h, w, l))).T 
        return coords

    def set_hole_trap_precise_locations(self):
        points = self.rng.random((self.N,3))
        self.precise_trap = (np.array(points) * (2**32 - 1)).astype(np.uint32)

        points = self.rng.random((self.HN,3))
        self.precise_hole = (np.array(points) * (2**32 - 1)).astype(np.uint32)

    def create_distance_matrix(self,trap_coords,hole_location):
        electrons = (trap_coords+(self.precise_trap.astype(np.float64)/(2**32 - 1)))*self.cell_dims 

        holes = (hole_location+(self.precise_hole.astype(np.float64)/(2**32 - 1)))*self.cell_dims 
        e = electrons[:,None,:]
        h = holes[None,:,:]
        self.dist = np.linalg.norm(e-h,axis=2)
    
    def define_new_d(self):
        """Creates a mask to only consider available holes and traps
        then creates the array of minimum distances"""
        mask = np.ones_like(self.dist,dtype=bool)
        mask[np.ix_(np.flatnonzero(self.occ_trap),np.flatnonzero(self.occ_hole))]=False
        self.d = np.ma.array(self.dist,mask=mask).min(axis=1).compressed()

    def trap_new_electron(self):
        """Function that randomly chooses a new electron trap.
        A index is chosen at random and then the nearest unavailable 
        electron hole is selected and added to the crystal"""
        if self.t_cnt >= self.N:
            return
        avail = np.flatnonzero(self.occ_trap==0)
        t_index = np.random.choice(avail)
        nn = self.nearest[t_index][~np.isin(self.nearest[t_index],np.flatnonzero(self.occ_hole))]
        if len(nn) > 0 : 
            h_index = nn[0]
        else: 
            avail = np.flatnonzero(self.occ_hole==0)
            h_index = np.random.choice(avail)
         
        self.occ_trap[t_index] = 1 
        self.occ_hole[h_index] = 1 
        self.define_new_d()
        self.t_cnt += 1 
        self.h_cnt += 1 

    def remove_electron(self,index: int | np.integer ):
        """Function to remove an electron-hole pair.
        The index from the self.d array of the pair removed 
        is passed into the function. This index is then used to find 
        the corresponding hole index. These are then both removed"""
        avail = np.flatnonzero(self.occ_trap)
        t_index = avail[index]
        h_index = int(np.where(self.dist[t_index,:]==self.d[index])[0])
        self.occ_trap[t_index] = 0 
        self.occ_hole[h_index] = 0
        self.d = np.delete(self.d,index)
        self.t_cnt -= 1 
        self.h_cnt -= 1 

   
# @dataclass(kw_only=True)
# class box(_electraps):
#     Height: float = field(default=50*mp.ang)
#     Width: float = field(default=50*mp.ang)
#     Length: float = field(default=50*mp.ang)
#     boundary_factor: float = field(default=1.5)
    

#     @property
#     def dims(self):
#         """Returns the dimensions of the crystal"""
#         return self.Length, self.Width, self.Height
    
#     @property
#     def bdims(self):
#         """Returns the dimensions of the crystal with boundary padding"""
#         return self.Length*self.boundary_factor, self.Width*self.boundary_factor, self.Height*self.boundary_factor
    
#     @property 
#     def volume(self):
#         """Returns the crystal volume"""
#         return self.Length*self.Width*self.Height
    
#     @property 
#     def dvolume(self):
#         """Returns the crystal volume"""
#         return self.Length*self.Width*self.Height*self.boundary_factor**3

#     def initialise_electrons(self,n):
#         """Intialises n free electrons in the crystal"""
#         self.electrons = np.random.rand(int(n), 3)*self.dims
#         self.n_el = n

#     def initialise_traps(self,n):
#         """Intialises n (altered to ensure similar density at boundaries) electron traps
#         in the crystal"""
#         tot = int(n * self.boundary_factor**3)
#         self.traps = np.random.rand(tot, 3)*self.bdims
#         self.n_trap = tot

#     def nearest_neighbour(self):
#         """Computes the complete distnaces matrix and finds the nearest neighbours r"""
#         self.distances = cdist(self.electrons,self.traps,'euclidean')
#         self.recalculate_nearest_neighbour()

#     def initialise_el_tr(self,ne,nt):
#         self.initialise_electrons(ne)
#         self.initialise_traps(nt)
#         self.nearest_neighbour()

#     def add_electron(self):
#         """Adds a single electron and hole to the crystal"""
#         # Generate new electron and trap
#         new_e = np.random.rand(1, 3)*self.dims
#         new_row = cdist(new_e,self.traps,'euclidean')
#         # if self.n_el == self.n_trap:  
#         new_t = np.random.rand(1, 3)*self.bdims
#         new_col = cdist(self.electrons,new_t,'euclidean')
#         new_elm = cdist(new_e,new_t)
#             # Add the new new row and column to the distance matrix
#         self.distances = np.hstack((self.distances,new_col))
#         full_new_row = np.hstack((new_row,new_elm))
#         self.traps = np.vstack((self.traps,new_t))
#         # else:
#             # full_new_row = new_row
#         self.distances = np.vstack((self.distances,full_new_row))
#         # Add the electron and trap to the relevant variable
#         self.electrons = np.vstack((self.electrons,new_e))
     
           
#         self.recalculate_nearest_neighbour()
#         # if self.n_el == self.n_trap: 
#         self.n_trap +=1
        
#         self.n_el +=1 
       

#     def remove_electrons(self, elec_idx):
#         """
#         Remove multiple electrons (and their paired traps) given by indices in `elec_idx`.
#         Mirrors the single-removal behavior but batch-optimized and RAM-friendly.
#         Assumes:
#         - self.t_index[e] gives the trap index paired to electron e
#         - self.electrons shape: (n_el, ...)
#         - self.traps     shape: (n_trap, ...)
#         - self.distances shape: (n_el, n_trap)
#         - self.r         shape: (n_el,)
#         """
#         import numpy as np

#         # --- sanitize indices ---
#         if np.isscalar(elec_idx):
#             elec_idx = np.array([int(elec_idx)], dtype=int)
#         else:
#             elec_idx = np.asarray(elec_idx, dtype=int).ravel()

#         if elec_idx.size == 0:
#             return

#         n_el_before  = self.electrons.shape[0]
#         n_tr_before  = self.traps.shape[0]

#         # clip invalid indices (optional: raise instead)
#         if np.any((elec_idx < 0) | (elec_idx >= n_el_before)):
#             raise IndexError("Some electron indices are out of bounds.")

#         # unique, sorted (so masks are deterministic)
#         elec_idx = np.unique(elec_idx)

#         # traps paired with those electrons
#         traps_idx = np.unique(self.t_index[elec_idx])

#         # --- build keep masks ---
#         keep_e = np.ones(n_el_before,  dtype=bool)
#         keep_t = np.ones(n_tr_before,  dtype=bool)
#         keep_e[elec_idx] = False
#         keep_t[traps_idx] = False

#         # --- apply removals (rows/cols) ---
#         # electrons & r
#         self.electrons = self.electrons[keep_e, :]
#         self.r         = self.r[keep_e]

#         # traps
#         self.traps     = self.traps[keep_t, :]

#         # distances: drop removed electron rows and trap columns
#         self.distances = self.distances[keep_e, :][:, keep_t]

#         # --- update t_index for remaining electrons ---
#         # 1) drop entries for removed electrons
#         t_index_kept = self.t_index[keep_e]

#         # 2) remap old trap indices -> new trap indices after column deletions
#         #    (no NN recompute; just reindexing)
#         old_to_new = np.full(n_tr_before, -1, dtype=int)
#         old_to_new[np.nonzero(keep_t)[0]] = np.arange(keep_t.sum(), dtype=int)

#         new_t_index = old_to_new[t_index_kept]   # removed-trap refs become -1

#         self.t_index = new_t_index

#         # If you want to mimic the original "no recompute" exactly,
#         # you can leave -1s as-is. Otherwise, you might trigger a recompute:
#         # if (new_t_index < 0).any():
#         #     self.recalculate_nearest_neighbour()

#         # --- counts ---
#         removed_el   = np.count_nonzero(~keep_e)
#         removed_trap = np.count_nonzero(~keep_t)
#         self.n_el   -= removed_el
#         self.n_trap -= removed_trap


#     # def remove_electron(self,elec_i):
#     #     """Removes a single electron and hole to the crystal"""
#     #     # Get the index of the trap that's nearest neighbour is electron
#     #     # indexed, elec_i    
#     #     trap_i = self.t_index[elec_i]
#     #     # Remove electron and trap indexed at ne and nt respectively 
#     #     self.electrons = np.delete(self.electrons,elec_i, axis=0)
#     #     self.traps = np.delete(self.traps,trap_i, axis=0)
#     #     # Delete row and column from the distances matrix
#     #     self.distances = np.delete(self.distances,elec_i,axis=0)
#     #     self.distances = np.delete(self.distances,trap_i,axis=1)
#     #     # Delete the nearest neighbour distances for ne and nt
#     #     self.r = np.delete(self.r,elec_i)
#     #     self.t_index[self.t_index > elec_i] -= 1
#     #     # self.recalculate_nearest_neighbour()
#     #     self.n_el -=1 
#     #     self.n_trap -=1

#     def recalculate_nearest_neighbour(self):
#         """Finds a new set of nearest neighbour distances and stores the 
#         trap indices"""
#         # Get indices of nearest neighbour lengths in the distance matrix
#         e_index, t_index = linear_sum_assignment(self.distances)
       
#         # Reorder so electron indices run from 0 to n 
#         sorted_e_index = np.argsort(e_index)
#         ei = e_index[sorted_e_index]
#         ti = t_index[sorted_e_index]
#         # Form new nearest neighbour matrix
#         self.r = self.distances[ei,ti]
#         self.t_index = ti
  

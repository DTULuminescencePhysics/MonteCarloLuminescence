from __future__ import annotations
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field, fields
from typing import Any, Dict
from math import ceil 
import numpy as np
from src.helper_functions import ArrayLike
from src.classes.physics.time_temperature import _temp 
from src.classes.constants import mp, cnst
from scipy.spatial.distance import cdist

from src.classes.physics.set_system import _ThermalParameters
from src.classes.physics.transition_process import EVENT_CODES 

@dataclass
class Box(_temp,_ThermalParameters):

    uc_h: float = field(default=8*mp.ang)
    uc_w: float = field(default=8*mp.ang)
    uc_l: float = field(default=8*mp.ang)
    dimension: float = field(default=7.5*mp.n)

    unit_cell_dims: np.ndarray =field(init=False)
    t_cnt: int = field(init=False) 
    h_cnt: int = field(init=False) 
   
    
    h: np.ndarray = field(init=False)
    w: np.ndarray = field(init=False)
    l: np.ndarray = field(init=False)
    volume: np.ndarray = field(init=False)
   
    h_pad: int = field(init=False)
    w_pad: int = field(init=False)
    l_pad: int = field(init=False)

    N: int = field(init=False)
    HN: int = field(init=False)
    occ_trap: np.ndarray = field(init=False)
    occ_hole: np.ndarray = field(init=False)
    trap_coords: np.ndarray = field(init=False)
    hole_coords: np.ndarray = field(init=False)
    nearest: np.ndarray = field(init=False)
    dist: np.ndarray = field(init=False)
    dist_ee: np.ndarray = field(init=False)
    d: np.ndarray = field(init=False)
    d_ee: np.ndarray = field(init=False)
    fill_time: ArrayLike = field(init=False)
    fill: ArrayLike = field(init=False)
    exec_time: ArrayLike = field(init=False)
    exec_index: int = field(init=False)
    event_bool: bool = field(init=False,default=False)
    rng: np.random.Generator = field(init=False)
    F1: np.ndarray = field(init=False)
    F2: np.ndarray = field(init=False)
    E_loc_array: np.ndarray = field(init=False)   # per-trap E_loc, shape (N,)
    E_cb_array:  np.ndarray = field(init=False)   # per-trap E_cb,  shape (N,)
    E_loc_occ:   np.ndarray = field(init=False)   # E_loc for occupied traps, shape (t_cnt,)
    E_cb_occ:    np.ndarray = field(init=False)   # E_cb  for occupied traps, shape (t_cnt,)
    R_tun: float|None = field(default=0)          # Retrapping cross section versus recombination of tunnelling
    R_CB:  float|None = field(default=1)          # Retrapping cross section versus recombination of conduction band
    retrap_mask_factor: float = field(default=0.8)
    boundary: str = field(default="padded")
    combine_when_fill: bool = field(default=False)
    recom_pre_fill: float = field(default=1.0)
    event_code: int = field(init=False, default=0)


    def __repr__(self):
        if self.kind == "constant":
            temp = f"at a constant temperature of : {self.T0} (K)\n"
        elif self.kind == "linear":
            temp = (f"with a linear temperature profile starting at : {self.T0} (K) \n"
                       f"and increasing at a rate of : {self.dT} (K/s)\n")
        else:
            temp = (f"with a multiple linear step temperature profile starting at : {self.T0} (K) \n")
                    #    f"with steps of : {self.dT_step} (K) at times : {self.times} (s) \n"
                    #    f"and linear rates of : {self.dT} (K/s)\n")
        string = ("Crystal Information: \n"
                  f"Unit cell dimensions (h,w,l) : {self.unit_cell_dims} (m)\n"
                  f"represented by an array of (h,w,l) : ({self.h[1]},{self.w[1]},{self.l[1]})\n"
                  f"with a total volume of : {self.volume[1]} (m^3)\n"
                  f"containing {self.N} traps and {self.HN} holes\n"
                  f"with a density of : {self.rho} (m^-3) or {self.urho} (unitless)\n"
                  f"The ground state to excited state energy gap is : {self.E_loc} (eV)\n"
                  f"and the conduction band gap is : {self.E_cb} (eV).\n"
                  f"Tunnelling frequency is : {self.b} (s^-1)\n"
                  f"and the tunneling rate constant is : {self.alpha_ES} (m^-1).\n"
                  f"The escape frequecy is : {self.s} (s^-1).\n"
                  f"The crystal is dosed at rate of : {self.D_dot} (Gy/s) \n"
                  f"with a characteristic dose of : {self.D0} (Gy)\n"
                  f"{temp}")
        
        return string
    
    @classmethod
    def from_config(cls, cfg: DictConfig, T_override: dict | None = None) -> "Box":
        """Create a ``Box`` instance from a Hydra ``DictConfig`` object."""
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"Expected DictConfig, received {type(cfg).__name__}")

        resolved = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(resolved, dict):
            raise ValueError("Resolved configuration must be a mapping")

        physics_cfg = resolved.get("physics", {})
        box_cfg = resolved.get("box", {})
        mc_cfg = resolved.get("mc", {})
        if T_override is None:
            temp_cfg = resolved.get("temp",{})
        else:
            temp_cfg = T_override

        init_fields = {f.name for f in fields(cls) if f.init}
        array_fields = {"times", "dT_step", "dT", "T_inf", "k"}

        kwargs: Dict[str, Any] = {}
        for section in (physics_cfg, temp_cfg, box_cfg, mc_cfg, resolved):
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if key not in init_fields or value is None:
                    continue
                if key in array_fields and isinstance(value, (list, tuple)):
                    kwargs[key] = np.asarray(value, dtype=float)
                else:
                    kwargs[key] = value
        return cls(**kwargs)
     

    def __post_init__(self):
        super().__post_init__()
        self.unit_cell_dims = np.array((self.uc_h,self.uc_w,self.uc_l))
        self.h= np.zeros(2)
        self.w= np.zeros(2)
        self.l= np.zeros(2) 
        self.set_dimensions()
      
    def set_dimensions(self, N_number: int = 100) -> None:
        """Set the dimensions of the box"""

        if self.boundary == "periodic":
            self.volume = np.array((self.dimension**3, self.dimension**3))
        else:
            self.volume = np.array((self.dimension**3, (self.dimension*1.5)**3))
        self.NumberofTraps()
        to_add = 1.5e-8
        while self.N < N_number:
            self.dimension += to_add
            if self.boundary == "periodic":
                self.volume = np.array((self.dimension**3, self.dimension**3))
            else:
                self.volume = np.array((self.dimension**3, (self.dimension*1.5)**3))
            self.NumberofTraps()
            to_add -= 1e-10

        self.NumberofHoles()
        
    def NumberofTraps(self)-> None:        
        """Returns the number of traps to generate"""
        self.N = round(self.rho*self.volume[0]) 
    
    def NumberofHoles(self)-> None:
        """Returns the number of holes to generate with the boundary padding"""
        self.HN = round(self.rho*self.volume[1])

    # def topk_manhattan(self,traps, holes, k=20):
    #     """Finds the k nearest holes that are not necerssarily
    #     unique based upon the manhattan distance"""
    #     idxs = np.empty((self.N, k), dtype=np.uint16)
    #     for i, p in enumerate(traps):
    #         d = np.abs(holes - p)*self.unit_cell_dims
    #         d = d.sum(axis=1)
    #         part = np.argpartition(d, k-1)[:k]
    #         order = np.argsort(d[part])
    #         best = part[order]
    #         idxs[i] = best
    #
    #     return idxs
    
    def set_random_generator(self, seed: int | None = None) -> None:
        """Set the random number genreator using the seed. The seed 
        will be a user set value plus the current simulation number"""
        self.rng = np.random.default_rng(seed=seed)

    def lattice_setup(self,seed:int, t_cnt: float, h_cnt: float, t:float = 0.0) -> None:
        """Function that generates the trape and hole locations
        and then creates a distance matrix to store them in."""
        self.set_random_generator(seed)

        # Draw E_loc and E_cb from Gaussian distributions for each trap
        if self.E_loc is not None:
            if self.E_loc_sigma > 0:
                self.E_loc_array = self.rng.normal(self.E_loc, self.E_loc_sigma, self.N)
            else:
                self.E_loc_array = np.full(self.N, self.E_loc)
        else:
            self.E_loc_array = np.zeros(self.N)

        if self.E_cb is not None:
            if self.E_cb_sigma > 0:
                self.E_cb_array = self.rng.normal(self.E_cb, self.E_cb_sigma, self.N)
            else:
                self.E_cb_array = np.full(self.N, self.E_cb)
        else:
            self.E_cb_array = np.zeros(self.N)

        size = np.array((self.dimension,self.dimension,self.dimension))

        if self.boundary == "periodic":
            hole_coords = self.rng.random((self.HN, 3)) * size
            trap_coords = self.rng.random((self.N, 3)) * size
        else:
            hole_coords = (self.rng.random((self.HN,3))*size*1.5)
            trap_coords = (self.rng.random((self.N,3))*size)+(self.dimension*0.25) 

        nn = self.HN

        self.create_distance_matrix(trap_coords,hole_coords)
        self.nearest = np.argsort(self.dist,axis=1)[:,:nn]
      
        self.occ_trap = np.zeros(self.N,dtype=np.uint8)
        self.occ_hole = np.zeros(self.HN,dtype=np.uint8)

        self.t_cnt = int(t_cnt*self.N)
        self.h_cnt = int(h_cnt*self.HN)
   
        if self.t_cnt > 0:
            if self.t_cnt == self.N:
                self.occ_trap[:] = 1
                hole_tot = np.flatnonzero(self.occ_hole).size
                if self.h_cnt == self.HN:
                    self.occ_hole[:] = 1
                    hole_tot = self.h_cnt
            else:
                hole_tot = np.flatnonzero(self.occ_hole).size 
                for _ in range(self.t_cnt):
                    avail = np.flatnonzero(self.occ_trap==0)
                    t_index = self.rng.choice(avail)
                    if hole_tot < self.h_cnt:
                        nn = self.nearest[t_index][~np.isin(self.nearest[t_index],np.flatnonzero(self.occ_hole))]
                        if len(nn) > 0 : 
                            h_index = nn[0]
                        else: 
                            avail2 = np.flatnonzero(self.occ_hole==0)
                            h_index = self.rng.choice(avail2)
                        
                        self.occ_hole[h_index] = 1
                        hole_tot += 1
                    self.occ_trap[t_index] = 1
            
            if hole_tot < self.h_cnt:
                while hole_tot < self.h_cnt:
                    avail2 = np.flatnonzero(self.occ_hole==0)
                    h_index = self.rng.choice(avail2)
                    self.occ_hole[h_index] = 1
                    hole_tot += 1
        elif self.h_cnt > 0:
            if self.h_cnt == self.HN:
                self.occ_hole[:] = 1
            else:
                hole_tot = np.flatnonzero(self.occ_hole).size 
                while hole_tot < self.h_cnt:
                    avail2 = np.flatnonzero(self.occ_hole==0)
                    h_index = self.rng.choice(avail2)
                    self.occ_hole[h_index] = 1
                    hole_tot += 1
        
        self.h_cnt = np.flatnonzero(self.occ_hole).size
        self.define_new_d_full()    # self.define_new_d()
        self.initial_times(t=t)

    # def float64_to_int32(self,points) -> np.ndarray:
    #     return (np.array(points) * (2**32 - 1)).astype(np.uint32)

    # def int32_to_float642(self,points: np.ndarray) -> np.ndarray:
    #     return points.astype(np.float64)/(2**32 - 1)

    # def set_hole_trap_precise_locations(self) -> tuple[np.ndarray,np.ndarray]:
    #     traps = self.rng.random((self.N,3))
    #     holes = self.rng.random((self.HN,3))
    #     return traps, holes

    # def xyz_coords_to_flat(self, arr: np.ndarray, h: int, w: int, l: int) -> np.ndarray:
    #     """Flattens array of xyz coordiantes into single list"""
    #     flat_coords = np.array(np.ravel_multi_index(arr.T, (h, w, l)),dtype=np.uint32)
    #     return flat_coords

    # def flat_to_xyz_coords(self, flat_coords: np.ndarray, h: int, w: int, l: int)-> np.ndarray:
    #     """Takes flattened coordinate list and converts them back to
    #     xyz coordinates"""
    #     coords = np.array(np.unravel_index(flat_coords, (h,w,l))).T
    #     return coords

    # def flat_i_to_xyz(self,flat_coord: np.ndarray, h: int, w: int, l: int)-> np.ndarray:
    #     """Takes single flat coordinate and converts it back to xyz"""
    #     return np.array(np.unravel_index(flat_coord, (h,w,l)))

    # def xyz_to_flat(self, arr: np.ndarray)-> np.ndarray:
    #     """Turns an array into arry of flat bits"""
    #     bits = np.packbits(arr.reshape(-1)).astype(np.uint8)
    #     return bits

    # def flat_to_xyz(self, bits: np.ndarray, h: int, w: int, l: int)-> np.ndarray:
    #     """Recovers binary 3D arry from flattened bits"""
    #     flat = np.unpackbits(bits)[:h*w*l]
    #     arr = flat.reshape(h, w, l).astype(bool)
    #     return arr

    # def flat_to_xyz_occupied(self, bits: np.ndarray, h: int, w: int, l: int)-> np.ndarray:
    #     """Returns the xyz coordinates of non-zero elements of a 3D array
    #     that has been flattened into a bit representation"""
    #     flat = np.unpackbits(bits)[:h*w*l]
    #     flat_indices = np.flatnonzero(flat)
    #     coords = np.array(np.unravel_index(flat_indices,(h, w, l))).T
    #     return coords
    
    # def create_distance_matrix(self,trap_coords: np.ndarray, 
    #                            hole_location: np.ndarray, 
    #                            ptraps: np.ndarray, pholes: np.ndarray) -> None:
    #     electrons = (trap_coords+ptraps)*self.unit_cell_dims 

    #     holes = (hole_location+pholes)*self.unit_cell_dims 
    #     e = electrons[:,None,:]
    #     h = holes[None,:,:]
    #     self.dist = np.linalg.norm(e-h,axis=2)

    def create_distance_matrix(self,trap_coords: np.ndarray,
                               hole_coords: np.ndarray) -> None:

        if self.boundary == "periodic":
            self.dist    = self._min_image_dist(trap_coords, hole_coords)
            self.dist_ee = self._min_image_dist(trap_coords, trap_coords)
        elif self.boundary == "padded":
            self.dist = cdist(trap_coords, hole_coords)
            self.dist_ee = cdist(trap_coords, trap_coords)
        else:
            raise ValueError(f"Invalid boundary condition: {self.boundary}")
        np.fill_diagonal(self.dist_ee, 1e-20)

    def _min_image_dist(self, coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances using the minimum-image
        convention for a cubic box of side length self.dimension."""
        L = self.dimension
        delta = coords_a[:, None, :] - coords_b[None, :, :]
        delta -= L * np.round(delta / L)            # round=0 if delta<L/2, round=1 if delta>L/2, round=-1 if delta<-L/2
        return np.linalg.norm(delta, axis=2)

        # m = self.dist.mean()
        # s = self.dist.std()
        # self.trial= np.max(np.min(self.dist,axis=1)) + s
        # r = 1e-9
        # p = np.exp((-4*np.pi*self.rho*np.power(r,3))/3)*(4*np.pi*self.rho*np.power(r,2))
        # while p > 5:
        #     r *=10
        #     p = np.exp((-4*np.pi*self.rho*np.power(r,3))/3)*(4*np.pi*self.rho*np.power(r,2))
        #     print(p,r)
       
        # print(m,s)
        # print(np.max(np.min(self.dist,axis=1)))
        
        
    # def distance_from_store(self) -> None:
    #     trap_coords = self.flat_to_xyz_coords(self.trap_coords, self.h[1], self.w[1], self.l[1])
    #     hole_coords = self.flat_to_xyz_coords(self.hole_coords, self.h[1], self.w[1], self.l[1])
    #     trap = self.int32_to_float642(self.precise_trap)
    #     hole = self.int32_to_float642(self.precise_hole)
    #     self.create_distance_matrix(trap_coords,hole_coords,trap,hole)

    # def define_new_d(self):
    #     """Creates a mask to only consider available holes and traps
    #     then creates the array of minimum distances"""
    #     row_m = self.occ_trap.astype(bool)
    #     col_m = self.occ_hole.astype(bool)
    #     full = self.dist[row_m][:,col_m]
    #     # self.d = full
    #     if full.size == 0: 
    #         self.d = np.zeros(0)
    #         return
    #     self.d = full.min(axis=1)

        # mask = np.ones_like(self.dist,dtype=bool)
        # mask[np.ix_(np.flatnonzero(self.occ_trap),np.flatnonzero(self.occ_hole))]=False
        # self.d = np.ma.array(self.dist,mask=mask)#.compressed()
        # self.d = np.ma.array(self.dist,mask=mask).min(axis=1).compressed()

    def define_new_d_full(self):
        """Updates the distance matrix between occupied electron trap and
        occupied hole trap pair and the distance between occupied electron trap
        and unoccupied electron trap"""
        occ_idx = np.flatnonzero(self.occ_trap == 1)
        self.E_loc_occ = self.E_loc_array[occ_idx]
        self.E_cb_occ  = self.E_cb_array[occ_idx]

        row_m = self.occ_trap.astype(bool)
        col_m = self.occ_hole.astype(bool)
        full = self.dist[row_m][:,col_m]
        # self.d = full
        if full.size == 0:
            self.d = np.zeros(0)
            return
        self.d = full   # .min(axis=1)

        col_m = np.invert(self.occ_trap.astype(bool))
        self.d_ee = self.dist_ee[row_m][:, col_m]
        if self.d_ee.size == 0:
            self.d_ee = np.zeros(0)
            return

        if self.retrap_mask_factor > 0.0:       # Retrapping is suppressed within a distance threshold
            threshold = self.retrap_mask_factor * self.d.min()
            mask = self.d_ee < threshold

            # If every empty trap in a row is masked, preserve the nearest one.
            all_masked_rows = mask.all(axis=1)    
            if all_masked_rows.any():
                nearest_col = self.d_ee.argmin(axis=1)
                for row_i in np.flatnonzero(all_masked_rows):
                    mask[row_i, nearest_col[row_i]] = False
            self.d_ee = np.where(mask, np.inf, self.d_ee)
        

    def trap_new_electron(self):
        """
        If recombination is allowed in filling, hole and electron annihilation are treated separately
        """
        if not self.combine_when_fill:
            
            if self.t_cnt >= self.N:
                return
            avail_t = np.flatnonzero(self.occ_trap == 0)
            t_index = self.rng.choice(avail_t)
            if self.h_cnt < self.HN:
                avail_h = np.flatnonzero(self.occ_hole == 0)
                h_index = self.rng.choice(avail_h)
                self.occ_hole[h_index] = 1
                self.h_cnt += 1
            self.occ_trap[t_index] = 1
            self.t_cnt += 1
            self.event_code = EVENT_CODES["fill"]
            self.define_new_d_full()
            return

        has_unoccupied_trap = self.t_cnt < self.N
        has_occupied_hole   = self.h_cnt > 0
        has_occupied_trap   = self.t_cnt > 0
        has_unoccupied_hole = self.h_cnt < self.HN

        pool_A_viable = has_unoccupied_trap or has_occupied_hole
        pool_B_viable = has_occupied_trap   or has_unoccupied_hole

        if not pool_A_viable and not pool_B_viable:
            return

        unoccupied_t = np.flatnonzero(self.occ_trap == 0) if has_unoccupied_trap else np.empty(0, int)
        occupied_h   = np.flatnonzero(self.occ_hole == 1) if has_occupied_hole   else np.empty(0, int)
        occupied_t   = np.flatnonzero(self.occ_trap == 1) if has_occupied_trap   else np.empty(0, int)
        unoccupied_h = np.flatnonzero(self.occ_hole == 0) if has_unoccupied_hole else np.empty(0, int)

        if pool_A_viable:
            n_ut     = len(unoccupied_t)
            n_oh     = len(occupied_h)
            n_oh_eff = round(n_oh * self.recom_pre_fill)
            choice   = self.rng.integers(0, n_ut + n_oh_eff)
            if choice < n_ut:
                self.occ_trap[unoccupied_t[choice]] = 1
                self.t_cnt += 1
            else:
                self.occ_hole[occupied_h[(choice - n_ut) % n_oh]] = 0
                self.h_cnt -= 1

        if pool_B_viable:
            n_ot     = len(occupied_t)
            n_uh     = len(unoccupied_h)
            n_ot_eff = round(n_ot * self.recom_pre_fill)
            choice   = self.rng.integers(0, n_ot_eff + n_uh)
            if choice < n_ot_eff:
                self.occ_trap[occupied_t[choice % n_ot]] = 0
                self.t_cnt -= 1
            else:
                self.occ_hole[unoccupied_h[choice - n_ot_eff]] = 1
                self.h_cnt += 1

        self.event_code = EVENT_CODES["fill"]
        self.define_new_d_full()

      
    # def remove_electron(self):
    #     """Function to remove an electron-hole pair.
    #     The index of the electron-hole pair is stored in self.fade_index 
    #     which is found when the times are updated. This index is then used to 
    #     find the corresponding hole index. These are then both removed"""
    #     if self.t_cnt <= 0:
    #         return
      
    #     avail = np.flatnonzero(self.occ_trap)
    #     t_index = avail[self.fade_index]
    #     idx_array = np.where(self.dist[t_index,:]==self.d[self.fade_index])[0]
    #     if idx_array.size == 0:
    #         raise ValueError("No match found for fade_index")
    #     h_index = int(idx_array[0])
       
    #     self.occ_trap[t_index] = 0 
    #     self.occ_hole[h_index] = 0
    #     self.define_new_d()
    #     self.t_cnt -= 1 
    #     self.h_cnt -= 1 

    # def remove_electron(self):
    #     h_index = np.flatnonzero(self.occ_hole)[self.selected_index]
    #     t_index = np.flatnonzero(self.occ_trap)[self.exec_index]
    #     self.occ_hole[h_index] = 0
    #     self.occ_trap[t_index] = 0
    #     h_cnt -= 1
    #     t_cnt -= 1

    # def move_electron(self):
    #     dest_index = np.where(self.occ_trap==0)[0][self.selected_index]
    #     t_index = np.flatnonzero(self.occ_trap)[self.exec_index]
    #     self.occ_trap[dest_index] = 1
    #     self.occ_trap[t_index] = 0

    def operate_electron(self):
        """
        Execute a fading transition for the selected electron.
        Collects rates from all active TransitionProcess objects, selects
        one channel via cumulative probability, and dispatches to its
        execute() method.
        """
        if self.t_cnt <= 0:
            return

        rate_blocks = []
        process_map = []       # list: (process, start_offset, end_offset)
        offset = 0

        for proc in self._processes:
            r = proc.rates(self, self.exec_index)
            if r.size > 0:
                rate_blocks.append(r)
                process_map.append((proc, offset, offset + r.size))
                offset += r.size

        if offset == 0:
            return

        all_rates = np.concatenate(rate_blocks)
        cum_prob = np.cumsum(all_rates) / all_rates.sum()
        idx = int(np.min(np.argwhere(cum_prob >= self.rng.random())))

        for proc, start, end in process_map:
            if idx < end:
                proc.execute(self, self.exec_index, idx - start)
                break

        self.define_new_d_full()

    # def move_electron(self):
    #     """
    #     Function to move an electron to another unoccupied electron trap.
    #     """
    #     if self.t_cnt <= 0:
    #         return
      
    #     avail = np.flatnonzero(self.occ_trap)
    #     orgn_index = avail[self.exec_index]
    #     idx_array = np.where(self.dist[orgn_index,:]==self.d[self.exec_index])[0]
    #     if idx_array.size == 0:
    #         raise ValueError("No match found for fade_index")
    #     dest_index = int(idx_array[0])
       
    #     self.occ_trap[orgn_index] = 0
    #     self.occ_trap[dest_index] = 1
    #     self.define_new_d_full()


    def timestep(self, dt) -> None:
        """Moves time forward by dt and updates the
        Temperature if needed. If the temperature has not changed
        the lifetimes are simply reduced by dt"""
        super().timestep(dt)
        if self.T_chng or self.event_code != 0: # self.event_bool
            self.recalc_times_full()
            self.select_electron()
        else:
            self.fill_time -= dt
            self.exec_time -= dt

    # def recalc_times(self):
    #     """Recalcualtes the lifetimes and fill times"""
    #     self._filltime = self._fill(self.N, self.t_cnt, self.D_dot)
    #     self._lifetimes = self._fade(self.T,self.d)

    def recalc_times_full(self):
        """Recalculates fill time. Transition rates are computed
        on demand by the process objects."""

        if self.D0 == None or self.D_dot == 0 or self.h_cnt >= self.HN:
            self.fill_time = 1e20
        else:
            fill_rate = self._fill(self.N, self.t_cnt, self.D_dot)
            # self._filltime = self.rng.exponential(1/fill_rate)
            self.fill_time = self.rng.exponential(1/fill_rate)

    # def random_fill_fade(self) -> None:
    #     """Generates new random fill and fade times"""
    #     if (self.h_cnt < self.HN) and self._filltime > 0:
    #         self.fill = self.rng.exponential(self._filltime) 
    #     else: 
    #         self.fill = 1e20
      
    #     if self.t_cnt == 0 :
    #         self.fade = 1e20
    #     elif self.t_cnt == 1:
    #         f = self.rng.exponential(self._lifetimes)
    #         self.fade_index = 0
    #         self.h_index = 0
    #     else: 
    #         f = self.rng.exponential(self._lifetimes)
    #         self.fade = np.min(f)
    #         self.fade_index = int(np.argmin(f)) 

    def select_electron(self) -> None:
        """Generate execution time for filling and fading transitions.

        For fading, the overall rate for each occupied electron is
        computed by summing bulk_rates_sum() across all active processes,
        then an exponential random time is drawn for each electron and
        the fastest one is selected.
        """
        # if (self.h_cnt < self.HN) and self.fill_time > 0:
        #     self.fill = self.fill_time
        # else:
        #     self.fill = 1e20

        if self.t_cnt == 0:
            self.exec_time = 1e20
            self.exec_index = None
        else:
            self.F1 = 1. / (1 + np.exp(-self.E_loc_occ / (cnst.k_b_ev * self.T)))
            self.F2 = 1. / (1 + np.exp( self.E_loc_occ / (cnst.k_b_ev * self.T)))

            total_rate = np.zeros(self.t_cnt)
            for proc in self._processes:
                total_rate += proc.bulk_rates_sum(self)

            f = self.rng.exponential(1.0 / total_rate)
            self.exec_time = np.min(f)
            self.exec_index = int(np.argmin(f))

    # def select_fading_event(self) -> None:
    #     """Generates new random transition"""
    #     if (self.h_cnt < self.HN) and self._filltime > 0:
    #         self.fill = self.rng.exponential(self._filltime) 
    #     else: 
    #         self.fill = 1e20
      
    #     if self.t_cnt == 0 :
    #         self.fade = 1e20
    #     elif self.t_cnt == 1:
    #         f = self.rng.exponential(self._lifetimes)
    #         self.fade_index = 0
    #         self.h_index = 0
    #     else: 
    #         f = self.rng.exponential(self._lifetimes)
    #         self.fade = np.min(f)
    #         self.fade_index = int(np.argmin(f)) 

     
    def initial_times(self, t: float = 0.0) -> None:
        """Generates the initial fill and fade times"""
        self.time = t
        self.T = self(self.time)
        self.recalc_times_full()    # self.recalc_times()
        self.select_electron()      # self.random_fill_fade()

    

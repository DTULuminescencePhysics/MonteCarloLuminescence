import sys
import numpy as np
from dataclasses import dataclass, field
from joblib import Parallel, delayed
# from multiprocessing import Pool
import time as tttt
from typing import Type
from src.errors import ErrorOutputHandler
from src.classes.physics import _thermal, Thermal, ThermalConductionBand, ThermalDose, ThermalConductionDose
from src.classes.crystal import box

from src.filesystem import output_monte_carlo_results
@dataclass(kw_only=True)
class MCBase(box):
    duration: float = field(default=160)
    dt_cap: float = field(default=1)
    initial_el: int = field(default=1e2)
    initial_tr: int = field(default=1e2)
    iso: bool = field(default=False) 
    
    store: np.ndarray = field(init=False)
    max_steps: int = field(init=False)
    _lifetime: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__() 
        self.max_steps = int(self.duration/self.dt_cap +1)*5
        if self.Height is None or self.Width is None or self.Length is None:
            self.Height = self.Width = self.Length = np.cbrt(self.initial_el/self.rho)

    @property
    def ur(self):
        """Convert distance r (in m) to unitless form"""
        return (np.cbrt((4*np.pi*self.rho)/3)*self.r)
    
    @property 
    def _recomb_wait(self):
        if not self.iso:
            self._lifetime = self.fading_rate

        return np.random.exponential(self._lifetime)
    
    def find_next_recombination(self):
        if self.n_el > 0:
            return np.min(self._recomb_wait), np.argmin(self._recomb_wait)
        else:
            return 1e30, 0

    def set_start(self,time=0):
        self.time = time 
        self.initialise_el_tr(self.initial_el,self.initial_tr)
        self.store = np.zeros((3,500))
        self.store[1,0] =  self.n_el
        self.store[2,0] = self.n_trap

    def run_simulation(self,rep: int, time=0):# err: ErrorOutputHandler,time=0): 

        self.set_start(time)
        i = 1 
        path = f"rep_{rep}.bin"
        step = 10 #self.dt_cap
        with open(path, "wb") as f:
            while self.time < self.duration: 
                recomb_tim, recomb_index = self.find_next_recombination() 
                dt = min(recomb_tim,self.fill,self.dt_cap)
                # dt = min(self.fill,step)
               
                # if dt == self.fill:
                #     self.add_electron()
                #     step = 10 - dt #self.dt_cap - dt
                # else:
                #     lives = self._recomb_wait - dt 
                #     remov = np.where(lives<1)
                #     # for r in remov:
                #     self.remove_electrons(remov)
                #     step =  10 #self.dt_cap
                if dt == recomb_tim:
                    self.remove_electrons(recomb_index)
                elif dt == self.fill:
                    self.add_electron()
                elif dt == self.dt_cap:
                    pass 

                self.timestep(dt)
                self.store[0,i] = self.time
                self.store[1,i] =  self.n_el
                self.store[2,i] = self.n_trap
                if (i == 499 ):
                    # to_append = np.zeros((3,self.max_steps))
                    # self.store = np.hstack((self.store,to_append))
                    self.store.T.tofile(f)
                    # np.array(self.store.T,dtype=np.float64).tofile(f)
                    self.store = np.zeros((3,500))
                    i = -1
              
                i +=1
            self.store[:,0:i].T.tofile(f)
        # code = output_monte_carlo_results(rep,self.store[:,0:i])
        # return code, rep

    def run_all(self,reps: int, err: ErrorOutputHandler, parl = True, time=0):
       
        start_time = tttt.perf_counter()
        if parl:
            Parallel(n_jobs=4)(delayed(self.run_simulation)(i) for i in range(reps))
        else: 
            code = []; additional = []
            for i in range(0,reps):
                c, a = self.run_simulation(i)
                code.append(c)
                additional.append(a)
        finish_time = tttt.perf_counter()
        err.output(f"Program finished in {finish_time-start_time} seconds")
        # if sum(code) > 0: 
        #     for i in range(len(code)): 
        #         if code[i] == 1:
        #             err.error(f"Error writing Monte Carlo Repetition, {additional[i]} output file",fatal=False)

Physics: dict[str, Type[_thermal]] = {
    "Thermal":   Thermal,
    "ThermalC":  ThermalConductionBand,
    "ThermalD":  ThermalDose,
    "ThermalCD": ThermalConductionDose
}

def make_mc_class(thermal_kind: str, name: str | None = None):
    phys = Physics[thermal_kind]
    cls_name = name or f"MC+{thermal_kind.capitalize()}"
    cls = type(cls_name,(MCBase,phys),{})
    cls = dataclass(kw_only=True)(cls)
    return cls 

def make_mc_instance(thermal_kind: str, **kwargs):
    MC = make_mc_class(thermal_kind)
    return MC(**kwargs)
    

def run_monte_carlo_simulation(phys_in : dict, mc_in: dict, err: ErrorOutputHandler):
    # input1 = {"E_loc": 0.8, 
    #          "s":1e10, 
    #          "rho":8e-4, 
    #          "factor":1e5, 
    #          "z":1.8, 
    #          "b":1e10,
    #          "c1":"black",
    #          "c2":"orange"
    #          }
    # input2 = {"E_loc": 0.8, 
    #          "s":1e10, 
    #          "rho":3e-4, 
    #          "factor":1e4, 
    #          "z":1.8, 
    #          "b":1e10,
    #          "c1":"blue",
    #          "c2":"pink"}
    # input3 = {"E_loc": 1.2, 
    #          "s":1e12, 
    #          "rho":3e-4, 
    #          "factor":2e6, 
    #          "z":1.8, 
    #          "b":1e12,
    #          "c1":"green",
    #          "c2":"grey"}

    # inputs = input1
    if mc_in.therm_type ==  "Thermal":
        MC = make_mc_class("Thermal")
        MonteCarlo = MC(duration=mc_in.duration,dt_cap=mc_in.max_dt,
                initial_el=mc_in.n_el,initial_tr=mc_in.n_tr,
                Height=mc_in.h, Width=mc_in.w, Length=mc_in.l,
                E_loc=phys_in.E_loc,alpha=phys_in.alpha,
                b=phys_in.b, s=phys_in.s, T_init=phys_in.T_init,
                dT=phys_in.dT,rho=phys_in.rho,urho=phys_in.urho)
    elif mc_in.therm_type == "ThermalC":
        MC = make_mc_class("ThermalC")
        MonteCarlo = MC(duration=mc_in.duration,dt_cap=mc_in.max_dt,
                initial_el=mc_in.n_el,initial_tr=mc_in.n_tr,
                Height=mc_in.h, Width=mc_in.w, Length=mc_in.l,
                E_loc=phys_in.E_loc,E_cb=phys_in.E_cb, alpha=phys_in.alpha,
                b=phys_in.b, s=phys_in.s, T_init=phys_in.T_init,
                dT=phys_in.dT,rho=phys_in.rho,urho=phys_in.urho)
    elif mc_in.therm_type == "ThermalD":
        MC = make_mc_class("ThermalD")
        MonteCarlo = MC(duration=mc_in.duration,dt_cap=mc_in.max_dt,
                initial_el=mc_in.n_el,initial_tr=mc_in.n_tr,
                Height=mc_in.h, Width=mc_in.w, Length=mc_in.l,
                E_loc=phys_in.E_loc,alpha=phys_in.alpha,
                D0=phys_in.D0, D_dot=phys_in.D_dot,
                b=phys_in.b, s=phys_in.s, T_init=phys_in.T_init,
                dT=phys_in.dT,rho=phys_in.rho,urho=phys_in.urho)
    elif mc_in.therm_type == "ThermalCD":
        MC = make_mc_class("ThermalCD")
        MonteCarlo = MC(duration=mc_in.duration,dt_cap=mc_in.max_dt,
                initial_el=mc_in.n_el,initial_tr=mc_in.n_tr,
                Height=mc_in.h, Width=mc_in.w, Length=mc_in.l,
                E_loc=phys_in.E_loc, E_cb=phys_in.E_cb, alpha=phys_in.alpha,
                D0=phys_in.D0, D_dot=phys_in.D_dot,
                b=phys_in.b, s=phys_in.s, T_init=phys_in.T_init,
                dT=phys_in.dT,rho=phys_in.rho,urho=phys_in.urho)
    else:
        err.error("Type of physics not known. Probably should never see this",fatal=True)

    err.checkpoint()

    MonteCarlo.run_all(mc_in.reps,err)
    
    return MonteCarlo

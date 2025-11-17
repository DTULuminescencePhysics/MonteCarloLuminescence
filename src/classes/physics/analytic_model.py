from __future__ import annotations
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict
import numpy as np 

from dataclasses import dataclass, field, fields
from scipy.integrate import solve_ivp 
from scipy.interpolate import interp1d
from src.classes.physics.time_temperature import _temp 
from src.classes.physics.set_system import _ThermalParameters 
from src.classes.constants import cnst

from src.process_plot import plot_analtyical_results

@dataclass
class analytical_crystal(_temp,_ThermalParameters):
    unitless:bool = field(default=False)
    ur: np.ndarray = field(init=False)
    ur_weights: np.ndarray = field(init=False)
    time_steps: np.ndarray = field(init=False)
    ratio: float = field(default=0.0)


    @classmethod
    def from_config(cls, cfg: DictConfig, T_override: dict | None = None) -> "analytical_crystal":
        """Create a ``analytical_crystal`` instance from a Hydra ``DictConfig`` object."""
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"Expected DictConfig, received {type(cfg).__name__}")

        resolved = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(resolved, dict):
            raise ValueError("Resolved configuration must be a mapping")

        physics_cfg = resolved.get("physics", {})
        box_cfg = resolved.get("box", {})
        if T_override is None:
            temp_cfg = resolved.get("temp",{})
        else: 
            temp_cfg = T_override

        init_fields = {f.name for f in fields(cls) if f.init}
        array_fields = {"times", "dT_step", "dT", "T_inf", "k"}

        kwargs: Dict[str, Any] = {}
        for section in (physics_cfg, temp_cfg, box_cfg, resolved):
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
        if self.phys_type == "king":
            self.phys_type = "king_ratio"
        else:
            self.phys_type = "ratio"
        # self.unitless = True
        if self.unitless:
            self.ur = np.linspace(0,2,10000)
        super().__post_init__()
        if not self.unitless:
            end = 2/(np.cbrt((4*np.pi*self.rho)/3))
            self.ur = np.linspace(0,end,100)

        self.time_steps = np.linspace(0.0, self.duration,10000)
        self.set_ur_values()

    def __repr__(self):
        if self.kind == "constant":
            temp = f"at a constant temperature of : {self.T0} (K)\n"
        elif self.kind == "linear":
            temp = (f"with a linear temperature profile starting at : {self.T0} (K) \n"
                    f"and increasing at a rate of : {self.dT} (K/s)\n") 
        elif self.kind == "step":
            temp = (f"with a step temperature profile starting at : {self.T0}"
                f"and a step of {self.dT_step} : \n")
        elif self.kind == "steps":
            temp = (f"with a multiple step temperature profile starting at : {self.T0} (K) \n"
                    f"with steps of : {self.dT_step} (K) at times : {self.times} (s)\n")
        elif self.kind == "linearsteps":
            temp = (f"with a multiple linear step temperature profile starting at : {self.T0} (K) \n"
                    f"with steps of : {self.dT_step} (K) at times : {self.times} (s) \n"
                    f"and linear rates of : {self.dT} (K/s)\n")
        elif self.kind == "exponential":
            temp = (f"with an exponential temperature profile starting at : {self.T0} (K) \n"
                    f"and approaching {self.T_inf} (K) with a rate constant of : {self.k} (s^-1)\n")
        elif self.kind == "lineardrops":
            temp = (f"with a linear drop temperature profile starting at : {self.T0} (K) \n"
                    f"with drops of : {self.dT_step} (K) at times : {self.times} (s) \n"
                    f"and linear rates of : {self.dT} (K/s)\n")
        string = ("Crystal Information: \n"
                f" A density of : {self.rho} (m^-3) or {self.urho} (unitless)\n"
                f"The ground state to excited state energy gap is : {self.E_loc} (eV)\n"
                f"and the conduction band gap is : {self.E_cb} (eV).\n"
                f"Tunnelling frequency is : {self.b} (s^-1)\n"
                f"and the tunneling rate constant is : {self.alpha} (m^-1).\n"
                f"The escape frequecy is : {self.s} (s^-1).\n"
                f"The crystal is dosed at rate of : {self.D_dot} (Gy/s) \n"
                f"with a characteristic dose of : {self.D0} (Gy)\n"
                f"{temp}")
        
        return string

        
    def set_ur_values(self):
        """Generates N unitless values of r and sets 
        the initial population weighting"""
        # self.ur /= 
        
        self.ur_weights = self.Pr(self.ur)
        # if not self.unitless:
        #     self.ur_weights *= 10000

    def Pr(self, r):
        """Probability density function for nearest neighbour distance r (in m)"""
        #
        if self.unitless:
            return 3*np.power(r,2)*np.exp(-np.power(r,3))
        else:
            return  np.exp((-4*np.pi*self.rho*np.power(r,3))/3)*(4*np.pi*self.rho*np.power(r,2))
        
    def get_analytical_solution(self):
        """Function that returns the analytical solution for the 
        given set of parameters."""

        def unitless_model(t,vars, ur ):
            """System of the ODEs that describe the filling and fading 
            of the traps in the crystal."""

            ratio = vars[0] 
            T = self.Tat(t)

            
            fill = (self.D_dot/self.D0)*(1-ratio)
            term1 = (self.b * np.exp(-(1/np.cbrt(self.urho)) * ur))*ratio
            term2 = (self.s * np.exp(-(self.E_loc - self.E_cb) / (cnst.k_b_ev * T)))*ratio
            fade = (term1 + term2)
            # fade = self._fade(T,ur)*ratio
            
            dn_dt = fill-fade
           
            
            return [dn_dt]
        
      
        def model(t,vars, r):
            """System of the ODEs that describe the filling and fading 
            of the traps in the crystal."""

            nN = vars[0] 
            T = self.Tat(t)

            fill = (self.D_dot/self.D0)*(1-nN)
            
            fade = self._fade(T,r)*nN
            dn_dt = fill-fade
           
            return [dn_dt]

       
        results = np.zeros((self.time_steps.size))
        cnt = 0
        # self.ur_weights /= self.ur_weights.sum()

        for i in range(len(self.ur)):
            if self.ur_weights[i] <=0:
                continue
            r = self.ur[i]

            cnt+=1 
            if self.unitless:
                solution = solve_ivp(unitless_model, (0, self.duration), [0], args=(r,), t_eval=self.time_steps, method = 'Radau',dense_output=True)
                res=  (solution.y[0]*self.ur_weights[i])
            else:
                solution = solve_ivp(model, (0, self.duration), [0], args=(r,), t_eval=self.time_steps, method = 'Radau',dense_output=False)
                res= solution.y[0]*self.ur_weights[i]
            
            try:
                results += res
               
            except:
                y = interp1d(solution.t, res, kind='linear', fill_value="extrapolate")
                res = y(self.time_steps)
                results += res
        
        if self.unitless:
            results /= cnt
        else:
            results /=np.sum(self.ur_weights)

        np.savetxt("Analytical_results.csv",np.column_stack((self.time_steps,results)), delimiter=",", header="Time, n/N (Analytic)")
        plot_analtyical_results("Analytical_results.csv", self.unit)


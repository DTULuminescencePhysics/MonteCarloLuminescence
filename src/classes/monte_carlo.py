import sys
from classes.physics import system
from classes.crystal import box
import numpy as np
from joblib import Parallel, delayed
from src.filesystem import output_monte_carlo_results
class MC:
    """Sets up universal parameters used for all 
    iterations of the Monte Carlo simulation"""

    def __init__(self,E_loc,b,s,alpha=None,E_cb=None,
                 T_init=273.15,dT=5, duration=160,
                 D0=None,D_dot=None,
                 rho = None, urho = None,
                 tot = None, h=None, w=None, l=None):
       
        
        self.phys = system(E_loc=E_loc,alpha=alpha,b=b,s=s,
                           E_cb=E_cb,D0=D0,D_dot=D_dot,
                           T_init=T_init,dT=dT)
        
        if tot is not None and urho is not None:
            self.tot = tot 
            self.phys.set_urho(urho)
            side_len = np.cbrt(self.tot/self.phys.rho)
            h=w=l=side_len
            self.crystal = box(h,w,l,1.5)
        else:
            if h is None or w is None or l is None:
                self.crystal = box(boundary_factor=1.5)
            else:
                self.crystal = box(h,w,l,1.5)
            if rho is None and urho is not None:
                self.phys.set_urho(urho)
                self.tot = int(self.phys.rho*self.crystal.volume)
            elif urho is None and rho is not None:
                self.tot = int(rho*self.crystal.volume)
                self.phys.set_rho(rho)
            elif rho is None and urho is None and tot is not None:
                self.tot = tot
                rho = tot/self.crystal.volume
                self.phys.set_rho(rho)
            else: 
                sys.exit("Need to define either a density or a number electrons")

        if D0 is not None and D_dot is not None:
            self.filling = True 
        else:
            self.filling = False

        self.duration = duration 
        self.dt_cap = 1
        
    def find_next_recombination(self,T):
        """Function that takes the list of nearest neighbours
        and then computes lifetimes and then draws the recombination
        times"""
        # if self.recalc: 
        #     self._alpha_r = self.phys.b*np.exp(-self.phys.alpha*self.crystal.r)
        #     self.re_calc = False
      
        # self._lifetime = 1/(self._alpha_r*self.phys.calc_p(T))
       
        self._lifetime = self.phys.fading_rate(self.crystal.r,T)
        self._recomb_wait = np.random.exponential(self._lifetime)
        return np.min(self._recomb_wait), np.argmin(self._recomb_wait)

    def find_next_filling(self,ne,nt):
        if self.filling and (ne != nt):
            filltime = self.phys.filling_rate(ne,nt)
            return  np.random.exponential(filltime)
        else:
            return 1e20

    def run_simulation(self,rep,folder,handler):
        """Function that runs a single iteration of the 
        Monte Carlo simulation"""

        self.crystal.initialise_el_tr(self.tot,self.tot)
        t_cur = 0.0 
        Temp = self.phys.T(t_cur)
        max_steps = int(self.duration/self.dt_cap +1)*5
        store = np.zeros((3,max_steps))
        # time = np.zeros(max_steps)
        # nel_store = np.zeros(max_steps)
        # tr_store = np.zeros(max_steps)
        store[1,0] =  self.crystal.n_el
        store[2,0] = self.crystal.n_trap
        i = 1
        self.recalc  = True
        while t_cur < self.duration: 
            if self.crystal.n_el > 0:
                recomb_tim, recomb_index = self.find_next_recombination(Temp)
            else:
                recomb_tim = 1e20
            fill_time = self.find_next_filling(self.crystal.n_el,self.crystal.n_trap)
            dt = min(recomb_tim,fill_time,self.dt_cap)

            if dt == recomb_tim:
                self.crystal.remove_electron(recomb_index)
                event = 1
            elif dt == fill_time:
                self.crystal.add_electron()
                event = 0
                self.re_calc = True
            elif dt == self.dt_cap:
                event = 0 
         
            t_cur += dt
            Temp = self.phys.T(t_cur)
            if (max_steps - i < 2):
                to_append = np.zeros((3,max_steps))
                store = np.vstack(store,to_append)
               
            store[0,i] = t_cur
            store[1,i] =  self.crystal.n_el
            store[2,i] = self.crystal.n_trap
            i +=1
        
        output_monte_carlo_results(rep,folder,store[:,0:i],handler)
       
        # df = pd.DataFrame({
        #     "Time": time[0:i],
        #     "Electrons": nel_store[0:i],
        #     "Traps": tr_store[0:i]
        # }) 

        # return df
    

def run_monte_carlo_simulation(folder,handler):
    input1 = {"E_loc": 0.8, 
             "s":1e10, 
             "rho":8e-4, 
             "factor":1e5, 
             "z":1.8, 
             "b":1e10,
             "c1":"black",
             "c2":"orange"
             }
    input2 = {"E_loc": 0.8, 
             "s":1e10, 
             "rho":3e-4, 
             "factor":1e4, 
             "z":1.8, 
             "b":1e10,
             "c1":"blue",
             "c2":"pink"}
    input3 = {"E_loc": 1.2, 
             "s":1e12, 
             "rho":3e-4, 
             "factor":2e6, 
             "z":1.8, 
             "b":1e12,
             "c1":"green",
             "c2":"grey"}

    inputs = input1
    MonteCarlo = MC(inputs["E_loc"],inputs["b"],inputs["s"],alpha=None,E_cb=None,
                 T_init=273.15,dT=5, duration=160,
                 D0=None,D_dot=None,
                 rho = None, urho = inputs["rho"],
                 tot = 1e2, h=None, w=None, l=None)
    
   
   
    reps = 2
    
    Parallel(n_jobs=-1)(delayed(MonteCarlo.run_simulation)(i,folder,handler) for i in range(reps))
   
    # dfs =[]
    # for i in range(reps):
    #     df = MonteCarlo.run_simulation()
    #     dfs.append(df)
 

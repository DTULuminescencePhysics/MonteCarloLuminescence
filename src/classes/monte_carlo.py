import sys
from dataclasses import dataclass, field
from classes.physics import system
from classes.crystal import box

class MC:

    def __init__(self,E_loc,b,s,alpha=None,E_cb=None,
                 T_init=273.15,dT=5,
                 D0=None,D_dot=None,
                 rho = None, urho = None,
                 tot = None, h=None, w=None, l=None):
        
        if h is None or w is None or l is None:
            self.crystal = box(boundary_factor=1.5)
        else:
            self.crystal = box(h,w,l,1.5)

        self.phys = system(E_loc=E_loc,alpha=alpha,b=b,s=s,
                           E_cb=E_cb,D0=D0,D_dot=D_dot,
                           T_init=T_init,dT=dT)

        if rho is None and urho is not None:
            self.phys.set_urho(urho)
            tot = self.phys.rho*self.crystal.volume
        elif urho is None and rho is not None:
            tot = rho*self.crystal.volume
            self.phys.set_rho(rho)
        elif rho is None and urho is None and tot is not None:
            rho = tot/self.crystal.volume
            self.phys.set_rho(rho)
        else: 
            sys.exit("Need to define either a density or a number electrons")

        self.crystal.initialise_el_tr(tot,tot)

        if D0 is not None and D_dot is not None:
            self.filling = True 
        else:
            self.filling = False



        
        
        


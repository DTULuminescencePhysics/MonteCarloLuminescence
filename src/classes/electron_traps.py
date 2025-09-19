from dataclasses import dataclass, field
import numpy as np


@dataclass(kw_only=True) 
class _electraps:
    n_el : int = field(init=False, repr=False) # Number of electrons
    n_trap : int = field(init=False, repr=False) # Number of electron traps
    electrons : np.ndarray = field(init=False, repr=False) # Store of electron coordinates
    traps     : np.ndarray = field(init=False, repr=False) # Store of trap coordinates
    distances : np.ndarray = field(init=False, repr=False) # Matrix of all distances between electrons and traps
    r         : np.ndarray = field(init=False, repr=False) # Store of the nearest neighbour distances in same order as electrons
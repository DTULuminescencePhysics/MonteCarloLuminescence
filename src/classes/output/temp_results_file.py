from __future__ import annotations
import os
import numpy as np

from typing import  Literal

modes = Literal["r", "r+", "w+", "c"]

class chronology_results:
        
    def __init__(self, iter: int, node_max: int, mode: modes,
                 temp_name:str = 'temp_time.dat', acc_name: str = 'accepted.dat') -> None:
        self.iter = iter

        lengths = (node_max+2)*iter
        self.temp_name = temp_name
        self.acc_name = acc_name
        self.temp_time = np.memmap(self.temp_name,dtype=np.float32, mode=mode, shape=(2,lengths))
        self.acc = np.memmap(self.acc_name,dtype=np.float32,mode=mode,shape=(3,self.iter))
        if mode == "w+":
            self.acc[:,:] = 0 
            self.temp_time[:,:] = 0
            self.flush()
        
        self.strt = 0
        self.end = 0

    def delete_stores(self,):
        if os.path.exists(self.acc_name): 
            os.remove(self.acc_name)
        if os.path.exists(self.temp_name): 
            os.remove(self.temp_name)

    def write_result(self, it: int, times: np.ndarray, temps: np.ndarray, 
                     accept: bool, val: float) -> None:
         
        self.end = self.strt+times.size
        self.temp_time[0,self.strt:self.end]=times
        self.temp_time[1,self.strt:self.end]=temps

        self.acc[0,it]=self.end
        if accept: 
            self.acc[1,it]=1
        else:
            self.acc[1,it]=0 

        self.acc[2,it]=val

        self.strt = self.end

    def flush(self): 

        self.temp_time.flush()
        self.acc.flush()
    
    def get_final_end(self): 
        val = -1 
        self.end = 0
        while self.end == 0: 
            self.end = int(self.acc[0,val])
            val -=1
            if abs(val) >= self.iter:
                self.end = 0 
                return 
        
    def set_true(self,i):
        self.acc[1,i]=1

    @property
    def all_times(self):
        self.get_final_end()
        return self.temp_time[0,0:self.end]
    
    @property
    def all_temps(self):
        self.get_final_end()
        return self.temp_time[1,:]
    
    @property
    def accepted(self):
        return self.acc[1,:].astype(np.bool_)
    
    @property
    def offset(self):
        return self.acc[0,:].astype(np.int32)
    
    @property
    def values(self):
        return self.acc[2,:]
    
    def get_time(self, i: int)-> np.ndarray:
        if i == 0:
            strt=0
        else:
            strt = int(self.acc[0,i-1])
        
        end = int(self.acc[0,i])
        return self.temp_time[0,strt:end]
    
    def get_temp(self, i: int)-> np.ndarray:
        if i == 0:
            strt=0
        else:
            strt = int(self.acc[0,i-1])
        
        end = int(self.acc[0,i])
        return self.temp_time[1,strt:end]

    def T_min_max(self):
        strt=0
        val_min = None
        val_max = None 
        for i in range(self.iter):
            if self.acc[1,i] == 1:
                if i == 0: 
                    strt = 0
                else: 
                    strt = int(self.acc[0,i-1])
                end = int(self.acc[0,i])
                val_min_try = np.amin(self.temp_time[1,strt:end])
                val_max_try = np.amax(self.temp_time[1,strt:end])

                if val_min is None: 
                    val_min = val_min_try
                elif val_min_try < val_min :
                    val_min = val_min_try

                if val_max is None: 
                    val_max = val_max_try
                elif val_max_try > val_max :
                    val_max = val_max_try

                

        if val_min is None: 
            val_min = 0
        
        if val_max is None: 
            val_min = 0
        return val_min, val_max
        

    







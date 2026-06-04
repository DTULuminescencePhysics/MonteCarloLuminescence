from __future__ import annotations
import os 
import numpy as np 
import matplotlib as mpl
from matplotlib import pyplot as plt
from typing import Tuple, TYPE_CHECKING

from src.classes.physics.temperature.temp_profile_class import TimeTempProfile
from src.classes.constants import time_to_seconds


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from src.classes.output.results_file import output_file


class plottingBase():
    """Class that contains the basic graphing functions setting the size and design of the 
    plots as well as the generic plotting function"""
    def __init__(self,unit: str = 's', celsius: bool = True, 
                 figDims: Tuple = (3.37,5.055), axDims: Tuple = (0.,0.,2.,1.))-> None: 
        
        self.timeUnit = unit
        self.celsius = celsius 
        self.figDims = figDims 
        self.axDims = axDims
        self.colors = ['g','b','k','c','m','y','r']
        self.lines  = ['-','--','-.',':']   

    def create_figure(self)-> Tuple[Figure,Axes]:
        """Creates figure and axes variables """
        fig = plt.figure(figsize= self.figDims)
        ax = fig.add_axes(self.axDims)

        return fig, ax 

    def save_figure(self,fig: Figure, file_name: str)-> None:
        """Saves the figure to a png file """

        fig.savefig(file_name,dpi=300, transparent=False,bbox_inches='tight')
        plt.close(fig)

    def time_sequence(self,times):
        if self.timeUnit != 's':
            if self.timeUnit == 'Ma':
                temp = times/time_to_seconds[self.timeUnit]
                last = temp[-1]
                return (((temp-last)*-1))
            else:
                return (times/time_to_seconds[self.timeUnit])
        else: 
            return times

    def plot_time_label(self, ax: Axes) -> None:
        """Sets Time axis according to the unit type"""

        match self.timeUnit: 
            case 's': 
                ax.set_xlabel("Time (s)")
            case 'm':
                ax.set_xlabel("Time (min)")
            case 'h':
                ax.set_xlabel("Time (hour)")
            case 'd':
                ax.set_xlabel("Time (day)")
            case 'y':
                ax.set_xlabel("Time (year)")
            case 'Ma':
                ax.set_xlabel("Time (Ma)")
                ax.invert_xaxis()
            case _:
                ax.set_xlabel("Time (s)")
    
    def temperature_sequence(self,temps):
        if self.celsius: 
            temps -= 273.15

    def plot_temp_label(self, ax: Axes, y_axis: bool=True) -> None:
        """Sets Temperature label accoridng to unit type"""
        if y_axis:
            if self.celsius: 
                ax.set_ylabel("Temperature (C)")
            else: 
                ax.set_ylabel("Temperature (K)")
        else:
            if self.celsius: 
                ax.set_xlabel("Temperature (C)")
            else: 
                ax.set_xlabel("Temperature (K)")

    def plot_x_vs_y(self, x: np.ndarray, y: np.ndarray, ax: Axes, 
             colour: str = "black", line: str = 'solid',
             label: str | None = None, alpha:float=1.0):
    
        ax.plot(x,y, color=colour, label = label, ls = line, alpha=alpha)


class MainPlot(plottingBase):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
       
        self.analytic_ratio_file = None
   
    def set_analytic_ratio_file(self,name) -> None:
        """Sets the namne of the forward_ratio_file"""
        if os.path.exists(f"{name}.csv"):
            self.analytic_ratio_file = []
            self.analytic_ratio_file.append(name)
            self.experiment_cnt = 1
    
    def set_multi_analytic_ratio_file(self,name): 
        if self.analytic_ratio_file is None: 
            self.analytic_ratio_file = []
            self.experiment_cnt = 0
        if os.path.exists(f"{name}.csv"):
            self.analytic_ratio_file.append(name)
            self.experiment_cnt += 1

    def main_ratio_body(self, ax: Axes, exp_cnt:int, mean:np.ndarray, x_vals: np.ndarray, 
                           std:np.ndarray, lo:np.ndarray, mi:np.ndarray, hi:np.ndarray, 
                           quant:bool=True, st:bool=True, temp:bool = False):
        if temp:
            x = np.r_[x_vals[:-1], x_vals[-1]]
        for i in range(exp_cnt):
            if not temp:
                x = np.r_[x_vals[:-1,i], x_vals[-1,i]]
            y_mean = np.r_[mean[:,i], mean[-1,i]]

            ax.step(x, y_mean, where="post", label="Mean fill %")
            if st: 
                y_std = np.r_[std[:,i], std[-1,i]]

                y_std_low = y_mean - y_std
                y_std_high = y_mean + y_std
                ax.fill_between(x, y_std_low, y_std_high, step="post", alpha=0.2,label="Mean ± 1 std")

            if quant: 
                y_low = np.r_[lo[:,i], lo[-1,i]]
                y_high = np.r_[hi[:,i], hi[-1,i]]
                ax.fill_between(x, y_low, y_high, step="post", alpha=0.25,label=f"q10-q90")
                y_mid = np.r_[mi[:,i], mi[-1,i]]
                ax.step(x, y_mid, where="post", linestyle="--", label=f"q50")

    def plot_forward_ratio(self, file_name:str, exp_cnt:int, mean:np.ndarray, times: np.ndarray, 
                           std:np.ndarray, lo:np.ndarray, mi:np.ndarray, hi:np.ndarray, 
                           quant:bool=True, st:bool=True): 
        fig, ax = self.create_figure()

        self.main_ratio_body(ax, exp_cnt, mean, times, std, lo, mi, hi, quant, st)
       
        self.plot_time_label(ax)
        ax.legend()
        ax.set_ylabel("n/N Trap ratio")
        ax.grid(True, alpha=0.3)
        self.save_figure(fig,file_name)

    def plot_forward_ratio_T(self, file_name:str, exp_cnt:int, mean:np.ndarray, temps: np.ndarray, 
                           std:np.ndarray, lo:np.ndarray, mi:np.ndarray, hi:np.ndarray, 
                           quant:bool=True, st:bool=True):
    
    
        fig, ax = self.create_figure()
        self.main_ratio_body(ax, exp_cnt, mean, temps, std, lo, mi, hi, quant, st,True)
       
        self.plot_temp_label(ax,False)
        ax.set_ylabel("n/N Trap ratio")
        ax.legend()
        self.save_figure(fig,file_name)


    def plot_T_profile(self, temp: np.ndarray, times: np.ndarray, file_name:str): 
        fig, ax = self.create_figure()
        self.plot_x_vs_y(times,temp,ax)
        self.plot_time_label(ax)
        self.plot_temp_label(ax)
        self.save_figure(fig,file_name)

    def create_new_time_series(self, xvalues, new_bin_width):
        t_min = xvalues[0]
        t_max = xvalues[-1]

        n_new = int(round((t_max - t_min) / new_bin_width))
        new_times = np.linspace(t_min,t_max,n_new+1)
        return n_new, new_times

    def rebin_event_values_fixed_width(self, xvalues, events, new_bin_width):
   
        n_new, new_xvalues = self.create_new_time_series(xvalues, new_bin_width)
        x_start = xvalues[:-1]
        x_end = xvalues[1:]
        new_x_start = new_xvalues[:-1]
        new_x_end = new_xvalues[1:]
        old_widths = x_end - x_start
        new_widths = new_x_end - new_x_start
      
        n_series = events.shape[1]
        accum = np.zeros((n_new, n_series), dtype=float)

        j_strt=0
        for i in range(n_new):
            for j in range(j_strt,x_end.size):
                b0 = x_end[j]
                if b0 < new_x_end[i]:
                    accum[i] += events[j]
                elif b0 == new_x_end[i]:
                    accum[i] += events[j]
                    j_strt = j+1
                    break
                else: 
                    overlap = (new_x_end[i] - x_start[j])
                    frac = overlap/old_widths[j]
                    accum[i] += events[j]*frac 
                    accum[i+1] += events[j]*(1-frac)
                    j_strt = j+1
                    break 
            accum[i] /= new_widths[i]

        return accum, new_xvalues
    
    def rebin_events_time(self, file_name:str, exp_cnt:int, times:np.ndarray, events: dict, event_type: dict, new_bin_width:float):

        accum, new_t = self.rebin_event_values_fixed_width(times[:,0],events[1],new_bin_width)
        new_times = {1: new_t}
        all_accum = {1: accum}
        if exp_cnt > 1:
            for i in range(1,exp_cnt):
                accum, new_t = self.rebin_event_values_fixed_width(times[:,i],events[i+1],new_bin_width)
                new_times[i+1] =  new_t
                all_accum[i+1] =  accum

        for i in range(1,exp_cnt+1): 
            file = f"{file_name}_stacked_intervals_exp_{i}.png"
            self.plot_stacked_time_intervals(file, new_times[i], all_accum[i], event_type[i])

            file = f"{file_name}_stacked_exp_{i}.png"
            self.plot_stacked_time(file, new_times[i], all_accum[i], event_type[i])

    def rebin_events_temp(self, file_name:str, exp_cnt:int, temps:np.ndarray, events: dict, event_type: dict, new_temp_width:float):
        accum, new_t = self.rebin_event_values_fixed_width(temps,events[1],new_temp_width)
        all_accum = {1: accum}
        if exp_cnt > 1:
            for i in range(1,exp_cnt):
                accum, new_t = self.rebin_event_values_fixed_width(temps,events[i+1],new_temp_width)
                all_accum[i+1] =  accum

        for i in range(1,exp_cnt+1): 
            file = f"{file_name}_stacked_intervals_exp_{i}.png"
            self.plot_stacked_temp_intervals(file, new_t, all_accum[i], event_type[i])

            file = f"{file_name}_stacked_exp_{i}.png"
            self.plot_stacked_temp(file, new_t, all_accum[i], event_type[i])

    def plot_stacked_time_intervals(self,file_name, xvalues, events, labels):
        fig, ax = self.create_figure()

        self.plot_stacked_intervals(ax,xvalues, events, labels)
        self.plot_time_label(ax)
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.save_figure(fig,file_name)
    
    def plot_stacked_temp_intervals(self,file_name, xvalues, events, labels):
        fig, ax = self.create_figure()

        self.plot_stacked_intervals(ax,xvalues, events, labels)
        self.plot_temp_label(ax,False)        
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.save_figure(fig,file_name)

    def plot_stacked_intervals(self, ax, xvalues, events, labels):
       
        for j, label in enumerate(labels):
            y = np.r_[events[0, j],events[:, j], ]
            ax.step(xvalues, y, where="pre",label=label)
          
    
    def plot_stacked_temp(self,file_name, xvalues, events, labels):
        fig, ax = self.create_figure()
        self.plot_stacked(ax, xvalues, events, labels)
        self.plot_temp_label(ax,False)        
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        self.save_figure(fig,file_name)


    def plot_stacked_time(self,file_name, xvalues, events, labels):
        fig, ax = self.create_figure()
        self.plot_stacked(ax, xvalues, events, labels)
        self.plot_time_label(ax)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        self.save_figure(fig,file_name)

    def plot_stacked(self,ax, xvalues, events, labels):
        x =  0.5*(xvalues[:-1] +xvalues[1:])
      
        for i in range(len(labels)):
            ax.plot(x,events[:,i],label=labels[i])

        
    def plot_analytic_comp_MC_single(self, analytic_file: str, MC_file: str):

        A_data = np.loadtxt(f"{analytic_file}", delimiter=",")
        MC_data = np.loadtxt(f"{MC_file}", delimiter=",")

        a_times = self.time_sequence(A_data[:,0])
        mc_times = self.time_sequence(MC_data[:,0])
        fig, ax = self.create_figure()

        self.plot_x_vs_y(a_times,A_data[:,-1],ax,self.colors[0],self.lines[0],"Analytic",alpha=0.25)
        self.plot_x_vs_y(mc_times,MC_data[:,-1],ax,self.colors[1],self.lines[0],"MC")
        self.plot_time_label(ax )
        ax.set_ylabel("n/N Trap ratio")
        ax.legend()
        self.save_figure(fig,"MC_vs_Analytic_result.png")
    
    def plot_analytic_comp_MC_multi(self, analytic_file: list[str], MC_file: list[str]):
        fig, ax = self.create_figure()
        
        j=0
        for i in range(len(MC_file)):
            if( i > 0 and i%7 == 0):
                j = ((j+1)%4) 
        
            file = MC_file[i]
            data = np.loadtxt(file, delimiter=",")
            times = self.time_sequence(data[:,0])
            self.plot_x_vs_y(times,data[:,-1],ax,self.colors[i%7],self.lines[j],f"MC Experiment no.{i+1}")

        j=2
        for i in range(len(analytic_file)):
            if( i > 0 and i%7 == 0):
                j = ((j+1)%4) 
        
            file = analytic_file[i]
            data = np.loadtxt(file, delimiter=",")
            times = self.time_sequence(data[:,0])
            self.plot_x_vs_y(times,data[:,1:],ax,self.colors[i%7],self.lines[j],f"Analytic Experiment no.{i+1}",alpha=0.5)

        self.plot_time_label(ax)
        ax.set_ylabel("n/N Trap ratio")
        ax.legend()
        self.save_figure(fig,"MC_vs_Analytic_result.png")


    def plot_forward(self, output:output_file, quant:bool=True, st:bool=True, new_time_width:float=1.0, new_temp_width:float=2.5) -> None:

        exp_cnt, times, mean, lo, mi, hi, std = output.output_result_get_all_ratioPlot(quant,st)
        self.plot_forward_ratio("Time_filling.png", exp_cnt, mean, times, std, lo, mi, hi, quant, st)
        temps = output.output_result_get_temperature(1)
        self.temperature_sequence(temps)
        self.plot_T_profile(temps,times[:,0],"Temperature_Profile.png")
        self.plot_forward_ratio_T("Temp_filling.png", exp_cnt, mean, temps, std, lo, mi, hi, quant, st)
        events, event_type = output.output_result_get_all_nonzero_events(exp_cnt)
        self.rebin_events_time("events", exp_cnt, times, events, event_type, new_time_width)
        self.rebin_events_temp("temp_events", exp_cnt, temps, events, event_type, new_temp_width)
        lum, lf_typ = output.output_result_get_all_lum_fill(1)
        self.rebin_events_time("all", exp_cnt, times, lum, lf_typ, new_time_width)
        self.rebin_events_temp("temp_all", exp_cnt, temps, lum, lf_typ, new_temp_width)

       
        # if self.analytic_ratio_file is not None: 
        #     for file in self.analytic_ratio_file: 
        #         data = np.loadtxt(f"{file}.csv", delimiter=",")
        #         times = self.time_sequence(data[:,0])
        #         self.plot_forward_ratio(data[:,2:],times,f"Time_filling_analytic_{file}.png")

        #     if self.experiment_cnt > 1:
        #         self.plot_forward_multi_experiment(self.analytic_ratio_file,'all.png') 


        # if self.forward_ratio_file is not None and self.analytic_ratio_file is not None:
        #     if self.experiment_cnt == 1:
        #         self.plot_analytic_comp_MC_single(self.analytic_ratio_file[0], self.forward_ratio_file[0])
        #     else:  
        #         self.plot_analytic_comp_MC_multi(self.analytic_ratio_file,self.forward_ratio_file)
    
    
   
class chronologyPlot_running(plottingBase): 

    def __init__(self, duration: float, **kwargs) -> None:

        super().__init__(**kwargs)
        self.duration = duration
       
        self.iters = None
        self.node_max = None
        self.tracking_lines = []
        self.t_common = np.linspace(0,self.duration, 1000)
        self.t_common_unit = self.t_common*time_to_seconds[self.timeUnit]

    def set_profile_files(self,iters, node_max):
        self.iters = iters 
        self.node_max = node_max
        

    def setup_simulation_tracker(self, temp):
        
        self.trackFig, self.trackAx = plt.subplots()
        
        self.trueT=temp
        self.temperature_sequence(self.trueT)
        self.plot_x_vs_y(self.t_common,self.trueT,self.trackAx,label="Actual")
        self.plot_temp_label(self.trackAx)
        self.plot_time_label(self.trackAx)
        if self.timeUnit == 'Ma':
            self.trackAx.invert_xaxis()
        
        self.trackFig.canvas.draw_idle()
        plt.show(block=False)

    def burn_clear_lines(self,):
        self.save_figure(self.trackFig,"burn_in_tracking_figure.png")

        for line in self.tracking_lines:
            line.remove()
        
        self.tracking_lines.clear()
        self.trackAx.legend()

        self.trackFig.canvas.draw_idle()
        plt.pause(0.1)

    def add_result(self, y, it):
        # fade existing lines
        self.temperature_sequence(y)
        for line in self.tracking_lines:
            current_alpha = line.get_alpha()
            new_alpha = max(current_alpha * 0.5, 0.1)
            line.set_alpha(0.1)
            line.set_color("red")
            line.set_label(None)

        # self.plot_x_vs_y(self.t_common,self.trueT,self.trackAx,label="Actual")
        
        line, = self.trackAx.plot(self.t_common, y, color="blue", alpha=1.0, label = it)
        # self.plot_time_label(self.trackAx)
        self.tracking_lines.append(line)
        self.trackAx.legend()
        self.trackFig.canvas.draw_idle()
        plt.pause(0.1) 

    def save_close_tracker(self):
        self.save_figure(self.trackFig,"tracking_figure.png")
        del(self.tracking_lines, self.t_common_unit)

    
class chronologyPlot(plottingBase): 

    def __init__(self, duration: float, **kwargs) -> None:

        super().__init__(**kwargs)
        self.duration = duration
        self.t_common = np.linspace(0,self.duration, 1000)

    def make_time_temp_grid(self, t_min: float, t_max: float, T_min: float,
        T_max: float, n_bins: int , n_samples: int ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Create a regular time-temperature grid.
        """
        t_edges = np.linspace(t_min, t_max, n_bins + 1)
        T_edges = np.linspace(T_min, T_max, n_bins + 1)
        grid = np.zeros((n_bins, n_bins), dtype=int)
        t_samples = np.linspace(t_min,t_max, n_samples)

        return t_edges,T_edges,grid,t_samples

    def accumulate_profile_on_grid(self, output: output_file, t_edges: np.ndarray, T_edges: np.ndarray ,grid: np.ndarray, 
                               t_samples: np.ndarray, ) -> int:
                               
                            #    temp_prof: np.ndarray, count: int) -> None:
        """
        Sample a time-temperature profile and increment grid squares that the
        profile passes through (once per profile per square).
        """
       
        t_bins = np.digitize((t_samples), t_edges) - 1
        n_bins = grid.shape[0]
        t_bins = np.clip(t_bins, 0, n_bins - 1)
        offsets = output.chronological_result_get_accepted_offsets()
        temps = output.chronological_result_get_temps()
        times = output.chronological_result_get_times()
        count = 0
        for i in range(0,offsets.size,2): 
            T_samples=self.make_temp_profile(times[offsets[i]:offsets[i+1]],temps[offsets[i]:offsets[i+1]])
        
            count +=1
    
            T_bins = np.digitize(T_samples, T_edges) - 1
            T_bins = np.clip(T_bins, 0, n_bins - 1)

            pairs = np.stack([T_bins, t_bins], axis=1)
            unique_pairs = np.unique(pairs, axis=0)
        
            for T_idx, t_idx in unique_pairs:
                grid[T_idx, t_idx] += 1
        
        return count

    
    def make_temp_profile(self,times: np.ndarray, temps: np.ndarray):

        tt = TimeTempProfile(kind="linearsteps",times=times,temps=temps)
        time_temp_profiles = tt(self.t_common)
          

        return time_temp_profiles

    def create_probability_density_grid(self, output: output_file, t_min: float, t_max: float, n_bins: int, n_samples: int) -> None: 
        
        T_min, T_max = output.chronological_result_max_T_min_T()
        t_edges,T_edges,grid,t_samples = self.make_time_temp_grid(t_min, t_max, T_min, T_max, n_bins, n_samples)
        
        count = self.accumulate_profile_on_grid(output, t_edges,T_edges,grid,t_samples)
                                        
                                      
        fig, ax = self.create_figure()
        im = ax.imshow(
            (grid/count),
            origin="lower",     
            aspect="auto",
            extent=(t_edges[0], t_edges[-1], T_edges[0], T_edges[-1])
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Probability")

        # ax.set_xlabel("Time")
        # ax.set_ylabel("Temperature")
       

        counts = grid.astype(float)    
        n_t = counts.shape[1]

        t_mids = 0.5 * (t_edges[:-1] + t_edges[1:])
        T_mids = 0.5 * (T_edges[:-1] + T_edges[1:])

        col_sums = counts.sum(axis=0)  

        cdf = np.cumsum(counts, axis=0) 
        with np.errstate(invalid="ignore", divide="ignore"):
            cdf = cdf / col_sums[None, :]

        probs=(0.6, 0.9)
        paths = {}
        valid = col_sums > 0
        for c in probs:
            p_low = (1.0 - c) / 2.0
            p_high = 1.0 - p_low
            idx_low = np.argmax(cdf >= p_low, axis=0) 
            idx_high = np.argmax(cdf >= p_high, axis=0)
            lower = np.full(n_t, np.nan, dtype=float)
            upper = np.full(n_t, np.nan, dtype=float)
            lower[valid] = T_mids[idx_low[valid]]
            upper[valid] = T_mids[idx_high[valid]]
            paths[c] = {"lower": lower, "upper": upper}

    
        idx = np.argmax(cdf >= 0.5, axis=0) 

        median = np.full(n_t, np.nan, dtype=float)  
        median[valid] = T_mids[idx[valid]]

       
        self.plot_x_vs_y(t_mids, paths[0.6]["lower"],ax,"green",":","60%")
        self.plot_x_vs_y(t_mids, paths[0.6]["upper"],ax,"green",":")
       
        self.plot_x_vs_y(t_mids, paths[0.9]["lower"],ax,"black",":","90%")
        self.plot_x_vs_y(t_mids, paths[0.9]["upper"],ax,"black",":")
        self.plot_x_vs_y(t_mids,median,ax,"red",":")
        
        trueT = self.make_temp_profile(output.tempertature_profile_get_times(),output.tempertature_profile_get_temps())
        # trueT = self.make_temp_profile(-1,Temp_points,time_points)
        self.plot_x_vs_y(self.t_common,trueT,ax,"white","--")

        self.plot_temp_label(ax)
        
        self.save_figure(fig,"weights.png")


    def make_weighting_plot(self, output: output_file,  n_bins: int = 50, n_samples: int = 10000):
        
        self.t_common = np.linspace(0,self.duration, n_samples)
       
        self.create_probability_density_grid(output, 0,self.duration,n_bins,n_samples)

        
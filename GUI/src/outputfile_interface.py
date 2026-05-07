from __future__ import annotations
import os
import numpy as np 
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING
from scipy.signal import savgol_filter
from src.classes.output.results_file import output_file
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtCore import Signal, Slot

os.environ["QT_API"] = "PySide6"
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure


time_to_seconds = {"seconds":1, "s": 1, "minutes": 60, "m": 60, "hours": 3600, "h": 3600, "days": 86400, "d": 86400, "years": 31556952, "y": 31556952, "ka":3.1556952e10 ,"ma":3.1556952e13,
                   "S": 1, "M": 60, "H": 3600, "D": 86400, "Y": 31556952, "Ka":3.1556952e10 ,"Ma":3.1556952e13,
                   "kA":3.1556952e10 ,"mA":3.1556952e13,"KA":3.1556952e10 ,"MA":3.1556952e13}

class MplCanvas(FigureCanvas):
    experimentcount = Signal(int)
    requestLineInfo = Signal(int,str)
    def __init__(self, parent=None,
                 width=3.37, height=5.055, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        self.fig.subplots_adjust(
            left=0.2,right=0.95,bottom=0.2,top=0.95)
        super().__init__(self.fig)

        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.units = {
        "xaxis" : "time",
        "xtime_unit" : "seconds",
        "xtemp_unit" : "Celsius",
        "std_num":1,
        "yaxis" : "ratio",
        "ytemp_unit" : "Celsius",
        "yevent_plot_type" : "steps",
        "smoothing" : False,
        "yevent_smoothing_unit":"seconds",
        "yevent_smoothing_value":0.0,
        "savgol_window":15,
        "savgol_poly":2
        }
        self.outputfile = None
        self.exp_count = 0
        self.data = None
        self._plot_list = defaultdict(dict)
        self.updateGeometry()
        self.set_axis_labels()

        
    def exp_data_retrieve(self, exp_num:int):
        lo,mi,hi = self.outputfile.output_result_get_quantiles(exp_num)
        data = {
            "time": self.outputfile.output_result_get_times(exp_num),
            "temperature": self.outputfile.output_result_get_temperature(exp_num),
            "mean": self.outputfile.output_result_get_meanTrap(exp_num),
            "std": self.outputfile.output_result_get_stdTrap(exp_num),
            "qlo": lo,
            "qmi": mi,
            "qhi": hi,
            "luminescence":self.outputfile.output_result_get_lum(exp_num),
            "fill":self.outputfile.output_result_get_filling(exp_num),
            "events":self.outputfile.output_result_get_events(exp_num)
        }
        return data
    
    def data_retrieve(self,):
        if self.outputfile is None:
            return 
        else:
            self.data = {}
            for i in range(self.exp_count):
                self.data[i+1] = self.exp_data_retrieve(i+1)
                self.rebin_events(i+1)
            self.add_all_lines()

    def add_all_lines(self):
        if self.units["yaxis"] == "ratio":
            self.add_all_ratio_lines()
        elif self.units["yaxis"] == "event":
            self.rebin_all()
            self.all_all_event_lines()
        elif self.units["yaxis"] == "temperature":
            self.add_all_temp_lines()

    @Slot(Path)
    def experiment_number_check(self, file_path: Path):

        self.outputfile = output_file(str(file_path),None)

        exp_num = self.outputfile.output_result_get_number_exp()
        self.exp_count = exp_num
        self.experimentcount.emit(exp_num)

    def clear_graph(self):
        for line in self._plot_list:
            line.remove()
        self.draw_idle()

    @Slot(str)
    def xaxis_update(self,new_val:str):
        if self.units["xaxis"] != new_val:
            self.units["xaxis"] = new_val
            self.rebin_all()
            self.update_x_units()
            self.set_axis_labels()
 
    @Slot(str)
    def yaxis_update(self,new_val:str):
        if self.units["yaxis"] != new_val:
            self.units["yaxis"] = new_val
            self.ax.cla()
            self.draw_idle()
            self._plot_list = defaultdict(dict)
            self.set_axis_labels()
            self.add_all_lines()

    @Slot(str)
    def xtime_update(self, new_val:str):
        if self.units["xtime_unit"] != new_val:
            self.units["xtime_unit"] = new_val
            self.update_x_units()
            self.set_axis_labels()

    @Slot(str)
    def xtemp_update(self, new_val:str):
        if self.units["xtemp_unit"] != new_val:
            self.units["xtemp_unit"] = new_val
            self.update_x_units()
            self.set_axis_labels()

    @Slot(int)
    def std_value_changed(self, new_val:int):
        if self.units["std_num"] != new_val:
            self.units["std_num"] = new_val
            for i in range(self.exp_count):
                x = self.get_x_data(i+1)
                self.update_std(i+1,x)
            self.set_axis_labels()
    @Slot(str)
    def ytemp_update(self, new_val:str):
        if self.units["ytemp_unit"] != new_val:
            self.units["ytemp_unit"] = new_val
            self.update_y_temp_units()
            self.set_axis_labels()

    @Slot(str)
    def yevent_plot_update(self, new_val:str):
        if self.units["yevent_plot_type"] != new_val:
            self.units["yevent_plot_type"] = new_val
            self.update_all_event_lines()
            self.set_axis_labels()
    
    @Slot(bool)
    def smoothing_update(self, new_val:bool):
        if self.units["smoothing"] != new_val:
            self.units["smoothing"] = new_val
            self.rebin_all()
            self.update_all_event_lines()
            self.set_axis_labels()
    
    @Slot(str)
    def yevent_smooting_uni_update(self, new_val:str):
        if self.units["yevent_smoothing_unit"] != new_val:
            self.units["yevent_smoothing_unit"] = new_val
            self.rebin_all()
            self.update_all_event_lines()
            self.set_axis_labels()

    @Slot(float)
    def yevent_smoothing_value_update(self, new_val:float):
        if self.units["yevent_smoothing_value"] != new_val:
            self.units["yevent_smoothing_value"] = new_val
            self.rebin_all()
            self.update_all_event_lines()
            self.set_axis_labels()

    @Slot(int,int)
    def savgol_change(self, new_val1: int, new_val2:int):
        if self.units["savgol_window"] != new_val1:
            self.units["savgol_window"] = new_val1
            self.units["savgol_poly"] = new_val2
            self.update_all_event_lines()
            self.set_axis_labels()
        elif self.units["savgol_poly"] != new_val2:
            self.units["savgol_window"] = new_val1
            self.units["savgol_poly"] = new_val2
            self.update_all_event_lines()
            self.set_axis_labels()
        
    def set_axis_labels(self): 
        if self.units["xaxis"] == "time":
            self.plot_time_label()
        elif self.units["xaxis"] == "temperature":
            self.plot_temp_label(False)
        
        if self.units["yaxis"] == "ratio":
            self.ax.set_ylabel("n/N Trap ratio")
        elif self.units["yaxis"] == "event":
            self.ax.set_ylabel("Intensity")
        elif self.units["yaxis"] == "temperature":
            self.plot_temp_label(True)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()

    def plot_time_label(self,) -> None:
        """Sets Time axis according to the unit type"""

        match self.units["xtime_unit"]: 
            case 'seconds': 
                self.ax.set_xlabel("Time (s)")
            case 'minutes':
                self.ax.set_xlabel("Time (min)")
            case 'hours':
                self.ax.set_xlabel("Time (hour)")
            case 'days':
                self.ax.set_xlabel("Time (day)")
            case 'years':
                self.ax.set_xlabel("Time (year)")
            case 'ma':
                self.ax.set_xlabel("Time (Ma)")
            case _:
                self.ax.set_xlabel("Time (s)")
        self.draw_idle()
    
    def plot_temp_label(self, y_axis: bool=True) -> None:
        """Sets Temperature label accoridng to unit type"""
        if y_axis:
            match self.units["ytemp_unit"]: 
                case 'Celsius':
                    self.ax.set_ylabel("Temperature (°C)")
                case 'Kelvin':
                    self.ax.set_ylabel("Temperature (K)")
        else:
            match self.units["xtemp_unit"]: 
                case 'Celsius':
                    self.ax.set_xlabel("Temperature (°C)")
                case 'Kelvin':
                    self.ax.set_xlabel("Temperature (K)")
        self.draw_idle()

    def add_all_ratio_lines(self,):
        for i in range(self.exp_count):
            self.requestLineInfo.emit(i,"ratio")
            self.requestLineInfo.emit(i,"STD")
            self.requestLineInfo.emit(i,"qlohi")
            self.requestLineInfo.emit(i,"qmi")

    def all_all_event_lines(self,):
        for i in range(self.exp_count):
            self.requestLineInfo.emit(i,"luminescence")
            self.requestLineInfo.emit(i,"fill")
            self.requestLineInfo.emit(i,"retrap")
            self.requestLineInfo.emit(i,"noEvent")
            self.requestLineInfo.emit(i,"GS_tun_recom")
            self.requestLineInfo.emit(i,"GS_CB_recom")
            self.requestLineInfo.emit(i,"GS_tun_retrap")
            self.requestLineInfo.emit(i,"GS_CB_retrap")
            self.requestLineInfo.emit(i,"GS_process")
            self.requestLineInfo.emit(i,"ES_tun_recom")
            self.requestLineInfo.emit(i,"ES_CB_recom")
            self.requestLineInfo.emit(i,"ES_tun_retrap")
            self.requestLineInfo.emit(i,"ES_CB_retrap")
            self.requestLineInfo.emit(i,"ES_process")
            self.requestLineInfo.emit(i,"bleach")

    def add_all_temp_lines(self):
        for i in range(self.exp_count):
            self.requestLineInfo.emit(i,"temperature")

    def update_std(self,exp_num:int, new_x:np.ndarray):
        old_item = self._plot_list[exp_num]["STD"]
        visible = old_item.get_visible()
        label = old_item.get_label()
        colour = old_item.get_facecolor()
        linestyle = old_item.get_linestyle()

        old_item.remove()
        self._plot_list[exp_num]["STD"]= self.std_plot(self.data[exp_num]["std"],self.data[exp_num]["mean"],new_x,colour,linestyle,label)
        if visible:
            self._plot_list[exp_num]["STD"].set_visible(True)
        else:
            self._plot_list[exp_num]["STD"].set_visible(False)
    
    def update_qlohi(self,exp_num:int, new_x:np.ndarray):
        old_item = self._plot_list[exp_num]["qlohi"]
        visible = old_item.get_visible()
        label = old_item.get_label()
        colour = old_item.get_facecolor()
        linestyle = old_item.get_linestyle()
        old_item.remove()
        self._plot_list[exp_num]["qlohi"] = self.quartile_plot(self.data[exp_num]["qlo"],self.data[exp_num]["qhi"],new_x,colour,linestyle,label)
        if visible:
            self._plot_list[exp_num]["qlohi"].set_visible(True)
        else:
            self._plot_list[exp_num]["qlohi"].set_visible(False)
    
    def update_lumfill_line(self,exp_num:int,line1:str,line2:str):
        x_vals = self.get_event_x_data(exp_num)
        if self.units["yevent_plot_type"] == "lines" :
            x =  0.5*(x_vals[:-1] + x_vals[1:])
            # win = min( self.units["savgol_window"], self.data[exp_num][line1].size if self.data[exp_num][line1].size % 2 == 1 else self.data[exp_num][line1].size - 1)
            y = savgol_filter(self.data[exp_num][line1], self.units["savgol_window"], self.units["savgol_poly"])

            # y = self.data[exp_num][line1]
        elif self.units["yevent_plot_type"] == "steps":
            x = x_vals
            y = np.r_[self.data[exp_num][line1][0],self.data[exp_num][line1][:],]
        
        self._plot_list[exp_num][line2].set_data(x,y)

    def update_event_line(self,exp_num:int,yvals:np.ndarray,line1:str):
        x_vals = self.get_event_x_data(exp_num)

        if self.units["yevent_plot_type"] == "lines" :
            x =  0.5*(x_vals[:-1] + x_vals[1:])
            # win = min(self.units["savgol_window"], yvals.size if yvals.size % 2 == 1 else yvals.size - 1)
            y = savgol_filter(yvals, self.units["savgol_window"], self.units["savgol_poly"])
            # y = yvals
        elif self.units["yevent_plot_type"] == "steps":
            x = x_vals
            y = np.r_[yvals[0],yvals[:],]
        
        self._plot_list[exp_num][line1].set_data(x,y)

    def update_all_event_lines(self):
        for i in range(self.exp_count):
            self.update_lumfill_line(i+1,"sLuminescence","luminescence")
            self.update_lumfill_line(i+1,"sFill","fill")
            self.update_event_line(i+1, self.data[i+1]["sEvent"][:,[3, 5, 7, 9]].sum(axis=1),"retrap")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,0] ,"noEvent")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,2] ,"GS_tun_recom")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,6] ,"GS_CB_recom")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,3] ,"GS_tun_retrap")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,7] ,"GS_CB_retrap")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,[2, 3, 6, 7]].sum(axis=1) ,"GS_process")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,4] ,"ES_tun_recom")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,8] ,"ES_CB_recom")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,5] ,"ES_tun_retrap")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,9] ,"ES_CB_retrap")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,[4, 5, 8, 9]].sum(axis=1) ,"ES_process")
            self.update_event_line(i+1,self.data[i+1]["sEvent"][:,10] ,"bleach")
            

        self.draw_idle()

    def get_x_data(self,exp_num):
        if self.units["xaxis"] == "time":
            return self.data[exp_num]["time"]/time_to_seconds[self.units["xtime_unit"]]
        elif self.units["xaxis"] == "temperature":
            match self.units["xtemp_unit"]: 
                case 'Celsius':
                    return self.data[exp_num]["temperature"] - 273.15  
                case 'Kelvin':
                   return self.data[exp_num]["temperature"]

    def get_event_x_data(self,exp_num):
        if self.units["xaxis"] == "time":
            return self.data[exp_num]["sX"]/time_to_seconds[self.units["xtime_unit"]]
        elif self.units["xaxis"] == "temperature":
            match self.units["xtemp_unit"]: 
                case 'Celsius':
                    return self.data[exp_num]["sX"] - 273.15  
                case 'Kelvin':
                   return self.data[exp_num]["sX"] 
    
    def update_x_units(self):
        for i in range(self.exp_count):
            x = self.get_x_data(i+1)
            if self.units["yaxis"] == "ratio":
                y = self._plot_list[i+1]["ratio"].get_ydata()
                self._plot_list[i+1]["ratio"].set_data(x,y)
                y = self._plot_list[i+1]["qmi"].get_ydata()
                self._plot_list[i+1]["qmi"].set_data(x,y)
                self.update_qlohi(i+1,x)
                self.update_std(i+1,x)
           
            elif  self.units["yaxis"] == "temperature":
                y = self._plot_list[i+1]["temperature"].get_ydata()
                self._plot_list[i+1]["temperature"].set_data(x,y)
        if self.units["yaxis"] == "event":
                self.update_all_event_lines()
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()

    def update_y_temp_units(self):
        match self.units["ytemp_unit"]: 
                case 'Celsius':
                    sub = 273.15  
                case 'Kelvin':
                    sub = 0  
        for i in range(self.exp_count):
            x = self.get_x_data(i+1)
            y = self.data[i+1]["temperature"]-sub
            self._plot_list[i+1]["temperature"].set_data(x,y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()
    
    @Slot(int,str)
    def hide_line(self,exp_num:int,line_to_plot: str):
        if self._plot_list[exp_num][line_to_plot]:
            self._plot_list[exp_num][line_to_plot].set_visible(False)
        self.update_legend()
        self.draw_idle()

    @Slot(bool,int,str,str,str,str)
    def update_line(self,plot:bool,exp_num:int,line_to_plot: str,colour:str, linestyle:str, label:str): 
        self._plot_list[exp_num][line_to_plot].set_color(colour)
        self._plot_list[exp_num][line_to_plot].set_linestyle(linestyle)
        self._plot_list[exp_num][line_to_plot].set_label(label)
        if plot:
            self._plot_list[exp_num][line_to_plot].set_visible(True)
        else:
            self._plot_list[exp_num][line_to_plot].set_visible
        self.update_legend()
        self.draw_idle()
    
    def update_legend(self):
        handles = []
        labels = []

        for exp_lines in self._plot_list.values():
            for item in exp_lines.values():
                label = item.get_label()
                if item.get_visible() and label:
                    handles.append(item)
                    labels.append(label)

        old_legend = self.ax.get_legend()
        if old_legend is not None:
            old_legend.remove()
        if labels:
            self.ax.legend(handles, labels)

    @Slot(bool,int,str,str,str,str) 
    def update_plot(self,plot:bool,exp_num:int,line_to_plot: str,colour:str, linestyle:str, label:str):
        match line_to_plot:
            case"temperature":
                x = self.get_x_data(exp_num)
                match self.units["ytemp_unit"]: 
                    case 'Celsius':
                        sub = 273.15  
                    case 'Kelvin':
                        sub = 0  
                y = self.data[exp_num]["temperature"]-sub
                line, = self.temperature_plot(y,x,colour,linestyle,label)
            case"ratio":
                x = self.get_x_data(exp_num)
                line, = self.mean_plot(self.data[exp_num]["mean"],x,colour,linestyle,label) 
            case"STD":
                x = self.get_x_data(exp_num)
                line = self.std_plot(self.data[exp_num]["std"],self.data[exp_num]["mean"],x,colour,linestyle,label)
            case"qlohi":
                x = self.get_x_data(exp_num)
                line = self.quartile_plot(self.data[exp_num]["qlo"],self.data[exp_num]["qhi"],x,colour,linestyle,label)
            case"qmi":
                x = self.get_x_data(exp_num)
                line, = self.median_plot(self.data[exp_num]["qmi"],x,colour,linestyle,label)
            case"luminescence":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sLuminescence"] ,colour,linestyle,label)
            case"fill":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sFill"] ,colour,linestyle,label)
            case"retrap":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,[3, 5, 7, 9]].sum(axis=1) ,colour,linestyle,label)
            case"noEvent":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,0] ,colour,linestyle,label)
            case"GS_tun_recom":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,2] ,colour,linestyle,label)
            case"GS_CB_recom":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,6] ,colour,linestyle,label)
            case"GS_tun_retrap":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,3] ,colour,linestyle,label)
            case"GS_CB_retrap":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,7] ,colour,linestyle,label)
            case"GS_process":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,[2, 3, 6, 7]].sum(axis=1) ,colour,linestyle,label)
            case"ES_tun_recom":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,4] ,colour,linestyle,label)
            case"ES_CB_recom":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,8] ,colour,linestyle,label)
            case"ES_tun_retrap":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,5] ,colour,linestyle,label)
            case"ES_CB_retrap":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,9] ,colour,linestyle,label)
            case"ES_process":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,[4, 5, 8, 9]].sum(axis=1) ,colour,linestyle,label)
            case"bleach":
                x = self.get_event_x_data(exp_num)
                line = self.event_plot(x,self.data[exp_num]["sEvent"][:,10] ,colour,linestyle,label)

                
        self._plot_list[exp_num][line_to_plot]= line
        if plot:
            self._plot_list[exp_num][line_to_plot].set_visible(True)
        else:
            self._plot_list[exp_num][line_to_plot].set_visible(False)
        self.update_legend()
        self.draw_idle()
    
    def mean_plot(self, mean:np.ndarray, x_vals: np.ndarray, colour:str, linestyle:str, label:str): 
       
        x = np.r_[x_vals[:-1], x_vals[-1]]
        y_mean = np.r_[mean[:], mean[-1]]

        return self.ax.step(x, y_mean, where="post", linestyle=linestyle, color=colour, label=label)
        
    def std_plot(self, std:np.ndarray, mean:np.ndarray, x_vals: np.ndarray, colour:str, linestyle:str, label:str, std_num:int=1):     
        std_num = self.units["std_num"]
        x = np.r_[x_vals[:-1], x_vals[-1]]
        y_std = np.r_[std[:], std[-1]]
        y_mean = np.r_[mean[:], mean[-1]]
        y_std_low = y_mean - y_std*std_num
        y_std_high = y_mean + y_std*std_num
        return self.ax.fill_between(x, y_std_low, y_std_high, step="post", alpha=0.2,label=label,linestyle=linestyle, color=colour)

    def quartile_plot(self, lo:np.ndarray, hi:np.ndarray, x_vals: np.ndarray, colour:str, linestyle:str, label:str): 
        x = np.r_[x_vals[:-1], x_vals[-1]]
        y_low = np.r_[lo[:], lo[-1]]
        y_high = np.r_[hi[:], hi[-1]]
        return self.ax.fill_between(x, y_low, y_high, step="post", alpha=0.25,label=label,linestyle=linestyle, color=colour)
        
    def median_plot(self, mi:np.ndarray, x_vals: np.ndarray, colour:str, linestyle:str, label:str):
        x = np.r_[x_vals[:-1], x_vals[-1]]
        y_mid = np.r_[mi[:], mi[-1]]
        return self.ax.step(x, y_mid, where="post", label=label,linestyle=linestyle, color=colour)

    def event_plot(self,x_vals:np.ndarray,event:np.ndarray,colour:str, linestyle:str, label:str):
        if self.units["yevent_plot_type"] == "lines" :
            x =  0.5*(x_vals[:-1] + x_vals[1:])
            
            glow_smoothed = savgol_filter(event, self.units["savgol_window"], self.units["savgol_poly"])

            line, =  self.ax.plot(x,glow_smoothed,color=colour, label = label, linestyle=linestyle)  

        elif self.units["yevent_plot_type"] == "steps":
            y = np.r_[event[0],event[:],]
            line, = self.ax.step(x_vals, y, where="pre",color=colour, label = label, linestyle=linestyle)
        
        return line 

    def temperature_plot(self, temp:np.ndarray, x_vals: np.ndarray, colour:str, linestyle:str, label:str):
        return self.ax.plot(x_vals,temp, color=colour, label = label, linestyle=linestyle,)

    def rebin_all(self):
        for i in range(self.exp_count):
            self.rebin_events(i+1)

    def rebin_events(self, exp_num):
        if  self.units["xaxis"] == "time":
            x = self.data[exp_num]["time"]
            if self.units["yevent_smoothing_value"] != 0:
                bin_width  = self.units["yevent_smoothing_value"]/time_to_seconds[self.units["yevent_smoothing_unit"]]
                n_new, new_xvalues = self.create_new_time_series(x, bin_width)
            else:
                bin_width=0
        elif  self.units["xaxis"] == "temperature":
            x = self.data[exp_num]["temperature"]
            if self.units["yevent_smoothing_value"] != 0:
                bin_width  = self.units["yevent_smoothing_value"]
                n_new, new_xvalues = self.create_new_time_series(x, bin_width)
            else:
                bin_width=0
        if bin_width == 0 or self.units["smoothing"] == False:
            self.data[exp_num]["sX"] = x
            self.data[exp_num]["sLuminescence"] = self.data[exp_num]["luminescence"]
            self.data[exp_num]["sFill"] = self.data[exp_num]["fill"]
            self.data[exp_num]["sEvent"] = self.data[exp_num]["events"]
            return 
       
            
        n_new, new_xvalues = self.create_new_time_series(x, bin_width)
        self.data[exp_num]["sX"] = new_xvalues
        filllum = np.column_stack((self.data[exp_num]["luminescence"],self.data[exp_num]["fill"]))
        accum = self.rebin_event_values_fixed_width(x,filllum,n_new,new_xvalues)
        maximum = np.max(accum[:,0])
        self.data[exp_num]["sLuminescence"] = accum[:,0]/maximum
        self.data[exp_num]["sFill"] = accum[:,1]/maximum
        self.data[exp_num]["sEvent"] = self.rebin_event_values_fixed_width(x,self.data[exp_num]["events"],n_new,new_xvalues)/maximum

    
    def create_new_time_series(self, xvalues, new_bin_width):
        t_min = xvalues[0]
        t_max = xvalues[-1]

        n_new = int(round((t_max - t_min) / new_bin_width))
        new_times = np.linspace(t_min,t_max,n_new+1)
        return n_new, new_times

    def rebin_event_values_fixed_width(self, xvalues, events, n_new, new_xvalues):
   
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

        return accum
    
    @Slot(str)
    def set_title(self, title:str):
        self.ax.set_title(title)
        self.draw_idle()

    @Slot(str)
    def save_image(self, file):
        invert = False
        if self.units["xaxis"] == "time" and self.units["xtime_unit"]=="ma":
            self.ax.invert_xaxis()
            invert = True
        self.fig.savefig(file,dpi=300, transparent=False,bbox_inches='tight')
        if invert:
            self.ax.invert_xaxis()
    
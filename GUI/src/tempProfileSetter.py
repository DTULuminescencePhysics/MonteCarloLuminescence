from __future__ import annotations
import os
import numpy as np
from typing import TYPE_CHECKING 
from PySide6.QtWidgets import QWidget, QVBoxLayout, QMessageBox, QLabel, QScrollArea, QSizePolicy
from PySide6.QtCore import Signal, QTimer, Qt

from GUI.src.designs.addtimeTempPoint import Ui_Form as Ui_timeTempPoint
from GUI.src.toggleButton import add_remove_button
os.environ["QT_API"] = "PySide6"
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure

class timeTempPointAdder(QWidget,Ui_timeTempPoint):
    tt_value = Signal(float,float)
    point_changed = Signal(int,float,float)
    point_removed = Signal(int)
    def __init__(self,ord:int, t:float, temp:float, t_units:int=0,
                 temp_units:bool=True, 
                 mode:str="pending"):
        super().__init__()
        self.setupUi(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        add_remove_button(self.addButton,True)
        add_remove_button(self.removeButton, False)
        self.ord = ord 
        self.time.setValue(t)
        self.temp.setValue(temp)
        self.set_temp_units(temp_units)
        self.set_t_units(t_units)
        self.mode = mode
        self.set_mode()
        self.rebuild_timer = QTimer(self)
        self.rebuild_timer.setSingleShot(True)
        self.rebuild_timer.setInterval(300) 
        self.rebuild_timer.timeout.connect(self.emit_point_changed)

        self.addButton.clicked.connect(self.set_point)
        self.removeButton.clicked.connect(self.remove_point)
        # self.time.valueChanged.connect(self.emit_point_changed)
        # self.temp.valueChanged.connect(self.emit_point_changed)
        self.time.valueChanged.connect(self.edit_begun)
        self.temp.valueChanged.connect(self.edit_begun)
       
    def edit_begun(self,):
        self.rebuild_timer.start()
    
    def set_mode(self):
        if self.mode == "fixed":
            self.setEnabled(False)
            self.addButton.hide()
            self.removeButton.hide()
        elif self.mode == "committed":
            self.setEnabled(True)
            self.addButton.hide()
            self.removeButton.show()
        elif self.mode == "pending":
            self.setEnabled(True)
            self.addButton.show()
            self.removeButton.hide()
    
    def remove_point(self):
        self.point_removed.emit(self.ord)

    def set_point(self):
        self.mode = "committed"
        self.set_mode()
        t = self.time.value()
        temp = self.temp.value()
        self.tt_value.emit(t,temp)
    
    def emit_point_changed(self):
        t = self.time.value()
        temp = self.temp.value()
        self.point_changed.emit(self.ord,t,temp)

    def set_temp_units(self,celsius:bool):
        if celsius:
            self.celsiusSymbol.show()
            self.kelviSymbol.hide()
        else:
            self.celsiusSymbol.hide()
            self.kelviSymbol.show()

    def set_t_units(self,units:int):
        if units == 0:
            self.timeUnits.setText("seconds")
        elif units ==1:
            self.timeUnits.setText("minutes")
        elif units ==2:
            self.timeUnits.setText("hours")
        elif units ==3:
            self.timeUnits.setText("days")
        elif units ==4:
            self.timeUnits.setText("years")
        elif units ==5:
            self.timeUnits.setText("Ka")
        elif units ==6:
            self.timeUnits.setText("Ma")

class timeTempList(QWidget):
    profile_changed = Signal(np.ndarray, np.ndarray)  # times, temps
    
    def __init__(self,times: list[float], temps: list[float], duration: float, 
                 t_units: int = 0,temp_units: bool = True,parent=None,):
        super().__init__(parent)
        self.kind = 0
        if len(times) != len(temps):
            raise ValueError("times and temps must have the same length.")

        if len(times) < 2:
            raise ValueError("A temperature profile needs at least start and end points.")

        if times[0] != 0:
            raise ValueError("The first time point must be 0.")

        if times[-1] != duration:
            raise ValueError("The final time point must equal duration.")
        self.times = np.array(times) 
        self.temps = np.array(temps)
        self.duration = duration
        self.t_units = t_units
        self.temp_units = temp_units
        self.point_widgets: list[timeTempPointAdder] = []
        self.pending_widget: timeTempPointAdder | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
      
        self.scrollarea = QScrollArea()
        self.scrollarea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scrollarea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scrollarea.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scrollarea.setMinimumWidth(800)
        self.scrollarea.setMinimumHeight(360)
        self.scrollarea.setWidgetResizable(True)
        self.scrollContent = QWidget()
        self.scrollContent.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.layout = QVBoxLayout(self.scrollContent)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(6)
        self.layout.addStretch() 
        self.scrollarea.setWidget(self.scrollContent)
        label = QLabel("Time Temperature List")
        self.mainLayout.addWidget(label)
        self.mainLayout.addWidget(self.scrollarea, 1)
        self.profile_changed.emit(self.times,self.temps)
        self.rebuild()
        
    def set_start_temperature(self, temp: float):
        if self.kind == 0:
            self.temps = np.array((temp,temp))
        else:
            self.temps[0] = temp
        
        self.point_widgets[0].temp.setValue(temp)
        
        self.profile_changed.emit(self.times,self.temps)

    def set_end_temperature(self, temp: float):
        self.temps[-1] = temp
        self.point_widgets[-1].temp.setValue(temp)
        self.profile_changed.emit(self.times,self.temps)


    def set_duration(self, duration: float):
        if duration <= 0:
            raise ValueError("duration must be greater than zero.")
        if self.t_units <5:
            if duration >= self.times[-2]:
                self.times[-1] = duration
                self.point_widgets[-1].time.setValue(duration)
                self.profile_changed.emit(self.times,self.temps)
            else: 
                for i in range(self.times.size):
                    if self.times[i]>= duration:
                        temp = self.temps[-1]
                        self.times = self.times[:i+1]
                        self.temps = self.temps[:i+1]
                        self.times[-1] = duration 
                        self.temps[-1] = temp
                        break
        else:
            if duration >= self.times[1]: 
                self.times[0] = duration
                self.point_widgets[0].time.setValue(duration)
                self.profile_changed.emit(self.times,self.temps)
            else:
                for i in range(self.times.size):
                    if self.times[i] <= duration:
                        temp = self.temps[0]
                        self.times = self.times[i-1:]
                        self.temps = self.temps[i-1:]
                        self.times[0] = duration 
                        self.temps[0] = temp
            self.profile_changed.emit(self.times,self.temps)
            self.rebuild()
    
    def resetPoints(self,times: list[float],temps: list[float], duration: float, t_units: int = 0,temp_units: bool = True):
        self.temp_units = temp_units
        self.duration = duration
        self.t_units = t_units

        if self.t_units <5:
            self.times = np.array(times) 
            self.temps = np.array(temps)
        else: 
            self.times = abs(np.array(times)-self.duration)
            self.temps = np.array(temps)
        self.profile_changed.emit(self.times,self.temps)

        self.duration = duration
        self.t_units = t_units
        self.rebuild()

    def rebuild(self):
        self.clear_layout()
        self.point_widgets = []

        for index  in range(self.times.size):
            if index  == 0: 
                mode = "fixed"
            elif index == self.times.size - 1:
                mode = "fixed"
            else:
                mode = "committed"
            

            widget = timeTempPointAdder(index, t=self.times[index],temp=self.temps[index], t_units=self.t_units,temp_units=self.temp_units,mode=mode,)
            widget.point_removed.connect(self.remove_point_widget)
            widget.point_changed.connect(self.update_internal_point)
            self.point_widgets.append(widget)
            self.layout.addWidget(widget)
        self.add_pending_widget()
    
    def add_pending_widget(self):

        default_t = self.suggest_new_time()
        default_temp = self.interpolate_temp(default_t)

        self.pending_widget = timeTempPointAdder(-1,t=default_t,temp=default_temp,t_units=self.t_units,temp_units=self.temp_units,mode="pending",)
        self.pending_widget.tt_value.connect(self.add_point)
        self.layout.addWidget(self.pending_widget)

    def add_point(self, t: float, temp: float):
        if self.t_units <5:
            if(t <= self.times[0] or t >= self.times[-1]):
                QMessageBox.warning(self,
                    "Invalid time point",
                    "Internal time points must be between the start and end times.",
                )
                return
            index = np.searchsorted(self.times, t, side="right")
            self.times = np.insert(self.times, index, t)
            self.temps = np.insert(self.temps,index, temp)
        else: 
            if(t >= self.times[0] or t <= self.times[-1]):
                QMessageBox.warning(self,
                    "Invalid time point",
                    "Internal time points must be between the start and end times.",
                )
                return
            index = np.searchsorted(-self.times, -t, side="right")
            self.times = np.insert(self.times, index, t)
            self.temps = np.insert(self.temps,index, temp)
       
        self.profile_changed.emit(self.times,self.temps)
        self.rebuild()
      

    def remove_point_widget(self, ord:int):
        np.delete(self.times,ord)
        np.delete(self.temps,ord)
        self.profile_changed.emit(self.times,self.temps)
        self.rebuild()
    
    def update_internal_point(self,ord:int,t:float,temp:float):
        if t == self.times[ord]: 
            self.temps[ord] = temp
            self.profile_changed.emit(self.times,self.temps)
            return
        if self.t_units <5:
            if t >= self.times[ord-1] and t <=self.times[ord+1]:
                self.times[ord] = t 
                self.temps[ord] = temp
                self.profile_changed.emit(self.times,self.temps)
                return
            else:
                if t >= self.times[0] and t <=self.times[-1]:
                    if t < self.times[ord-1]:
                        for i in range(ord-1,0,-1):
                            self.times[i+1]=self.times[i]
                            self.temps[i+1]=self.temps[i]
                            if t>= self.times[i-1]:
                                self.times[i]=t
                                self.temps[i]=temp
                                break 
                    elif t > self.times[ord+1]:
                        for i in range(ord+1,self.times.size):
                            self.times[i-1]= self.times[i]
                            self.temps[i-1]=self.temps[i]
                            if t <= self.times[i+1]:
                                self.times[i]=t
                                self.temps[i]=temp
                                break
            self.profile_changed.emit(self.times,self.temps)
            self.rebuild()
        else: 
            if t >= self.times[ord+1]  and t <= self.times[ord-1]:
                self.times[ord] = t 
                self.temps[ord] = temp
                self.profile_changed.emit(self.times,self.temps)
                return
            else:
                if t <= self.times[0] and t >= self.times[-1]:
                    if t > self.times[ord-1]:
                        for i in range(ord-1,0,-1):
                            self.times[i+1]=self.times[i]
                            self.temps[i+1]=self.temps[i]
                            if t<= self.times[i-1]:
                                self.times[i]=t
                                self.temps[i]=temp
                                break 
                    elif t < self.times[ord+1]:
                        for i in range(ord+1,self.times.size):
                            self.times[i-1]= self.times[i]
                            self.temps[i-1]=self.temps[i]
                            if t >= self.times[i+1]:
                                self.times[i]=t
                                self.temps[i]=temp
                                break
            self.profile_changed.emit(self.times,self.temps)
            self.rebuild()

    def clear_layout(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)
                child.deleteLater()

    def suggest_new_time(self) -> float:
        largest_gap = np.argmax(np.diff(self.times))
        suggested_time = (self.times[largest_gap] + self.times[largest_gap+1]) / 2
        return suggested_time

    def interpolate_temp(self, t: float) -> float:
        for i in range(self.times.size-1):
            if self.times[i] <= t <= self.times[i+1]:
                if self.times[i] == self.times[i+1]:
                    return self.temps[i]
                frac = (t - self.times[i]) / (self.times[i+1] - self.times[i])
                return self.temps[i] + frac * (self.temps[i+1] - self.temps[i])

        return self.temps[-1]

    def set_temp_units(self, celsius: bool):
        if celsius == self.temp_units:
            return
        else:
            if celsius: 
                self.temps = self.temps -273.15 
            else: 
                self.temps = self.temps +273.15

            self.temp_units = celsius
            self.profile_changed.emit(self.times,self.temps)
            self.rebuild()

    def set_t_units(self, units: int):
        if (units < 5 and self.t_units > 4)or(units>4 and self.t_units <5):
            self.times = abs(self.times - self.duration)
            self.profile_changed.emit(self.times,self.temps)
            self.t_units = units
            self.rebuild()
        else:
            self.t_units = units
            for widget in self.point_widgets:
                widget.set_t_units(units)

            if self.pending_widget is not None:
                self.pending_widget.set_t_units(units)
    
    def get_temps(self): 
        return self.temps
    
    def get_times(self):
        if self.t_units < 5: 
            return self.times 
        else: 
            return abs(self.times - self.duration)
     
class TempMplCanvas(FigureCanvas):

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
        self.isflipped = False
        self.reverse = False
        self.times = None
        self.temps = None
        self.set_x_units(0)
        self.set_y_units(True)
       
       

    def set_x_units(self, units:int):
        self.reverse = False
        if units == 0:
            self.ax.set_xlabel("Time (s)")
        elif units ==1:
            self.ax.set_xlabel("Time (min)")
        elif units ==2:
            self.ax.set_xlabel("Time (hour)")
        elif units ==3:
            self.ax.set_xlabel("Time (day)")
        elif units ==4:
            self.ax.set_xlabel("Time (year)")
        elif units ==5:
            self.ax.set_xlabel("Time (Ka)")
            self.reverse = True
        elif units ==6:
            self.ax.set_xlabel("Time (Ma)")
            self.reverse = True
        self.plot_profile()


    def set_y_units(self,celsius:bool):
        if celsius:
            self.ax.set_ylabel("Temperature (°C)")
        else:
            self.ax.set_ylabel("Temperature (K)")
        self.draw_idle()

    def update_plot(self,times, temps):
        self.times = times
        self.temps = temps
        self.plot_profile()

    def plot_profile(self):
        for line in list(self.ax.lines):
            line.remove()

        if self.times is not None and self.temps is not None:
         
         
            self.ax.plot(self.times, self.temps, marker="o")
            if self.reverse:
                self.ax.set_xlim(self.times.max(), self.times.min())
            else:
                self.ax.set_xlim(self.times.min(), self.times.max())
        self.ax.relim()
        self.ax.autoscale_view()
            
        # self.get_right_flip(to_plot=False)
     
        self.draw_idle()

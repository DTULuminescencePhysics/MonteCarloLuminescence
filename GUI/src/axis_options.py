from __future__ import annotations

from PySide6.QtWidgets import QWidget, QButtonGroup
from PySide6.QtCore import Signal, Slot
import numpy as np
from GUI.src.designs.time_buttons import Ui_Form as TimeButtonsUi
from GUI.src.designs.error_buttons import Ui_Form as ErrorButtonsUi
from GUI.src.designs.event_buttons import Ui_Form as EventButtonsUi
from GUI.src.designs.temperature_buttons import Ui_Form as TemperatureButtonsUi


class TimeOptions(QWidget, TimeButtonsUi): 
    timeUnitChanged = Signal(str)
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.buttonGroup = QButtonGroup(self)
        self._button_setup()
        self.buttonGroup.buttonToggled.connect(self._on_button_toggled)

    def _on_button_toggled(self, button, checked):
        if checked:
            self.timeUnitChanged.emit(self._button_values[button])
       
    def _button_setup(self): 
        self._button_values = {
            self.secs: "seconds",
            self.mins: "minutes",
            self.hours: "hours",
            self.days : "days",
            self.years: "years",
            self.ka : "ka",
            self.ma: "ma",
        }
        self.buttonGroup.setExclusive(True)
        self.buttonGroup.addButton(self.secs)
        self.buttonGroup.addButton(self.mins)
        self.buttonGroup.addButton(self.days)
        self.buttonGroup.addButton(self.hours)
        self.buttonGroup.addButton(self.years)
        self.buttonGroup.addButton(self.ka)
        self.buttonGroup.addButton(self.ma)
        self.secs.setChecked(True)
    

    def checked_check(self,): 
        button = self.buttonGroup.checkedButton()
        return self._button_values[button]
    
    @Slot(float)
    def check_minimum_units(self, time:float):
        self.secs.setEnabled(True)
        self.mins.setEnabled(True)
        self.hours.setEnabled(True)
        self.days.setEnabled(True)
        self.years.setEnabled(True)
        self.ka.setEnabled(True)
        self.ma.setEnabled(True)
        if time < 60: #secs
            self.mins.setEnabled(False)
            self.hours.setEnabled(False)
            self.days.setEnabled(False)
            self.years.setEnabled(False)
            self.ka.setEnabled(False)
            self.ma.setEnabled(False)
        elif time < 3600: #hours
            self.days.setEnabled(False)
            self.years.setEnabled(False)
            self.ka.setEnabled(False)
            self.ma.setEnabled(False)
        elif time < 86400: #days
            self.years.setEnabled(False)
            self.ka.setEnabled(False)
            self.ma.setEnabled(False)
            self.mins.setChecked(True)
        elif time < 31556952: #years
            self.secs.setEnabled(False)
            self.mins.setEnabled(False)
            self.hours.setEnabled(False)
            self.ka.setEnabled(False)
            self.ma.setEnabled(False)
            self.years.setChecked(True)
        elif time < 3.155695200: #10 years
            self.secs.setEnabled(False)
            self.mins.setEnabled(False)
            self.hours.setEnabled(False)
            self.days.setEnabled(False)
            self.ka.setEnabled(False)
            self.ma.setEnabled(False)
            self.years.setChecked(True)       
        else: 
            self.secs.setEnabled(False)
            self.mins.setEnabled(False)
            self.hours.setEnabled(False)
            self.days.setEnabled(False)
            self.years.setEnabled(False)
            self.ka.setChecked(True)


        

class ErrorOptions(QWidget, ErrorButtonsUi): 
    OptionToggled = Signal(str,bool)
    STD_value = Signal(int)
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.stdSlider.setEnabled(False)
        self.std.setChecked(False)
        self.median.setChecked(False)
        self.quartile.setChecked(False)
        self.std.toggled.connect(self.stdSlider.setEnabled)
        self.std.toggled.connect(self.stdOptionEmitter)
        self.median.toggled.connect(self.medianOptionEmitter)
        self.quartile.toggled.connect(self.quartileOptionEmitter)
        self.stdSlider.sliderMoved.connect(self.slider_moved)

    def stdOptionEmitter(self,choice:bool):
        self.OptionToggled.emit('std',choice)
    
    def medianOptionEmitter(self,choice:bool):
        self.OptionToggled.emit('m',choice)
    
    def quartileOptionEmitter(self,choice:bool):
        self.OptionToggled.emit('q',choice)
    
    def slider_moved(self,value:int):
        self.stds.setText(f"{value}")
        self.STD_value.emit(value)

class TemperatureOptions(QWidget, TemperatureButtonsUi): 
    TemperatureChange = Signal(str)
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.buttonGroup = QButtonGroup(self)
        self.buttonGroup.addButton(self.celsius)
        self.buttonGroup.addButton(self.kelvin)
        self.buttonGroup.setExclusive(True)
        self.celsius.setChecked(True)
        self.buttonGroup.buttonToggled.connect(self._Temp_toggeled)
        self._tempbutton_values = {
            self.celsius: "Celsius",
            self.kelvin: "Kelvin",
        }

    def _Temp_toggeled(self,button, checked):
       if checked:
            self.TemperatureChange.emit(self._tempbutton_values[button])
    
    def checked_check(self,): 
        button = self.buttonGroup.checkedButton()
        return self._tempbutton_values[button]

class EventOptions(QWidget, EventButtonsUi):
    lineType = Signal(str)
    smoothingUnit = Signal(str)
    smoothingSignal = Signal(float)
    savgolSignal = Signal(int,int)
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.buttonGroup = QButtonGroup(self)
        self.buttonGroup.addButton(self.steps)
        self.buttonGroup.addButton(self.lines)
        self.buttonGroup.setExclusive(True)
        self.steps.setChecked(True)
        self.changeTempUnit("Celsius")
        self.changeTimeUnitList("seconds")
        self.smoothing.setChecked(False)
        self.smootingVal.setEnabled(False)
        self.timeUnits.setEnabled(False)
        self.timeUnits.currentIndexChanged.connect(self.emit_smoothing_unit)
        self.smoothing.toggled.connect(self.smootingVal.setEnabled)
        self.smoothing.toggled.connect(self.timeUnits.setEnabled)
        self.buttonGroup.buttonToggled.connect(self.plot_toggeled)
        self.smootingVal.valueChanged.connect(self.smoothing_changed)
        self.savgol_window.setValue(15)
        self.savgol_window.setMinimum(1)
        self.savgol_window.setEnabled(False)
        self.savgol_poly.setValue(2)
        self.savgol_poly.setMinimum(0)
        self.savgol_poly.setEnabled(False)
        self.savgol_poly.setMaximum(14)

        self.savgol_poly.valueChanged.connect(self.savgol_changed)
        self.savgol_window.valueChanged.connect(self.savgol_changed)

        self.savgolWinVals = np.array((14,np.inf,np.inf,np.inf))
        self._button_values = {
            self.steps: "steps",
            self.lines: "lines",
        }
    @Slot(int,int)
    def savgol_window_max_setter(self, exp_num:int, length:int):
        self.savgolWinVals[exp_num] = length
        self.savgol_window.setMaximum(self.savgolWinVals.min())

    def savgol_changed(self):
        self.savgol_poly.setMaximum(self.savgol_window.value()-1)
        self.savgolSignal.emit(self.savgol_window.value(),self.savgol_poly.value())

    def plot_toggeled(self,button, checked):
       if checked:
            self.lineType.emit(self._button_values[button])
            if self._button_values[button] == "lines":
                self.savgol_window.setEnabled(True)
                self.savgol_poly.setEnabled(True)
                self.savgolSignal.emit(self.savgol_window.value(),self.savgol_poly.value())
            else:
                self.savgol_window.setEnabled(False)
                self.savgol_poly.setEnabled(False)

    def checked_check(self,): 
        button = self.buttonGroup.checkedButton()
        return self._button_values[button]
    
    @Slot(str)
    def changeTempUnit(self, sig:str): 
        self.tempUnits.setText(sig)
        self.smoothingUnit.emit(sig)
    
    @Slot(str)
    def changeTimeUnitList(self, sig:str):
        match sig: 
            case "seconds":
                self.timeUnits.model().item(0).setEnabled(True)
                self.timeUnits.model().item(1).setEnabled(False)
                self.timeUnits.model().item(2).setEnabled(False)
                self.timeUnits.model().item(3).setEnabled(False)
                self.timeUnits.model().item(4).setEnabled(False)
                self.timeUnits.model().item(5).setEnabled(False)
                self.timeUnits.model().item(6).setEnabled(False)
                self.timeUnits.setCurrentIndex(0)
            case "minutes":
                self.timeUnits.model().item(0).setEnabled(True)
                self.timeUnits.model().item(1).setEnabled(True)
                self.timeUnits.model().item(2).setEnabled(False)
                self.timeUnits.model().item(3).setEnabled(False)
                self.timeUnits.model().item(4).setEnabled(False)
                self.timeUnits.model().item(5).setEnabled(False)
                self.timeUnits.model().item(6).setEnabled(False)
                self.timeUnits.setCurrentIndex(1)
            case "hours":
                self.timeUnits.model().item(0).setEnabled(True)
                self.timeUnits.model().item(1).setEnabled(True)
                self.timeUnits.model().item(2).setEnabled(True)
                self.timeUnits.model().item(3).setEnabled(False)
                self.timeUnits.model().item(4).setEnabled(False)
                self.timeUnits.model().item(5).setEnabled(False)
                self.timeUnits.model().item(6).setEnabled(False)
                self.timeUnits.setCurrentIndex(2)
            case "days":
                self.timeUnits.model().item(0).setEnabled(True)
                self.timeUnits.model().item(1).setEnabled(True)
                self.timeUnits.model().item(2).setEnabled(True)
                self.timeUnits.model().item(3).setEnabled(True)
                self.timeUnits.model().item(4).setEnabled(False)
                self.timeUnits.model().item(5).setEnabled(False)
                self.timeUnits.model().item(6).setEnabled(False)
                self.timeUnits.setCurrentIndex(3)
            case "years":
                self.timeUnits.model().item(0).setEnabled(False)
                self.timeUnits.model().item(1).setEnabled(False)
                self.timeUnits.model().item(2).setEnabled(True)
                self.timeUnits.model().item(3).setEnabled(True)
                self.timeUnits.model().item(4).setEnabled(True)
                self.timeUnits.model().item(5).setEnabled(False)
                self.timeUnits.model().item(6).setEnabled(False)
                self.timeUnits.setCurrentIndex(4)
            case "ka": 
                self.timeUnits.model().item(0).setEnabled(False)
                self.timeUnits.model().item(1).setEnabled(False)
                self.timeUnits.model().item(2).setEnabled(False)
                self.timeUnits.model().item(3).setEnabled(False)
                self.timeUnits.model().item(4).setEnabled(True)
                self.timeUnits.model().item(5).setEnabled(True)
                self.timeUnits.model().item(6).setEnabled(False)
                self.timeUnits.setCurrentIndex(5)
            case "ma": 
                self.timeUnits.model().item(0).setEnabled(False)
                self.timeUnits.model().item(1).setEnabled(False)
                self.timeUnits.model().item(2).setEnabled(False)
                self.timeUnits.model().item(3).setEnabled(False)
                self.timeUnits.model().item(4).setEnabled(True)
                self.timeUnits.model().item(5).setEnabled(True)
                self.timeUnits.model().item(6).setEnabled(True)
                self.timeUnits.setCurrentIndex(6)


    def emit_smoothing_unit(self):
        if self.timeUnits.currentIndex() == 0:
            self.smoothingUnit.emit("seconds")
        elif self.timeUnits.currentIndex() == 1:
            self.smoothingUnit.emit("minutes")
        elif self.timeUnits.currentIndex() == 2:
            self.smoothingUnit.emit("hours")

        elif self.timeUnits.currentIndex() == 3:
            self.smoothingUnit.emit("days")

        elif self.timeUnits.currentIndex() == 4:
            self.smoothingUnit.emit("years")

        elif self.timeUnits.currentIndex() == 5:
            self.smoothingUnit.emit("ka")

        elif self.timeUnits.currentIndex() == 6:
            self.smoothingUnit.emit("ma")
    
    def smoothing_changed(self):
        self.smoothingSignal.emit(self.smootingVal.value())

         



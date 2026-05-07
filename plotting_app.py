from __future__ import annotations
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QButtonGroup
from PySide6.QtCore import Signal, Slot, QTimer

from GUI.src.axis_options import TimeOptions, ErrorOptions, TemperatureOptions, EventOptions
from GUI.src.linePicker import experimentLineList
from GUI.src.file_load import FileSelector
from GUI.src.designs.mainWindow import Ui_MainWindow
from GUI.src.outputfile_interface import MplCanvas

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class MainWindow(QMainWindow, Ui_MainWindow):
    XAxisToggled = Signal(int)
    YAxisToggled = Signal(int)
    XAxisValues = Signal(str)
    YAxisValues = Signal(str)
    titleSignal = Signal(str)
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.line_choosers = {}
        self.x_axis_buttons = QButtonGroup(self)
        self.y_axis_buttons = QButtonGroup(self)
        self._set_xaxis_buttons()
        self._set_yaxis_buttons()
       
        self.fileloader = FileSelector()
        self.FileChoosebutton.clicked.connect(self.fileloader.open_dialog)
       
        self.fileloader.fileSelected.connect(self.new_file_selected)

        self.DisplayArea = MplCanvas( self)
        self.TopSection.addWidget(self.DisplayArea)
        self.DisplayArea.experimentcount.connect(self.set_line_list)
        self.fileloader.fileSelected.connect(self.DisplayArea.experiment_number_check)
        
        self.x_axis_buttons.buttonToggled.connect(self._Xaxis_toggeled)
        self.y_axis_buttons.buttonToggled.connect(self._Yaxis_toggeled)


        self.XAxisToggled.connect(self.xaxis_button_stackedWidget.setCurrentIndex)
        self.XAxisToggled.connect(self.YEventOptions.unit.setCurrentIndex)
        self.XAxisToggled.connect(self._xdouble_temp_check)

        self.YAxisToggled.connect(self.yaxis_button_stackedWidget.setCurrentIndex)
        self.YAxisToggled.connect(self._ydouble_temp_check)

        self.XTempOptions.TemperatureChange.connect(self.YEventOptions.changeTempUnit)
        self.XTimeOptions.timeUnitChanged.connect(self.YEventOptions.changeTimeUnitList)
        self.YFillOptions.OptionToggled.connect(self.filling_option_changes)

        self.DisplayArea.requestLineInfo.connect(self.line_request)
        self.xaxis_button_stackedWidget.currentChanged.connect(self.Axis_value_is_altered)
        self.yaxis_button_stackedWidget.currentChanged.connect(self.Axis_value_is_altered)
        self.YFillOptions.STD_value.connect(self.DisplayArea.std_value_changed)
        self.XAxisValues.connect(self.DisplayArea.xaxis_update)
        self.YAxisValues.connect(self.DisplayArea.yaxis_update)
        self.XTimeOptions.timeUnitChanged.connect(self.DisplayArea.xtime_update)
        self.XTempOptions.TemperatureChange.connect(self.DisplayArea.xtemp_update)
        self.YTempOptions.TemperatureChange.connect(self.DisplayArea.ytemp_update)
        self.YEventOptions.lineType.connect(self.DisplayArea.yevent_plot_update)
        self.YEventOptions.smoothingUnit.connect(self.DisplayArea.yevent_smooting_uni_update)
        self.YEventOptions.smoothing.toggled.connect(self.DisplayArea.smoothing_update)
        self.YEventOptions.smoothingSignal.connect(self.DisplayArea.yevent_smoothing_value_update)
        self.YEventOptions.savgolSignal.connect(self.DisplayArea.savgol_change)

        self.text_change_timer = QTimer(self)
        self.text_change_timer.setSingleShot(True)
        self.text_change_timer.setInterval(1000)

        self.TitleInput.textChanged.connect(self._text_change_delayed)
        self.text_change_timer.timeout.connect(self._apply_text_change)
        self.titleSignal.connect(self.DisplayArea.set_title)

        self.saveButton.clicked.connect(self.fileloader.open_save_dialog)
        self.fileloader.savedPlot.connect(self.DisplayArea.save_image)
    @Slot(Path)
    def new_file_selected(self, file_path: Path):
        self.FileName.setText(str(file_path.name))

    @Slot(int)
    def set_line_list(self,exp_num:int):
        choice = self.yaxis_button_stackedWidget.currentIndex()
       
        if exp_num == 1: 
            self.LineChooserSetup(0,choice, False)
        else:
            for i in range(exp_num):
                self.LineChooserSetup(i,choice, True)
        
        self.DisplayArea.data_retrieve()
        self.Axis_value_is_altered()

    def _set_xaxis_buttons(self,):
      
        self.x_axis_buttons.addButton(self.XTime)
        self.x_axis_buttons.addButton(self.Xtemp)
        self.x_axis_buttons.setExclusive(True)
        self.XTime.setChecked(True)
        self.XTimeOptions = TimeOptions()
        self.XTempOptions = TemperatureOptions()
        self.xaxis_button_stackedWidget.addWidget(self.XTimeOptions)
        self.xaxis_button_stackedWidget.addWidget(self.XTempOptions)
        self.xaxis_button_stackedWidget.setCurrentIndex(0)
        self._xbutton_values = {
            self.XTime: 0,
            self.Xtemp: 1,
        }

    def _Xaxis_toggeled(self,button, checked):
        if checked: 
            self.XAxisToggled.emit(self._xbutton_values[button])

    
    def Axis_value_is_altered(self,):
       
        if self.xaxis_button_stackedWidget.currentIndex() == 0:
            self.XAxisValues.emit("time")
        else:
            self.XAxisValues.emit("temperature")
        if self.yaxis_button_stackedWidget.currentIndex() == 0:
            self.YAxisValues.emit("ratio")
            y_axis_type = 0 
        elif self.yaxis_button_stackedWidget.currentIndex() == 1:
            self.YAxisValues.emit("event")
            y_axis_type = 1
        else:
            self.YAxisValues.emit("temperature")
            y_axis_type = 2
        self.line_options_on_off(y_axis_type)

    def line_options_on_off(self,choice:int):
        
        for exp in self.line_choosers.values():
            exp.on_off_selector(choice) 

    def _Yaxis_toggeled(self,button, checked):
        if checked: 
            self.YAxisToggled.emit(self._ybutton_values[button])

    def _set_yaxis_buttons(self,):
        self.y_axis_buttons.addButton(self.Yfill)
        self.y_axis_buttons.addButton(self.Yevents)
        self.y_axis_buttons.addButton(self.Ytemp)
        self.y_axis_buttons.setExclusive(True)
        self.Yfill.setChecked(True)
        self.YFillOptions = ErrorOptions()
        self.YEventOptions = EventOptions()
        self.YTempOptions = TemperatureOptions()
        self.yaxis_button_stackedWidget.addWidget(self.YFillOptions)
        self.yaxis_button_stackedWidget.addWidget(self.YEventOptions)
        self.yaxis_button_stackedWidget.addWidget(self.YTempOptions)
        self.yaxis_button_stackedWidget.setCurrentIndex(0)
        self._ybutton_values = {
            self.Yfill: 0,
            self.Yevents: 1,
            self.Ytemp: 2
        }
    
    @Slot(int)
    def _xdouble_temp_check(self, unit:int):
        if unit == 1:
            if self.yaxis_button_stackedWidget.currentIndex() == 2:
                self.Yfill.click()

    @Slot(int)
    def _ydouble_temp_check(self, unit:int):
        if unit == 2:
            if self.xaxis_button_stackedWidget.currentIndex() == 1:
                self.XTime.click()

    def LineChooserSetup(self, exp_num:int, choice: int, multi_exp: bool = False):
        exp = experimentLineList(exp_num,multi_exp)
        exp.on_off_selector(choice)
        exp.LineInfo.connect(self.DisplayArea.update_line)
        exp.LineInfoShort.connect(self.DisplayArea.update_plot)
        exp.LineOff.connect(self.DisplayArea.hide_line)
        exp_name = f"exp{exp_num+1}"
        self.line_choosers[exp_name] = exp
        if multi_exp:
            tab_name = f"Experiment {exp_num+1}"
        else:
            tab_name = ""
        self.LinePlotter.insertTab(exp_num, exp, tab_name)

        setattr(self,exp_name,exp)
    
    @Slot(int,str)
    def line_request(self,exp_num,name):
        exp = f"exp{exp_num+1}"
        self.line_choosers[exp].emit_short_info(name)
    
    @Slot(str,bool) 
    def filling_option_changes(self,opt:str,choice:bool):
        if opt == 'std':
            for exp in self.line_choosers.values():
                exp.std_check(choice)
        elif opt == 'm':
            for exp in self.line_choosers.values():
                exp.median_check(choice)
        elif opt == 'q':
            for exp in self.line_choosers.values():
                exp.quartile_check(choice)

    def _text_change_delayed(self, text: str):
        self._pending_name = text
        self.text_change_timer.start()

    def _apply_text_change(self):
        name = self._pending_name
        self.titleSignal.emit(name)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec())
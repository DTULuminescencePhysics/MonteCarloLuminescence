from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLineEdit
from PySide6.QtCore import QTimer, Signal
from GUI.src.designs.Lineplotter import Ui_Form as LinePlotterUi

linestyles = {
    0 : "solid",
    1 : "dotted",
    2 : "dashed",
    3 : "dashdot",
}
colourList = {
    0 : "black",
    1 : "gray",
    2 : "darkred",
    3 : "red",
    4 : "orange",
    5 : "olive",
    6 : "yellow",
    7 : "green",
    8 : "cyan",
    9 : "blue",
    10 : "darkviolet",
    11 : "deeppink",
}

class SingleLine(QWidget, LinePlotterUi):
    LineInfo = Signal(bool,int,str,str,str,str)
    LineOff = Signal(int,str)
    LineInfoShort = Signal(bool,int,str,str,str,str)
    def __init__(self, name: str, linetype: int, col: int, exp_num:int, lineName:str):
        super().__init__()
        self.setupUi(self)
        self.lineName = lineName
        self.exp_num = exp_num
        self.backup_name = name
        self.name = name
        self.lineEdit.setEnabled(False)
        self.Colour.setEnabled(False)
        self.LineStyle.setEnabled(False)
        self.PlotCheck.setEnabled(False)
        self.lineEdit.setPlaceholderText(name)
        self.lineEdit.setText(name)
        self.linetype = linestyles[linetype]
        self.colourname = colourList[col]

        self.Colour.setCurrentIndex(col)
        self.LineStyle.setCurrentIndex(linetype)

        self.LineStyle.currentIndexChanged.connect(self._linetype_change)
        self.Colour.currentIndexChanged.connect(self._colour_change)

        self.text_change_timer = QTimer(self)
        self.text_change_timer.setSingleShot(True)
        self.text_change_timer.setInterval(1000)

        self.lineEdit.textChanged.connect(self._text_change_delayed)
        self.text_change_timer.timeout.connect(self._apply_text_change)
        self.PlotCheck.toggled.connect(self.line_toggled)
    
    def emit_short_info(self):
        if self.PlotCheck.isChecked():
            self.LineInfoShort.emit(True,self.exp_num,self.lineName,self.colourname,self.linetype,self.name)
        else:
            self.LineInfoShort.emit(False,self.exp_num,self.lineName,self.colourname,self.linetype,self.name)

    def line_change_signal_emit(self):
        if self.PlotCheck.isChecked():
            self.LineInfo.emit(True,self.exp_num,self.lineName,self.colourname,self.linetype,self.name)
        else:
            self.LineInfo.emit(False,self.exp_num,self.lineName,self.colourname,self.linetype,self.name)

    def line_toggled(self):
        if self.PlotCheck.isChecked():
            self.line_change_signal_emit() 
        else:
            self.LineOff.emit(self.exp_num,self.lineName)

    def _linetype_change(self, index: int):
        self.linetype = linestyles[index]
        self.line_change_signal_emit() 


    def _colour_change(self, index: int):
        self.colourname = colourList[index]
        self.line_change_signal_emit() 


    def _text_change_delayed(self, text: str):
        self._pending_name = text
        self.text_change_timer.start()

    def _apply_text_change(self):
        self.name = self._pending_name
        
        self.line_change_signal_emit() 

    def on_off(self,switch:bool): 
        self.lineEdit.setEnabled(switch)
        self.Colour.setEnabled(switch)
        self.LineStyle.setEnabled(switch)
        self.PlotCheck.setEnabled(switch)
        if switch:
            self.show()
        else:
            self.hide()

  
class experimentLineList(QWidget):
    LineInfo = Signal(bool,int,str,str,str,str)
    LineOff = Signal(int,str)
    LineInfoShort = Signal(bool,int,str,str,str,str)
 
    def __init__(self,exp_num:int, multi_exp:bool = False):
        super().__init__()
        self.multi_exp = multi_exp
        self.exp_num = exp_num+1
        mainLayout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        verticalLayout = QVBoxLayout(scroll_content)
        
       
        self.Experimentname = QLineEdit()
        self.Experimentname.setPlaceholderText(f"Experiment {self.exp_num}")
        self.text_change_timer = QTimer(self)
        self.text_change_timer.setSingleShot(True)
        self.text_change_timer.setInterval(1000)
        self.std_toggle = False
        self.median_toggle = False 
        self.quartiles_toggle = False

        single_line_defs = [
            ("temperature", "Temperature", 0, 0),
            ("ratio", "n/N", 1, 1),
            ("STD", "Standard deviation", 1, 5),
            ("qlohi", "Quartile 10-90%", 2, 2),
            ("qmi", "Median", 3, 3),
            ("luminescence", "Luminescence", 0, 5),
            ("fill", "Filling", 0, 4),
            ("retrap", "Retrapping", 3, 8), 
            ("noEvent", "No event", 2, 6),
            ("GS_tun_recom", "Ground State tunneling recombination", 3, 7),
            ("GS_CB_recom", "Ground State Conduction Band recombination", 3, 11),
            ("GS_tun_retrap", "Ground State tunneling re-trapping", 0, 8),
            ("GS_CB_retrap", "Ground State Conduction Band re-trapping", 0, 0),
            ("GS_process", "Ground State processes", 1, 6),
            ("ES_tun_recom", "Excited State tunneling recombination", 1, 9),
            ("ES_CB_recom", "Excited State Conduction Band recombination", 1, 1),
            ("ES_tun_retrap", "Excited State tunneling re-trapping", 2, 10),
            ("ES_CB_retrap", "Excited State Conduction Band re-trapping", 2, 2),
            ("ES_process", "Excited State processes", 2, 7),
            ("bleach", "Bleaching", 3, 3),]

        self.single_lines = {}
        verticalLayout.addWidget(self.Experimentname)
        if not self.multi_exp:
            self.Experimentname.hide()
        for name, label, x, y in single_line_defs:
            if self.multi_exp:
                full_label = f"Experiment {self.exp_num} {label}"
            else:
                full_label = label
            widget = SingleLine(full_label, x, y, self.exp_num, name)
            widget.LineInfo.connect(self.LineInfo.emit)
            widget.LineOff.connect(self.LineOff.emit)
            widget.LineInfoShort.connect(self.LineInfoShort.emit)
            self.single_lines[name] = widget

            setattr(self, name, widget)
            verticalLayout.addWidget(widget)

        verticalLayout.addStretch()
        scroll_area.setWidget(scroll_content)
        mainLayout.addWidget(scroll_area)
        self.ratio.PlotCheck.toggled.connect(self.ratio_toggled)

    def on_off_selector(self, choice:int):
        if choice == 0:
            self.on_ratio()
        elif choice ==  1:
            self.on_events()
        elif choice ==2:
            self.on_temperature()

    def disbale_all(self):
        for line in self.single_lines.values(): 
            line.hide()

    def on_ratio(self):
        self.disbale_all()
        self.STD.show()
        self.ratio.on_off(True)
        self.qlohi.on_off(True)
        self.qmi.on_off(True)
    
    def on_temperature(self):
        self.disbale_all()
        self.temperature.on_off(True)
    
    def on_events(self):
        self.disbale_all()
        self.luminescence.on_off(True)
        self.fill.on_off(True)
        self.retrap.on_off(True)
        self.noEvent.on_off(True)
        self.GS_tun_recom.on_off(True)
        self.GS_CB_recom.on_off(True)
        self.GS_tun_retrap.on_off(True)
        self.GS_CB_retrap.on_off(True)
        self.GS_process.on_off(True)
        self.ES_tun_recom.on_off(True)
        self.ES_CB_recom.on_off(True)
        self.ES_tun_retrap.on_off(True)
        self.ES_CB_retrap.on_off(True)
        self.ES_process.on_off(True)
        self.bleach.on_off(True)

    def ratio_toggled(self,choice:bool):
        if choice:
            if self.std_toggle:
                self.STD.PlotCheck.setChecked(True)
            if self.median_toggle:
                self.qmi.PlotCheck.setChecked(True)
            if self.quartiles_toggle:
                self.qlohi.PlotCheck.setChecked(True)

    def std_check(self,choice:bool):
        self.STD.on_off(choice)
        self.STD.show()
        if choice:
            self.std_toggle = True
            if self.ratio.PlotCheck.isChecked():
                self.STD.PlotCheck.setChecked(True)
        else:
            self.std_toggle = False
            self.STD.PlotCheck.setChecked(False)

    def median_check(self,choice:bool):
        if choice:
            self.median_toggle= True
            if self.ratio.PlotCheck.isChecked():
                self.qmi.PlotCheck.setChecked(True)
        else:
            self.median_toggle = False
            self.qmi.PlotCheck.setChecked(False)

    def quartile_check(self,choice:bool):
        if choice:
            self.quartiles_toggle= True
            if self.ratio.PlotCheck.isChecked():
                self.qlohi.PlotCheck.setChecked(True)
        else:
            self.quartiles_toggle = False
            self.qlohi.PlotCheck.setChecked(False)

    def emit_short_info(self,name):
        self.single_lines[name].emit_short_info()

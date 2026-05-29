from __future__ import annotations
from pathlib import Path
from PySide6.QtWidgets import QComboBox, QWidget, QToolBox, QVBoxLayout, QInputDialog, QPushButton, QMessageBox
from PySide6.QtCore import Signal, Slot, QTimer
from conf.config_utils import discover_exisitng_profiles, load_profile, save_profile, update_main_user_config
from GUI.src.toggleButton import set_toggle
from GUI.src.designs.setupWidget import Ui_ToolBox as Ui_Setup
from GUI.src.designs.fwdbckrunButtons import Ui_Form as Ui_buttons
from GUI.src.tempProfileSetter import timeTempList, TempMplCanvas
from conf.config_setup_model import SetupConfig
from pydantic import ValidationError
from conf.config_physics_model import PhysicsProfile
from conf.config_temp_model import TemperatureProfile
from conf.config_chrono_model import ChronologyProfile
from src.filesystem import CONFIG_DIR, USER_CONFIG_DIR, TEMP_DIR, PHYS_DIR, CHRON_DIR


class setupWindow(QToolBox, Ui_Setup):
    pageCleared = Signal(int,str)
    page = Signal(int)
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setupProf = SetupConfig() 
        self.tempProf = TemperatureProfile()
        self.physProf = PhysicsProfile()
        self.chronProf = ChronologyProfile()
        self.currentpage = self.currentIndex()
        set_toggle(self.boundary)
        set_toggle(self.tempUnit)
        self.tempUnit.toggled.connect(self.tempProfileTUnitSet)
        self.timeUnit.currentIndexChanged.connect(self.tempProfiletimeUnitSet)
        self.tempProfileKind.currentIndexChanged.connect(self.tempProfileKindSet)
        self.useDefaultCrystal.toggled.connect(self.defaultDim)
        self.densityCheck.toggled.connect(self.rhoselect)
        self.enableFilling.toggled.connect(self.fillingWidget.setEnabled)
        self.tunnelEnable.toggled.connect(self.tunnelingWidget.setEnabled)
        self.cndctionEnable.toggled.connect(self.cbInputWidget.setEnabled)
        self.bandTailenable.toggled.connect(self.bandTailSelect)
        self.useDefault_b_BT.toggled.connect(self.bandTailB)
        self.use_default_alpha_BT.toggled.connect(self.bandTailalpha)
        self.setup_expInfo()
        self.setup_tempInfo()
        self.setup_physInfo()
        self.tempUnit.toggled.connect(self.tempPlotter.set_y_units)
        self.timeUnit.currentIndexChanged.connect(self.tempPlotter.set_x_units)
        self.timeTemperatureListWidget.profile_changed.connect(self.tempPlotter.update_plot)
        self.exp_name_box.currentIndexChanged.connect(self.load_previousConfig)
        self.tempBox.currentIndexChanged.connect(self.load_previousTemp)
        self.physBox.currentIndexChanged.connect(self.load_previousPhys)

        self.expEditButton.toggled.connect(self.loadedExpProfileEdit)
        self.tempEditProfileButton.toggled.connect(self.loadedTempProfileEdit)
        self.physEditCheck.toggled.connect(self.loadedPhysProfileEdit)
        self.duration.valueChanged.connect(self.timeTemperatureListWidget.set_duration)
        self.startT.valueChanged.connect(self.timeTemperatureListWidget.set_start_temperature)
        self.endT.valueChanged.connect(self.timeTemperatureListWidget.set_end_temperature)
        self.endT.valueChanged.connect(self.dT_update)
        self.startT.valueChanged.connect(self.dT_update)
        self.duration.valueChanged.connect(self.dT_update)
        self.dT.valueChanged.connect(self.dT_changed)
        self.b.valueChanged.connect(self.bandTailBchange)
        self.alphaGS.valueChanged.connect(self.bandTailAlphachange)
        self.currentChanged.connect(self.change_current_page)
       

    def change_current_page(self, new_page:int):
        if self.currentpage == 0:
            checker = self.get_expInfoToConfig()
            if checker: 
                self.currentpage = new_page  
        elif self.currentpage == 1:
            checker = self.get_tempInfoToConfig()
            if checker: 
                self.currentpage = new_page
        elif self.currentpage == 2:
            checker = self.get_physInfoToConfig()
            if checker: 
                self.currentpage = new_page
        elif self.currentpage == 3: 
            checker = False
            if checker: 
                # self.pageCleared.emit(0,self..text()) 
                self.currentpage = new_page
            else:
                self.pageCleared.emit(0,None) 
       
        self.page.emit(self.currentpage) 
        self.blockSignals(True)
        self.setCurrentIndex(self.currentpage)
        self.blockSignals(False)
    
    def set_drop_down_options(self, dropdown:QComboBox, dir: str | Path ):
        available = discover_exisitng_profiles(dir)
        for entry in available: 
            dropdown.addItem(entry)

        return available
    
    ##########################################################################################
    def setup_expInfo(self): 
        self.userInputLists = self.set_drop_down_options(self.exp_name_box, USER_CONFIG_DIR)
        self.expEditButton.hide()
        self.set_expInfoFromConfig()

    def set_expInfoFromConfig(self,): 
        self.expNameLineEdit.setText(self.setupProf.project)
        self.seedBox.setValue(self.setupProf.seed)
        self.trapRatio.setValue(self.setupProf.t_pcnt)
        self.holeRatio.setValue(self.setupProf.h_pcnt)
        self.reps.setValue(self.setupProf.reps)
        self.n_jobs.setValue(self.setupProf.n_jobs)
        self.max_dt.setValue(self.setupProf.max_dt)
        if self.setupProf.boundary == "periodic":
            self.boundary.setChecked(True)
        if self.setupProf.mc:
            self.mcRun.setChecked(True)
        if self.setupProf.ac:
            self.acRun.setChecked(True)
        if self.setupProf.TC:
            self.tcRun.setChecked(True)

    def get_expInfoToConfig(self) -> bool:
        """Read experiment setup widgets, validate them, and update self.setupProf."""
        if self.expNameLineEdit.text() == "":
            QMessageBox.warning(
                self,
                "No Experiment profile Name",
                "Experiment needs a file name. Enter one now to proceed"
            )
            self.pageCleared.emit(0,None)  
            return False
        
        boundary = "periodic" if self.boundary.isChecked() else "padded"

        data = {
            "project": self.expNameLineEdit.text().strip(),
            "seed": self.seedBox.value(),
            "t_pcnt": self.trapRatio.value(),
            "h_pcnt": self.holeRatio.value(),
            "reps": self.reps.value(),
            "n_jobs": self.n_jobs.value(),
            "max_dt": self.max_dt.value(),
            "boundary": boundary,
            "mc": self.mcRun.isChecked(),
            "ac": self.acRun.isChecked(),
            "TC": self.tcRun.isChecked(),
        }

        try:
            self.setupProf = SetupConfig.model_validate(data)
            self.pageCleared.emit(0,self.expNameLineEdit.text())
            return True

        except ValidationError as exc:
            messages = []
            for error in exc.errors():
                field = ".".join(str(part) for part in error.get("loc", []))
                message = error.get("msg", "Invalid value")
                if field:
                    messages.append(f"{field}: {message}")
                else:
                    messages.append(message)

            QMessageBox.warning(
                self,
                "Invalid experiment setup",
                "Please fix the following setup values:\n\n" + "\n".join(messages),
            )
            self.pageCleared.emit(0,None)  
            return False

        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid experiment setup",
                str(exc),
            )
            self.pageCleared.emit(0,None)  

            return False

    def load_previousConfig(self, listnum: int):
        if listnum == 0: 
            self.setupProf = SetupConfig()
            self.expInputWidget.setEnabled(True)
            self.expEditButton.hide()
        else:
            self.expEditButton.show()
            self.expEditButton.setChecked(False)
            self.expInputWidget.setEnabled(False) 
            profile_name = self.userInputLists[listnum-1]
            data = load_profile(profile_name, USER_CONFIG_DIR)
            phys = (data["defaults"][0]["/physics"])
            phys_index = self.userPhysLists.index(phys)
            self.physBox.setCurrentIndex(phys_index+1)
            temp = (data["defaults"][1]["/temp"])
            temp_index = self.userTempLists.index(temp)
            self.tempBox.setCurrentIndex(temp_index+1)
            try:
                chron = (data["defaults"][2]["/chronology"])
                # chron_index = self.chronLists.index(temp)
                # self.chronBox.setCurrentIndex(chron_index+1)
            except:
                chron = None
            self.setupProf = SetupConfig.model_validate(data["setup"])
        
        self.set_expInfoFromConfig()   

    def loadedExpProfileEdit(self, edit:bool):
        if edit:
            self.expInputWidget.setEnabled(True)
            text = self.expNameLineEdit.text()
            self.expNameLineEdit.setText(f"{text} copy")
        else:
            self.load_previousConfig(self.exp_name_box.currentIndex()) 
            self.expInputWidget.setEnabled(False)

    ##########################################################################################
    def setup_tempInfo(self):
        self.tempPlotter = TempMplCanvas()
        self.horizontalLayout_2.addWidget(self.tempPlotter)
        self.timeTemperatureListWidget = timeTempList([0.0,self.duration.value()],[0.0,0.0],self.duration.value(),0,True)
        spacer_index = self.verticalLayout_6.indexOf(self.verticalSpacer)
        self.verticalLayout_6.insertWidget(spacer_index,self.timeTemperatureListWidget,1)
        self.tempProfiletimeUnitSet(0)
        self.tempProfileKindSet(0)
        self.TempInputWidget.setEnabled(True)
        self.tempEditProfileButton.hide()
        self.tempUnit.setChecked(True)
        self.userTempLists = self.set_drop_down_options(self.tempBox, TEMP_DIR)
        self.timeTemperatureListWidget.hide()
        self.tempPlotter.plot_profile()

    def set_tempInfoFromConfig(self):
        if self.tempProf.kind == "Constant":
            self.tempProfileKind.setCurrentIndex(0)
        elif self.tempProf.kind == "Linear":
            self.tempProfileKind.setCurrentIndex(1)
        else: 
            self.tempProfileKind.setCurrentIndex(2)
        
        self.duration.setValue(self.tempProf.duration)
        
        if self.tempProf.unit == "s":
            self.timeUnit.setCurrentIndex(0)
        elif self.tempProf.unit == "m":
            self.timeUnit.setCurrentIndex(1)
        elif self.tempProf.unit == "h":
            self.timeUnit.setCurrentIndex(2)
        elif self.tempProf.unit == "d":
            self.timeUnit.setCurrentIndex(3)
        elif self.tempProf.unit == "y":
            self.timeUnit.setCurrentIndex(4)
        elif self.tempProf.unit == "Ka":
            self.timeUnit.setCurrentIndex(5)
        elif self.tempProf.unit == "Ma":
            self.timeUnit.setCurrentIndex(6)
        else:
            self.timeUnit.setCurrentIndex(0)
        self.tempUnit.blockSignals(True)
        if self.tempProf.celsius:
            self.tempUnit.setChecked(True)
        else:
            self.tempUnit.setChecked(False)
        self.tempUnit.blockSignals(False)
        
        self.startT.setValue(self.tempProf.T0)
        self.timeTemperatureListWidget.hide()
        if self.tempProfileKind.currentIndex() == 0:
            self.endT.setValue(self.tempProf.T0)
        elif self.tempProfileKind.currentIndex() == 1:
            if self.tempProf.dT is not None: 
                self.dT.setValue(self.tempProf.dT)
            if self.tempProf.temps is not None:
                self.endT.setValue(self.tempProf.temps[1])
        elif self.tempProfileKind.currentIndex() == 2:
            self.endT.setValue(self.tempProf.temps[-1])
            self.timeTemperatureListWidget.resetPoints(self.tempProf.times,self.tempProf.temps,self.tempProf.duration,self.timeUnit.currentIndex(),self.tempProf.celsius)
            self.timeTemperatureListWidget.show()

    def get_tempInfoToConfig(self) -> bool:
        """Read temperature profile widgets, validate them, and update self.tempProf."""
        if self.tempProfileName.text() == "":
            QMessageBox.warning(
                self,
                "No Temperature profile Name",
                "Temperature profile needs a file name. Enter one now to proceed"
            )
            self.pageCleared.emit(1,None)
            return False
        
        unit_map = {
            0: "s",
            1: "m",
            2: "h",
            3: "d",
            4: "y",
            5: "Ka",
            6: "Ma",
        }
        kind_map = {
            0: "Constant",
            1: "Linear",
            2: "Other",
        }

        unit = unit_map.get(self.timeUnit.currentIndex(), "s")
        kind = kind_map.get(self.tempProfileKind.currentIndex(), "Other")
        duration = self.duration.value()
        celsius = self.tempUnit.isChecked()
        T0 = self.startT.value()

        times = None
        temps = None
        dT = None

        if kind == "Linear":
            times = [0.0, duration]
            temps = [T0, self.endT.value()]
            dT = self.dT.value()
        elif kind == "Other":
            times = self.timeTemperatureListWidget.get_times()
            temps = self.timeTemperatureListWidget.get_temps()
        elif kind == "Constant":
            times = [0.0, duration]
            temps = [T0, T0]

        data = {
            "unit": unit,
            "duration": duration,
            "celsius": celsius,
            "T0": T0,
            "kind": kind,
            "times": times,
            "temps": temps,
            "dT": dT,
        }

        try:
            self.tempProf = TemperatureProfile.model_validate(data)
            self.pageCleared.emit(1,self.tempProfileName.text())  
            return True

        except ValidationError as exc:
            messages = []
            for error in exc.errors():
                field = ".".join(str(part) for part in error.get("loc", []))
                message = error.get("msg", "Invalid value")
                if field:
                    messages.append(f"{field}: {message}")
                else:
                    messages.append(message)

            QMessageBox.warning(
                self,
                "Invalid temperature profile",
                "Please fix the following temperature values:\n\n" + "\n".join(messages),
            )
            self.pageCleared.emit(1,None)  
            return False

        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid temperature profile",
                str(exc),
            )
            self.pageCleared.emit(1,None)  
            return False
            
    def tempProfileKindSet(self,choice:int):
        if choice == 0:
            self.diffTBox.hide()
            self.TendBox.hide()
            self.timeTemperatureListWidget.kind = 0
            self.timeTemperatureListWidget.resetPoints([0,self.duration.value()],[self.startT.value(),self.startT.value()],self.duration.value(),self.timeUnit.currentIndex(),self.tempProf.celsius)
            self.timeTemperatureListWidget.hide()
        elif choice == 1:
            self.diffTBox.show()
            self.TendBox.show()
            self.timeTemperatureListWidget.kind = 1
            self.timeTemperatureListWidget.resetPoints([0,self.duration.value()],[self.startT.value(),self.endT.value()],self.duration.value(),self.timeUnit.currentIndex(),self.tempProf.celsius)
            self.timeTemperatureListWidget.hide()
        elif choice == 2:
            self.timeTemperatureListWidget.kind = 2
            self.timeTemperatureListWidget.show()
            self.diffTBox.hide()
            self.TendBox.show()

    def tempProfileTUnitSet(self,cel:bool):
        self.timeTemperatureListWidget.set_temp_units(cel)
        if cel:
            self.startT.blockSignals(True)
            t = self.startT.value()
            self.startT.setValue(t-273.15)
            self.startT.blockSignals(False)
            self.endT.blockSignals(True)
            t = self.endT.value()
            self.endT.setValue(t-273.15)
            self.endT.blockSignals(False)
            self.kelvinSymbol.hide()
            self.kelvinSymbol2.hide()
            self.kelvinSymbol3.hide()
            self.celsiusSymbol.show()
            self.celsiusSymbol2.show()
            self.celsiusSymbol3.show()
        else:
            self.startT.blockSignals(True)
            t = self.startT.value()
            self.startT.setValue(t+273.15)
            self.startT.blockSignals(False)
            self.endT.blockSignals(True)
            t = self.endT.value()
            self.endT.setValue(t+273.15)
            self.endT.blockSignals(False)
            self.kelvinSymbol.show()
            self.kelvinSymbol2.show()
            self.kelvinSymbol3.show()
            self.celsiusSymbol.hide()
            self.celsiusSymbol2.hide()
            self.celsiusSymbol3.hide()
    
    def tempProfiletimeUnitSet(self,choice:int):
        self.timeTemperatureListWidget.set_t_units(choice)
        self.maSymbol.hide()
        self.kaSymbol.hide()
        self.yearSymbol.hide()
        self.dSymbol.hide()
        self.hourSymbol.hide()
        self.minSymbol.hide()
        self.sSymbol.hide()
        if choice == 0:
            self.sSymbol.show()
        elif choice == 1:
            self.minSymbol.show()
        elif choice == 2:
            self.hourSymbol.show()
        elif choice == 3:
            self.dSymbol.show()
        elif choice == 4:
            self.yearSymbol.show()
        elif choice == 5:
            self.kaSymbol.show()
        elif choice == 6:
            self.maSymbol.show()
    
    def load_previousTemp(self, listnum: int):
        if listnum == 0: 
            self.tempProf = TemperatureProfile()
            self.TempInputWidget.setEnabled(True)
            self.tempEditProfileButton.hide()
        else: 
            self.TempInputWidget.setEnabled(False)
            self.tempEditProfileButton.show()
            self.tempEditProfileButton.setChecked(False)
            profile_name = self.userTempLists[listnum-1]
            self.tempProfileName.setText(str(profile_name))
            data = load_profile(profile_name, TEMP_DIR)
            self.tempProf = TemperatureProfile.model_validate(data)
        
        self.set_tempInfoFromConfig()

    def loadedTempProfileEdit(self, edit:bool):
        if edit:
            self.TempInputWidget.setEnabled(True)
            text = self.tempProfileName.text()
            self.tempProfileName.setText(f"{text} copy")
        else:
            self.load_previousTemp(self.tempBox.currentIndex())
            self.TempInputWidget.setEnabled(False)

    def dT_changed(self):
        end = self.startT.value() + self.dT.value()*self.duration.value()
        self.endT.setValue(end)
    
    def dT_update(self, ):
        dt = (self.endT.value() - self.startT.value() )/self.duration.value()
        self.dT.blockSignals(True)
        self.dT.setValue(dt)
        self.dT.blockSignals(False)
    ##########################################################################################

    def setup_physInfo(self): 
        self.userPhysLists = self.set_drop_down_options(self.physBox, PHYS_DIR)
        self.physInputWidget.setEnabled(True)
        self.physEditCheck.hide()
        self.useDefaultCrystal.setChecked(True)
        self.rhoselect(False)
        self.fillingWidget.setEnabled(False)
        self.tunnelingWidget.setEnabled(False)
        self.cbInputWidget.setEnabled(False)
        self.bandTailSelect(False)
        self.useDefault_b_BT.setChecked(True)
        self.use_default_alpha_BT.setChecked(True)

    def defaultDim(self,select:bool):
        if select:
            self.crystalDimension.setValue(7.5)
            self.crystalDimension.setEnabled(False)
            self.crystaldimensionUnit.setCurrentIndex(4)
            self.crystaldimensionUnit.setEnabled(False)
        else: 
            self.crystalDimension.setEnabled(True)
            self.crystaldimensionUnit.setEnabled(True)

    def rhoselect(self,select:bool):
        if select: 
            self.unitlessLabel.show()
            self.unitLabel.hide()
            self.rhoUnitLabel.show()
        else:
            self.unitlessLabel.hide()
            self.unitLabel.show()
            self.rhoUnitLabel.hide()

    def bandTailSelect(self,select:bool):
        if select:
            self.bandTailInputWidget.setEnabled(True)
            self.bandTailInputWidget.show()
        else:
            self.bandTailInputWidget.setEnabled(False)
            self.bandTailInputWidget.hide()
    
    def bandTailB(self,select:bool): 
        if select: 
            b = self.b.value()
            self.b_BT.setValue(b)
            self.b_BT.setEnabled(False)
        else:
            self.b_BT.setEnabled(True)
    
    def bandTailBchange(self,b:float):
        if self.useDefault_b_BT.isChecked():
            self.b_BT.setValue(b)

    def bandTailalpha(self,select:bool): 
        if select: 
            a = self.alphaGS.value()
            self.alpha_BT.setValue(a)
            self.alpha_BT.setEnabled(False)
        else:
            self.alpha_BT.setEnabled(True)

    def bandTailAlphachange(self,alpha:float):
        if self.use_default_alpha_BT.isChecked():
            self.alpha_BT.setValue(alpha)
    
    def load_previousPhys(self, listnum: int):
        if listnum == 0: 
            self.physProf = PhysicsProfile()
            self.physInputWidget.setEnabled(True)
            self.physEditCheck.hide()
        else:
            self.physEditCheck.show()
            self.physEditCheck.setChecked(False)
            self.physInputWidget.setEnabled(False) 
            profile_name = self.userPhysLists[listnum-1]
            self.physName.setText(str(profile_name))
            data = load_profile(profile_name, PHYS_DIR)
            self.physProf = PhysicsProfile.model_validate(data)
           
        
        self.set_physInfoFromConfig()   

    def set_physInfoFromConfig(self):
        self.ucH.setValue(self.physProf.uc_h/1e-10)
        self.ucW.setValue(self.physProf.uc_w/1e-10)
        self.ucL.setValue(self.physProf.uc_l/1e-10)
        if self.physProf.dimension is None or self.physProf.dimension == 7.5e-9:
            self.useDefaultCrystal.setChecked(True)
            self.defaultDim(True)
        else:
            self.crystalDimension.setValue(self.physProf.dimension/1e-9)
            self.crystaldimensionUnit.setCurrentIndex(4)
            self.useDefaultCrystal.setChecked(False)
            self.defaultDim(False)
        
        if self.physProf.rho is not None:
            self.densityval.setValue(self.physProf.rho)
            self.densityCheck.setChecked(False)
        elif self.physProf.urho is not None:
            self.densityval.setValue(self.physProf.urho)
            self.densityCheck.setChecked(True)
        
        self.eLoc.setValue(self.physProf.E_loc)
        self.eCB.setValue(self.physProf.E_cb)
        self.sigmaELoc.setValue(self.physProf.E_loc_sigma)
        self.sigmaECB.setValue(self.physProf.E_cb_sigma)
        
        if self.physProf.enable_fill: 
            self.enableFilling.setChecked(True)
            self.D0.setValue(self.physProf.D0)
            self.d_dot.setValue(self.physProf.D_dot)
            if self.physProf.Dd_unit == "s":
                self.dd_unit.setCurrentIndex(0)
            elif self.physProf.Dd_unit == "m":
                self.dd_unit.setCurrentIndex(1)
            elif self.physProf.Dd_unit == "h":
                self.dd_unit.setCurrentIndex(2)
            elif self.physProf.Dd_unit == "d":
                self.dd_unit.setCurrentIndex(3)
            elif self.physProf.Dd_unit == "y":
                self.dd_unit.setCurrentIndex(4)
            elif self.physProf.Dd_unit == "ka" or "Ka" or "KA" or "kA":
                self.dd_unit.setCurrentIndex(5)
            elif self.physProf.Dd_unit == "ma"or "Ma" or "MA" or "mA":
                self.dd_unit.setCurrentIndex(6)

        self.combine_when_fill.setChecked(self.physProf.combine_when_fill)
        self.recom_pre_fill.setValue(self.physProf.recom_pre_fill)

      
        self.tunnelEnable.setChecked(self.physProf.enable_tunneling)
        
        if self.physProf.b is not None: 
            self.b.setValue(self.physProf.b)
        else: 
            self.b.setValue(1e12)

        if self.physProf.alpha_GS is not None: 
            self.alphaGS.setValue(self.physProf.alpha_GS)
        else:
            self.alphaGS.setValue(0)
        
        if self.physProf.alpha_ES is not None: 
            self.alphaEX.setValue(self.physProf.alpha_ES)
        else:
            self.alphaEX.setValue(0)

        self.retrapRatio.setValue(self.physProf.R_tun)
        self.VRH.setChecked(self.physProf.VRH)

        self.cndctionEnable.setChecked(self.physProf.enable_cb)
        if self.physProf.s is not None: 
            self.s.setValue(self.physProf.s)
        else: 
            self.s.setValue(1e12)
        if self.physProf.mu is not None: 
            self.mu.setValue(self.physProf.mu)
        else:
            self.mu.setValue(0.1)

        self.RCB.setValue(self.physProf.R_CB)
        self.bandTailenable.setChecked(self.physProf.enable_BT)
        self.retrap_mask_factor.setValue(self.physProf.retrap_mask_factor)
        if self.physProf.shallow_deep_ratio is not None:
            self.shallow_deep_ratio.setValue(self.physProf.shallow_deep_ratio)
        if self.physProf.threshold_depth is not None:
            self.threshold_depth.setValue(self.physProf.threshold_depth)
        if self.physProf.E_u is not None:
            self.E_u.setValue(self.physProf.E_u)
        if self.physProf.b_BT is None: 
            self.useDefault_b_BT.setChecked(True)
            self.bandTailB(True)
        else:
            self.useDefault_b_BT.setChecked(False)
            self.b_BT.setValue(self.physProf.b_BT)
            self.bandTailB(False)

        if self.physProf.alpha_BT is None: 
            self.use_default_alpha_BT.setChecked(True)
            self.bandTailalpha(True)
        else:
            self.use_default_alpha_BT.setChecked(False)
            self.alpha_BT.setValue(self.physProf.alpha_BT)
            self.bandTailalpha(False)

            
        self.init_shallow.setChecked(self.physProf.init_shallow)
        if self.physProf.sh_pcnt is not None:
            self.sh_pcnt.setValue(self.physProf.sh_pcnt)

    def get_physInfoToConfig(self) -> bool:
        if self.physName.text() == "":
            QMessageBox.warning(
                self,
                "No Physics profile Name",
                "Physics profile needs a file name. Enter one now to proceed"
            )
            self.pageCleared.emit(2,None)  
            return False
        """Read physics widgets, validate them, and update self.physProf."""
        dimension = None
        if not self.useDefaultCrystal.isChecked():
            dimension_unit_scale = {
                0: 1.0,
                1: 1e-3,
                2: 1e-6,
                3: 1e-9,
                4: 1e-9,
                5: 1e-10,
            }
            scale = dimension_unit_scale.get(self.crystaldimensionUnit.currentIndex(), 1e-9)
            dimension = self.crystalDimension.value() * scale

        if self.densityCheck.isChecked():
            rho = None
            urho = self.densityval.value()
        else:
            rho = self.densityval.value()
            urho = None

        dd_unit_map = {
            0: "s",
            1: "m",
            2: "h",
            3: "d",
            4: "y",
            5: "ka",
            6: "ma",
        }

        enable_fill = self.enableFilling.isChecked()
        enable_tunneling = self.tunnelEnable.isChecked()
        enable_cb = self.cndctionEnable.isChecked()
        enable_BT = self.bandTailenable.isChecked()
        
        b_BT = None if self.useDefault_b_BT.isChecked() else self.b_BT.value()
        alpha_BT = None if self.use_default_alpha_BT.isChecked() else self.alpha_BT.value()
        shallow_deep_ratio = None if not enable_BT else self.shallow_deep_ratio.value()
        threshold_depth = None if not enable_BT else self.threshold_depth.value()
        E_u = None if not enable_BT else self.E_u.value()
        sh_pcnt = None if not enable_BT else  self.sh_pcnt.value()

        data = {
            "uc_h": self.ucH.value() * 1e-10,
            "uc_w": self.ucW.value() * 1e-10,
            "uc_l": self.ucL.value() * 1e-10,
            "dimension": dimension,
            "rho": rho,
            "urho": urho,
            "E_loc": self.eLoc.value(),
            "E_cb": self.eCB.value(),
            "E_loc_sigma": self.sigmaELoc.value(),
            "E_cb_sigma": self.sigmaECB.value(),
            "enable_fill": enable_fill,
            "D0": self.D0.value(),
            "D_dot": self.d_dot.value(),
            "Dd_unit": dd_unit_map.get(self.dd_unit.currentIndex(), "s"),
            "combine_when_fill": self.combine_when_fill.isChecked(),
            "recom_pre_fill": self.recom_pre_fill.value(),
            "enable_tunneling": enable_tunneling,
            "b": self.b.value(),
            "alpha_GS": self.alphaGS.value(),
            "alpha_ES": self.alphaEX.value(),
            "R_tun": self.retrapRatio.value(),
            "VRH": self.VRH.isChecked(),
            "enable_cb": enable_cb,
            "s": self.s.value(),
            "mu": self.mu.value(),
            "R_CB": self.RCB.value(),
            "retrap_mask_factor": self.retrap_mask_factor.value(),
            "enable_BT": enable_BT,
            "shallow_deep_ratio": shallow_deep_ratio,
            "threshold_depth": threshold_depth,
            "E_u": E_u,
            "b_BT": b_BT,
            "alpha_BT": alpha_BT,
            "init_shallow": self.init_shallow.isChecked(),
            "sh_pcnt": sh_pcnt,
        }

        try:
            self.physProf = PhysicsProfile.model_validate(data)
            self.pageCleared.emit(2,self.physName.text()) 

            return True
        except ValidationError as exc:
            messages = []
            for error in exc.errors():
                field = ".".join(str(part) for part in error.get("loc", []))
                message = error.get("msg", "Invalid value")
                if field:
                    messages.append(f"{field}: {message}")
                else:
                    messages.append(message)

            QMessageBox.warning(
                self,
                "Invalid physics profile",
                "Please fix the following physics values:\n\n" + "\n".join(messages),
            )
            self.pageCleared.emit(2,None)  
            return False

        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid physics profile",
                str(exc),
            )
            self.pageCleared.emit(2,None)  
            return False

    def loadedPhysProfileEdit(self, edit:bool):
        if edit:
            self.physInputWidget.setEnabled(True)
            text = self.physName.text()
            self.physName.setText(f"{text} copy")
        else:
            self.load_previousPhys(self.physBox.currentIndex()) 
            self.physInputWidget.setEnabled(False)

class FwdBckButtons(QWidget, Ui_buttons):
    pageSelect = Signal(int)
    # FinalChecks
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.runNow.hide()
        self.setupRun.hide()
        self.back.hide()
        self.index = 0
        self.experimentName = None 
        self.temperatureName = None
        self.physicsName = None 
        self.chronName = None
        self.chronNeeded = False
        self.forward.clicked.connect(self.forwardPushed)
        self.back.clicked.connect(self.backPushed)

    def chronToggled(self,checked:bool):
        self.chronNeeded = checked

    def backPushed(self,):
        self.pageSelect.emit(self.index-1)
    
    def forwardPushed(self,):
        self.pageSelect.emit(self.index+1)
    
    def pageUpdated(self, index:int, name:str|None): 
        if index == 0:
            self.experimentName = name
        elif index == 1:
            self.temperatureName = name
        elif index == 2:
            self.physicsName = name
        elif index == 3:
            self.chronName = name
        
        self.runNow.hide()
        self.setupRun.hide()

        if self.experimentName is not None and self.temperatureName is not None: 
            if not self.chronNeeded:
                self.runNow.show()
                self.setupRun.show()
            else: 
                if self.physicsName is not None:
                    self.runNow.show()
                    self.setupRun.show()
            
    def pageChanged(self, index:int): 
        self.index = index
        if index == 0:
            self.back.hide()
            self.forward.show()
        elif((index == 2 and not self.chronNeeded) or (index == 3)):
            self.forward.hide()
            self.back.show()
        else: 
            self.forward.show()
            self.back.show()
        
        
class fullSetupWindow(QWidget):
    runSignal = Signal(int)
    def __init__(self):
        super().__init__()
        self.verticalLayout = QVBoxLayout(self)

        self.setWindow = setupWindow()
        self.buttons = FwdBckButtons()
       
        self.verticalLayout.addWidget(self.setWindow)
        self.verticalLayout.addWidget(self.buttons)   

        self.setWindow.tcRun.toggled.connect(self.buttons.chronToggled)
        self.buttons.pageSelect.connect(self.changePage)
        self.setWindow.pageCleared.connect(self.buttons.pageUpdated)
        self.setWindow.page.connect(self.buttons.pageChanged)
        self.buttons.runNow.clicked.connect(self.runExperiment)
        self.buttons.setupRun.clicked.connect(self.setupExperiment)

    def changePage(self,index:int):
       self.setWindow.setCurrentIndex(index)

    
    def finalChecks(self): 
        chronCheck = self.buttons.chronNeeded
        checker =  self.setWindow.get_expInfoToConfig()
        if not checker:
            self.changePage(0)
            return False 
        
        checker =  self.setWindow.get_tempInfoToConfig()
        if not checker:
            self.changePage(1)
            return False 
        
        checker =  self.setWindow.get_physInfoToConfig()
        if not checker:
            self.changePage(2)
            return False 

        if chronCheck: 
            checker =  True #self.setWindow.get_chronInfoToConfig()
            if not checker:
                self.changePage(3)
                return False 
            
        return True 
    
    def warning(self,safe_name,directory):
        new_name, ok = QInputDialog.getText(
                self,
                "Profile already exists",
                (
                    f"A profile named '{safe_name}.yaml' already exists.\n\n"
                    "Enter a new name to save a copy, or press Cancel / leave unchanged "
                    "to overwrite the existing file."
                ),
                text=safe_name,
            )
        if ok and new_name.strip() and new_name.strip() != safe_name:
            safe_name = new_name.strip().removesuffix(".yaml")
        
        path = Path(directory, f"{safe_name}.yaml")
        return path, safe_name

    
    def profile_save(self, name: str, directory: str | Path, profile, exclude_none: bool = False,) -> str:

        directory = Path(directory)
        # directory.mkdir(parents=True, exist_ok=True)
        safe_name = name.removesuffix(".yaml")
        path = directory / f"{safe_name}.yaml"
        if path.exists():
            path, safe_name = self.warning(safe_name,directory)
           
        data = profile.model_dump(exclude_none=exclude_none)

        save_profile(data, path)
        return safe_name
    

    def save_setup(self,):
        safe_name = self.buttons.experimentName.removesuffix(".yaml")
        path = Path(USER_CONFIG_DIR,f"{safe_name}.yaml")
        if path.exists():
            path, safe_name = self.warning(safe_name,USER_CONFIG_DIR)


        self.buttons.temperatureName = self.profile_save(self.buttons.temperatureName, TEMP_DIR, self.setWindow.tempProf, True)
        self.buttons.physicsName = self.profile_save(self.buttons.physicsName, PHYS_DIR, self.setWindow.physProf, False)
        defaults = [{"/physics": self.buttons.physicsName},{"/temp": self.buttons.temperatureName},]
        if self.buttons.chronNeeded: 
            self.buttons.chronName = self.profile_save(self.buttons.chronName, CHRON_DIR, self.setWindow.chronProf, False)
            defaults.append({"/chronology": self.buttons.chronName})

        data = {"defaults": defaults, "setup": self.setWindow.setupProf.model_dump(exclude_none=True),}
        save_profile(data, path)
        update_main_user_config(CONFIG_DIR,"config.yaml", safe_name)

    def runExperiment(self,):
        
        check = self.finalChecks()
        if check: 
            self.save_setup()
            self.runSignal.emit(1)
    
    def setupExperiment(self,):
        
        check = self.finalChecks()
        if check: 
            self.save_setup()
            self.runSignal.emit(0)
   
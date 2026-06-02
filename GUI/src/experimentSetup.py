from __future__ import annotations
import math
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
        set_toggle(self.method)
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
        self.setup_chronInfo()
        self.tempUnit.toggled.connect(self.tempPlotter.set_y_units)
        self.timeUnit.currentIndexChanged.connect(self.tempPlotter.set_x_units)
        self.timeTemperatureListWidget.profile_changed.connect(self.tempPlotter.update_plot)
        self.exp_name_box.currentIndexChanged.connect(self.load_previousConfig)
        self.tempBox.currentIndexChanged.connect(self.load_previousTemp)
        self.physBox.currentIndexChanged.connect(self.load_previousPhys)
        self.chronBox.currentIndexChanged.connect(self.load_previousChronProfile)

        self.expEditButton.toggled.connect(self.loadedExpProfileEdit)
        self.tempEditProfileButton.toggled.connect(self.loadedTempProfileEdit)
        self.physEditCheck.toggled.connect(self.loadedPhysProfileEdit)
        self.chronEditButton.toggled.connect(self.loadedChronProfileEdit)

        self.duration.valueChanged.connect(self.timeTemperatureListWidget.set_duration)
        self.startT.valueChanged.connect(self.timeTemperatureListWidget.set_start_temperature)
        self.endT.valueChanged.connect(self.timeTemperatureListWidget.set_end_temperature)
        self.endT.valueChanged.connect(self.dT_update)
        self.startT.valueChanged.connect(self.dT_update)
        self.duration.valueChanged.connect(self.dT_update)
        self.dT.valueChanged.connect(self.dT_changed)
        
        self.b.valueChanged.connect(self.bandTailBchange)
        self.b_scientific.valueChanged.connect(self.bandTailBchange)
        self.alphaGS.valueChanged.connect(self.bandTailAlphachange)
        self.alphaGS_scientific.valueChanged.connect(self.bandTailAlphachange)
        self.currentChanged.connect(self.change_current_page)

        self.method.toggled.connect(self.chronMethodToggled)
        self.burn_in.toggled.connect(self.burnInToggled)
        self.overall_check.toggled.connect(self.overallToggled)
        self.individual_check.toggled.connect(self.individualToggled)


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
            checker = self.get_chronInfoToConfig()
            if checker: 
                self.currentpage = new_page
       
        self.page.emit(self.currentpage) 
        self.blockSignals(True)
        self.setCurrentIndex(self.currentpage)
        self.blockSignals(False)
    
    def set_drop_down_options(self, dropdown:QComboBox, dir: str | Path ):
        available = discover_exisitng_profiles(dir)
        for entry in available: 
            dropdown.addItem(entry)

        return available
    
    def split_scientific(self, value: float) -> tuple[float, int]:
        if value == 0:
            return 0.0, 0

        sign = -1 if value < 0 else 1
        abs_value = abs(value)
        exponent = math.floor(math.log10(abs_value))
        mantissa = abs_value / (10 ** exponent)

        return sign * mantissa, exponent

    def combine_scientific(self, mantissa: float, exponent: int) -> float:
        return mantissa * (10 ** exponent)

    
    ##########################################################################################
    def setup_expInfo(self): 
        self.userInputLists = self.set_drop_down_options(self.exp_name_box, USER_CONFIG_DIR)
        self.expEditButton.hide()
        self.expEditButton.setChecked(False)
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
            self.expEditButton.setChecked(False)
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
                chron_index = self.chronInputsList.index(chron)
                self.chronBox.setCurrentIndex(chron_index+1)
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
        self.tempEditProfileButton.setChecked(False)
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
        self.chronTempUnits(cel)
    
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
            self.tempEditProfileButton.setChecked(False)
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
        self.physEditCheck.setChecked(False)
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
            bexp = self.b_scientific.value()
            self.b_BT.setValue(b)
            self.b_BT_scientific.setValue(bexp)
            self.b_BT.setEnabled(False)
            self.b_BT_scientific.setEnabled(False)
        else:
            self.b_BT.setEnabled(True)
            self.b_BT_scientific.setEnabled(True)

    
    def bandTailBchange(self,):
        self.bandTailB(self.useDefault_b_BT.isChecked())
       

    def bandTailalpha(self,select:bool): 
        if select: 
            a = self.alphaGS.value()
            aexp = self.alphaGS_scientific.value()
            self.alpha_BT.setValue(a)
            self.alpha_BT_scientific.setValue(aexp)
            self.alpha_BT.setEnabled(False)
            self.alpha_BT_scientific.setEnabled(False)
        else:
            self.alpha_BT.setEnabled(True)
            self.alpha_BT_scientific.setEnabled(True)


    def bandTailAlphachange(self):
        self.bandTailalpha(self.use_default_alpha_BT.isChecked())
    
    
    def load_previousPhys(self, listnum: int):
        if listnum == 0: 
            self.physProf = PhysicsProfile()
            self.physInputWidget.setEnabled(True)
            self.physEditCheck.hide()
            self.physEditCheck.setChecked(False)
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
            density, dexp = self.split_scientific(self.physProf.rho)
            self.densityval.setValue(density)
            self.density_scientific.setValue(dexp)
            self.densityCheck.setChecked(False)
        elif self.physProf.urho is not None:
            density, dexp = self.split_scientific(self.physProf.urho)
            self.densityval.setValue(density)
            self.density_scientific.setValue(dexp)
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
            b, bexp = self.split_scientific(self.physProf.b)
            self.b.setValue(b)
            self.b_scientific.setValue(bexp)
        else:
            self.b.setValue(1.0)
            self.b_scientific.setValue(12) 
           

        if self.physProf.alpha_GS is not None:
            alpha, alphaE = self.split_scientific(self.physProf.alpha_GS) 
            self.alphaGS.setValue(alpha)
            self.alphaGS_scientific.setValue(alphaE)
        else:
            self.alphaGS.setValue(9)
            self.alphaGS_scientific.setValue(12)
        
        if self.physProf.alpha_ES is not None:
            alpha, alphaE = self.split_scientific(self.physProf.alpha_ES) 
            self.alphaEX.setValue(alpha)
            self.alphaEX_scientific.setValue(alphaE)
        else:
            self.alphaEX.setValue(9)
            self.alphaEX_scientific.setValue(9)
            

        self.retrapRatio.setValue(self.physProf.R_tun)
        self.VRH.setChecked(self.physProf.VRH)

        self.cndctionEnable.setChecked(self.physProf.enable_cb)

        if self.physProf.s is not None:
            s, secp = self.split_scientific(self.physProf.s) 
            self.s.setValue(s)
            self.s_scientific.setValue(secp)
        else: 
            self.s.setValue(1.0)
            self.s_scientific.setValue(12)

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
            b_bT, b_BTexp = self.split_scientific(self.physProf.b_BT)
            self.b_BT.setValue(b_bT)
            self.b_BT_scientific.setValue(b_BTexp)
            self.bandTailB(False)

        if self.physProf.alpha_BT is None: 
            self.use_default_alpha_BT.setChecked(True)
            self.bandTailalpha(True)
        else:
            self.use_default_alpha_BT.setChecked(False)
            alphags, alphaexp = self.split_scientific(self.physProf.alpha_BT)
            self.alpha_BT.setValue(alphags)
            self.alpha_BT_scientific.setValue(alphaexp)
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
            urho = self.combine_scientific(self.densityval.value(),self.density_scientific.value())
            rho = None
        else:
            rho = self.combine_scientific(self.densityval.value(),self.density_scientific.value())
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
        b = self.combine_scientific(self.b.value(),self.b_scientific.value())
        alphaGS = self.combine_scientific(self.alphaGS.value(),self.alphaGS_scientific.value())
        alphaEX = self.combine_scientific(self.alphaEX.value(),self.alphaEX_scientific.value())
        s = self.combine_scientific(self.s.value(),self.s_scientific.value())

        b_BT = b if self.useDefault_b_BT.isChecked() else self.combine_scientific(self.b_BT.value(),self.b_BT_scientific.value())
        alpha_BT = alphaGS if self.use_default_alpha_BT.isChecked() else self.combine_scientific(self.alpha_BT.value(),self.alpha_BT_scientific.value())
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
            "b": b,
            "alpha_GS": alphaGS,
            "alpha_ES": alphaEX,
            "R_tun": self.retrapRatio.value(),
            "VRH": self.VRH.isChecked(),
            "enable_cb": enable_cb,
            "s": s,
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
    ##########################################################################################
    
    def setup_chronInfo(self,):
        self.chronInputsList = self.set_drop_down_options(self.chronBox, CHRON_DIR)
        self.chronEditButton.hide()
        self.chronEditButton.setChecked(False)
        self.chronTempUnits(True)
        self.set_chronInfoFromConfig() 
    
    def set_chronInfoFromConfig(self,):
        if self.chronProf.method == "RJMCMC":
            self.method.setChecked(True)
        else: 
            self.method.setChecked(False)
        
        self.chronIterations.setValue(self.chronProf.iters)
        self.T_Target.setValue(self.chronProf.T_Target)
        self.T_tolerance.setValue(self.chronProf.T_tolerance)
        self.T0_lo.setValue(self.chronProf.T0_lo)
        self.T0_hi.setValue(self.chronProf.T0_hi)
        if self.chronProf.monotonic == "free":
            self.monotonic.setCurrentIndex(0)
        elif self.chronProf.monotonic == "increasing":
            self.monotonic.setCurrentIndex(1)
        elif self.chronProf.monotonic == "decreasing":
            self.monotonic.setCurrentIndex(2)

        self.min_internal.setValue(self.chronProf.min_internal)
        self.max_internal.setValue(self.chronProf.max_internal)

        if self.chronProf.method == "RJMCMC":
            self.RJMCMCwidget.show()
        else:
            self.RJMCMCwidget.hide()   
                
        if self.chronProf.rjmcmc.logLikeSigma is not None:
            self.logLikeSigma.setValue(self.chronProf.rjmcmc.logLikeSigma)
        self.p_birth.setValue(self.chronProf.rjmcmc.parameters.p_birth)
        self.p_death.setValue(self.chronProf.rjmcmc.parameters.p_death)
        self.p_move_times.setValue(self.chronProf.rjmcmc.parameters.p_move_time)
        self.p_move_temp.setValue(self.chronProf.rjmcmc.parameters.p_move_temp)
        self.p_move_endpoints.setValue(self.chronProf.rjmcmc.parameters.p_move_endpoints)
        self.sigma_birth.setValue(self.chronProf.rjmcmc.parameters.sigma_birth)
        self.sigma_t_birth.setValue(self.chronProf.rjmcmc.parameters.sigma_t_birth)
        self.sigma_temp.setValue(self.chronProf.rjmcmc.parameters.sigma_temp)
        self.sigma_time_frac.setValue(self.chronProf.rjmcmc.parameters.sigma_time_frac)
        self.sigma_endpoints.setValue(self.chronProf.rjmcmc.parameters.sigma_endpoints)
        if self.chronProf.rjmcmc.burn_in.burn: 
            self.burn_in.setChecked(True)
            self.BurnInWidget.show()
        else:
            self.BurnInWidget.hide()
        
        self.max_steps.setValue(self.chronProf.rjmcmc.burn_in.max_steps)
        self.window.setValue(self.chronProf.rjmcmc.burn_in.window)
        self.patience_windows.setValue(self.chronProf.rjmcmc.burn_in.patience_windows)

        self.adjustment_factor.setValue(self.chronProf.rjmcmc.burn_in.adjustment_factor) 
        self.eta_prob.setValue(self.chronProf.rjmcmc.burn_in.eta_prob) 
        self.eta_sigma.setValue(self.chronProf.rjmcmc.burn_in.eta_sigma)
        if self.chronProf.rjmcmc.burn_in.move_bounds is not None:
            self.movePMin.setValue(self.chronProf.rjmcmc.burn_in.move_bounds[0])
            self.moveP_max.setValue(self.chronProf.rjmcmc.burn_in.move_bounds[1])
        
        self.overall_check.setChecked(self.chronProf.rjmcmc.burn_in.overall_check)
        self.overallToggled(self.chronProf.rjmcmc.burn_in.overall_check)
        self.individual_check.setChecked(self.chronProf.rjmcmc.burn_in.individual_check)
        self.individualToggled(self.chronProf.rjmcmc.burn_in.individual_check)
        self.verbose.setChecked(self.chronProf.rjmcmc.burn_in.verbose)

        if self.chronProf.rjmcmc.burn_in.overall_accept_target is not None:
            self.overall_min.setValue(self.chronProf.rjmcmc.burn_in.overall_accept_target[0])
            self.overall_max.setValue(self.chronProf.rjmcmc.burn_in.overall_accept_target[1])
    
        if self.chronProf.rjmcmc.burn_in.birth_accept_target is not None:
            self.birthMin.setValue(self.chronProf.rjmcmc.burn_in.birth_accept_target[0])
            self.birthMax.setValue(self.chronProf.rjmcmc.burn_in.birth_accept_target[1])
        if self.chronProf.rjmcmc.burn_in.death_accept_target is not None:
            self.deathMin.setValue(self.chronProf.rjmcmc.burn_in.death_accept_target[0])
            self.deathMax.setValue(self.chronProf.rjmcmc.burn_in.death_accept_target[1])
        if self.chronProf.rjmcmc.burn_in.move_time_accept_target is not None:
            self.timeMin.setValue(self.chronProf.rjmcmc.burn_in.move_time_accept_target[0])
            self.timeMax.setValue(self.chronProf.rjmcmc.burn_in.move_time_accept_target[1])
        if self.chronProf.rjmcmc.burn_in.move_temp_accept_target is not None:
            self.tempMin.setValue(self.chronProf.rjmcmc.burn_in.move_temp_accept_target[0])
            self.tempMax.setValue(self.chronProf.rjmcmc.burn_in.move_temp_accept_target[1])
        if self.chronProf.rjmcmc.burn_in.move_endpoints_accept_target is not None:
            self.endpointMin.setValue(self.chronProf.rjmcmc.burn_in.move_endpoints_accept_target[0])
            self.endpointMax.setValue(self.chronProf.rjmcmc.burn_in.move_endpoints_accept_target[1])

        if self.chronProf.rjmcmc.burn_in.sigma_birth_bounds is not None:
            self.sigmabirthtempMax.setValue(self.chronProf.rjmcmc.burn_in.sigma_birth_bounds[1])
            self.sigmabirthtempMin.setValue(self.chronProf.rjmcmc.burn_in.sigma_birth_bounds[0])
        if self.chronProf.rjmcmc.burn_in.sigma_birth_t_bounds is not None:
            self.sigmabirthtimeMax.setValue(self.chronProf.rjmcmc.burn_in.sigma_birth_t_bounds[1])
            self.sigmabirthtimeMin.setValue(self.chronProf.rjmcmc.burn_in.sigma_birth_t_bounds[0])
        if self.chronProf.rjmcmc.burn_in.sigma_time_bounds is not None:
            self.sigmatimeMin.setValue(self.chronProf.rjmcmc.burn_in.sigma_time_bounds[0])
            self.sigmatimeMax.setValue(self.chronProf.rjmcmc.burn_in.sigma_time_bounds[1])
        if self.chronProf.rjmcmc.burn_in.sigma_temp_bounds is not None:
            self.sigmaTempMin.setValue(self.chronProf.rjmcmc.burn_in.sigma_temp_bounds[0])
            self.sigmatempMax.setValue(self.chronProf.rjmcmc.burn_in.sigma_temp_bounds[1])
        if self.chronProf.rjmcmc.burn_in.sigma_endpoints_bounds is not None:
            self.sigmaEndpointMax.setValue(self.chronProf.rjmcmc.burn_in.sigma_endpoints_bounds[1])
            self.sigmaEndpointMin.setValue(self.chronProf.rjmcmc.burn_in.sigma_endpoints_bounds[0])

    def get_chronInfoToConfig(self) -> bool:
        if self.chronName.text() == "":
            QMessageBox.warning(
                self,
                "No Chronology profile Name",
                "Chronology profile needs a file name. Enter one now to proceed"
            )
            self.pageCleared.emit(3, None)
            return False

        method = "RJMCMC" if self.method.isChecked() else "MCMC"

        monotonic_map = {
            0: "free",
            1: "increasing",
            2: "decreasing",
        }

        move_bounds = None
        if self.individual_check.isChecked():
            move_bounds = [self.movePMin.value(), self.moveP_max.value()]

        overall_accept_target = None
        if self.overall_check.isChecked():
            overall_accept_target = [self.overall_min.value(), self.overall_max.value()]

        birth_accept_target = None
        death_accept_target = None
        move_time_accept_target = None
        move_temp_accept_target = None
        move_endpoints_accept_target = None
        sigma_birth_bounds = None
        sigma_birth_t_bounds = None
        sigma_time_bounds = None
        sigma_temp_bounds = None
        sigma_endpoints_bounds = None

        if self.individual_check.isChecked():
            birth_accept_target = [self.birthMin.value(), self.birthMax.value()]
            death_accept_target = [self.deathMin.value(), self.deathMax.value()]
            move_time_accept_target = [self.timeMin.value(), self.timeMax.value()]
            move_temp_accept_target = [self.tempMin.value(), self.tempMax.value()]
            move_endpoints_accept_target = [self.endpointMin.value(), self.endpointMax.value()]
            sigma_birth_bounds = [self.sigmabirthtempMin.value(), self.sigmabirthtempMax.value()]
            sigma_birth_t_bounds = [self.sigmabirthtimeMin.value(), self.sigmabirthtimeMax.value()]
            sigma_time_bounds = [self.sigmatimeMin.value(), self.sigmatimeMax.value()]
            sigma_temp_bounds = [self.sigmaTempMin.value(), self.sigmatempMax.value()]
            sigma_endpoints_bounds = [self.sigmaEndpointMin.value(), self.sigmaEndpointMax.value()]

        data = {
            "method": method,
            "iters": self.chronIterations.value(),
            "T_Target": self.T_Target.value(),
            "T_tolerance": self.T_tolerance.value(),
            "T0_lo": self.T0_lo.value(),
            "T0_hi": self.T0_hi.value(),
            "monotonic": monotonic_map.get(self.monotonic.currentIndex(), "free"),
            "min_internal": self.min_internal.value(),
            "max_internal": self.max_internal.value(),
            "rjmcmc": {
                "logLikeSigma": self.logLikeSigma.value(),
                "parameters": {
                    "p_birth": self.p_birth.value(),
                    "p_death": self.p_death.value(),
                    "p_move_time": self.p_move_times.value(),
                    "p_move_temp": self.p_move_temp.value(),
                    "p_move_endpoints": self.p_move_endpoints.value(),
                    "sigma_birth": self.sigma_birth.value(),
                    "sigma_t_birth": self.sigma_t_birth.value(),
                    "sigma_temp": self.sigma_temp.value(),
                    "sigma_time_frac": self.sigma_time_frac.value(),
                    "sigma_endpoints": self.sigma_endpoints.value(),
                },
                "burn_in": {
                    "burn": self.burn_in.isChecked(),
                    "max_steps": self.max_steps.value(),
                    "window": self.window.value(),
                    "patience_windows": self.patience_windows.value(),
                    "adjustment_factor": self.adjustment_factor.value(),
                    "eta_prob": self.eta_prob.value(),
                    "eta_sigma": self.eta_sigma.value(),
                    "move_bounds": move_bounds,
                    "overall_check": self.overall_check.isChecked(),
                    "individual_check": self.individual_check.isChecked(),
                    "verbose": self.verbose.isChecked(),
                    "overall_accept_target": overall_accept_target,
                    "birth_accept_target": birth_accept_target,
                    "death_accept_target": death_accept_target,
                    "move_time_accept_target": move_time_accept_target,
                    "move_temp_accept_target": move_temp_accept_target,
                    "move_endpoints_accept_target": move_endpoints_accept_target,
                    "sigma_birth_bounds": sigma_birth_bounds,
                    "sigma_birth_t_bounds": sigma_birth_t_bounds,
                    "sigma_time_bounds": sigma_time_bounds,
                    "sigma_temp_bounds": sigma_temp_bounds,
                    "sigma_endpoints_bounds": sigma_endpoints_bounds,
                },
            },
        }

        try:
            self.chronProf = ChronologyProfile.model_validate(data)
            self.pageCleared.emit(3, self.chronName.text())
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
                "Invalid chronology profile",
                "Please fix the following chronology values:\n\n" + "\n".join(messages),
            )
            self.pageCleared.emit(3, None)
            return False

        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid chronology profile",
                str(exc),
            )
            self.pageCleared.emit(3, None)
            return False



    def burnInToggled(self, checked:bool):
        if checked: 
            self.BurnInWidget.show()
        else:
            self.BurnInWidget.hide() 

    def overallToggled(self, checked:bool):
        if checked: 
            self.overall_min.setEnabled(True)
            self.overall_max.setEnabled(True)
        else:
            self.overall_min.setEnabled(False)
            self.overall_max.setEnabled(False)
    
    def individualToggled(self,checked:bool):
        if checked: 
            self.birthMin.setEnabled(True)
            self.birthMax.setEnabled(True)
            self.deathMin.setEnabled(True)
            self.deathMax.setEnabled(True)
            self.timeMin.setEnabled(True)
            self.timeMax.setEnabled(True)
            self.tempMin.setEnabled(True)
            self.tempMax.setEnabled(True)
            self.endpointMin.setEnabled(True)
            self.endpointMax.setEnabled(True)
            self.sigmabirthtempMax.setEnabled(True)
            self.sigmabirthtempMin.setEnabled(True)
            self.sigmabirthtimeMax.setEnabled(True)
            self.sigmabirthtimeMin.setEnabled(True)
            self.sigmatimeMin.setEnabled(True)
            self.sigmatimeMax.setEnabled(True)
            self.sigmaTempMin.setEnabled(True)
            self.sigmatempMax.setEnabled(True)
            self.sigmaEndpointMax.setEnabled(True)
            self.sigmaEndpointMin.setEnabled(True)
        else:
            self.birthMin.setEnabled(False)
            self.birthMax.setEnabled(False)
            self.deathMin.setEnabled(False)
            self.deathMax.setEnabled(False)
            self.timeMin.setEnabled(False)
            self.timeMax.setEnabled(False)
            self.tempMin.setEnabled(False)
            self.tempMax.setEnabled(False)
            self.endpointMin.setEnabled(False)
            self.endpointMax.setEnabled(False)
            self.sigmabirthtempMax.setEnabled(False)
            self.sigmabirthtempMin.setEnabled(False)
            self.sigmabirthtimeMax.setEnabled(False)
            self.sigmabirthtimeMin.setEnabled(False)
            self.sigmatimeMin.setEnabled(False)
            self.sigmatimeMax.setEnabled(False)
            self.sigmaTempMin.setEnabled(False)
            self.sigmatempMax.setEnabled(False)
            self.sigmaEndpointMax.setEnabled(False)
            self.sigmaEndpointMin.setEnabled(False)

    def chronMethodToggled(self, checked:bool):
        if checked:
            self.RJMCMCwidget.show()
        else:
            self.RJMCMCwidget.hide()

    def chronTempUnits(self,cel:bool):
        if cel: 
            self.chronDegree.show()
            self.chronDegree1.show()
            self.chronDegree2.show()
            self.chronKelvin.hide()
            self.chronKelvin1.hide()
            self.chronKelvin2.hide()
        else: 
            self.chronDegree.hide()
            self.chronDegree1.hide()
            self.chronDegree2.hide()
            self.chronKelvin.show()
            self.chronKelvin1.show()
            self.chronKelvin2.show()


    def load_previousChronProfile(self, listnum:int):
        if listnum == 0: 
            self.chronProf = ChronologyProfile()
            self.chronWidget.setEnabled(True)
            self.chronEditButton.hide()
            self.chronEditButton.setChecked(False)
        else: 
            self.chronWidget.setEnabled(False)
            self.chronEditButton.show()
            self.chronEditButton.setChecked(False)
            profile_name = self.chronInputsList[listnum-1]
            self.chronName.setText(str(profile_name))
            data = load_profile(profile_name, CHRON_DIR)
            self.chronProf = ChronologyProfile.model_validate(data)
        
        self.set_chronInfoFromConfig()
        
    def loadedChronProfileEdit(self, edit:bool):
        if edit:
            self.chronWidget.setEnabled(True)
            text = self.chronName.text()
            self.chronName.setText(f"{text} copy")
        else:
            self.load_previousChronProfile(self.chronBox.currentIndex())
            self.chronWidget.setEnabled(False)
        



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
            checker = self.setWindow.get_chronInfoToConfig()
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

    
    def profile_save(self, name: str, directory: str | Path, profile,  tocheck:bool = False) -> str:

        directory = Path(directory)
        # directory.mkdir(parents=True, exist_ok=True)
        safe_name = name.removesuffix(".yaml")
        path = directory / f"{safe_name}.yaml"
        if path.exists() and tocheck:
            path, safe_name = self.warning(safe_name,directory)
           
        data = profile.model_dump(exclude_none=False)

        save_profile(data, path)
        return safe_name
    

    def save_setup(self,):
        safe_name = self.buttons.experimentName.removesuffix(".yaml")
        path = Path(USER_CONFIG_DIR,f"{safe_name}.yaml")
        if path.exists():
            if self.setWindow.expEditButton.isChecked():
                path, safe_name = self.warning(safe_name,USER_CONFIG_DIR)

        
        self.buttons.temperatureName = self.profile_save(self.buttons.temperatureName, TEMP_DIR, self.setWindow.tempProf, self.setWindow.tempEditProfileButton.isChecked())
        self.buttons.physicsName = self.profile_save(self.buttons.physicsName, PHYS_DIR, self.setWindow.physProf, self.setWindow.physEditCheck.isChecked())
        defaults = [{"/physics": self.buttons.physicsName},{"/temp": self.buttons.temperatureName},]
        if self.buttons.chronNeeded: 
            self.buttons.chronName = self.profile_save(self.buttons.chronName, CHRON_DIR, self.setWindow.chronProf)
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
   
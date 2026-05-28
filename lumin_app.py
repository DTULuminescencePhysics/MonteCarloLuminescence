from __future__ import annotations
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, Slot, QTimer, Qt
from PySide6.QtCore import QProcess
from src.filesystem import PROJECT_ROOT
from PySide6.QtGui import QAction, QPixmap
from GUI.src.plottingWindow import PlottingWindow
from GUI.src.experimentSetup import fullSetupWindow
from GUI.src.designs.mainWindow import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
   
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.menubar.setNativeMenuBar(False)
        self.actionSetup_Experiment = QAction("Setup Experiment", self)
        self.actionGenerate_Plots = QAction("Generate Plots", self)
        self.menuFile.clear()
        self.menuFile.addAction(self.actionSetup_Experiment)
        self.menuFile.addAction(self.actionGenerate_Plots)

        self.plotWindow = PlottingWindow()
        self.setWindow = fullSetupWindow()
        self.runningWindow = self.create_running_page()
        self.previous_page = self.setWindow
        self.pages = QStackedWidget(self)
        self.pages.addWidget(self.setWindow)
        self.pages.addWidget(self.plotWindow)
        self.pages.addWidget(self.runningWindow)

        self.setCentralWidget(self.pages)
        self.setWindow.runSignal.connect(self.runMainprogram)
        self.actionGenerate_Plots.triggered.connect(self.setPlotWindow)
        self.actionSetup_Experiment.triggered.connect(self.setExperiment)

        self.process = None
        self.current_run_code = None


    def create_running_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_label = QLabel(page)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_path = Path(PROJECT_ROOT, "GUI", "src", "designs", "all_logo.png")
        if image_path.exists():
            pixmap = QPixmap(str(image_path))
            image_label.setPixmap(
                pixmap.scaled(
                    320,
                    320,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            image_label.setText("Experiment running...")
            image_label.setStyleSheet("font-size: 24px; font-weight: bold;")

        message_label = QLabel("Please wait until the current process finishes.", page)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(image_label)
        layout.addWidget(message_label)

        return page

    def set_navigation_enabled(self, enabled: bool):
        self.actionSetup_Experiment.setEnabled(enabled)
        self.actionGenerate_Plots.setEnabled(enabled)
        self.menuFile.setEnabled(enabled)


    def setExperiment(self):
        if self.process is not None:
            return
        self.pages.setCurrentWidget(self.setWindow)

    def setPlotWindow(self):
        if self.process is not None:
            return
        self.pages.setCurrentWidget(self.plotWindow)

    def runMainprogram(self, runCode: int):
        self.current_run_code = runCode

        if runCode == 1:
            self.start_main_process([])
        elif runCode == 0:
            self.start_main_process(["--save"])


    def start_main_process(self, args: list[str]):
        if self.process is not None:
            QMessageBox.information(
                self,
                "Process already running",
                "Please wait for the current process to finish.",
            )
            return

        self.previous_page = self.pages.currentWidget()
        self.pages.setCurrentWidget(self.runningWindow)
        self.set_navigation_enabled(False)

        project_root = PROJECT_ROOT
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(project_root))
        self.process.readyReadStandardOutput.connect(self.read_process_stdout)
        self.process.readyReadStandardError.connect(self.read_process_stderr)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        program = sys.executable
        arguments = ["main.py", *args]
        self.process.start(program, arguments)
        if not self.process.waitForStarted(3000):
            self.set_navigation_enabled(True)
            self.pages.setCurrentWidget(self.previous_page)
            self.process = None
            QMessageBox.critical(
                self,
                "Process failed",
                "Could not start main.py.",
            )

    def read_process_stdout(self):

        if self.process is None:
            return

        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        print(text, end="")

        # Optional: append to a QTextEdit in your UI

        # self.logTextEdit.append(text)

    def read_process_stderr(self):

        if self.process is None:
            return

        text = bytes(self.process.readAllStandardError()).decode(errors="replace")
        print(text, end="")

        # Optional: append to a QTextEdit in your UI
        # self.logTextEdit.append(text)

    def latest_result_file(self) -> Path | None:
        run_dir = Path(PROJECT_ROOT, "run")

        if not run_dir.exists():
            return None

        result_files = list(run_dir.rglob("result.hdf5"))

        if not result_files:
            return None

        return max(result_files, key=lambda path: path.stat().st_mtime)

    def load_result_into_plotting_page(self):
        result_file = self.latest_result_file()

        if result_file is None:
            QMessageBox.warning(
                self,
                "Result file not found",
                "The experiment finished, but no result.hdf5 file was found in the run folder.",
            )
            return

        self.pages.setCurrentWidget(self.plotWindow)
        self.previous_page = self.plotWindow

        self.plotWindow.new_file_selected(result_file)
        self.plotWindow.DisplayArea.experiment_number_check(result_file)

    def process_finished(self, exit_code: int, exit_status):
        self.set_navigation_enabled(True)
        self.pages.setCurrentWidget(self.previous_page)
        if exit_code == 0:
            if self.current_run_code == 1:
                self.load_result_into_plotting_page()
                QMessageBox.information(
                    self,
                    "Process complete",
                    "The process finished successfully. The result file has been loaded into the plotting page.",
                )
            else:
                QMessageBox.information(
                    self,
                    "Process complete",
                    "The process finished successfully.",
                )
        else:
            QMessageBox.critical(
                self,
                "Process failed",
                f"The process exited with code {exit_code}.",
            )
        self.process = None
        self.current_run_code = None

    def process_error(self, error):
        self.set_navigation_enabled(True)
        self.pages.setCurrentWidget(self.previous_page)
        self.process = None
        self.current_run_code = None
        QMessageBox.critical(
            self,
            "Process error",
            f"Could not run main.py: {error}",
        )

app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec())
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QFileDialog


class FileSelector(QWidget):
    fileSelected = Signal(Path)  
    savedPlot = Signal(str)
    def __init__(self, parent=None,):
        super().__init__(parent)

        self.file_path = None
        self.file_name = None
        self.parent_folder = None
    def open_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select output file",
            "",
            "Hierarchical Data Format version 5 (*.hdf5)"
        )

        if file_path:
            path = Path(file_path)

            if(self.file_path is None) or (self.file_path != str(path)):
                self.file_path = str(path)
                self.file_name = path.name
                self.fileSelected.emit(path)
                self.parent_folder = path.parent
    
    def open_save_dialog(self):
        if(self.file_path is None):
            return
        else:
            self.save_dialog_launch()
    def save_dialog_launch(self):
        if self.parent_folder is None:
            folder = str(Path.home())
        else:
            folder = str(self.parent_folder)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            folder,
            "PNG Files (*.png);;"

        )

        if not file_path:
            return

        self.savedPlot.emit(file_path)
        return 
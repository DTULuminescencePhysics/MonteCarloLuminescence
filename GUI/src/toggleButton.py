from __future__ import annotations
from pathlib import Path
from PySide6.QtWidgets import QCheckBox, QPushButton
from PySide6.QtGui import QIcon
from PySide6.QtCore import QSize

svgFolder =  Path(__file__).parent / "designs" / "SVG" 
leftbuttonPath = svgFolder / "leftToggle.svg"
rightbuttonPath = svgFolder / "rightToggle.svg"
plusbuttonPath = svgFolder / "plus.svg"
minusbuttonPath = svgFolder / "minus.svg"

TOGGLE_STYLE = f"""
    QCheckBox {{
        spacing: 8px;
    }}

    QCheckBox::indicator {{
        width: 52px;
        height: 28px;
    }}

    QCheckBox::indicator:unchecked {{
        image: url("{leftbuttonPath.as_posix()}");
    }}

    QCheckBox::indicator:checked {{
        image: url("{rightbuttonPath.as_posix()}");
    }}
"""

def set_toggle(checkbox:QCheckBox):
    checkbox.setStyleSheet(TOGGLE_STYLE)
    return 


BUTTON_STYLE = """
    QPushButton {
        border: none;
        background: transparent;
    }
    QPushButton:hover {
        background: rgba(0, 0, 0, 20);
        border-radius: 4px;
    }
    """
def add_remove_button(button:QPushButton,add:bool):
    if add: 
        button.setIcon(QIcon(str(plusbuttonPath)))
        button.setIconSize(QSize(24,24))
    else: 
        button.setIcon(QIcon(str(minusbuttonPath)))
        button.setIconSize(QSize(24,24))
    button.setStyleSheet(BUTTON_STYLE)

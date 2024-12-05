from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import sys
import os

class InfoAPP(QDialog):

    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), "info_config.ui")
        uic.loadUi(ui_path, self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ventana = InfoAPP()
    _ventana.show()
    app.exec_()
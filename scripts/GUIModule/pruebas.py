from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import pyqtgraph as pg
import sys
import os
import numpy as np
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import time

class SupervisionAPP(QDialog):
    """
    Interfaz gráfica para supervisar las sesiones
    """
    def __init__(self, clases:list, tiempo_eeg:float, board_shim):
        super().__init__()
        
        ui_path = os.path.join(os.path.dirname(__file__), 'supervision.ui')
        uic.loadUi(ui_path, self)
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        pg.setConfigOptions(antialias=True)

        # create a new QGraphicsScene
        scene = QGraphicsScene()

        # create a new QGraphicsView and set the scene
        self.graphicsView.setScene(scene)

        # create a new pg.GraphicsLayoutWidget
        self.graphics_window = pg.GraphicsLayoutWidget(title='BrainFlow Plot', size=(800, 600))

        # add the graphics layout widget to the scene
        scene.addWidget(self.graphics_window)

        self.graphicsView.resize(800, 600)

        self.traces = dict()
        self.x = np.array([0])
        self.y = np.array([0])
        self.tiempo_eeg = tiempo_eeg
        self.clases = clases

        self._init_timeseries()
        
    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.graphics_window.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        # print(data)
        for count, channel in enumerate(self.exg_channels):
            self.curves[count].setData(data[channel].tolist())


app = QApplication(sys.argv)


board_id = brainflow.board_shim.BoardIds.SYNTHETIC_BOARD.value
sampling_rate = 250
board = BoardShim(board_id, BrainFlowInputParams())

board.prepare_session()
board.start_stream()

time.sleep(2)

_ventana = SupervisionAPP(['AD', 'DP'], 2, board)

timer = QtCore.QTimer()
timer.timeout.connect(_ventana.update)
timer.start(_ventana.update_speed_ms)

_ventana.show()
app.exec_()
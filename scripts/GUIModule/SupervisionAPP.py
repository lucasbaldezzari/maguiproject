from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import pyqtgraph as pg
import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph import LabelItem

class SupervisionAPP(QDialog):
    """
    Interfaz gráfica para supervisar las sesiones
    """
    def __init__(self, clases:list, canales:list, umbral_clasificacion = 75):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(__file__), 'supervision.ui')
        uic.loadUi(ui_path, self)

        self.canales = canales
        self.clases = clases

        self.umbral_calsificacion = umbral_clasificacion #% umbral de clasificación

        self.sample_rate = 250.
        self.t_lenght = 2 # segundos de señal
        #Creo un numpyarray para hacer de buffer de datos que se mostrará en pantalla
        #El shape es (canales, tiempo)
        self.data = np.zeros((len(self.canales), int(self.sample_rate*self.t_lenght)))
        self.acumForFFT_smooth = 10 #cantidad a acumular para suavizar la FFT
        self.acumForFFT = 0

        #creo eje tiempo
        self.tline = np.linspace(0, self.t_lenght, int(self.sample_rate*self.t_lenght))

        #creo eje frecuencia
        self.fline = np.linspace(0, self.sample_rate/2, int(self.sample_rate*self.t_lenght/2))
        #frecuencias mínima y máxima a graficar
        self.fmin = 5
        self.fmax = 30

        pg.setConfigOptions(antialias=True)

        self.desplegable_escala.currentIndexChanged.connect(self.change_scale)

        self.escalas = ['Auto', 50e-6, 100e-6, 200e-6, 400e-6, 1e-3, 10e-3, 100e-3, 1, 10, 100]

        # new QGraphicsScene for timeseries EEG data
        scene = QGraphicsScene()
        self.graphicsView.setScene(scene)
        self.graphics_window = pg.GraphicsLayoutWidget(title='EEG Plot', size=(950, 390))
        self.graphics_window.setBackground('w')
        scene.addWidget(self.graphics_window)

        # new QGraphicsScene for probabilties bars
        scene2 = QGraphicsScene()
        self.graphicsBars.setScene(scene2)
        self.graphics_window2 = pg.GraphicsLayoutWidget(title='Bars', size=(400, 250))
        self.graphics_window2.setBackground('w')
        scene2.addWidget(self.graphics_window2)

        # new QGraphicsScene for FFT data
        scene3 = QGraphicsScene()
        self.graphicsFFT.setScene(scene3)
        self.graphics_window3 = pg.GraphicsLayoutWidget(title='FFT', size=(500, 250))
        self.graphics_window3.setBackground('w')
        scene3.addWidget(self.graphics_window3)

        #8 colores para los canales de EEG
        self.colores_eeg = ['#fb7e7b', '#ebcb5b', '#77aa99', '#581845', '#F7DC6F', '#F1C40F', '#9B59B6','#8E44AD']

        #colores para las barras de probabilidad
        self.colores_barras = ['#8199c8', '#b58fbb', '#77aa99', '#edcf5b', '#fa7f7c', '#F1C40F', '#9B59B6','#8E44AD']

        import matplotlib
        #obtengo la paleta de colores summer_r
        self.colormap_timeBar = matplotlib.colormaps["Blues"]
        del matplotlib

        self.tiempo_actual = 0 #tiempo actual del trial. Su usa para la barra de progreso
        self.reset_timeBar = False #para resetear la barra de progreso
        self.tipo_sesiones = ["Entrenamiento", "Calibración/Feedback", "Online"]
        self.fases = {
            0: "Preparación",
            1: "Apagando texto preparación",
            2: "Prendiendo tarea",
            3: "Acción",
            4: "Apagando tarea",
            5: "Mensaje de fin de tarea",
            6: "Apagando mensaje de fin de tarea",
            6: "Guardando EEG / descanso"
        } ## Los nombres de las fases se defasan para estar en concordancia con el Core.py

        self._init_timeseries()
        self._init_barras()
        self._init_FFT()

        self.update_propbars([0.0 for i in range(len(self.clases))])

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()

        for i in range(len(self.canales)):
            p = self.graphics_window.addPlot(row=i, col=0)
            p.showAxis('left', True)
            p.setMenuEnabled('left', True)
            p.showGrid(x=True, y=True)

            ax0 = p.getAxis('left') #para manipular el axis izquierdo (eje y)
            ax0.setStyle(showValues=True)
            ax0.setLabel(f"C{self.canales[i]}", color=self.colores_eeg[i], size='14pt', bold=True)

            ## Creo eje inferior
            ax1 = p.getAxis('bottom') #para manipular el eje inferior (eje x)
            ax1.setStyle(showValues=True)
            ax1.setTickFont(QFont('Arial', 8))
            ax1.setRange(0, self.t_lenght)
            
            p.showAxis('top', False)  # Ocultar el eje superior
            p.showAxis('bottom', True)

            self.plots.append(p)

            # p.addLegend(size=(2,0), offset=(-0.5,-0.1))# agrego leyenda por cada gráfico
            curve = p.plot(pen = self.colores_eeg[i], name = f'Canal {i}')
            self.curves.append(curve)
    
    def change_scale(self):
        indice = self.desplegable_escala.currentIndex()

        if indice == 0:
            for p in self.plots:
                p.enableAutoRange(axis='y')

        else:
            limites = self.escalas[indice]
            for p in self.plots:
                p.setRange(yRange=(-limites, limites))

    def _init_FFT(self):
        self.plots3 = list()
        self.curves2 = list()

        p = self.graphics_window3.addPlot(row=0, col=0)
        p.showAxis('left', True)
        p.setMenuEnabled('left', False)
        p.showAxis('bottom', True)
        p.setMenuEnabled('bottom', False)
        p.showGrid(x=True, y=True)

        ax1 = p.getAxis('left') #para manipular el eje inferior (eje x)
        ax1.setStyle(showValues=False)
        ax1.setLabel(f"Amplitud (uv)", color="k", size='14pt', bold=True)

        self.plots3.append(p)

        for i in range(len(self.canales)):
            curve = p.plot(pen = self.colores_eeg[i])
            self.curves2.append(curve)

    def update_plots(self, newData):

        samplesToRemove = newData.shape[1] #muestras a eliminar del buffer interno de datos
        ## giro el buffer interno de datos
        self.data = np.roll(self.data, -samplesToRemove, axis=1)
        ## reemplazo los ultimos datos del buffer interno con newData
        self.data[:, -samplesToRemove:] = newData

        for canal in range(len(self.canales)):
            self.curves[canal].setData(self.tline,self.data[canal])

        self.update_FFT(self.data)


    def _init_barras(self):
        self.plots2 = list()
        self.bars = list()

        br = self.graphics_window2.addPlot(row=0, col=0)
        br.showAxis('left', True)
        br.setMenuEnabled('left', False)
        br.showAxis('bottom', True)
        br.setMenuEnabled('bottom', False)
        self.plots2.append(br)
        bottom_axis = br.getAxis('bottom') #eje inferior
        bottom_axis.setTicks([[(i, self.clases[i]) for i in range(len(self.clases))]])

        ## Manipulando eje izquierdo
        lef_axis = br.getAxis('left')
        lef_axis.setStyle(showValues=True)
        lef_axis.setPen('k')
        lef_axis.setTickFont(QFont('Arial', 10))
        lef_axis.setRange(0, 100)

        ## Agrego una línea horizontal en 100%
        self.hline100 = pg.InfiniteLine(pos=100, angle=0, movable=False, pen='k')
        br.addItem(self.hline100)

        ## Agrego una lina para el umbral
        hline_umbral = pg.InfiniteLine(pos=self.umbral_calsificacion, angle=0, movable=False, pen='k')
        hline_umbral.setPen(pg.mkPen('#548711', width=1, style=Qt.DashLine))
        br.addItem(hline_umbral)

        for i in range(len(self.clases)):
            bar = pg.BarGraphItem(x=[i], height=[0], width=0.8, brush=pg.mkBrush(self.colores_barras[i]))
            br.addItem(bar)
            self.bars.append(bar)

    def update_propbars(self, data = [0.5, 0.5]):
        for i, bar in enumerate(self.bars):
            bar.setOpts(height=round(data[i]*100,2))

    def update_FFT(self, data):
        """Calcula la FFT de los datos y los grafica en self.fmin y self.fmax"""

        #tomo los datos de la fft y los grafico en self.fmin y self.fmax
        data = abs(np.fft.fft(data).real)[:,int(self.fmin*self.t_lenght):int(self.fmax*self.t_lenght)]

        fline = np.linspace(self.fmin, self.fmax, data.shape[1])

        if self.acumForFFT == 0:
            self.smoothFFT = data
            self.acumForFFT += 1
        else:
            self.smoothFFT += data
            self.acumForFFT += 1

        if self.acumForFFT == self.acumForFFT_smooth:
            data = self.smoothFFT/self.acumForFFT_smooth
            for canal in range(len(self.canales)):
                        self.curves2[canal].setData(fline, data[canal])
            self.acumForFFT = 0

        

        # for canal in range(len(self.canales)):
        #     self.curves2[canal].setData(fline, data[canal])

    def update_order(self, texto:str):
        """
        Actualiza la etiqueta que da la orden
            texto (str): texto de la orden
        """
        self.label_orden.setText(texto)

    def update_timebar(self, tiempo_total, delta_tiempo, phase):
        """
        Actualiza la barra de progreso del trial
            tiempo_actual (float): tiempo actual del trial en segundos. No debe ser mayor al tiempo de trial total
            phase (int): phase actual cuando se actualiza la barra.
                0: preparacion;
                1: accion;
                2: descanso;
        """
        try:
            self.tiempo_actual += delta_tiempo
            progreso = int(self.tiempo_actual *100/tiempo_total)
            colormap = self.colormap_timeBar(progreso)
            color_barra = f"background-color: rgb({colormap[0]*255}, {colormap[1]*255}, {colormap[2]*255});"
            self.progressBar.setValue(progreso)

            self.progressBar.setStyleSheet("QProgressBar::chunk {" + color_barra + "}")

            if phase == 0 or self.reset_timeBar:
                self.tiempo_actual = 0
                self.reset_timeBar = False

        except:
            print("Error al actualizar la barra de progreso")

    def update_info(self, sesion:int, trial_duration:float, phase:int, trial_actual:int, trials_totales:int):
        """
        Para actualizar los campos de información en la interfaz de superivisión
        """
        self.label_sesion_type.setText(f'Tipo de Sesión: {self.tipo_sesiones[sesion]}')
        self.label_trial_duration.setText(f'Tiempo del Trial: {trial_duration} s')
        if phase == 0 or self.reset_timeBar:
            self.label_trial_time.setText(f'Progreso: {0.0} s')
        else:
            self.label_trial_time.setText(f'Progreso: {round(self.tiempo_actual,1)} s')
        self.label_trial_phase.setText(f'Fase Actual: {self.fases[phase]}')
        self.label_actual_trial.setText(f'Trial actual / Trials Totales: {trial_actual+1}/{trials_totales}')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ventana = SupervisionAPP(['AD', 'DP'], [1,2,3])

    _ventana.update_info(1, 10, 1, 10, 100)
    _ventana.update_timebar(10, 1, 2)

    _ventana.show()
    app.exec_()
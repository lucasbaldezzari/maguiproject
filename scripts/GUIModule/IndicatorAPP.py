from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import sys
import os
import winsound

class IndicatorAPP(QDialog):
    """
    Interfaz gráfica que tiene como fin únicamente mostrar la orden para la adquisición de datos 
    de entrenamiento para el clasificador
    """
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), 'registro.ui')
        uic.loadUi(ui_path, self)

        screen_resolution = QApplication.desktop().screenGeometry()
        self.width, self.height = screen_resolution.width(), screen_resolution.height()

        # Establecer las dimensiones de la ventana
        self.setGeometry(0, 0, self.width, self.height)

        self.Centrar(self.cruz)
        self.showCruz(False) #Si seleccionas False no hará ruido al instanciar la interfaz
        self.showCueOnSquare(False)
        self.showCueOffSquare(True)
        self.showBar(False)
        self.Bajar(self.progressBar, 200)

        #obtenemos el background de label_orden
        self.background_color = self.label_orden.palette().color(QPalette.Background).name()
        self.font_color = "rgb(25,50,200)"

        import matplotlib
        #obtengo la paleta de colores summer_r
        self.colormap_barra = matplotlib.colormaps["summer_r"]
        del matplotlib
        self.cue_on_square.setVisible(False)
        self.cue_off_square.setVisible(True)
        self.session_on_square.setVisible(False)
        self.session_off_square.setVisible(True)

    def update_order(self, texto, fontsize = 36, background = None, border = "0px", font_color = "black",
                     opacity = 1.0):
        """
        Actualiza la etiqueta que da la orden
            texto (str): texto de la orden
        """
        self.label_orden.setFont(QFont('Berlin Sans', fontsize))
        self.label_orden.setText(texto)
        if background:
            self.label_orden.setStyleSheet(f"background-color: {background};border: {border} solid black;color: {font_color}")
        else:
            self.label_orden.setStyleSheet(f"background-color: {self.background_color}; border: 0px solid black;color: {self.font_color}")
        
        # self.opacity_effect.setOpacity(opacity)
        # self.label_orden.setGraphicsEffect(self.opacity_effect)
        
    def actualizar_barra(self, probabilidad:float = 0.5):
        """
        Actualiza la etiqueta que da la orden
            texto (str): texto de la orden
        """
        porcentaje = int(probabilidad*100)

        self.progressBar.setValue(porcentaje)
        colormap = self.colormap_barra(probabilidad)
        color_barra = f"background-color: rgb({colormap[0]*250}, {colormap[1]*250}, {colormap[2]*250});"

        self.progressBar.setStyleSheet("QProgressBar::chunk {" + color_barra + "}")

    def Centrar(self, objeto):
        """
        Centra el objeto (widget) en la pantalla
        """

        # Centrar el objeto en la ventana
        objeto.setGeometry(int(self.width/2 - objeto.width()/2), int(self.height/2 - objeto.height()/2), 
                           int(objeto.width()),int(objeto.height()))
        
    def Subir(self, objeto, pixeles:int = 100):
        """
        Baja el objeto desde el centro de la pantalla
        """
        objeto.setGeometry(int(self.width/2 - objeto.width()/2), int(self.height/2 - objeto.height()/2) - pixeles, 
                           int(objeto.width()),int(objeto.height()))
        
    def Bajar(self, objeto, pixeles:int = 100):
        """
        Baja el objeto desde el centro de la pantalla
        """
        objeto.setGeometry(int(self.width/2 - objeto.width()/2), int(self.height/2 - objeto.height()/2) + pixeles, 
                           int(objeto.width()),int(objeto.height()))
        
    def showCruz(self, mostrar:bool):
        """
        Muestra o no la cruz de preparación en la interfaz
        """
        if mostrar:
            self.cruz.setVisible(True)
            self.Subir(self.label_orden)
            # winsound.Beep(440, 1000)
        else:
            self.cruz.setVisible(False)
            self.Centrar(self.label_orden)

    def showCueOffSquare(self, mostrar:bool = False):
        """
        Muestra o no la cruz de preparación en la interfaz
        """
        if mostrar:
            self.cue_off_square.setVisible(True)
            # self.Subir(self.label_orden)
            # winsound.Beep(440, 1000)
        else:
            self.cue_off_square.setVisible(False)
            # self.Centrar(self.label_orden)

    def showCueOnSquare(self, mostrar:bool = False):
        """
        Muestra o no la cruz de preparación en la interfaz
        """
        if mostrar:
            self.cue_on_square.setVisible(True)
        else:
            self.cue_on_square.setVisible(False)

    def showSessionOffSquare(self, mostrar:bool = False):
        """
        Muestra o no la cruz de preparación en la interfaz
        """
        if mostrar:
            self.session_off_square.setVisible(True)
        else:
            self.session_off_square.setVisible(False)

    def showSessionOnSquare(self, mostrar:bool = False):
        """
        Muestra o no la cruz de preparación en la interfaz
        """
        if mostrar:
            self.session_on_square.setVisible(True)
            # self.Subir(self.label_orden)
            # winsound.Beep(440, 1000)
        else:
            self.session_on_square.setVisible(False)
            # self.Centrar(self.label_orden)

    def showBar(self, mostrar:bool):
        """
        Muestra o no la barra de éxito del clasificador
        """
        if mostrar:
            self.progressBar.setVisible(True)
        else:
            self.progressBar.setVisible(False)
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ventana = IndicatorAPP()
    _ventana.show()
    app.exec_()
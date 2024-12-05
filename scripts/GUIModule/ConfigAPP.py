from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import sys
import json
import os

class ConfigAPP(QDialog):

    def __init__(self, fileName, InfoAPP):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), "config_registro.ui")
        uic.loadUi(ui_path, self)
        self.is_open = True
        self.fileName = fileName
        self.btn_iniciar.clicked.connect(self.Inicio) #Guarda en .json e inicia la interfaz de entrenamiento
        self.btn_regresar.clicked.connect(self.Cerrar) #Cierra la interfaz
        self.btn_guardar.clicked.connect(self.Guardar) #Solo reescribe el .json
        self.btn_info.clicked.connect(self.Informacion) #Solo reescribe el .json
        self.configParameters = self.Cargar() #establece los valores al inicio en base al archivo .json
        self.info_app = InfoAPP

    def Cargar(self):
        with open(self.fileName, 'r') as fp:
            configParameters = json.load(fp)
            self.desplegable_sesion.setCurrentIndex(configParameters["typeSesion"])
            self.desplegable_paradigma.setCurrentIndex(configParameters["cueType"])
            self.line_trials.setText(f'{configParameters["ntrials"]}')
            self.line_valorclases.setText(f'{configParameters["classes"]}')
            self.line_nombreclases.setText(f'{configParameters["clasesNames"]}')
            self.line_inicio.setText(f'{configParameters["startingTimes"]}')
            self.line_tiempo_accion.setText(f'{configParameters["cueDuration"]}')
            self.line_tiempo_descanso.setText(f'{configParameters["finishDuration"]}')
            self.line_tiempoClasif.setText(f'{configParameters["lenToClassify"]}')
            self.onlineCommandSendingTime.setText(f'{configParameters["onlineCommandSendingTime"]}')
            self.line_ventanaClasif.setText(f'{configParameters["lenForClassifier"]}')
            self.umbralClassifier.setText(f'{configParameters["umbralClassifier"]}')
            self.numberOfRounds.setText(f'{configParameters["numberOfRounds"]}')
            self.line_sujeto.setText(f'{configParameters["subjectName"]}')
            self.line_sesion.setText(f'{configParameters["sesionNumber"]}')

            #Parámetros para inicar la placa openbci
            self.desplegable_placa.setCurrentText(configParameters["boardParams"]['boardName'])
            self.line_canales.setText(f'{configParameters["boardParams"]["channels"]}')
            self.line_puerto.setText(f'{configParameters["boardParams"]["serialPort"]}')

            #parámetros del filtro
            self.line_lowpass.setText(f'{configParameters["filterParameters"]["lowcut"]}')
            self.line_highpass.setText(f'{configParameters["filterParameters"]["highcut"]}')
            self.line_notch.setText(f'{configParameters["filterParameters"]["notch_freq"]}')
            self.line_anchonotch.setText(f'{configParameters["filterParameters"]["notch_width"]}')
            self.line_frecuencia.setText(f'{configParameters["filterParameters"]["sample_rate"]}')
            self.line_axis.setText(f'{configParameters["filterParameters"]["axisToCompute"]}')

            self.desplegable_extractor.setCurrentText(configParameters['featureExtractorMethod'])

            self.events_file.setText(f'{configParameters["events_file"]}')
            self.line_clasificador.setText(f'{configParameters["classifierFile"]}')

            if configParameters["customPipeline"] == True:
                self.desplegable_pipeline.setCurrentText('SI')
            else:
                self.desplegable_pipeline.setCurrentText('NO')

            self.line_pipeline.setText(f'{configParameters["pipelineFile"]}')

            self.arduinoFlag.setCurrentText(f'{configParameters["arduinoFlag"]}')
            self.arduinoPort.setText(f'{configParameters["arduinoPort"]}')

        return configParameters

    def Guardar(self):
        with open(self.fileName, 'w') as fp:
            configParameters = dict()
            configParameters["typeSesion"] = int(self.desplegable_sesion.findText(self.desplegable_sesion.currentText()))
            configParameters["cueType"] = int(self.desplegable_paradigma.findText(self.desplegable_paradigma.currentText()))
            configParameters["ntrials"] = int(self.line_trials.text())
            configParameters["classes"] = eval(self.line_valorclases.text())
            configParameters["clasesNames"] = eval(self.line_nombreclases.text())
            configParameters["startingTimes"] = eval(self.line_inicio.text())
            configParameters["cueDuration"]= float(self.line_tiempo_accion.text())
            configParameters["finishDuration"] = float(self.line_tiempo_descanso.text())
            configParameters["lenToClassify"] = float(self.line_tiempoClasif.text())
            configParameters["onlineCommandSendingTime"] = float(self.onlineCommandSendingTime.text())
            configParameters["lenForClassifier"] = float(self.line_ventanaClasif.text())
            configParameters["umbralClassifier"] = float(self.umbralClassifier.text())
            configParameters["numberOfRounds"] = float(self.numberOfRounds.text())
            configParameters["subjectName"] = self.line_sujeto.text()
            configParameters["sesionNumber"] = int(self.line_sesion.text())

            #Parámetros para inicar la placa openbci
            configParameters["boardParams"] = dict()
            configParameters["boardParams"]['boardName'] = self.desplegable_placa.currentText()
            configParameters["boardParams"]['channels'] = eval(self.line_canales.text()) 
            configParameters["boardParams"]["serialPort"] = self.line_puerto.text()

            #parámetros del filtro
            configParameters["filterParameters"] = dict()
            configParameters["filterParameters"]["lowcut"] = float(self.line_lowpass.text())
            configParameters["filterParameters"]["highcut"] = float(self.line_highpass.text())
            configParameters["filterParameters"]["notch_freq"] = float(self.line_notch.text())
            configParameters["filterParameters"]["notch_width"] = float(self.line_anchonotch.text())
            configParameters["filterParameters"]["sample_rate"] = float(self.line_frecuencia.text())
            configParameters["filterParameters"]["axisToCompute"] = int(self.line_axis.text())

            configParameters['featureExtractorMethod'] = self.desplegable_extractor.currentText()

            configParameters["events_file"] = self.events_file.text()
            configParameters["classifierFile"] = self.line_clasificador.text()

            if self.desplegable_extractor.currentText() == 'SI':
                configParameters["customPipeline"] = True
            else:
                configParameters["customPipeline"] = False

            configParameters["pipelineFile"] = self.line_pipeline.text()

            configParameters["arduinoFlag"] = self.arduinoFlag.currentText()
            configParameters["arduinoPort"] = self.arduinoPort.text()

            json.dump(configParameters, fp, indent = 4)

        self.configParameters = configParameters #actualizamos los parámetros de la sesión

    def getParameters(self):
        return self.configParameters

    def Inicio(self):
        self.Guardar()
        self.hide()
        self.is_open = False
    
    def Informacion(self):
        self.info_app.show()

    def Cerrar(self):
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    from InfoAPP import InfoAPP
    _ventana = ConfigAPP("config.json", InfoAPP())
    _ventana.show()
    app.exec_()
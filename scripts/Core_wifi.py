from EEGLogger.EEGLogger import EEGLogger, setupBoard

from SignalProcessor.Filter import Filter
# from SignalProcessor.RavelTransformer import RavelTransformer
# from SignalProcessor.FeatureExtractor import FeatureExtractor
from ArduinoCommunication.ArduinoCommunication import ArduinoCommunication

import json
import os
import time
import random
import logging

import numpy as np
import pandas as pd
import pickle

import sys
from PyQt5.QtCore import QTimer#, QThread, pyqtSignal, pyqtSlot, QObject, QRunnable, QThreadPool, QTime, QDate, QDateTime
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from GUIModule.IndicatorAPP import IndicatorAPP
from GUIModule.ConfigAPP import ConfigAPP
from GUIModule.SupervisionAPP import SupervisionAPP
from GUIModule.InfoAPP import InfoAPP

from sklearn.pipeline import Pipeline

class Core(QMainWindow):
    """Esta clase es la clase principal del sistema.
    Clase para manejar el bloque de procesamiento de los datos (filtrado, extracción de característica y clasificación),
    las GUI y el bloque de comunicación con el dispositivo de control. 
    Esta clase usará las clases EEGLogger, Filter, FeatureExtractor y Classifier para procesar la señal de EEG.
    NOTA: Queda pendiente ver la implementación de al menos un hilo para controlar el Filter, CSPMultcilass, FeatureExtractor,
    RavelTransformer y Classifier. Este hilo deberá ser el que controle los tiempos de de inicio/inter-trial, el tiempo
    del cue y el de descanso. Además, deberá ser el encargado de enviar los comandos al dispositivo de control (o bien,
    se puede hacer otro para el control de la comunicación con el dispositivo.)
    

    FLUJO BÁSICO DEL PROGRAMA
    Si bien se pueden tener tres tipos de sesiones diferentes, el flujo básico del programa es el siguiente:
    1. Se inicia el contrsuctor. Dentro del mismo se configuran parámetros importantes y se inicializan las clases
    ConfigAPP, IndicatorAPP y SupervisionAPP. Además, se configuran los timers para controlar el inicio de la sesión,
    el final de la sesión y el tiempo de cada trial. Finalmente se llama showGUIAPPs() para mostrar las GUI el cual
    inica las app y el timer configAppTimer.
    2. Una vez que se inicia la sesión, se cierra la app de configuración y se llama a la función self.start() para iniciar
    el timer self.iniSesionTimer el cual controla self.startSesion() quien inicia la comunicación con la placa y dependiendo
    el tipo de sesión iniciará el timer self.trainingEEGThreadTimer o self.feedbackThreadTimer.
    3. Estos timers controla los procesos dependiendo del tipo de sesión. Pero de manera general se encargan de controlar
    el tiempo de cada trial y de cada fase del trial. Por ejemplo, en el caso de la sesión de entrenamiento, el timer
    self.trainingEEGThreadTimer controla el tiempo de cada trial y de cada fase del trial. En la fase de inicio, se muestra la cruz
    y se espera un tiempo aleatorio entre self.startingTimes[0] y self.startingTimes[1]. Luego, se pasa a la fase de cue
    y finalmente a la fase de finalización del trial. En esta última fase se guardan los datos de EEG y se incrementa el
    número de trial. Este proceso se repite hasta que se alcanza el último trial de la sesión.

    """
    def __init__(self, configParameters, configAPP, indicatorAPP, supervisionAPP):
        """Constructor de la clase
        - Parameters (dict): Diccionario con los parámetros a ser cargados. Los parámetros son:
            -typeSesion (int): Tipo de sesión. 0: Entrenamiento, 1: Feedback o calibración, 2: Online.
            -cueType (int): 0: se ejecuta movimiento, 1: se imaginan movimientos.
            -ntrials (int): Número de trials a ejecutar.
            -classes (list): Lista de valores enteros de las clases a clasificar.
            -clasesNames (list): Lista con los nombres de las clases a clasificar.
            -startingTimes (lista): Lista con los valores mínimo y máximo a esperar antes de iniciar un nuevo cue o tarea. 
            Estos valores se usan para generar un tiempo aleatorio entre estos valores.
            -cueDuration (float): Duración del cue en segundos.
            -finishDuration (float): Duración del tiempo de finalización en segundos.
            -lenToClassify (float): Tiempo a usar para clasificar la señal de EEG.
            -lenForClassifier (float): Tiempo total de EEG para alimentar el clasificador.
            -subjectName (str): Nombre del sujeto.
            -sesionNumber (int): Número de la sesión.
            -boardParams (dict): Diccionario con los parámetros de la placa. Los parámetros son:
                -boardName (str): Nombre de la placa. Puede ser cyton, ganglion o synthetic.
                -serialPort (str): Puerto serial de la placa. Puede ser del tipo /dev/ttyUSB0 o COM3 (para windows).
            -filterParameters (dict): Diccionario con los parámetros del filtro. Los parámetros son:
                l-owcut (float): Frecuencia de corte inferior del filtro pasa banda.
                -highcut (float): Frecuencia de corte superior del filtro pasa banda.
                -notch_freq (float): Frecuencia de corte del filtro notch.
                -notch_width (float): Ancho de banda del filtro notch.
                -sample_rate (float): Frecuencia de muestreo de la señal.
                -axisToCompute (int): Eje a lo largo del cual se calculará la transformada.
            -featureExtractorMethod (str): Método de extracción de características. Puede ser welch o hilbert.
            -training_events_file (str): Ruta al archivo txt con los eventos registrados durante las sesiones
            -classifierFile (str): Ruta al archivo pickle con el clasificador. IMPORTANTE: Se supone que este archivo ya fue generado con la sesión
            de entrenamiento y será usado durante las sesiones de feedback y online.
            -arduinoFlag: Flag para indicar si se usará arduino para controlar el dispositivo de control. Puede ser SI o NO
            -arduinoPort: Puerto serial del arduino. Puede ser del tipo /dev/ttyUSBx o COMx (para windows).
        Un trial es la suma de startingTimes + cueDuration + finishDuration

        - indicatorAPP (QWidget): Objeto de la clase Entrenamiento. Se usa para enviar señales a la GUI.
        - supervisionAPP (QWidget): Objeto de la clase Supervision. Se usa para supervisar eventos, señal de EEG entre otros.
        
        NOTA: Definir qué parámetros se necesitan inicar dentro del constructor."""

        super().__init__() #Inicializamos la clase padre

        self.configAPP = configAPP
        self.indicatorAPP = indicatorAPP
        self.__supervisionAPPClass = supervisionAPP

        #Parámetros generales para la sesións
        self.configParameters = configParameters

        self.typeSesion = configParameters["typeSesion"]
        self.cueType = configParameters["cueType"]
        self.ntrials = configParameters["ntrials"]
        self.classes = configParameters["classes"]
        self.clasesNames = configParameters["clasesNames"]
        self.startingTimes = configParameters["startingTimes"]
        self.cueDuration = configParameters["cueDuration"]
        self.finishDuration = configParameters["finishDuration"]
        self.lenToClassify = configParameters["lenToClassify"]
        self.onlineCommandSendingTime = configParameters["onlineCommandSendingTime"]
        self.numberOfRounds = configParameters["numberOfRounds"] #rondas para estimar la probabilidad de cada clase
        self.numberOfRounds_Accum = 0
        self.lenForClassifier = configParameters["lenForClassifier"]
        self.umbralClassifier = configParameters["umbralClassifier"] #umbral de probabilidad para enviar un comando
        self.subjectName = configParameters["subjectName"]
        self.sesionNumber = configParameters["sesionNumber"]

        #Parámetros para inicar la placa openbci
        self.boardParams = configParameters["boardParams"]
        self.channels = self.boardParams["channels"]
        self.serialPort = self.boardParams["serialPort"]
        self.boardName = self.boardParams["boardName"]

        #parámetros del filtro
        self.filterParameters = configParameters["filterParameters"]
        #Chequeamos el tipo de placa para corregir el sample rate
        if self.boardParams["boardName"] == "cyton":
            self.filterParameters["sample_rate"] = 250.
        elif self.boardParams["boardName"] == "ganglion":
            self.filterParameters["sample_rate"] = 200.
        elif self.boardParams["boardName"] == "synthetic":
            self.filterParameters["sample_rate"] = 250.

        self.__preparacion_opacity_value = 0.0 #valor de opacidad para el texto de preparación
        self.__cue_opacity_value = 0.0 #valor de opacidad para el texto de tarea
        self.__start_opactity_value = 1.0 #valor de opacidad para el texto de preparación
        self.__final_opactity_value = 1.0 #valor de opacidad para el texto de finalización

        self.trialDuration = 0 #timestamp para guardar el tiempo de cada trial
        self.sample_rate = self.filterParameters["sample_rate"]

        ## Archivo de eventos de una sesión de entrenamiento
        self.training_events_file = configParameters["events_file"]

        ## Archivo para cargar el clasificador
        self.classifierFile = configParameters["classifierFile"]

        #archivo para cargar el pipeline
        self.pipelineFile = configParameters["pipelineFile"]

        self.arduinoFlag = 1 if configParameters["arduinoFlag"] == "SI" else 0
        self.arduinoPort = configParameters["arduinoPort"]
        self.arduino = None #Al iniciar no hay arduino conectado
        self.comando = b'0'

        self.__trialPhase = 0 #0: Inicio, 1: Cue, 2: Finalización
        self.__trialNumber = 0 #Número de trial actual
        self.__startingTime = self.startingTimes[0]
        self.__deltat = 50 #ms
        self.delta_opacity = 1/((self.__startingTime*1000)/self.__deltat)
        self.__trial_init_time = 0
        self.__cue_init_time = 0
        self.rootFolder = "data/"

        self.session_started = False #Flag para indicar si se inició la sesión

        self.prediction = -1 #iniciamos el valor de predicción
        self.probas = [0 for i in range(len(self.classes))]

        #Configuramos timers del Core
        """
        Funcionamiento QTimer
        https://stackoverflow.com/questions/42279360/does-a-qtimer-object-run-in-a-separate-thread-what-is-its-mechanism
        """

        #timer para controlar el inicio de la sesión
        self.iniSesionTimer = QTimer()
        self.iniSesionTimer.setInterval(1000) #1 segundo1
        self.iniSesionTimer.timeout.connect(self.startSesion)

        #timer para controlar si se alcanzó el último trial y así cerrar la app
        self.checkTrialsTimer = QTimer()
        self.checkTrialsTimer.setInterval(10) #Chequeamos cada 10ms
        self.checkTrialsTimer.timeout.connect(self.checkLastTrial)

        #timer para controlar las fases de cada trial
        self.trainingEEGThreadTimer = QTimer() #Timer para control de tiempo de las fases de trials
        self.trainingEEGThreadTimer.setInterval(int(self.startingTimes[1]*1000))
        self.trainingEEGThreadTimer.timeout.connect(self.trainingEEGThread)

        self.feedbackThreadTimer = QTimer() #Timer para control de tiempo de las fases de trials
        self.feedbackThreadTimer.setInterval(int(self.startingTimes[1]*1000))
        self.feedbackThreadTimer.timeout.connect(self.feedbackThread)

        self.onlineThreadTimer = QTimer() #Timer para control de tiempo de las fases de trials
        self.onlineThreadTimer.setInterval(int(self.onlineCommandSendingTime*1000))
        self.onlineThreadTimer.timeout.connect(self.onlineThread)

        #timer para controlar el tiempo para clasificar el EEG
        self.classifyEEGTimer = QTimer()
        self.classifyEEGTimer.setInterval(int(self.lenToClassify*1000)) #Tiempo en milisegundos
        self.classifyEEGTimerStarted = False
        self.classifyEEGTimer.timeout.connect(self.classifyEEG)

        #timer para controlar la app de configuración
        self.configAppTimer = QTimer()
        self.configAppTimer.setInterval(5) #ms
        self.configAppTimer.timeout.connect(self.checkConfigApp)

        #timer para actualizar la supervisionAPP
        self.__supervisionAPPTime = 10 #ms
        self.supervisionAPPTimer = QTimer()
        self.supervisionAPPTimer.setInterval(self.__supervisionAPPTime)
        self.supervisionAPPTimer.timeout.connect(self.updateSupervisionAPP)

        self.showGUIAPPs()

    def updateParameters(self,newParameters):
        """Actualizamos cada valor dentro del diccionario
        configParameters a partir de newParameters"""
        self.typeSesion = newParameters["typeSesion"]
        self.cueType = newParameters["cueType"]
        self.ntrials = newParameters["ntrials"]
        self.classes = newParameters["classes"]
        self.clasesNames = newParameters["clasesNames"]
        self.startingTimes = newParameters["startingTimes"]
        self.cueDuration = newParameters["cueDuration"]
        self.finishDuration = newParameters["finishDuration"]
        self.lenToClassify = newParameters["lenToClassify"]
        self.onlineCommandSendingTime = newParameters["onlineCommandSendingTime"]
        self.numberOfRounds = newParameters["numberOfRounds"]
        self.lenForClassifier = newParameters["lenForClassifier"]
        self.umbralClassifier = newParameters["umbralClassifier"]
        self.subjectName = newParameters["subjectName"]
        self.sesionNumber = newParameters["sesionNumber"]
        self.boardParams = newParameters["boardParams"]
        self.channels = self.boardParams["channels"]
        self.serialPort = self.boardParams["serialPort"]
        self.boardName = self.boardParams["boardName"]

        self.channels = self.boardParams["channels"]

        self.filterParameters = newParameters["filterParameters"]
        #Chequeamos el tipo de placa para corregir el sample rate
        if self.boardParams["boardName"] == "cyton":
            self.filterParameters["sample_rate"] = 250.
        elif self.boardParams["boardName"] == "ganglion":
            self.filterParameters["sample_rate"] = 200.
        elif self.boardParams["boardName"] == "synthetic":
            self.filterParameters["sample_rate"] = 250.

        self.sample_rate = self.filterParameters["sample_rate"]
        
        self.training_events_file = newParameters["events_file"]
        self.classifierFile = newParameters["classifierFile"]

        #archivo para cargar el pipeline
        self.__customPipeline = newParameters["customPipeline"]
        self.pipelineFile = newParameters["pipelineFile"]

        self.__startingTime = self.startingTimes[1]

        self.classifyEEGTimer.setInterval(int(self.lenToClassify*1000)) #Tiempo en milisegundos
        self.onlineThreadTimer.setInterval(int(self.onlineCommandSendingTime*1000))

        self.probas = [0 for i in range(len(self.classes))]

        self.arduinoFlag = 1 if newParameters["arduinoFlag"] == "SI" else 0
        self.arduinoPort = newParameters["arduinoPort"]

        #actualizamos el diccionario
        self.configParameters = newParameters

    def saveConfigParameters(self, fileName = None):
        """Guardamos el diccionario configParameters en un archivo json"""

        if not fileName:
            with open(self.eegStoredFolder+self.eegFileName+"config.json", 'w') as f:
                json.dump(self.configParameters, f, indent = 4)
        else:
            with open(fileName, 'w') as f:
                json.dump(self.configParameters, f, indent = 4)

    def setFolders(self, rootFolder = "data/"):
        """Función para chequear que existan las carpetas donde se guardarán los datos de la sesión.
        En caso de que no existan, se crean.
        
        La carpeta base es la rootFolder. Dentro de esta carpeta se crean las carpetas para cada sujeto.
        
        Se usa el nombre del sujeto para crear una subcarpeta. Dentro de esta se crean las carpetas para cada sesión."""

        #si la carpeta rootFolder no existe, se crea
        if not os.path.exists(rootFolder):
            os.makedirs(rootFolder)

        #Si la carpeta rootFolder/self.subjectName no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName):
            os.makedirs(rootFolder + self.subjectName)

        #Carpeta para almacenar la señal de EEG
        #Si la carpeta rootFolder/self.subjectName/eegdata/self.sesionNumber no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/eegdata" + f"/sesion{str(self.sesionNumber)}"):
            os.makedirs(rootFolder + self.subjectName + "/eegdata" + f"/sesion{str(self.sesionNumber)}")
        self.eegStoredFolder = self.rootFolder + self.subjectName + "/eegdata/" + f"/sesion{str(self.sesionNumber)}/"

        #Chequeamos si el eegFileName existe. Si existe, se le agrega un número al final para no repetir
        #el nombre del archivo por defecto es self.eegFileName =  "s{self.sesionNumber}_t{self.cueType}_r1.npy"
        #Donde, s = sesión_number, ts = type_sesion, ct = cue_type, r = run_number
        self.eegFileName =  f"sn{self.sesionNumber}_ts{self.typeSesion}_ct{self.cueType}_r1.npy"
        if os.path.exists(self.eegStoredFolder + self.eegFileName):
            i = 2
            while os.path.exists(self.eegStoredFolder + self.eegFileName):
                self.eegFileName =  f"sn{self.sesionNumber}_ts{self.typeSesion}_ct{self.cueType}_r{i}.npy"
                i += 1
        
        #Cramos un archivo txt que contiene la siguiente cabecera:
        #"trialNumber,classNumber,className,startingTime,cueInitTime,trialDuration,cueDuration,finishDuration,startingTime(legible)\n"
        #Primero creamos el archivo y agregamos la cabecera. Lo guardamos en rootFolder/self.subjectName/eegdata/self.sesionNumber
        #con el mismo nombre que self.eegFileName pero con extensión .txt
        self.eventsFileName = self.eegStoredFolder + self.eegFileName[:-4] + "_events" + ".txt"
        eventsFile = open(self.eventsFileName, "w")
        eventsFile.write("trialNumber,classNumber,className,startingTime,cueInitTime,trialDuration,cueDuration,finishDuration,startingTime(legible)\n")
        eventsFile.close()

    def saveEvents(self):
        """Función para almacenar los eventos de la sesión en el archivo txt self.eventsFileName
        La función almacena los siguientes eventos, self.trialNumber, self.classNumber, self.class,
        self.trialPhase, self.trialTime, self.trialStartTime, self.trialEndTime.
        Cada nuevo dato se agrega en una nueva linea. Se abre el archivo en modo append (a)"""
        
        eventsFile = open(self.eventsFileName, "a")
        
        claseActual = self.trialsSesion[self.__trialNumber]
        classNameActual = self.clasesNames[self.classes.index(claseActual)]

        #obtenemos el timestamp actual
        self.trialDuration = time.time()
        #formateamos el timestamp actual a formato legible del tipo DD/MM/YYYY HH:MM:SS
        startingTimeLegible = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(self.__trial_init_time))
        
        ##Cabecera de archivo de eventos
        ###"trialNumber,classNumber,className,startingTime,cueInitTime,trialDuration,cueDuration,finishDuration,startingTime(legible)\n"

        if self.typeSesion == 0:
            eventos = f"{self.__trialNumber+1},{claseActual},{classNameActual},{self.__trial_init_time},{self.__cue_init_time},{self.trialDuration},{self.cueDuration},{self.finishDuration},{startingTimeLegible}\n"

        eventsFile.write(eventos)
        eventsFile.close()

    def setEEGLogger(self, startStreaming = True):
        """Seteamos EEGLogger para lectura de EEG desde placa.
        Iniciamos streaming de EEG."""
        
        print("Seteando EEGLogger...")
        board, board_id = setupBoard(boardName = self.boardName, serial_port = self.serialPort)
        self.eeglogger = EEGLogger(board, board_id)
        self.eeglogger.connectBoard()
        self.eeglogger.setStreamingChannels(self.channels)
        time.sleep(1) #esperamos 1 segundo para que se conecte la placa
        print("Iniciando streaming de EEG...")

        # channels_names = self.eeglogger.board.get_eeg_channels(board_id)

        if startStreaming:
            self.eeglogger.startStreaming()#iniciamos streaming de EEG
            print("Esperamos para estabilizar señal de EEG...")
            time.sleep(3) #Esperamos unos segundos para estabilizar la señal de EEG
            
            #iniciamos timer para actualizar grafico de EEG de la supervisionAPP
            self.supervisionAPPTimer.start()

    def setFilter(self):
        """Función para setear el filtro de EEG que usaremos en la supervisiónAPP
        - Los parámetros del filtro se obtienen a partir de self.parameters['filterParameters']"""

        lowcut = self.filterParameters['lowcut']
        highcut = self.filterParameters['highcut']
        notch_freq = self.filterParameters['notch_freq']
        notch_width = self.filterParameters['notch_width']
        sample_rate = self.filterParameters['sample_rate']
        axisToCompute = self.filterParameters['axisToCompute']

        self.filter = Filter(lowcut=lowcut, highcut=highcut, notch_freq=notch_freq, notch_width=notch_width,
                             sample_rate=sample_rate, axisToCompute = axisToCompute,
                             padlen = int(self.__supervisionAPPTime * sample_rate/1000)-1)

    def setPipeline(self, **pipelineBlocks):
        """Función para setear el pipeline para el procesamiento y clasificación de EEG.
        Parametros:
        - filename (str): nombre del archivo (pickle) donde se encuentra el pipeline guardado. Si es None
        se setea el pipeline con los parámetros dados en pipelineObject.
        - pipelineBlocks (dict): diccionario con los diferentes objetos para el pipeline.
        """
        
        #Si pipelineBlocks esta vacío, se carga el pipeline desde el archivo self.pipelineFileName
        if not pipelineBlocks: #cargamos pipeline desde archivo. ESTO ES LO RECOMENDABLE
            self.pipeline = pickle.load(open(self.pipelineFile, "rb")) #cargamos el pipeline

        #Si pipelineBlocks no esta vacío, se setea el pipeline con los parámetros dados en pipelineObject
        else:
            self.pipeline = Pipeline([(step, pipelineBlocks[step]) for step in pipelineBlocks.keys()])

    def makeAndMixTrials(self):
        """Clase para generar los trials de la sesión. La cantidad total de trials
        está dada por [ntrials * len(self.classes)].
        Se genera un numpy array de valores correspondiente a cada clase y se mezclan.
        
        Retorna:
            -trialsSesion (list): numpyarray con los trials de la sesión.
            
        VERSIONES FUTURAS:
        Implementar un generador de trials para que no sea necesario generar todos los trials de una vez.  

        Ejemplo:

        # def makeAndMixTrials(self):
                trials = np.array([[i] * self.ntrials for i in self.classes]).ravel()
        #       np.random.shuffle(trials)
        #       for trial in trials:
        #           yield trial

        # newClass = getNewClass(ntrials, classes) 
        """

        self.trialsSesion = np.array([[i] * self.ntrials for i in self.classes]).ravel()
        random.shuffle(self.trialsSesion)

    def checkLastTrial(self):
        """Función para chequear si se alcanzó el último trial de la sesión.
        Se compara el número de trial actual con el número de trials totales dado en self.trialsSesion"""
        if self.__trialNumber == len(self.trialsSesion):
            print("Último trial alcanzado")
            print("Sesión finalizada")
            self.checkTrialsTimer.stop()
            self.trainingEEGThreadTimer.stop()
            self.feedbackThreadTimer.stop()
            self.eeglogger.stopBoard()
            self.supervisionAPPTimer.stop()
            self.closeApp()
        else:
            pass

    def show_fase_preparacion(self):
        """Función para mostrar el mensaje de preparación de manera paulatina"""
        self.__preparacion_opacity_value += 0.05 #0.1/self.__deltat = 1000ms = 1.0s
        if self.__preparacion_opacity_value <= 1.0:
            mensaje = "Preparate..."
            background = f"rgba(255,255,255,{self.__preparacion_opacity_value*100}%)"
            font_color = f"rgba(38,38,38,{self.__preparacion_opacity_value*100}%)"
            self.indicatorAPP.update_order(mensaje, fontsize = 46,
                                            background = background, font_color = font_color)
            self.trainingEEGThreadTimer.setInterval(self.__deltat)
        else:
            self.__trialPhase =  1 # pasamos a la siguiente fase -> Fase de tarea o cue
            self.__preparacion_opacity_value = 0.0
            self.trainingEEGThreadTimer.setInterval(1)

    def fase_preparacion(self):
        self.__trial_init_time = time.time()
        print(f"Trial {self.__trialNumber + 1} de {len(self.trialsSesion)}")
        self.indicatorAPP.showCueOnSquare(False)
        self.indicatorAPP.showCueOffSquare(True)
        mensaje = "Preparate..."
        self.indicatorAPP.update_order(mensaje)
        self.__trialPhase = 2 # pasamos a la siguiente fase -> Mostrar cue progresivamente
        self.trainingEEGThreadTimer.setInterval(1) #esperamos el tiempo aleatorio
    
    def hide_preparación(self):
        self.__start_opactity_value -= self.delta_opacity
        mensaje = "Preparate..."
        if self.__start_opactity_value >= 0.0:
            background = f"rgba(255,255,255,{self.__start_opactity_value*100}%)"
            font_color = f"rgba(38,38,38,{self.__start_opactity_value*100}%)"
            self.indicatorAPP.update_order(mensaje, fontsize = 46,
                                            background = background, font_color = font_color)
            self.trainingEEGThreadTimer.setInterval(self.__deltat) #esperamos 50ms
        else:
            background = f"rgba(255,255,255,0%)"
            font_color = f"rgba(38,38,38,0%)"
            self.indicatorAPP.update_order(mensaje, fontsize = 46,
                                            background = background, font_color = font_color)
            self.__trialPhase =  3 ##guardamos los datos de EEG
            self.__start_opactity_value = 1.0
            hidding_final_time = round(random.uniform(self.startingTimes[0], self.startingTimes[1]), 3)
            self.trainingEEGThreadTimer.setInterval(int(hidding_final_time*1000))
            # self.trainingEEGThreadTimer.setInterval(2000)

    def show_cue(self):
        claseActual = self.trialsSesion[self.__trialNumber]
        classNameActual = self.clasesNames[self.classes.index(claseActual)]
        self.__cue_opacity_value += 0.1 #0.1/self.__deltat = 500ms = 0.5s
        if self.__cue_opacity_value <= 1.0:
            background = f"rgba(38,38,38,{self.__cue_opacity_value*100}%)"
            font_color = f"rgba(255,255,255,{self.__cue_opacity_value*100}%)"
            self.indicatorAPP.update_order(f"{classNameActual}", fontsize = 46,
                                            background = background, font_color = font_color)
        else:
            self.__trialPhase =  4 # pasamos a la siguiente fase -> Fase de tarea o cue
            self.__cue_opacity_value = 1
        self.trainingEEGThreadTimer.setInterval(self.__deltat)

    def fase_cue(self):
        self.__cue_init_time = time.time()
        self.indicatorAPP.showCueOnSquare(True)
        self.indicatorAPP.showCueOffSquare(False)
        claseActual = self.trialsSesion[self.__trialNumber]
        classNameActual = self.clasesNames[self.classes.index(claseActual)]
        self.indicatorAPP.update_order(f"{classNameActual}", fontsize = 46,
                                            background = "rgba(38,38,38,100%)", font_color = "white")
        self.__trialPhase = 5 # Empezamos a apagar el estímulo
        self.trainingEEGThreadTimer.setInterval(int(self.cueDuration * 1000))

    def hide_cue(self):
        claseActual = self.trialsSesion[self.__trialNumber]
        classNameActual = self.clasesNames[self.classes.index(claseActual)]
        self.__cue_opacity_value -= 0.05 ##0.05/self.__deltat = 1000ms
        if self.__cue_opacity_value >= 0.0:
            background = f"rgba(38,38,38,{self.__cue_opacity_value*100}%)"
            font_color = f"rgba(255,255,255,{self.__cue_opacity_value*100}%)"
            self.indicatorAPP.update_order(f"{classNameActual}", fontsize = 46,
                                            background = background, font_color = font_color)
            self.trainingEEGThreadTimer.setInterval(self.__deltat)
        else:
            background = f"rgba(38,38,38,0%)"
            font_color = f"rgba(255,255,255,0%)"
            self.indicatorAPP.update_order(f"{classNameActual}", fontsize = 46,
                                            background = background, font_color = font_color)
            self.__trialPhase =  6 ##guardamos los datos de EEG
            self.__cue_opacity_value = 0.0
            self.trainingEEGThreadTimer.setInterval(3000) #esperamos 3 segundos
        
    def fase_end(self):
        mensaje = "Podes descansar"
        background = "rgba(255,255,255,100%)"
        font_color = "rgba(38,38,38,100%)"
        self.indicatorAPP.update_order(mensaje, fontsize = 46,
                                       background = background, font_color = font_color)
        self.__trialPhase = 7 #Fase para guardar datos de EEG
        self.trainingEEGThreadTimer.setInterval(int(self.finishDuration * 1000))

    def hide_fase_end(self):
        self.__final_opactity_value -= 0.05
        mensaje = "Podes descansar"
        if self.__final_opactity_value >= 0.0:
            background = f"rgba(255,255,255,{self.__final_opactity_value*100}%)"
            font_color = f"rgba(38,38,38,{self.__final_opactity_value*100}%)"
            
            self.indicatorAPP.update_order(mensaje, fontsize = 46,
                                            background = background, font_color = font_color)
            self.trainingEEGThreadTimer.setInterval(self.__deltat) #esperamos 50ms
        else:
            background = f"rgba(255,255,255,0%)"
            font_color = f"rgba(38,38,38,0%)"
            self.indicatorAPP.update_order(mensaje, fontsize = 46,
                                            background = background, font_color = font_color)
            self.__trialPhase =  8 ##guardamos los datos de EEG
            self.__final_opactity_value = 1.0
            self.trainingEEGThreadTimer.setInterval(2000)

    def save_data(self):
        newData = np.array([1]) ##fake data
        self.eeglogger.saveData(newData, fileName = self.eegFileName, path = self.eegStoredFolder, append=True)
        # self.trialDuration = time.time() #timestamp para guardar el tiempo de cada trial
        self.saveEvents() #guardamos los eventos de la sesión
        self.__trialPhase = 0 #volvemos a la fase inicial del trial
        self.supervisionAPP.reset_timeBar = True
        self.__trialNumber += 1 #incrementamos el número de trial
        ##sacamos el texto de la pantalla
        background = f"rgba(38,38,38,0%)"
        font_color = f"rgba(255,255,255,0%)"
        self.indicatorAPP.update_order(f"", fontsize = 46,
                                        background = background, font_color = font_color)
        self.trainingEEGThreadTimer.setInterval(10)

    def trainingEEGThread(self):
        """Función para hilo de lectura de EEG durante fase de entrenamiento.
        Sólo se almacena trozos de EEG correspondientes a la suma de startTrainingTime y cueDuration.
        """

        if self.__trialPhase == 0:
            self.show_fase_preparacion()

        if self.__trialPhase == 1:
            self.fase_preparacion()
            ##pasamos a la fase 2 para esconder el texto de preparación

        if self.__trialPhase == 2:
            self.hide_preparación()
            ##pasamos a la fase 3

        elif self.__trialPhase == 3: #empezamos a mostrar estímulos
            self.show_cue()
            ##pasamos a la fase 4

        elif self.__trialPhase == 4: ##Fase de tarea o cue
            self.fase_cue()
            ##pasamos a la fase 5

        elif self.__trialPhase == 5: ##apagamos el estímulo
            self.hide_cue()
            ##pasamos a la fase 6

        elif self.__trialPhase == 6: ##fase de finalización
            self.fase_end()
            ##pasamos a la fase 7

        elif self.__trialPhase == 7: ##fase de finalización
            self.hide_fase_end()
            ##pasamos a la fase 8

        elif self.__trialPhase == 8:
            self.save_data()
            ##pasamos a la fase 0

    def feedbackThread(self):
        """Función para hilo de lectura de EEG durante fase de entrenamiento.
        Sólo se almacena trozos de EEG correspondientes a la suma de startTrainingTime y cueDuration.
        """
        pass

    def onlineThread(self):
        """Función para hilo de lectura de EEG durante fase de entrenamiento.
        Para esta versión se propone que durante la sesión online se entre al onlineThread()
        en el tiempo seteado en self.lenToClassify, es decir, en el tiempo que se desea obtener un valor
        de clasificación.
        """
        ##TODO
        pass
            
    def showGUIAPPs(self):
        """Función para configurar la sesión de entrenamiento usando self.confiAPP.
        """
        self.indicatorAPP.show() #mostramos la APP
        self.indicatorAPP.update_order("Configurando la sesión...")
        self.configAPP.show() #mostramos la APP
        self.configAppTimer.start()

    def checkConfigApp(self):
        """Función para comprobar si la configuración de la sesión ha finalizado."""
        if not self.configAPP.is_open:
            print("CONFIG APP CERRADA")
            newParameters = self.configAPP.getParameters()
            self.updateParameters(newParameters)
            self.configAppTimer.stop()
            self.start() #iniciamos la sesión

    def updateSupervisionAPP(self):
        """Función para actualizar la APP de supervisión."""
        ##Actualizamos gráficas de EEG y FFTgit
        pass

    def classifyEEG(self):
        """Función para clasificar EEG
        La función se llama cada vez que se activa el timer self.classifyEEGTimer. La duración
        del timer esta dada por self.classifyEEGTimer.setInterval().

        Se obtiene un nuevo trozo de EEG de longitud self.lenToClassify segundos, se añade al
        buffer de datos a clasificar y se clasifica. El resultado de la clasificación se almacena
        en self.prediction.

        Por cada entrada a la función, se elimina el primer trozo de datos del buffer de datos a
        clasificar y se añade el nuevo trozo de datos. La idea es actualizar los datos mientras la persona ejecuta
        la tarea.
        """
        #not used
        pass

    def start(self):
        """Método para iniciar la sesión"""
        print(f"Preparando sesión {self.sesionNumber} del sujeto {self.subjectName}")

        self.supervisionAPP = self.__supervisionAPPClass([str(clase) for clase in self.classes], self.channels)
        # self.supervisionAPP.show() #mostramos la APP de supervisión

        self.indicatorAPP.showSessionOnSquare(True)
        self.indicatorAPP.showSessionOffSquare(False)
        
        if self.typeSesion == 0:
            self.indicatorAPP.update_order("Iniciando sesión de entrenamiento") #actualizamos app
        
        if self.typeSesion == 1:
            self.indicatorAPP.update_order("Iniciando sesión de feedback") #actualizamos app

        self.iniSesionTimer.start()

    def sanityChecks(self):
        """Método chequear diferentes parámetros antes de iniciar la sesión para no crashear
        durante la sesión o de Entrenamiento, de Calibración u Online cuando estas están en marcha."""

        print("Iniciando Sanity Check...")

        training_events = pd.read_csv(self.training_events_file, sep = ",")
        # trained_classesNames = training_events["className"].unique()
        trained_classesValues = np.sort(training_events["classNumber"].unique())
        train_samples = int(training_events["cueDuration"].unique()*self.sample_rate)

        n_channels = len(self.channels)
        classify_samples = int(self.sample_rate * self.lenForClassifier)

        ## Chequeos
        ## Chequeamos que self.classes y self.clasesNames tengan la misma cantidad de elementos
        if len(self.classes) != len(self.clasesNames):
            self.closeApp()
            raise Exception("La cantidad de clases y la cantidad de nombres de clases deben ser iguales")
        
        ## chequeamos que no se repitan nombres en self.clasesNames
        if len(self.clasesNames) != len(set(self.clasesNames)):
            self.closeApp()
            raise Exception("Hay nombres de clases repetidos")
        
        ## chequeamos que nos e repitan valores en self.classes
        if len(self.classes) != len(set(self.classes)):
            self.closeApp()
            raise Exception("Hay valores de clases repetidos")
        
        ## Chequeamos que la duración del trial sea igual al utilizado para entrenar el clasificador
        if train_samples != classify_samples:
            self.closeApp()
            raise Exception("La duración del trial a clasificar debe ser igual al utilizado para entrenar el clasificador")

        ## Chequeamos que los trained_classesValues estén presentes dentro de self.classes
        if not np.any(np.isin(trained_classesValues, self.classes)):
            ## me quedo con los valores que no están en self.classes
            values_not_in_classes = trained_classesValues[~np.isin(trained_classesValues, self.classes)]
            self.closeApp()
            raise Exception("Hay una o más clases a utilizar que no se usaron durante en la sesión de entrenamiento", values_not_in_classes)
 
        ## generamos un numpy array con valores enteros igual a 1. El shape es de la forma [1, n_channels, classify_samples]
        ## Este array representa un trial de EEG
        trial = np.ones((1, n_channels, classify_samples), dtype=np.int8)

        ##chequeamos si self.pipeline posee el método predict_proba
        if not hasattr(self.pipeline, "predict_proba"):
            self.closeApp()
            raise Exception("El pipeline no posee el método predict_proba")
        else: #si lo posee, chequeamos que la cantidad de probabilidades retornada sea igual a la cantidad de clases
            probas = self.pipeline.predict_proba(trial)
            if len(probas[0]) != len(self.classes):
                self.closeApp()
                mensaje = "La cantidad de probabilidades retornada por el pipeline es diferente a la cantidad de clases que se intenta clasificar. \nLa cantidad y el tipo de clases a clasificar debe corresponderse con la usada durante el entrenamiento del clasificador"
                raise Exception(mensaje)
        
        if self.arduinoFlag:
            try:# Intentamos conectaros a Arduino
                ## inicializamos la clase ArduinoCommunication
                self.arduino = ArduinoCommunication(port = self.arduinoPort)
                time.sleep(0.5) #esperamos 0.5seg
                print("Iniciamos sesión con Arduino")
                self.arduino.iniSesion()
            except Exception as e:
                print(e)
                print("No se pudo establecer comunicación con arduino")
                self.arduinoFlag = 0
            
        print("Sanity Check finalizado. Todo OK")

    def startSesion(self):
        """Método para iniciar timers del Core, además
        se configuran las carpetas de almacenamiento y se guarda el archivo de configuración de la sesión.
        Se setea el setEEGLogger para comunicación con la placa.
        """
        self.iniSesionTimer.stop() #detenemos timer de inicio de sesión

        ##si el tipo de sesión no es la 2, generamos setFolders
        if self.typeSesion != 2:
            self.setFolders(rootFolder = self.rootFolder) #configuramos las carpetas de almacenamiento
            self.saveConfigParameters(self.eegStoredFolder+self.eegFileName[:-4]+"_config.json") #guardamos los parámetros de configuración
        
        self.setEEGLogger() #seteamos EEGLogger
        self.makeAndMixTrials() #generamos y mezclamos los trials de la sesión
        self.checkTrialsTimer.start()

        if self.typeSesion == 0: #sesión de Entrenamiento
            print("Inicio de sesión de entrenamiento")
            self.trainingEEGThreadTimer.start() #iniciamos timer para controlar hilo entrenamiento
            self.session_started = True
            
        elif self.typeSesion == 1: #sesión de Calibración
            print("Inicio de sesión de Feedback")
            self.setPipeline() #seteamos el pipeline
            self.sanityChecks() ## sanity check
            self.feedbackThreadTimer.start() #iniciamos timer para controlar hilo calibración
            self.session_started = True

        elif self.typeSesion == 2: #sesión Online
            print("Inicio de sesión Online")
            ##Cerramos indicatorAPP ya que no se usa en modo Online
            self.indicatorAPP.close()            
            self.setPipeline() #seteamos el pipeline5
            self.sanityChecks() ## sanity check
            self.onlineThreadTimer.start() #iniciamos timer para controlar hilo calibración
            self.session_started = True

    def closeApp(self):
        print("Cerrando aplicación...")
        if self.arduino is not None:
            self.arduino.endSesion()
        self.indicatorAPP.close()
        self.supervisionAPP.close()
        self.close()

if __name__ == "__main__":

    debbuging = False
    if debbuging:
        logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    ##cargamos los parametros desde el archivo config.json
    with open("config.json", "r") as f:
        parameters = json.load(f)

    core = Core(parameters, ConfigAPP("config.json", InfoAPP), IndicatorAPP(), SupervisionAPP)

    sys.exit(app.exec_())

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import logging, argparse
import time
import os


class EEGLogger():
    """Clase para adquirir/registrar señales de EEG a partir de las placas Cyton, Ganglion o Synthetic de OpenBCI"""

    def __init__(self, board, board_id) -> None:
        """Constructor de la clase
        - board: objeto de la clase BoardShim
        - board_id: id de la placa. Puede ser BoardIds.CYTON_BOARD.value o BoardIds.GANGLION_BOARD.value o BoardIds.SYNTHETIC_BOARD.value
        - channelsSelected: Lista con los ids de los canales seleccionados. Por defecto son los canales 1, 2 y 3.
        """
        self.board = board
        self.board_id = board_id
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id) #frecuencia de muestreo
        self.acel_channels = BoardShim.get_accel_channels(board_id) #canales de acelerometro
        self.rawData = np.zeros((len(self.eeg_channels), 0)) #datos crudos    
        
    def connectBoard(self):
        """Nos conectamos a la placa y empezamos a transmitir datos"""
        self.board.prepare_session()
        print("***********")
        print(f"Channles: {self.eeg_channels}")
        print("***********")

    def startStreaming(self):
        logging.info("Starting streaming")
        self.board.start_stream()

    def stopBoard(self):
        """Paramos la transmisión de datos de la placa"""
        self.board.stop_stream()
        self.board.release_session()
    
    def getData(self, sampleLength = 4, removeDataFromBuffer = True):
        """Obtenemos algunas muestras de la placa. La cantidad de muestras que devuelve el método depende del timeLength y de la frecuencia
        de muestro de la placa. 
        Los datos se entregan en un numpy array de forma [canales, muestras]. Los datos están en microvolts.
        - sampleLength: duración (en segundos) de la señal a adquirir de la placa. Por defecto son 6 segundos."""

        num_samples = int(self.board.get_sampling_rate(self.board_id) * sampleLength)

        if removeDataFromBuffer:
            return self.board.get_board_data(num_samples) #devuelve los datos y los borra del buffer
        else:
            return self.board.get_current_board_data(num_samples) #devuelve los datos sin borrarlos del buffer
    
    def addData(self, newdata):
        """Agregamos datos a la variable rawData. 
        - newdata: numpy array de forma [canales, muestras]"""
        self.rawData = np.concatenate((self.rawData, newdata), axis = 1)
    
    def saveData(self, eegdata, fileName = "subject1.npy", path = "recordedEEG/", append = True):
        """Guardamos los datos crudos en un archivo .npy
        - fileName: nombre del archivo
        - path: carpeta donde se guardará el archivo
        - append: si es True, los datos se agregan al archivo. Si es False, se sobreescribe el archivo."""

        #Usamos try/except para enviar un mensaje de error pero no cerrar el programa
        try:
            if append:
                #chequeamos si el archivo existe
                if os.path.isfile(path + fileName):
                    storedData = np.load(path + fileName, allow_pickle = True)
                    # storedData = np.concatenate((storedData, eegdata), axis = 1)
                    with open(path + fileName, "wb") as f:
                        np.save(f, storedData)
                else:
                    with open(path + fileName, "wb") as f:
                        np.save(f, eegdata)
            else:
                with open(path + fileName, "wb") as f:
                    np.save(f, eegdata)

        except Exception as e:
            print("Error al guardar los datos")
            print(e)

    def setStreamingChannels(self, channelsSelected = [1,2,3]):
        """
        Función para desactivar los canales que no se van a utilizar.
        - channelsSelected: lista con los ids de los canales a utilizar. Por defecto son los canales 1, 2 y 3.
        
        Doc:
        - https://docs.openbci.com/Cyton/CytonSDK/#channel-setting-commands
        - https://docs.openbci.com/Ganglion/GanglionSDK/
        """
        #chqueamos que la boardid sea la cyton
        if self.board_id == BoardIds.CYTON_BOARD.value:
            #obtengo los canales de la cyton
            cytonChannels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
            #Aquellos canalos que no utilizamos son desactivados
            for channel in cytonChannels:
                if channel not in channelsSelected:
                    print(self.board.config_board(f"x{channel}100000X"))
                    time.sleep(0.1)

        elif self.board_id == BoardIds.GANGLION_BOARD.value:
            pass #para implementar en caso de usar la Ganglion

def setupBoard(boardName = "synthetic", serial_port = None):
    """Función para configurar la conexión a la placa.
    - boardName (str): tipo de placa. Puede ser cyton, ganglion o synthetic.
    - serial_port (str): puerto serial de la placa. Por defecto es None. Recibe puertos del tipo 'COM5'.

    NOTA: En esta primera versión, la función sólo recibe los parámetros de la placa a utilizar y el puerto serial.
    En caso de que se necesiten parámetros adicionales para configurar la conexión a la placa, se deben pasar parámetros adicionales.
    """

    boards = {"cyton": BoardIds.CYTON_BOARD, #IMPORTANTE: frecuencia muestreo 250Hz
            "ganglion": BoardIds.GANGLION_BOARD, #IMPORTANTE: frecuencia muestro 200Hz
            "synthetic": BoardIds.SYNTHETIC_BOARD} #IMPORTANTE: frecuencia muestreo 250Hz
    
    board_id = boards[boardName]
    
    parser = argparse.ArgumentParser()
    
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default = serial_port)
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default = board_id)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    board = BoardShim(board_id, params)

    return board, board_id    

if __name__ == "__main__":

    debbuging = True
    if debbuging:
        logging.basicConfig(level=logging.DEBUG)

    boardName = "cyton"
    selectedChannels = [1,2,3]

    #IMPORTENTE: Chequear en que puerto esta conectada la OpenBCI.  
    puerto = "COM5"

    #usamos setupBoard para generar un objeto BoardShim y un id de placa
    board, board_id = setupBoard(boardName = boardName, serial_port = "COM5")

    eeglogger = EEGLogger(board, board_id) #instanciamos un objeto para adquirir señales de EEG desde la placa OpenBCI

    eeglogger.connectBoard() #nos conectamos a la placa
    eeglogger.setStreamingChannels(selectedChannels) #configuramos los canales que vamos a utilizar
    
    trialDuration = 2 #duración del trial en segundos
    
    ## extraemos los ids de los canales seleccionados
    eegChannels = eeglogger.board.get_eeg_channels(board_id)

    eeglogger.startStreaming() #iniciamos la adquisición de datos

    time.sleep(1) #esperamos un segundo para que se estabilice la señal

    print("Adquiriendo datos por primera vez...")
    print("Debemos esperar para completar el buffer")
    time.sleep(trialDuration) #esperamos a que se adquieran los datos

    newData = eeglogger.getData(trialDuration)[eegChannels]
    print("Forma del array de datos [canales, muestras]: ",newData.shape)

    print("Guardando datos...")
    # eeglogger.saveData(newData, fileName = "subject1.npy", path = "", append=True) #guardamos los datos en un archivo .npy

    print("Detenemos la adquisición de datos")
    eeglogger.stopBoard()
"""
Created on Monday 16 september 2023
@author: Lucas BALDEZZARI
Clase ArduinoCommunication
Clase para comunicación entre Arduino y PC utilizando la libreria PYSerial
        VERSIÓN: SCT-01-RevB (24/9/2021)
        Se agrega lista de movimientos en variable self.movements para enviar comandos a través del puerto serie
"""

import os
import serial
import time

class ArduinoCommunication:
    """Clase para comunicación entre Arduino y PC utilizando la libreria PYSerial.
        Constructor del objeto ArduinoCommunication
    """

    def __init__(self, port, baudrate = 19200, commands = [b'1',b'2',b'3',b'4',b'5'], timeout = 0, write_timeout = 0):
        """
        Constructor del objeto ArduinoCommunication

        Parametros
        ----------
        port: String
            Puerto serie por el cual nos conectaremos
        commands: list of bytes
            Lista de comandos que se enviarán a Arduino.
        baudrate: int
            Velocidad de comunicación serie
        sleep: int
            Tiempo en segundos de espera para establecer comunicación serie
        timeout: int
            Tiempo en segundos de espera para recibir respuesta desde Arduino
        Retorna
        -------
        Nada
            
        """
        self.baudrate = baudrate
        ## intentamos conectar con el puerto serie con Try/Except
        try:
            self.dev = serial.Serial(port, baudrate = baudrate, timeout = timeout,
                                     write_timeout = write_timeout) # Abrimos puerto serie con baudrate de 19200
            print("Conexión establecida con Arduino en el puerto", port)
        except:
            raise Exception("No se pudo establecer conexión con Arduino en el puerto", port)

        #chequeamos que la lista de comandos no esté vacía con assert
        assert commands != [], "La lista de comandos no puede estar vacía"
        self.commands = commands #lista de comandos

        self.sessionStatus = None #variable para guardar el estado de la sesión
        self._estadoRobot = None #variable para guardar el estado del robot/actuador
        
    def query(self, byte):
        """Enviamos un byte a arduino y recibimos un byte desde arduino
        
        Parametros
        ----------
        message (byte):
            Byte que se desea enviar por puerto serie.
        Cada vez que se envía un byte con write, se recibe un byte desde Arduino.
        """
        self.dev.write(byte) #enviamos byte por el puerto serie
        respuesta = self.dev.readline().decode('ascii').strip() #recibimos una respuesta desde Arduino
        return respuesta
    
    def sendMessage(self, message):
        """Función para enviar una lista de bytes con diferentes variables de estado
        hacia Arduino. Estas variables sirven para control de flujo del programa del
        Arduino.
        """
        incomingData = []
        for byte in message:
            incomingData.append(self.query(byte))

        ##chequeamos que incomingData[0] no sea '' (vacio). De ser así se asigna '' a incomingData sino se hace format(int(incomingData[0]), '04b')
        if incomingData[0] == '':
            return ''
        else:
            return format(int(incomingData[0]), '04b')
        
    def iniSesion(self):
        """Se inicia sesión."""
        
        self.sessionStatus = b"1" #sesión en marcha
        self.moveOrder = b"0" #EL robot empieza en STOP
        self.systemControl = [self.sessionStatus, self.moveOrder]
        self._estadoRobot = self.sendMessage(self.systemControl)
        print("Estado del ROBOT:", self._estadoRobot)
        
    def endSesion(self):
        """Se finaliza sesión.
            Se envía información a Arduino para finalizar sesión. Se deben tomar acciones en el Arduino
            una vez recibido el mensaje.
        """
        
        self.sessionStatus = b"0" #sesión finalizada
        self.moveOrder = b"0" #enviamos un STOP
        self.systemControl = [self.sessionStatus, self.moveOrder]
        self._estadoRobot = self.sendMessage(self.systemControl)
        self.close() #cerramos comunicación serie

    def checkConnection(self):
        """función para chequear si estamos conectados al puerto serial con arduino"""
        ##intentamos leer con self.dev.read() usando try/except
        try:
            self.dev.read()
            return True
        except:
            return False

    def getRobotStatus(self):
        """Obtenemos el estado del robot"""
        return self._estadoRobot

    def close(self):
        """Cerramos comunicción serie"""
        self.dev.close()

if __name__ == "__main__":
    ard = ArduinoCommunication('COM7')
    ard.iniSesion()
    ##enviamos un mensaje de prueba
    sessionStatus = b"1" #sesión en marcha
    moveOrder = b"3" #Comando 1
    ard.sendMessage([sessionStatus,moveOrder])
    estadoRobot = ard.getRobotStatus()
    print("Estado del ROBOT:", estadoRobot)
    #terminamos sesión
    ard.endSesion()
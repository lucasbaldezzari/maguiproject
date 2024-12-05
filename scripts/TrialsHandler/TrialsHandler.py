import numpy as np
import logging
import pandas as pd

class TrialsHandler():
    """Clase para obtener los trials a partir de raw data"""

    def __init__(self, rawEEG, eventos, tinit = 1, tmax = 4, reject = None, sample_rate = 250., trialsToRemove = None) -> None:
        """Constructor de la clase Trials
        Parametros:
            - rawEEG (numpy.array): array de numpy con la señal de EEG de la forma [channels, samples]
            - eventos: dataframe con los eventos. El dataframe posee las columnas:
                trialNumber,classNumber,className,startingTime,cueDuration,trialTime,trialTime(legible)
            - tinit, tmax: tiempo inicial del trial y tiempo final del trial. Reltivos al inicio de la tarea (cue)
            - reject (float): Valor de umbral para rechazar trials. Si el valor absoluto de alguno de los canales
            supera este valor, el trial es rechazado. Si es None, no se rechazan trials."""
        
        self.rawEEG = rawEEG
        self.eventos = eventos.set_index("trialNumber")
        self.tinit = tinit
        self.tmax = tmax
        self.reject = reject
        self.sample_rate = sample_rate
        self.labels = self.getLabels()
        self.trials = self.getTrials() #array de numpy con los trials de la forma [trials, channels, samples]
        self.classesName = self.getClassesName() #tupla con los nombres de las clases y su número de clase
        #chequeamos si hay trials que remover
        if trialsToRemove is not None:
            self.removeTrials(trialsToRemove)

    def getTrials(self):
        """Función para extraer los trials dentro de self.rawEEG"""
        ## Recorremos los eventos y extraemos los trials considerando el tinit y tmax
        #Calculamos la cantidad de muestras que representa el cueDuration.
        #Es importante tener en cuenta que el startingTime es variable. 

        #calculamos la cantidad de muestras que representa el cueDuration. Este tiempo es fijo
        cueDuration_samples = int(self.eventos["cueDuration"].to_numpy()[0] * self.sample_rate)
        #calculamos la cantidad de muestras que representa el finishDuration
        finishDuration_samples = int(self.eventos["finishDuration"].to_numpy()[0] * self.sample_rate)
        #calculamos la cantidad de muestras que representa el tinit
        tinit_samples = int(self.tinit * self.sample_rate)
        #encontramos el mínimo tinit en el dataframe
        max_tinit_samples = int(self.eventos["startingTime"].max()*self.sample_rate)
        min_tinit_samples = int(self.eventos["startingTime"].min()*self.sample_rate)
        if tinit_samples > max_tinit_samples:
            print("El tiempo mínimo supera lo marcado en Eventos. Se reemplaza el tiempo mínimo dentro de Eventos")
            tinit_samples = min_tinit_samples
        #calculamos la cantidad de muestras que representa el tmax
        tmax_samples = int(self.tmax * self.sample_rate)
        if tmax_samples > cueDuration_samples + finishDuration_samples:
            print("tmax_samples > cueDuration_samples. Se reemplaza tmax_samples por cueDuration_samples")
            tmax_samples = cueDuration_samples + finishDuration_samples

        #calculamos la cantidad de trials
        trials = self.eventos.shape[0]
        #calculamos la cantidad de canales
        channels = self.rawEEG.shape[0]
        #calculamos la cantidad de muestras por trial
        total_samples = tinit_samples + tmax_samples

        #Creamos un array de numpy para almacenar los trials
        trialsArray = np.zeros((trials, channels, total_samples))

        startingAccumulator = 0
        delaySamples = 0 #variable para almacenar la cantidad de muestras que se deben mover para extraer el siguiente trial

        #Recorremos los trials
        for trial in self.eventos.index:
            #calculamos la cantidad de muestras que representa el startingTime.
            #Recordar que el startingTime es variable
            startingTime_samples = int(self.eventos.loc[trial]["startingTime"] * self.sample_rate)
            startingAccumulator += startingTime_samples
            delaySamples = startingAccumulator + (cueDuration_samples + finishDuration_samples)*(trial-1)
            trialsArray[trial-1] = self.rawEEG[:, delaySamples - tinit_samples : delaySamples + tmax_samples]

        print("Se han extraido {} trials".format(trials))
        print("Se han extraido {} canales".format(channels))
        print("Se han extraido {} muestras por trial".format(total_samples))

        return trialsArray
            
    def getClassesName(self):
        """Función para obtener las etiquetas de los trials"""
        #Obtenemos los nombres de las clases. Cada nomrbe de clase se asocia con su número de clase.
        #Formamos una tupla con los valores únicos de la columna className y classNumber
        #Ordenamos los de la tupla por el número de clase

        #Nos quedamos con las columnas className y classNumber del dataframe de eventos
        clases = self.eventos[["className", "classNumber"]]
        #Eliminamos los duplicados
        clases = clases.drop_duplicates()
        #Ordenamos por classNumber
        clases = clases.sort_values(by="classNumber")
        # #Convertimos a tupla
        # clases = clases.to_records(index=False)
        # #Convertimos a lista
        # clases = tuple(clases)

        return clases["className"].values.tolist(), clases["classNumber"].values.tolist()
        
    def getLabels(self):
        """Función para obtener las etiquetas de los trials"""
        #Nos quedamos con la columna classNumber del dataframe de eventos. La pasamos a un array de numpy
        labels = self.eventos["classNumber"].to_numpy()
        return labels
    
    def saveTrials(self, filename):
        """Función para guardar los trials en un archivo .npy"""
        np.save(filename, self.trials)
        print("Se han guardado los trials en {}".format(filename))

    def removeTrials(self, trialsToRemove:list):
        """Función para remover trials usando la lista de trialsToRemove.
        A partir de trialsToRemove removemos los indices de self.eventos, luego actualiamos
        self.trials y self.labels"""

        ##chequeamos que trialsToRemove sea una lista
        if not isinstance(trialsToRemove, list):
            raise TypeError("trialsToRemove debe ser una lista")
        
        ##chequeamos que los valores de los trials existan cómo indices
        if not all(trial in self.eventos.index for trial in trialsToRemove):
            raise ValueError("Los valores de trialsToRemove no existen cómo índices en self.eventos")
        
        else:
            #removemos los trials de self.eventos
            self.eventos = self.eventos.drop(trialsToRemove)
            #eliminamos los trials de self.trials
            self.trials = np.delete(self.trials, trialsToRemove, axis=0)
            #eliminamos los trials de self.labels
            self.labels = np.delete(self.labels, trialsToRemove, axis=0)

            print("Se han removido los trials {}".format(trialsToRemove))
    
if __name__ == "__main__":

    file = "data\mordiendo\eegdata\sesion1\sn1_ts0_ct1_r2.npy"
    rawEEG = np.load(file)

    eventosFile = "data\mordiendo\eegdata\sesion1\sn1_ts0_ct1_r2_events.txt"
    eventos = pd.read_csv(eventosFile, sep = ",")

    trialhandler = TrialsHandler(rawEEG, eventos, tinit = 0.5, tmax = 4, reject=None, sample_rate=250.)
    print(trialhandler.trials.shape)

    # trials.saveTrials("data/dummyTest/eegdata/sesion1/sn1_ts0_ct0_r1_trials.npy")

    t = np.arange(0, rawEEG[1].shape[0]/250., 1/250.)

    import matplotlib.pyplot as plt
    plt.plot(t[:], rawEEG[0,:])
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (uV)")
    plt.show()

    trials = trialhandler.trials
    t = np.arange(0, trials.shape[2]/250., 1/250.)

    #graficamos cada trial en un subplot. Cada subplot en una fila

    #stilo ggplot
    plt.style.use('ggplot')
    fig, axs = plt.subplots(trials.shape[0], 1, sharex=True, sharey=True)
    for i in range(trials.shape[0]):
        axs[i].plot(t[:], trials[i,0,:])
        axs[i].set_title("Trial {}".format(i+1))
        axs[i].set_xlabel("Tiempo (s)")
        axs[i].set_ylabel("Amplitud (uV)")
    #titulo del gráfico
    fig.suptitle("Trials para board synthetic - Sesión Calibración")
    plt.show()
    


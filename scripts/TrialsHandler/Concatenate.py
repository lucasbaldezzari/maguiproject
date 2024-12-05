import numpy as np
import pandas as pd
import logging

class Concatenate():
    """Clase para concatenar objetos del tipo TrialsHandler"""

    def __init__(self, thlist:list) -> None:
        """Constructor de la clase Concatenate
        
        Parametros:
            - thlist (list): lista de objetos TrialsHandler a concatenar
        """
        
        self.thlist = thlist
        self.eventos = self._concatenate()
        self.trials = self.getTrials()
        self.labels = self.getLabels()
        self.classesName = self.getClassesName() 

    def _concatenate(self):
        """Función para concatenar los DataFrame.
        Por cada dataframe dentro de self.thlist, se concatenan los mismos.
        Deben modificarse los índices de tal manera de que no se repitan, el primer dataframe
        mantiene los índices originales, los siguientes se les suma el máximo índice del dataframe anterior.
        """
        #obtenemos el máximo índice del dataframe anterior
        max_index = 0
        #creamos una lista vacía para guardar los dataframes concatenados
        df_list = []
        for th in self.thlist:
            #obtenemos el dataframe del objeto TrialsHandler
            df = th.eventos
            #modificamos los índices
            df.index = df.index + max_index
            #agregamos el dataframe a la lista
            df_list.append(df)
            #actualizamos el máximo índice
            max_index += df.index.max() + 1
        #concatenamos los dataframes
        df_concatenado = pd.concat(df_list)
        return df_concatenado

    def getTrials(self):
        """Función para obtener los trials concatenados
        Por cada TrialsHandler.trials, concatenamos los trials para mantenerlo de la forma
        [trials, canales, muestras]
        """
        return np.concatenate([th.trials for th in self.thlist], axis=0)
        
    def getLabels(self):
        """Función para obtener los labels concatenados
        Concatenamos los lables de cada trialsHandler uno seguido al otro en un array 1D
        """

        total = sum([len(th.labels) for th in self.thlist])
        labels = np.zeros(total, dtype=int)

        i = 0
        for th in self.thlist:
            labels[i:i+len(th.labels)] = th.labels
            i += len(th.labels)

        return labels

    
    def getClassesName(self):
        """Función para obtener los nombres de las clases concatenados.
        IMPORTANTE: Se supone que los TrialsHandler tienen las mismas clases y en el mismo orden.
        Por lo tanto nos quedamos con los nombres de las clases del primer TrialsHandler.
        """

        return self.thlist[0].classesName


if __name__ == "__main__":
    from TrialsHandler.TrialsHandler import TrialsHandler
    from TrialsHandler.Concatenate import Concatenate
    import numpy as np
    import pandas as pd

    file = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct1_r1.npy"
    eventosFile = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct1_r1_events.txt"
    rawEEG_1 = np.load(file)
    eventos_1 = pd.read_csv(eventosFile, sep = ",")

    file = "data\sujeto_1\eegdata\sesion2\sn2_ts0_ct0_r1.npy"
    eventosFile = "data\sujeto_1\eegdata\sesion2\sn2_ts0_ct0_r1_events.txt"
    rawEEG_2 = np.load(file)
    eventos_2 = pd.read_csv(eventosFile, sep = ",")

    th_1 = TrialsHandler(rawEEG_1, eventos_1, tinit = 0.5, tmax = 4, reject=None, sample_rate=250., trialsToRemove = [29,30])
    th_2 = TrialsHandler(rawEEG_2, eventos_2, tinit = 0.5, tmax = 4, reject=None, sample_rate=250.)

    concat = Concatenate([th_1, th_2])

    print(concat.eventos.shape)
    print(concat.trials.shape)
    print(concat.labels.shape)
    print(concat.classesName)

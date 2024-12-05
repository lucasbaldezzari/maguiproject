import numpy as np
from scipy.signal import windows
from sklearn.base import BaseEstimator, TransformerMixin

class Window(BaseEstimator, TransformerMixin):
    """Clase para aplicar una ventana a la señal. Las señales llegan en un numpy array de la forma
    [canales, muestras]. La idea es aplicar una ventana a toda la señal y retornarla con el mismo shape.
    Se utilizan las ventanas de numpy, por lo que se puede usar cualquier ventana que se encuentre en la
    documentación de numpy. La clase se puede usar como un objeto de sklearn, por lo que se puede usar en
    un pipeline de sklearn.
    
    NOTA: Al 11/9/2023 se hacen pruebas del uso de ventanas en el pipeline y no se obtienen buenos resultados
    """

    def __init__(self, windowName = "Hamming", axisToCompute = 2):
        """
        Inicializa el objeto con los parámetros de la ventana.
        -windowName: Nombre de la ventana a aplicar.
        -axisToCompute: Eje sobre el cual aplicar la ventana.
        """

        self.windowName = windowName
        self.axisToCompute = axisToCompute

        #chqueamos que el nombre de la ventana sea correcto
        if hasattr(windows, self.windowName):
            self.window = getattr(windows, windowName)
        else:
            raise ValueError("El nombre de la ventana no es correcto. Debe ser una ventana de numpy.")
        ##a partir de windowName importamos la ventana desde windows
        

    def fit(self, X = None, y=None):
        """Métod fit. No hace nada."""

        return self
    
    def transform(self, signal, y = None):
        """Función para aplicar la ventana a la señal.
        -signal: Es la señal en un numpy array de la forma [canales, muestras]."""

        return signal*self.window(signal.shape[self.axisToCompute])
    
if __name__ == "__main__":

    from SignalProcessor.Filter import Filter
    from SignalProcessor.Window import Window
    from TrialsHandler.TrialsHandler import TrialsHandler
    from SignalProcessor.CSPMulticlass import CSPMulticlass
    from SignalProcessor.FeatureExtractor import FeatureExtractor
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline

    fm = 250.
    file = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct0_r1.npy"
    rawEEG = np.load(file)

    eventosFile = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct0_r1_events.txt"
    eventos = pd.read_csv(eventosFile, sep = ",")

    trialhandler = TrialsHandler(rawEEG, eventos, tinit = 0., tmax = 4, reject=None, sample_rate=250.)

    trials = trialhandler.trials
    trials.shape
    labels = trialhandler.labels

    filtro = Filter(lowcut=8.0, highcut=28.0, notch_freq=50.0, notch_width=2.0, sample_rate=100.0)
    window = Window(windowName = "hann")
    cspmulticlass = CSPMulticlass(n_components=2, method = "ovo", n_classes = len(np.unique(labels)), reg = 0.01)
    featureExtractor = FeatureExtractor(method = "welch", sample_rate = fm, band_values=[8,18])

    pipeline_sin_window = Pipeline([
        ('pasabanda', filtro),
        ('cspmulticlase', cspmulticlass),
        # ('featureExtractor', featureExtractor),
    ])

    pipeline_sin_window.fit(trials, labels)

    trials_sin_ventana = pipeline_sin_window.transform(trials)
    trials_sin_ventana.shape

    #Grafico
    trial = 1
    plt.figure(figsize=(10,5))
    plt.plot(trials_sin_ventana[trial-1,0,:])
    plt.plot(trials_sin_ventana[trial-1,1,:])
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud (uV)")
    plt.show()

    pipeline_con_ventana = Pipeline([
        ('window', window),
        ('pasabanda', filtro),
        ('cspmulticlase', cspmulticlass),
        # ('featureExtractor', featureExtractor),
    ])

    pipeline_con_ventana.fit(trials, labels)

    trials_con_ventana = pipeline_con_ventana.transform(trials)

    #Grafico
    trial = 1
    plt.figure(figsize=(10,5))
    plt.plot(trials_con_ventana[trial-1,0,:])
    plt.plot(trials_con_ventana[trial-1,1,:])
    plt.xlabel("Muestras")
    plt.show()

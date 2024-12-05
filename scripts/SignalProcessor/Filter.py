import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.base import BaseEstimator, TransformerMixin

class Filter(BaseEstimator, TransformerMixin):
    """Clase para filtrar señales provenientes de la placa openBCI. Las señales llegan en un numpy array de la forma
    [trials, canales, muestras]. La idea es aplicar un filtro pasa banda y un filtro notch a la señal a todo el array.
    La clase se puede usar como un objeto de sklearn, por lo que se puede usar en un pipeline de sklearn."""

    def __init__(self, lowcut = 8.0, highcut = 30.0, notch_freq = 50.0, notch_width = 2.0, sample_rate = 250.0,
                 axisToCompute = 2, padlen = None, order = 4, discard_samples = 0):
        """Inicializa el objeto con los parámetros de filtrado.
        -lowcut: Frecuencia de corte inferior del filtro pasa banda.
        -highcut: Frecuencia de corte superior del filtro pasa banda.
        -notch_freq: Frecuencia de corte del filtro notch.
        -notch_width: Ancho de banda del filtro notch.
        -sample_rate: Frecuencia de muestreo de la señal.
        -X: Señal de entrada. No se usa en este caso.
        -y: No se usa en este caso."""

        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.notch_width = notch_width
        self.sample_rate = sample_rate
        self.axisToCompute = axisToCompute
        self.padlen = padlen
        self.order = order
        self.discard_samples = int(discard_samples*self.sample_rate) #muestras a descartar al inicio de la señal filtrada

        self.b, self.a = butter(self.order, [self.lowcut, self.highcut], btype='bandpass', fs=self.sample_rate)
        self.b_notch, self.a_notch = iirnotch(self.notch_freq, 20, self.sample_rate)

    def fit(self, X = None, y=None):
        """Métod fit. No hace nada."""

        return self #el método fit siempre debe retornar self!

    def transform(self, signal, y = None):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un numpy array de la forma [n-trials, canales, muestras]."""

        signal = signal - np.mean(signal, axis=self.axisToCompute, keepdims=True)
        signal = filtfilt(self.b, self.a, signal, axis = self.axisToCompute, padlen = self.padlen) #aplicamos el filtro pasa banda
        signal = filtfilt(self.b_notch, self.a_notch, signal, axis = self.axisToCompute, padlen = self.padlen) #aplicamos el filtro notch
        return signal[..., self.discard_samples:] #descartamos las primeras muestras

if __name__ == "__main__":

    with open('SignalProcessor/testData/all_left_trials.npy', 'rb') as f:
        signalleft = np.load(f)

    with open('SignalProcessor/testData/all_right_trials.npy', 'rb') as f:
        signalright = np.load(f)

    filtro = Filter(lowcut=8.0, highcut=28.0, notch_freq=50.0, notch_width=2.0, sample_rate=100.0)

    signalleftFiltered = filtro.fit_transform(signalleft)
    sognalrightFiltered = filtro.fit_transform(signalright)

    ### Grafico para comparar señal original y señal filtrada
    import matplotlib.pyplot as plt
    plt.plot(signalleftFiltered[0,28,:], label = "Filtrada")
    plt.plot(signalleft[0,28,:], label = "Original")
    plt.show()
import numpy as np
from scipy.signal import hilbert, welch
from sklearn.base import BaseEstimator, TransformerMixin
# from matplotlib import mlab

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """La clase para extraer características de desincronización relacionada con eventos (ERD) y sincronización relacionada con eventos (ERS) de señales EEG.
    - La clase puede extraer la envolvente de la señal a partir de Hilber, o,
    - La clase puede extraer la potencia de la señal a partir de la transformada de Welch.

    La clase se comporta como un transformer de sklearn, por lo que puede ser usada en un pipeline de sklearn.
    """

    def __init__(self, method = "welch", sample_rate = 250., axisToCompute = 2, band_values = None):
        """No se inicializan atributos.
        - method: método por el cual extraer las caracerísticas. Puede ser welch o hilbert.
        - sample_rate: frecuencia de muestreo de la señal.
        - axisToCompute: eje a lo largo del cual se calculará la transformada.
        - utils_freqs: lista de dos valores con el 
        rango de freucencias a devolver cuando se aplique transform o fit_transform"""

        assert method in ["hilbert", "welch"], "method debe ser 'hilbert' o 'welch'"
        self.method = method
        self.sample_rate = sample_rate
        self.axisToCompute = axisToCompute
        self.band_values = band_values
        

    def fit(self, X = None, y=None):
        """No hace nada"""

        return self #este método siempre debe retornar self!

    def transform(self, signal):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un arreglo de numpy de la forma [n_trials, canales, muestras].
        
        Retorna: Un arreglo de numpy con las características de la señal. La forma del arreglo es [canales, power_sample, n_trials]"""
        
        if self.method == "welch":
            """Retorna la potencia de la señal en la forma [n_trials, canales, power_samples]"""
            self.freqs, self.power = welch(signal, axis=self.axisToCompute) #trnasformada de Welch
            self.freqs = self.freqs*self.sample_rate

            if self.band_values:
                self.freqs_indexes = np.where((self.freqs >= self.band_values[0]) & (self.freqs <= self.band_values[1]))[0]
                #retornamos los valores de frecuencias que se encuentran en el rango de interés
                self.freqs = self.freqs[self.freqs_indexes]
                self.power = self.power[:, :, self.freqs_indexes]
                return self.power
        
            else:
                return self.power
        
        if self.method == "hilbert":
            """Retorna la potencia de la señal en la forma [n_trials, canales, power_samples]"""
            analyticSignal = hilbert(signal, axis=self.axisToCompute)
            self.envolvente = np.abs(analyticSignal) #envolvente de la señal analítica
            return self.envolvente
        
    def fit_transform(self, signal, y = None):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un arreglo de numpy de la forma [canales, muestras]."""

        self.fit()
        return self.transform(signal)
    

if __name__ == '__main__':

    with open("SignalProcessor\TestData\All_left_trials.npy", "rb") as f:
        signalLeft = np.load(f)

    with open("SignalProcessor\TestData\All_right_trials.npy", "rb") as f:
        signalRight = np.load(f)
    

    ## Extraemos envolventes de las señals
    featureExtractor = FeatureExtractor(method="welch", sample_rate=100., band_values = [8,12]) #instanciamos el extractor de características
    featuresleft = featureExtractor.fit_transform(signalLeft) #signal [n_channels, n_samples, n_trials]
"""Clase para aplicar CSP

La clase hace uso de mne.decoding.csp (https://mne.tools/stable/generated/mne.decoding.CSP.html)

Si bien esta clase puede parecer redundante, en realidad se agregan algunos métodos para el correcto funcionamiento
del pipeline del Hackathon 2.

References
    ----------

    Christian Andreas Kothe,  Lecture 7.3 Common Spatial Patterns
    https://www.youtube.com/watch?v=zsOULC16USU

    Información acerca de "Common spatial pattern"
    https://en.wikipedia.org/wiki/Common_spatial_pattern


    Optimizing Spatial filters for Robust EEG Single-Trial Analysis
    https://ieeexplore.ieee.org/document/4408441

    https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html#sphx-glr-auto-examples-decoding-decoding-csp-eeg-py

    ----------

    #IMPORTANTE: DEBEMOS PRESTAR ESPECIAL ANTENCIÓN AL ORDEN DE LAS CLASES YA QUE EL CSP NOS DEVOLVERÁ UN SET DE FILTROS ESPACIALES
    #CADA FILTRO ESPACIAL SE CORRESPONDE A UNA DE LAS CLASES EN EL ORDEN CON QUE SE ENTRENA EL CSP

"""

import numpy as np
import pickle
from sklearn import base

from mne.decoding import CSP

class commonSpatialPattern(base.BaseEstimator, base.TransformerMixin):
    """Clase CSP. El código base usa mne.decoding.csp. Se agregan algunos métodos para funcionalidad
    del sistema correspondiente al segundo hackathon"""

    def __init__(self, n_components=4, reg=None, log=None, cov_est='concat',
                 transform_into='csp_space', norm_trace=False,
                 cov_method_params=None, rank=None,
                 component_order='mutual_info') -> None:
        """Constructor de clase

        Para información de los parámetros reg, log y transform_into referenciarse a mne.decoding.csp"""

        self.n_components = n_components
        self.rank = rank
        self.reg = reg
        self.cov_est = cov_est
        self.log = log
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = component_order
        self.transform_into = transform_into

        self._csp = CSP(n_components = self.n_components, reg=reg, log=log, cov_est=cov_est,
                 transform_into=transform_into, norm_trace=norm_trace, cov_method_params=cov_method_params,
                 rank=rank, component_order=component_order)

    @staticmethod
    def loadCSP(self, filename = "csp_model.pkl"):
        """Cargamos un modelo ya entrenado a partir de un archivo .pickle"""

        with open(filename, "rb") as f:
            self._csp = pickle.load(f)

    def fit(self, X, y):
        """Determina los filtros espaciales a partir de las épocas dentro de nuestros

        - X: numpy.array de la forma [n_trials, n_channels, n_samples]
        - y: array con las etiquetas de clase. El array es de la forma [n_trials].
        """

        self._csp.fit(X,y)

        return self #El método fit siempre retorna self

    def transform(self, X, y = None):
        """Retorna los datos transformados a partir de los filtros espaciales determinados por el método fit
        La forma de los datos retornados es [n_trials, n_components, n_samples]"""

        return self._csp.transform(X)
    
    def fit_transform(self, X, y=None, **fit_params):
        """Sobreescribir este método para que se comporte según lo esperado cuando se llame a fit_transform()"""
        ##TODO: implementar este método

    def saveCSPFilters(self, filname = "csp_model.pkl"):
        """Guardamos el modelo"""
        with open(filname, 'wb') as f:
            pickle.dump(self._csp, f)


if __name__ == "__main__":

    sample_frec = 100.
    folder = "testData/"
    left = np.load(folder+"all_left_trials.npy", allow_pickle=True)
    right = np.load(folder+"all_right_trials.npy", allow_pickle=True)
    print(left.shape) #[n_channels, n_samples, ntrials]
    print(right.shape) #[n_channels, n_samples, ntrials]

    c3, cz, c4 = 26, 28, 30 #canales de interés
    class_info = {1: "left", 2: "right"} #diccionario para identificar clases. El orden se corresponde con lo que hay eneegmatrix
    n_clases = len(list(class_info.keys()))
    
    #los datos deben pasarse al csp de la forma [n_epochs, n_channels, n_samples]
    #los datos que cargamos en eegmatrix están de la forma [n_clases, n_channels, n_samples]

    #Contactemos en un sólo array
    eegmatrix = np.concatenate((left,right), axis=0) #importante el orden con el que concatenamos
    print(eegmatrix.shape) #[ n_trials (o n_epochs), n_channels, n_samples]

    #En este ejemplo tenemos los trials ordenados, es decir, tenemos 100 trials para la clase left y 100 para la clase right.
    #Por lo tanto, las etiquetas que crearemos estarán ordenadas.
    n_trials = left.shape[0]
    totalTrials = eegmatrix.shape[0]
    labels = np.array([np.full(n_trials, label) for label in class_info.keys()]).reshape(totalTrials)
    print(labels.shape)
    print(labels)

    csp = commonSpatialPattern(n_components = 59, transform_into="csp_space", reg = 0.01)

    csp.fit(eegmatrix, labels) #entrenamos csp

    #projectamos un sólo trial.
    left_csp = csp.transform(left)
    right_csp = csp.transform(right)

    import matplotlib.pyplot as plt
    left_var = np.log(np.var(left.mean(axis = 0), axis = 1))
    right_var = np.log(np.var(right.mean(axis = 0), axis = 1))
    plt.bar(np.arange(1,60), left_var, label = "left")
    plt.bar(np.arange(1,60), np.flip(right_var), label = "right")
    plt.legend()
    plt.show()

    #ahora graficamos los datos después de aplicar csp
    leftvar_csp = np.log(np.var(left_csp.mean(axis = 0), axis = 1))
    rightvar_csp = np.log(np.var(right_csp.mean(axis = 0), axis = 1))
    plt.bar(np.arange(1,60), leftvar_csp, label = "left_csp")
    plt.bar(np.arange(1,60), np.flip(rightvar_csp), label = "right_csp")
    plt.legend()
    plt.show()

    ## Vamos a utilizar un nuevo csp con menos componentes
    csp = commonSpatialPattern(n_components = 4, transform_into="csp_space", reg = 0.01)

    csp.fit(eegmatrix, labels) #entrenamos csp

    #projectamos un sólo trial.
    n_channels = left.shape[1]
    n_samples = left.shape[2]
    left_csp = csp.transform(left.mean(axis = 0).reshape(1,n_channels,n_samples))
    right_csp = csp.transform(right.mean(axis = 0).reshape(1,n_channels,n_samples))

    ##Extraemos características de los datos sin transformar y transformados. Promediamos los eeg trials
    from FeatureExtractor import FeatureExtractor

    left_medio = left.mean(axis = 0)
    right_medio = right.mean(axis = 0)

    #extrameos features sin csp
    featureExtractor = FeatureExtractor(sample_rate = 100)
    left_features_no_csp = featureExtractor.transform(left_medio.reshape(1,n_channels,n_samples))[0]

    featureExtractor = FeatureExtractor(sample_rate = 100)
    right_features_no_csp = featureExtractor.transform(right_medio.reshape(1,n_channels,n_samples))[0]

    #graficando las features sin csp para canales c3, cz y c4. En tres axes
    fig, axes = plt.subplots(1,3, figsize = (10,10))
    fig.suptitle("Canales C3, Cz y C4")
    axes[0].plot(left_features_no_csp[c3,:], label = "left")
    axes[0].plot(right_features_no_csp[c3,:], label = "right")
    axes[0].legend()
    axes[1].plot(left_features_no_csp[cz,:], label = "left")
    axes[1].plot(right_features_no_csp[cz,:], label = "right")
    axes[1].legend()
    axes[2].plot(left_features_no_csp[c4,:], label = "left")
    axes[2].plot(right_features_no_csp[c4,:], label = "right")
    axes[2].legend()
    plt.show()

    #extraemos las features con CSP
    featureExtractor = FeatureExtractor(sample_rate = 100)
    left_features_csp = featureExtractor.transform(left_csp).reshape(4,-1) #4 son las componentes
    right_features_csp = featureExtractor.transform(right_csp).reshape(4,-1)

    plt.plot(left_features_csp[0], label = "componente 1 - left")
    plt.plot(right_features_csp[0], label = "componente 1 - right")
    plt.show()

    #entrenando rápidamente un clasificador y probandolo con cross validation
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import ShuffleSplit, cross_val_score
    from sklearn.model_selection import train_test_split

    from RavelTransformer import RavelTransformer
    from FeatureExtractor import FeatureExtractor

    ravel = RavelTransformer(method="mean")
    lda = LinearDiscriminantAnalysis()
    svm = SVC(kernel='linear', C=1, gamma='scale')

    #creamos el pipeline
    pipeline = Pipeline([
        ("csp", commonSpatialPattern(n_components = 6, transform_into="csp_space", reg = 0.01)),
        ("feature_extractor", FeatureExtractor(sample_rate = 100, axisToCompute = 2)),
        ("ravel", ravel),
        ("classifier", lda)
    ])

    #entrenamos el pipeline
    pipeline.fit(eegmatrix, labels)

    #predecimos con el pipeline un trial
    n_channels = left.shape[1]
    n_samples = left.shape[2]
    prediction = pipeline.predict(left[50].reshape(1,n_channels,n_samples))
    print(prediction)

    #hacemos cross validation

    eegtotrain = eegmatrix[:,[c3,cz,c4],:]
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)
    cv_split = cv.split(eegmatrix)

    scores = cross_val_score(pipeline, eegmatrix, labels, cv = cv, n_jobs=None)

    print(scores)

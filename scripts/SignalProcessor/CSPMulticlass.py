"""
Clase para generar y/o aplicar filtros CSP para múltiples movimientos (clases). Se puede seleecionar entre el método "one vs one" o "one vs all. En el primer cao
tendremos K(K-1)/2 filtros (donde K es la cantidad de movimientos/clases) y en el segundo se tendrán tantos filtros como clases/movimientos tenga la BCI.

Esta clase hará uso de la clase mne.decoding.csp para generar los filtros CSP. Para más información sobre esta clase, ver la documentación de mne.

1) El constructor de CSPMulticlass recibe los parámetros que recibe la clase mne.decoding.csp. Estos parámetros se utilizarán para entrenar cada clasificador
2) El método fit recibe los datos de entrenamiento y las etiquetas de clase. Estos datos se utilizarán para entrenar cada filtro CSP. El método fit retorna self
3) El método transform recibe los datos a transformar y retorna los datos transformados. Este método no se utiliza para entrenar los filtros CSP.

IMPORTANTE: Debemos tener en cuenta que el método mne.decoding.csp posee un atributo filters_. El mismo posee la matriz de filtros espaciales luego de aplicar
fit(). La clase CSP devuelve sólo los m filtros con las mayores autovalores. Estos m filtros son los que capturan la información discriminante más importante
entre las dos clases. Cuando transform_into='csp_space', el método transform() aplica estos m filtros a los datos de entrada, resultando en un nuevo espacio de
características que está proyectado sobre los componentes CSP. A diferencia de otras implementaciones, el método de mne sólo devuelve los m filtros con las mayores
autovalores. Esto significa que si queremos utilizar además los m filtros correspondientes a los m menores autovalores, debemos hacerlo manualmente utilizando
los atributos filters_, eigenvalues_ y eigenvectors_ de la clase mne.decoding.csp, ejemplo:


eigenvalues = csp.eigenvalues_
eigenvectors = csp.filters_

sorted_indices = np.argsort(eigenvalues)[::-1]  # ordenamos los índices
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

m = 3
top_filters = sorted_eigenvectors[:, :m]
bottom_filters = sorted_eigenvectors[:, -m:]
"""

import numpy as np
import pickle
from sklearn import base
import matplotlib.pyplot as plt

from mne.decoding import CSP

import itertools
from mne import EvokedArray
from mne import create_info
from mne.channels import make_standard_montage
import copy

class CSPMulticlass(base.BaseEstimator, base.TransformerMixin):
    """TODO: Documentación de clase"""

    def __init__(self, method = "ovo", n_classes = 5, n_components=2, reg=None, log=None, cov_est='concat',
                 transform_into='csp_space', norm_trace=False, cov_method_params=None, rank=None,
                 component_order='mutual_info', format_names = "C%svsC%s") -> None:
        """Constructor de clase

        Parámetros:
        ----------------
        - method: str. Es el método a utilizar para entrenar los filtros CSP. Puede ser "ovo" u "ova". Por defecto es "ovo"
        - n_classes: int. Es la cantidad de clases que se pretende discriminar con CSP. Por defecto es 5
        - n_components: int. Es la cantidad de componentes a extraer. Por defecto es 2
        - reg: float. Es el parámetro de regularización. Por defecto es None
        - log: bool. Si es True, se aplica logaritmo a la matriz de covarianza. Por defecto es None
        - cov_est: str. Es el método a utilizar para estimar la matriz de covarianza. Puede ser "concat", "epoch" o "auto". Por defecto es "concat"
        - transform_into: str. Es el método a utilizar para transformar los datos. Puede ser "average_power" o "csp_space". Por defecto es "csp_space"
        - norm_trace: bool. Si es True, se normaliza la traza de la matriz de covarianza. Por defecto es False
        - cov_method_params: dict. Es un diccionario con los parámetros para el método de estimación de covarianza. Por defecto es None
        - rank: int. Es el rango de la matriz de covarianza. Por defecto es None
        - component_order: str. Es el método a utilizar para ordenar los componentes. Puede ser "mutual_info" o "shuffle". Por defecto es "mutual_info"

        Para más información sobre los parámetros (excepto method y n_classes), ver la documentación de mne.decoding.csp"""

        #chequeamos si method es válido
        assert method in ["ovo", "ova"], "method debe ser 'ovo' u 'ova'"
        self.method = method
        self.n_classes = n_classes
        self.n_components = n_components
        self.rank = rank
        self.reg = reg
        self.cov_est = cov_est
        self.log = log
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = component_order
        self.transform_into = transform_into
        self.class_combinations = None

        assert format_names.count("%") == 2, "format_names debe tener dos y sólo dos %s, ejemplo: 'C%svsC%s'"
        self.format_names = format_names

        #lista de filtros CSP
        if method == "ovo":
            self.csplist = [CSP(n_components = self.n_components, reg=reg, log=log, cov_est=cov_est,
                 transform_into=transform_into, norm_trace=norm_trace, cov_method_params=cov_method_params,
                 rank=rank, component_order=component_order) for i in range(int((n_classes*(n_classes-1))/2))]

        if method == "ova":
            self.csplist = [CSP(n_components = self.n_components, reg=reg, log=log, cov_est=cov_est,
                 transform_into=transform_into, norm_trace=norm_trace, cov_method_params=cov_method_params,
                 rank=rank, component_order=component_order) for i in range(n_classes)]

    def saveCSPList(self, filename, folder = "filtrosCSP"):
        """Método para guardar los filtros CSP en un archivo pickle

        Parámetros:
        ----------------
        - filename: str. Es el nombre del archivo donde se guardarán los filtros CSP
        - folder: str. Es el nombre de la carpeta donde se guardarán los filtros CSP. Por defecto es "filtrosCSP"

        Retorna:
        ----------------
        - None"""

        #guardamos en un archivo pickle
        with open(folder + "/" + filename, "wb") as f:
            pickle.dump(self.csplist, f)

    @staticmethod #método estático para poder cargar filtros CSP sin instanciar la clase
    def loadCSPList(self, filename, folder = "filtrosCSP"):
        """Método para cargar los filtros CSP desde un archivo pickle

        Parámetros:
        ----------------
        - filename: str. Es el nombre del archivo donde se guardarán los filtros CSP
        - folder: str. Es el nombre de la carpeta donde se guardarán los filtros CSP. Por defecto es "filtrosCSP"

        Retorna:
        ----------------
        - None"""

        with open(folder + "/" + filename, "rb") as f:
            self.csplist = pickle.load(f)


    def fit(self, X, y):
        """Método para entrenar los filtros CSP

        - Las señales de EEG vienen en el formato (n_trials, n_channels, n_samples).
        - Si el método es "ovo", se entrena un filtro CSP para cada combinación de clases. Clase1 vs Clase2, Clase1vsClase3, etc.
        - Si el método es "ova", se entrena un filtro CSP para cada clase. Donde se toma clase1 vs todas las demás clases,
        clase 2 vs todas las demás clases, etc.
        - Los filtros a entrenar se encuentran dentro de la lista self.csplist

        Parámetros:
        ----------------
        - X: ndarray. Es un arreglo de numpy con los datos de entrenamiento. Tiene el formato (n_trials, n_channels, n_samples)
        - y: ndarray. Es un arreglo de numpy con las etiquetas de clase. Tiene el formato (n_trials,)

        Retorna:
        ----------------
        - self"""

        assert X.shape[0] == y.shape[0], "X e y deben tener la misma cantidad de trials"
        assert self.n_classes == len(np.unique(y)), "La cantidad de clases debe ser igual a la cantidad de labels únicas en y"

        ##ordeno los trials en base a los valores dentro de y
        #obtengo los índices de los trials ordenados
        indices = np.argsort(y)
        #ordeno los trials en base a los índices
        X = X[indices]
        y = y[indices]

        if self.method == "ovo":
            classlist = np.unique(y)
            ## ordenamos classlist de menor a mayor
            classlist = np.sort(classlist)
            class_combinations = list(itertools.combinations(classlist, 2))
            self.class_combinations = [f"{self.format_names % (c1,c2)}" for c1, c2 in class_combinations]

            for i, (c1, c2) in enumerate(class_combinations):
                #índices de las muestras con clase c1 y clase c2
                index_c1 = np.where(y == c1)
                index_c2 = np.where(y == c2)

                #trials correspondientes a las clases c1 y c2
                trials_c1 = X[index_c1]
                trials_c2 = X[index_c2]

                #concatenamos los trials a utilizar para entrenar el filtro CSP
                trials = np.concatenate((trials_c1, trials_c2), axis = 0)

                #concatenamos las etiquetas de los trials
                labels = np.concatenate((np.ones(trials_c1.shape[0]), np.zeros(trials_c2.shape[0])), axis = 0)

                #fitteamos el filtro CSP
                self.csplist[i].fit(trials, labels)

        if self.method == "ova":
            classlist = np.unique(y)
            classlist = np.sort(classlist)
            self.class_combinations = [f"C{str(c)}"  + ".vsA" for c in classlist]
            for i, c in enumerate(classlist):
                c_index = np.where(y == c) #índices de la clase de interés
                others_index = np.where(y != c) #índices de las demás clases

                c_trials = X[c_index] #trials de la clase de interes
                others_trials = X[others_index] #trials de las demás clases

                trials = np.concatenate((c_trials, others_trials), axis = 0)

                c_labels = np.zeros(c_trials.shape[0]) #etiquetas de los trials de la clase de interés
                others_labels = np.ones(others_trials.shape[0]) #etiquetas de los trials de las demás clases
                labels = np.concatenate((c_labels, others_labels), axis = 0) #concatenamos las etiquetas

                #fitteamos el filtro CSP
                self.csplist[i].fit(trials, labels)

        self.is_fitted = True

        return self #el método fit siempre debe retornar self

    def transform(self, X, y = None):
        """Para cada csp en self.csplist, se transforman los trials de X

        - X: ndarray. Es un arreglo de numpy con los datos de entrenamiento. Tiene el formato (n_trials, n_channels, n_samples)"""

        #transformamos los trials de cada clase
        X_transformed = [csp.transform(X) for csp in self.csplist]

        #concatenamos los trials de cada clase
        X_transformed = np.concatenate(X_transformed, axis = 1)

        return X_transformed
    
    def plot_patterns(self, channelsName, fm, montage = 'standard_1020',scalings=1.0,
                      cbar_fmt = "%3.1f", cmap = "coolwarm",nrows="auto", ncols = "auto",
                      sensors = "kx",size = 1, cspnames = False, ynames = False,font_ylab = 10,
                      save = False, filename = "patterns.png", dpi  = 300, contours = 6,
                      title = "Patrones CSP", title_size = 14, colorbar = True, show = True):
            
            import matplotlib.pyplot as pyplot
            
            #chequeamos self.is_fitted sino raise error
            assert self.is_fitted, "La clase CSPMulticlass no está entrenada. Debe llamar al método fit antes de graficar los patrones CSP"
            
            # Chequeamos que la cantidad de columnas y filas sea válida
            if nrows == "auto":
                nrows = len(self.class_combinations)
            if ncols == "auto":
                ncols = self.n_components
            else:
                mensaje = "La cantidad de filas y columnas debe ser igual a la cantidad de componentes por la cantidad de combinaciones de clases"
                assert nrows*ncols == self.n_components*len(self.class_combinations), mensaje

            #creamos info para poder utilizar la función plot_topomap
            info = create_info(channelsName, fm, "eeg")
            montage = make_standard_montage(montage)
            info.set_montage(montage)

            #obtenemos los patrones CSP
            patterns_array = np.array([csp.patterns_ for csp in self.csplist])
            patterns_array = patterns_array.reshape(-1, patterns_array.shape[2])
            
            #obtenemos los límites del colorbar
            vlim = (patterns_array.min(), patterns_array.max())

            #obtenemos la cantidad de componentes a graficar
            n_combintaions = len(self.class_combinations)
            offset = patterns_array.T.shape[0]
            qty = np.arange(self.n_components)
            components = np.array([qty + offset*k for k in range(n_combintaions)]).ravel()

            #creamos un objeto EvokedArray para poder utilizar la función plot_topomap
            info2 = copy.deepcopy(info)
            with info2._unlock():
                    info2["sfreq"] = 1.0

            #eliminamos objeto info
            del info

            patterns = EvokedArray(patterns_array.T, info2, tmin=0)

            topomap = patterns.plot_topomap(times = components, colorbar=colorbar, size = size, scalings=scalings, time_format="",
                                            nrows = nrows, ncols = ncols, sensors=sensors, vlim = vlim, contours = contours,
                                            cbar_fmt=cbar_fmt, cmap=cmap, show=False)
            
            #agregamos nombre de las clases
            if ynames:
                j = 0
                for i in range(len(self.class_combinations)):
                    if colorbar and j == ncols:
                        j+=1
                    topomap.axes[j].set_ylabel(self.class_combinations[i], size=font_ylab)
                    j+=self.n_components

            #agregamos nombre de los filtros
            if cspnames:
                j = 0
                l = 0
                for i in range(len(self.class_combinations)):
                    if colorbar and j == ncols:
                        j+=1
                    for k in range(self.n_components):
                        topomap.axes[j+k].set_title(f"{self.class_combinations[l]}:P{k+1}", fontsize=10)
                    j+=self.n_components
                    l+=1

            #modifico los límites del ejey del colorbar
            topomap.axes[ncols].set_ylim(vlim[0],vlim[1])

            #agregamos título
            pyplot.suptitle(title, fontsize=title_size)

            #guardamos figura si es necesario
            if save:
                pyplot.savefig(filename, dpi=dpi)

            #mostarmos figura si es necesario
            if show:
                pyplot.show()

            del pyplot

    def plot_filters(self, channelsName, fm, montage = 'standard_1020',scalings=1.0,
                        cbar_fmt = "%3.1f", cmap = "coolwarm",nrows="auto", ncols = "auto",
                        sensors = "kx",size = 1, cspnames = False, ynames = False,font_ylab = 10,
                        save = False, filename = "filters.png", dpi  = 300, contours = 6,
                        title = "Filtros CSP", title_size = 14, colorbar = True, show = True):
            
            import matplotlib.pyplot as pyplot
                
            #chequeamos self.is_fitted sino raise error
            assert self.is_fitted, "La clase CSPMulticlass no está entrenada. Debe llamar al método fit antes de graficar los filtros CSP"
            
            # Chequeamos que la cantidad de columnas y filas sea válida
            if nrows == "auto":
                nrows = len(self.class_combinations)
            if ncols == "auto":
                ncols = self.n_components
            else:
                mensaje = "La cantidad de filas y columnas debe ser igual a la cantidad de componentes por la cantidad de combinaciones de clases"
                assert nrows*ncols == self.n_components*len(self.class_combinations), mensaje

            #creamos info para poder utilizar la función plot_topomap
            info = create_info(channelsName, fm, "eeg")
            montage = make_standard_montage(montage)
            info.set_montage(montage)

            #obtenemos los patrones CSP
            filters_array = np.array([csp.filters_ for csp in self.csplist])
            filters_array = filters_array.reshape(-1, filters_array.shape[2])
            
            #obtenemos los límites del colorbar
            vlim = (filters_array.min(), filters_array.max())

            #obtenemos la cantidad de componentes a graficar
            n_combintaions = len(self.class_combinations)
            offset = filters_array.T.shape[0]
            qty = np.arange(self.n_components)
            components = np.array([qty + offset*k for k in range(n_combintaions)]).ravel()

            #creamos un objeto EvokedArray para poder utilizar la función plot_topomap
            info2 = copy.deepcopy(info)
            with info2._unlock():
                    info2["sfreq"] = 1.0

            #eliminamos objeto info
            del info

            filters = EvokedArray(filters_array.T, info2, tmin=0)

            topomap = filters.plot_topomap(times = components, colorbar=colorbar, size = size, scalings=scalings, time_format="",
                                            nrows = nrows, ncols = ncols, sensors=sensors, vlim = vlim, contours = contours,
                                            cbar_fmt=cbar_fmt, cmap=cmap, show=False)
            
            #agregamos nombre de las clases
            if ynames:
                j = 0
                for i in range(len(self.class_combinations)):
                    if colorbar and j == ncols:
                        j+=1
                    topomap.axes[j].set_ylabel(self.class_combinations[i], size=font_ylab)
                    j+=self.n_components

            #agregamos nombre de los filtros
            if cspnames:
                j = 0
                l = 0
                for i in range(len(self.class_combinations)):
                    if colorbar and j == ncols:
                        j+=1
                    for k in range(self.n_components):
                        topomap.axes[j+k].set_title(f"{self.class_combinations[l]}:F{k+1}", fontsize=10)
                    j+=self.n_components
                    l+=1

            #modifico los límites del ejey del colorbar
            topomap.axes[ncols].set_ylim(vlim[0],vlim[1])

            #agregamos título
            pyplot.suptitle(title, fontsize=title_size)

            #guardamos figura si es necesario
            if save:
                pyplot.savefig(filename, dpi=dpi)

            #mostarmos figura si es necesario
            if show:
                pyplot.show()

if __name__ == "__main__":
    import numpy as np
    from SignalProcessor.Filter import Filter
    from SignalProcessor.CSPMulticlass import CSPMulticlass
    from sklearn.model_selection import train_test_split
    from mne.channels import make_standard_montage
    import matplotlib.pyplot as plt
    from mne.decoding import CSP

    eeg_file = "SignalProcessor/testData/trials_sujeto8_trainingEEG.npy"
    trials = np.load(eeg_file)
    labels = np.load("SignalProcessor/testData/labels_sujeto8_training.npy")

    channelsName = ["P3", "P4", "C3", "C4", "F3", "F4", "Pz", "Cz"]
    channelsSelected = [0,1,2,3,4,5,6,7]
    channelsName = [channelsName[i] for i in channelsSelected]
    trials = trials[:,channelsSelected,:]

    ##filtramos los trials para las clases que nos interesan
    trials = trials[np.where((labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 5))]
    labels = labels[np.where((labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 5))]

    fm = 250.
    filter = Filter(lowcut=8, highcut=18, notch_freq=50.0, notch_width=2, sample_rate=fm, axisToCompute=2, padlen=None, order=4)

    n_components = 3
    n_classes = len(np.unique(labels))
    cspmulticlass = CSPMulticlass(n_components=n_components, method = "ovo", n_classes = n_classes, reg = 0.01)

    trials_filtered = filter.fit_transform(trials)

    eeg_train, eeg_test, labels_train, labels_test = train_test_split(trials_filtered, labels, test_size=0.2, stratify=labels, random_state=42)
    eeg_train, eeg_val, labels_train, labels_val = train_test_split(eeg_train, labels_train, test_size=0.2, stratify=labels_train, random_state=42)

    cspmulticlass.fit(eeg_train, labels_train)

    nrows = len(cspmulticlass.class_combinations)//2
    ncols = n_components*2
    
    cspmulticlass.plot_patterns(channelsName, fm, size = 1, nrows=nrows, ncols=ncols, cspnames=True, save=True, contours = 10)
    cspmulticlass.plot_filters(channelsName, fm, size = 1, nrows=nrows, ncols=ncols, cspnames=True, save=True, contours = 10)
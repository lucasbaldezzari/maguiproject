
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split

from TrialsHandler.TrialsHandler import TrialsHandler
from TrialsHandler.Concatenate import Concatenate


import pickle
import os

### ********** Cargamos los datos **********
sujeto = "sujeto_11" #4 no, 5 no
tipoTarea = "imaginado" #imaginado
ct = 0 if tipoTarea == "ejecutado" else 1 #0 ejecutado, 1 imaginado
comb = 4
r = 1
best_classi = "svc" #lda o svc

baseFolder = f"data\{sujeto}"
eventosFile = f"{baseFolder}\eegdata\sesion1\sn1_ts0_ct{ct}_r{r}_events.txt"
file = f"{baseFolder}\eegdata\sesion1\sn1_ts0_ct{ct}_r{r}.npy"
rawEEG_1 = np.load(file)
eventos_1 = pd.read_csv(eventosFile, sep = ",")

eventosFile = f"{baseFolder}\eegdata\sesion2\sn2_ts0_ct{ct}_r{r}_events.txt"
file = f"{baseFolder}\eegdata\sesion2\sn2_ts0_ct{ct}_r{r}.npy"
rawEEG_2 = np.load(file)
eventos_2 = pd.read_csv(eventosFile, sep = ",")


#Creamos objetos para manejar los trials
th_1 = TrialsHandler(rawEEG_1, eventos_1, tinit = 0, tmax = 4, reject=None, sample_rate=250., trialsToRemove = [])
th_2 = TrialsHandler(rawEEG_2, eventos_2, tinit = 0, tmax = 4, reject=None, sample_rate=250., trialsToRemove = [])

dataConcatenada = Concatenate([th_1,th_2])#concatenamos datos

channelsSelected = [0,1,2,3,6,7]

trials = dataConcatenada.trials

#me quedo con channelsSelected
trials = trials[:,channelsSelected,:]
labels = dataConcatenada.labels
classesName, labelsNames = dataConcatenada.classesName

comb1 = np.where((labels == 1) | (labels == 2))
comb2 = np.where((labels == 1) | (labels == 2) | (labels == 5))
comb3 = np.where((labels == 1) | (labels == 2) | (labels == 4) | (labels == 5))
comb4 = np.where((labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 5))

comb_labels1 = [1,2]
comb_labels2 = [1,2,5]
comb_labels3 = [1,2,4,5]
comb_labels4 = [1,2,3,4,5]

combs = [comb1, comb2, comb3, comb4]
comb_labels = [comb_labels1, comb_labels2, comb_labels3, comb_labels4]

#filtramos los trials para las clases que nos interesan
trials = trials[combs [comb-1]]
labels = labels[combs [comb-1]]

### ********** Separamos los datos en train, validation y test **********
eeg_train, eeg_test, labels_train, labels_test = train_test_split(trials, labels, test_size=0.1, stratify=labels, random_state=42)


### ********** GUARDAMOS DATOS **********

pipsFolder = "pipelines" #carpeta donde guardaremos los pipelines

## cargamos el pipeline desde f"{baseFolder}\{pipsFolder}\\best_lda_{tipoTarea}_comb{comb}.pkl"
with open(f"{baseFolder}\{pipsFolder}\\best_{best_classi}_{tipoTarea}_comb{comb}.pkl", 'rb') as file:
    best_pipe = pickle.load(file)

##importamos cross_val_predict
from sklearn.model_selection import cross_val_predict

## usamos cross_vall_predict para obtener las probabilidades de las predicciones
y_scores = cross_val_predict(best_pipe, eeg_train, labels_train, cv=3, method="predict")

plotsFolder = "plots" #carpeta donde guardaremos los dataframes
## chequeamos si la carpeta f"{baseFolder}\{dfFolder}" existe, si no existe la creamos
if not os.path.exists(f"{baseFolder}\{plotsFolder}"):
    os.makedirs(f"{baseFolder}\{plotsFolder}")

fig, ax = plt.subplots(figsize=(10, 7))
ConfusionMatrixDisplay.from_predictions(labels_train, y_scores, cmap="YlGn", normalize="true",
                                        ax = ax, values_format=".0%")
##quitamos los nombres de los ejes x e y
ax.set_xlabel("")
ax.set_ylabel("")
##tamaño de los xticks e yticks
ax.tick_params(axis='both', which='major', labelsize=12)
plt.title(f"MC de {sujeto} - C{comb} - {best_classi.upper()} - {tipoTarea}", fontsize=15)
filename = f"{baseFolder}\\{plotsFolder}\\cm_{best_classi}_{tipoTarea}_comb{comb}.png"
plt.savefig(filename, dpi=300)
# plt.show()

##guardamos los valores de la matriz de confusión
cm = ConfusionMatrixDisplay.from_predictions(labels_train, y_scores, cmap="Greens", normalize="true",
                                        ax = ax, values_format=".0%").confusion_matrix

##guardamos la matriz de confusión en un dataframe
df = pd.DataFrame(cm, index = comb_labels[comb-1], columns = comb_labels[comb-1])
df.to_csv(f"{baseFolder}\{plotsFolder}\\cm_{best_classi}_{tipoTarea}_comb{comb}.csv", sep = ";", decimal = ",")
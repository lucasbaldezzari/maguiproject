import numpy as np
import pandas as pd

from TrialsHandler.TrialsHandler import TrialsHandler
from TrialsHandler.Concatenate import Concatenate

import seaborn as sns

from SignalProcessor.Filter import Filter
from SignalProcessor.CSPMulticlass import CSPMulticlass
from SignalProcessor.FeatureExtractor import FeatureExtractor
from SignalProcessor.RavelTransformer import RavelTransformer

import matplotlib.pyplot as plt
    
## Clasificadores LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

## Librerias para entrenar y evaluar el modelo
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import pickle
import os

sujeto = f"sujeto_{11}" #4 no, 5 no
tipoTarea = "imaginado" #imaginado
ct = 0 if tipoTarea == "ejecutado" else 1 #0 ejecutado, 1 imaginado
comb = 4
r = 1

nrows = 4#"auto" ## auto para comb 1 y 2, 3 filas x 4 columnas para comb 3.... y 5 filas x f columnas para comb4
ncols = 3#"auto"

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
th_1 = TrialsHandler(rawEEG_1, eventos_1, tinit = 0.5, tmax = 4, reject=None, sample_rate=250., trialsToRemove = [])
th_2 = TrialsHandler(rawEEG_2, eventos_2, tinit = 0.5, tmax = 4, reject=None, sample_rate=250., trialsToRemove = [])

dataConcatenada = Concatenate([th_1,th_2])#concatenamos datos

channelsSelected = [0,1,2,3,4,5,6,7]

trials = dataConcatenada.trials

#me quedo con channelsSelected
trials = trials[:,channelsSelected,:]
labels = dataConcatenada.labels
classesName, labelsNames = dataConcatenada.classesName

comb1 = np.where((labels == 1) | (labels == 2))
comb2 = np.where((labels == 1) | (labels == 2) | (labels == 5))
comb3 = np.where((labels == 1) | (labels == 2) | (labels == 4) | (labels == 5))
comb4 = np.where((labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 5))

combs = [comb1, comb2, comb3, comb4]

#filtramos los trials para las clases que nos interesan
trials = trials[combs [comb-1]]
labels = labels[combs [comb-1]]

### ********** Analizamos los datos **********
##calculo la varianza de cada canal para cada trial
filter = Filter(lowcut=8, highcut=12, notch_freq=50.0, notch_width=2, sample_rate=250.,
                axisToCompute=2, padlen=None, order=4)

trials_filtered = filter.fit_transform(trials) #filtramos los trials

##calculo la varianza de cada canal para cada trial
se = np.std(trials_filtered, axis=2) ##VARIANZA

##genero un dataframe con la varianza de cada canal para cada trial
df_se = pd.DataFrame(se) 

##las columnas del dataframe son los channelsSelected + 1
df_se.columns = [f"ch{i+1}" for i in channelsSelected]
df_se["labels"] = labels

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=df_se.iloc[:,:-1], orient="h", palette="Blues", whis=[0,95], linewidth=1)
plt.title(f"Varianza de cada canal para todos los trials - Sesión '{tipoTarea}'", fontsize=16)
plt.xlabel("Varianza", fontsize=16)
plt.ylabel("Canal", fontsize=16)
plt.show()

## eliminamos las columnas ch5 y ch6
df_se = df_se.drop(["ch5", "ch6"], axis=1)

##genero una figure de 2x3 y a cada ax pongo un boxplot de la varianza de cada canal para cada clase
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
c = 0
for i in range(2):
    for j in range(3):
        sns.boxplot(ax=axs[i,j], data=df_se, x = "labels", y=df_se.columns[c], palette="Blues", whis=[0,95], linewidth=1)
        axs[i,j].set_title(f"Canal {df_se.columns[c][-1]}", fontsize=12)
        axs[i,j].set_xlabel("Clase", fontsize=10)
        axs[i,j].set_ylabel("Varianza", fontsize=10)
        c += 1
##agrego titulo general al gráfico
fig.suptitle(f"Varianza de cada canal para cada clase - Sesión '{tipoTarea}'", fontsize=14)
plt.show()

# ##gráfico de densidad de la varianza de cada canal para cada clase
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
# c = 0
# for i in range(2):
#     for j in range(3):
#         sns.kdeplot(ax=axs[i,j], data=df_se, x=df_se.columns[c], hue="labels", palette="Accent", fill=True)
#         axs[i,j].set_title(f"Canal {df_se.columns[c][-1]}", fontsize=11)
#         axs[i,j].set_xlabel("Varianza", fontsize=10)
#         axs[i,j].set_ylabel("Densidad", fontsize=10)
#         c += 1
# plt.show()

### ELIMINANDO TRIALS CON VARANZA MUY ALTA

##calculo el percentil 95 de la varianza de cada canal para cada trial
se = se[:,[0,1,2,3,6,7]]

q=np.percentile(se, q=95)
bad_trials = []
for i in range(len(se)):
    if np.any(se[i]>q):
        bad_trials.append(i)

#bad_trials (imaginado) = [13, 15, 17, 25, 26, 28, 30, 31, 38, 45, 46, 48, 50, 52, 56, 59, 70, 73, 74, 84, 88, 95, 101, 102, 105, 117]
#bad_trials (ejecutado) = [5, 7, 9, 10, 14, 17, 22, 24, 27, 33, 35, 50, 60, 64, 66, 73, 91, 97, 101, 102, 103, 120, 121, 124, 138, 140, 149]
labels[9:11]
##elimino bad_trials de trials y labels
trials = np.delete(trials, bad_trials, axis=0)
labels = np.delete(labels, bad_trials, axis=0)


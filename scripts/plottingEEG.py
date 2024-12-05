import numpy as np
import pandas as pd
from SignalProcessor.Filter import Filter
from EEGPlotter.EEGPlotter import EEGPlotter
from TrialsHandler.TrialsHandler import TrialsHandler
from SignalProcessor.CSPMulticlass import CSPMulticlass

fm = 250.
filtro = Filter(8, 12, 50, 2, fm, 1, padlen=None, order=4)

## cargamos archivos
file = "data\sujeto_11\eegdata\sesion1\sn1_ts0_ct1_r1.npy"
rawEEG = np.load(file)

##slected channels
selected_channels = np.array([1,2,3,4,7,8]) -1 
rawEEG = rawEEG[selected_channels,:]

eeg = filtro.fit_transform(rawEEG)

eventosFile = "data\sujeto_11\eegdata\sesion1\sn1_ts0_ct1_r1_events.txt"
eventos = pd.read_csv(eventosFile, sep = ",")

##convertimos la coluna eventos["trialTime(legible)"][0] a datetime
tiempos = pd.to_datetime(eventos["trialTime(legible)"])

#nos quedamos con los minutos y segundos y se lo vamos sumando a medida que recorremos los trials
tiempos = tiempos.dt.minute*60 + tiempos.dt.second
tiempos = tiempos - tiempos[0]

tinit = 0.
tmax = 4.

## Instanciamos la clase TrialsHandler para extraer los trials, labels, nombre de clases, etc.
trialhandler = TrialsHandler(rawEEG, eventos,
                             tinit = tinit, tmax = tmax,
                             reject=None, sample_rate=fm,
                             trialsToRemove=None)

labels = trialhandler.labels
#agregamos un número creciente de 1 al largo de lables delante de cada label
labels = [str(i)+"-"+"C"+str(labels[i-1]) for i in range(1,len(labels)+1)]

ini_trial = 12
final_trial = 14
trial_duration = 10

ti = int(ini_trial*trial_duration*fm)
tf = int(final_trial*trial_duration*fm)

trials_times = tiempos.values[ini_trial:final_trial] - tiempos.values[ini_trial]

paso = 2 #segundos
window_size = 10 #segundos
eeg_plotter = EEGPlotter(eeg[:,ti:tf], fm, paso, window_size,task_window = (3,4),
                         labels = labels[ini_trial:final_trial], trials_start = trials_times)

eeg_plotter.plot()



# cspmulticlass = CSPMulticlass(n_components=6, method = "ova", n_classes = len(np.unique(labels)),
#                                     reg = 0.01, transform_into = "csp_space")

# trials = trialhandler.trials

# #me quedo con los trials donde trialhandler.labels sean 1 o 2
# selected_trials = np.array([trial for trial, label in zip(trialhandler.trials, trialhandler.labels) if label in [1, 2]])
# labels = np.array([label for trial, label in zip(trialhandler.trials, trialhandler.labels) if label in [1, 2]])

# cspmulticlass.fit(selected_trials, labels)

# eeg_trials_csp = cspmulticlass.transform(selected_trials)

# eeg_trials_csp.shape

# ##eeg_trials_csp tiene shape (n_trials, n_components x n_classes, n_samples). 
# eeg_trials_csp_comp1 = eeg_trials_csp[:,0:450:6,:]
# eeg_trials_csp_comp2 = eeg_trials_csp[:,1:450:6,:]
# eeg_trials_csp_comp3 = eeg_trials_csp[:,2:450:6,:]
# eeg_trials_csp_comp4 = eeg_trials_csp[:,3:450:6,:]
# eeg_trials_csp_comp5 = eeg_trials_csp[:,4:450:6,:]
# eeg_trials_csp_comp6 = eeg_trials_csp[:,5:450:6,:]

##concatenamos eeg_trials_csp_comp1 de tal forma de pasar 
450/6
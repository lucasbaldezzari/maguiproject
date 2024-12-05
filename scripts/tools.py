import numpy as np
import scipy.io
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from mne import create_info
from mne.io import RawArray

"""Nota, las funcones get_names, prepareData y getTrials se utilizan para trabajar con los datos de la competencia de BCI IV
Para más información: https://bbci.de/competition/iv/desc_1.html"""

def get_names(path):
    import os
    names = []
    for file in os.listdir(path):
        if file.endswith(".mat"):
            names.append(file)
    return names

def prepareData(rawData, path = "dataset/"):
    """El argumento datos contiene una lista de diccionarios. Cada diccionario contiene los EEG e información referente al registro.
    La función devuelve un numpy array con todos los EEGs y un diccionario que contiene diferente información que se utilizará para 
    análisis de los registros de EEG"""

    dataReady = dict()

    for i, data in enumerate(rawData):
        sample_rate = data['nfo']['fs'][0][0][0]
        EEG = data['cnt'].T
        nchannels, nsamples = EEG.shape

        channel_names = [s[0] for s in data['nfo']['clab'][0][0][0]]
        event_starting = data['mrk'][0][0][0]
        event_codes = data['mrk'][0][0][1]
        labels = np.zeros((1, nsamples), int)
        labels[0, event_starting] = event_codes

        cl_lab = [s[0] for s in data['nfo']['classes'][0][0][0]]
        cl1 = cl_lab[0]
        cl2 = cl_lab[1]
        nclasses = len(cl_lab)
        nevents = len(event_starting)

        xpos = data["nfo"]["xpos"][0,0]
        ypos = data["nfo"]["ypos"][0,0]

        dataReady[f"subject{i+1}"] = {
            "eeg": EEG,
            "sample_rate": sample_rate,
            "nchannels": nchannels,
            "nsamples": nsamples,
            "channelsNames": channel_names,
            "event_starting": event_starting,
            "event_codes": event_codes,
            "labels": labels,
            "class1": cl1,
            "class2": cl2,
            "nclasses": nclasses,
            "nevents": nevents,
            "xpos": xpos,
            "ypos": ypos,
        } 

    return dataReady

def getTrials(EEG, cl_lab, event_codes, event_onsets, nchannels,
              w1 = 0.5, w2 = 2.5, sample_rate = 100.0):
    """Obtenemos los trials a partir del EEG
    
    - EEE: numpy array con la señal EEG en la forma [channels, samples]
    - cl_lab: lista con los nombres de las clases
    - event_codes: numpy array con los códigos de los eventos
    - event_onsets: numpy array con los comienzos de los eventos
    - nchannels: número de canales
    - w1: tiempo en segundos antes del evento
    - w2: tiempo en segundos después del evento
    - sample_rate: frecuencia de muestreo de la señal EEG"""

    # Diccionario con los datos de los registros EEG
    trials = {}

    #La ventana de tiempo se define en segundos. En este caso, 0.5 segundos antes del evento y 2.5 segundos después del evento
    win = np.arange(int(w1*sample_rate), int(w2*sample_rate))

    # Length of the time window
    nsamples = len(win)

    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        
        # Extraemos los comienzos de los eventos de la clase cl
        cl_onsets = event_onsets[event_codes == code]
        
        # Guardamos memoria para los trials
        trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
        
        # Extraemos cada trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:,:,i] = EEG[:, win+onset]

    #changing the order of numpy in the way [trials, channels, samples]    
    for clase in trials.keys():
        trials[clase] = trials[clase].swapaxes(1, 2).swapaxes(0, 1)

    #los datos dentro de trials son de la forma [trials, channels, samples]
    return trials

def plot_signals(signals, title, legend):
    """"Grafica una lista de señales
    - signals: lista de señales con numpy arrays de la forma [samples]
    - title: título del gráfico
    - legend: lista con los nombres de las señales o leyenda que se desea mostrar en el gráfico"""
    plt.figure()
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    for signal in signals:
        plt.plot(signal)
    plt.legend(legend)
    plt.show()


def fromNpyToFif(raweeg, channels_names, filename = "eeginfif.fif", sf = 250., chan_types="eeg"):
    """Genera archivos del tipo fif que luego pueden ser procesados utilizando MNE
    - raweeg: numpyarray con los datos de eeg de la forma [n_channels, n_samples]
    - channels_names: lista de strings con los nombres de canales
    - sf: frecuencia de muestreo. Por defecto es 250.
    - chan_types: string con el tipo de canal a almacenar. Por defecto es "eeg"

    Ejemplo
    channel_names = ['Fp1', 'Fp2', 'C3', 'F4', 'C3', 'C4', 'P3', 'P4']
    """
    ch_types = ch_types*len(channels_names) #["eeg"]*n_channels
    info = create_info(ch_names = channels_names, sfreq = sf, ch_types = chan_types)
    
    rawmne = RawArray(raweeg.T, info)

    rawmne.save(filename, overwrite = True)

    return rawmne
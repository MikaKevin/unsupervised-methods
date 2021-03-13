
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.signal import butter, lfilter, kaiserord, firwin, freqz, iirnotch
from scipy.fftpack import fft, fftfreq
from math import pi
from scipy.signal import find_peaks
from sklearn.decomposition import FastICA


fs = 360  # sampling frecuency
# --------------------------------load Data-------------------------------------------
def get_samples(filename):
    mat = loadmat(filename)
    data = mat['val']
    samples1 = np.array(data[0, 0, :])
    samples2 = np.array(data[0, 1, :])
    samples3 = np.array(data[0, 2, :])
    samples4 = np.array(data[0, 3, :])
    num = data.shape[2]
    return num, samples1, samples2, samples3, samples4

# ------------------------------Visualizing raw data Samples--------------------------
def show_signals(filename, s_f):

    number_samples, samples1, samples2, samples3, samples4 = get_samples(
        filename)
    duration = number_samples / s_f
    time = np.linspace(0, duration, number_samples)
    fig, ax = plt.subplots(4, 1, squeeze=False, figsize=(15, 15))

    ax[0][0].plot(time, samples1)
    ax[0][0].set_title('Electrode one', fontsize=20)
    ax[0][0].set_ylabel('Amplitute [uV]', fontsize=20)

    ax[1][0].plot(time, samples2)
    ax[1][0].set_title('Electrode two', fontsize=20)
    ax[1][0].set_ylabel('Amplitute [uV]', fontsize=20)

    ax[2][0].plot(time, samples3)
    ax[2][0].set_title('Electrode three', fontsize=20)
    ax[2][0].set_ylabel('Amplitute [uV]', fontsize=20)

    ax[3][0].plot(time, samples4)
    ax[3][0].set_title('Electrode four', fontsize=20)
    ax[3][0].set_xlabel('time[s]', fontsize=20)
    ax[3][0].set_ylabel('Amplitute [uV]', fontsize=20)
    plt.show()
filename = '000.mat'
show_signals(filename, fs)
#----------------------- unmixing the signals using  FastICA------------------------ 
def unmixing_signals(filename , s_f):
    number_samples, samples1, samples2, samples3, samples4 = get_samples(filename)
    mixedsignal = np.array([samples1, samples2, samples3, samples4])
    ica = FastICA(n_components=5)
    recunSignals = ica.fit_transform(mixedsignal.transpose())
    return recunSignals , number_samples

recunSignl , num_samples = unmixing_signals(filename,fs)
print(recunSignl.shape)
duration = num_samples / fs
time = np.linspace(0, duration, num_samples)
fig, ax = plt.subplots(4, 1, squeeze=False, figsize=(30, 15))
ax[0][0].plot(time, recunSignl[:, 0])
ax[0][0].set_title('recunSignal1', fontsize=20)
ax[0][0].set_ylabel('Amplitute [uV]', fontsize=20)

ax[1][0].plot(time, recunSignl[:, 1])
ax[1][0].set_title('recunSignal2', fontsize=20)
ax[1][0].set_ylabel('Amplitute [uV]', fontsize=20)

ax[2][0].plot(time, recunSignl[:, 2])
ax[2][0].set_title('recunSignal3', fontsize=20)
ax[2][0].set_ylabel('Amplitute [uV]', fontsize=20)

ax[3][0].plot(time, recunSignl[:, 3])
ax[3][0].set_title('recunSignal4', fontsize=20)
ax[3][0].set_xlabel('time[s]', fontsize=20)
ax[3][0].set_ylabel('Amplitute [uV]', fontsize=20)

plt.show()


i = 0
max = np.array([])

while i<4:
    max_ind = 0
    max_ind = np.argmax(abs(recunSignl[:,i]))
    max = np.append(max,recunSignl[max_ind,i])
    i = i+1
i = 0
while i<4:
    if max[i]< 0:
        recunSignl[:,i] = -recunSignl[:,i]
        max[i] = -max[i]
    i = i+1

whereIsSignal = np.array([])
i = 0
while i<2:
    whereIsSignal = np.append(whereIsSignal,np.argmax(max))
    max[np.argmax(max)] = 0
    i = i+1
whereIsSignal = whereIsSignal.astype(int)
Signal = np.array([recunSignl[:,whereIsSignal[0]],recunSignl[:,whereIsSignal[1]]])
fig = plt.figure()
plt.plot(time,Signal[0,:]);
plt.plot(time,Signal[1,:]);
plt.show()
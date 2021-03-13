# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:01:38 2020

@author: User
"""
import scipy.io
import numpy as np
from detecta import detect_peaks
import matplotlib.pyplot as plt
sampledata = scipy.io.loadmat('sample_1.mat',mat_dtype = True,squeeze_me=True)
samplingInterval = sampledata["samplingInterval"]
sf = 24*10**3
data = sampledata["data"]

%matplotlib inline

detect_peaks(x, mph=0, mpd=80, show=True, threshhold = np.mean(np.abs(data)))

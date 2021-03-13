import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import FastICA
import scipy.io
import pandas as pd

sampledata = scipy.io.loadmat('002.mat',mat_dtype = True,squeeze_me=True)
data = pd.DataFrame(data=sampledata["val"])
Electrode1 = np.array([data.iloc[0]])
Electrode1data = np.array([])
Electrode1data = np.append(Electrode1data,Electrode1)
Electrode2 = data.iloc[1]
Electrode2data = np.array([])
Electrode2data = np.append(Electrode2data,Electrode2)

Electrode3 = data.iloc[2]
Electrode3data = np.array([])
Electrode3data = np.append(Electrode3data,Electrode3)

Electrode4 = data.iloc[3]
Electrode4data = np.array([])
Electrode4data = np.append(Electrode4data,Electrode4)

x = np.arange(0,len(Electrode4),1)


test = np.array([Electrode1data,Electrode2data,Electrode3data,Electrode4data])


ica = FastICA()
S_test = ica.fit(test.transpose()).transform(test.transpose())
A_ = ica.mixing_  # Get estimated mixing matrix  # Reconstruct signals


plt.plot(x,S_test[:,3])

plt.tight_layout()
plt.show()
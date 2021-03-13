import pandas as pd
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#----------------------------LOAD DATA--------------------------------------------
sampledata = scipy.io.loadmat('sample_1.mat',mat_dtype = True,squeeze_me=True)
samplingInterval = sampledata["samplingInterval"]
sf = 24*10**3
data = sampledata["data"]
#----------------------------SEARCH PEAKS ----------------------------------------
def get_spikes(data, spike_window=80, tf=5, offset=10, max_thresh=350):

    # Calculate threshold based on data mean
    thresh = np.mean(np.abs(data)) *tf

    # Find positions wherere the threshold is crossed
    pos = np.where(data > thresh)[0]
    pos = pos[pos > spike_window]

    # Extract potential spikes and align them to the maximum
    spike_samp = []
    wave_form = np.empty([1, spike_window*2])
    for i in pos:
        if i < data.shape[0] - (spike_window+1):
            # Data from position where threshold is crossed to end of window
            tmp_waveform = data[i:i+spike_window*2]

            # Check if data in window is below upper threshold (artifact rejection)
            if np.max(tmp_waveform) < max_thresh:
                # Find sample with maximum data point in window
                tmp_samp = np.argmax(tmp_waveform) +i

                # Re-center window on maximum sample and shift it by offset
                tmp_waveform = data[tmp_samp-(spike_window-offset):tmp_samp+(spike_window+offset)]

                # Append data
                spike_samp = np.append(spike_samp, tmp_samp)
                wave_form = np.append(wave_form, tmp_waveform.reshape(1, spike_window*2), axis=0)
        i = i + spike_window*2
    # Remove duplicates
    ind = np.where(np.diff(spike_samp) <= 1)[0]
    spike_samp = spike_samp[ind]
    wave_form = wave_form[ind]

    return spike_samp, wave_form

spike_samp, wave_form = get_spikes(data, spike_window=50, tf=8, offset=20)

spike_mean = np.array([])
spike_std = np.array([])
spike_max = np.array([])
spike_low = np.array([])
spike_dis = np.array([])
spike_number = np.array([])

def extremadistance(x):
    ind_max=np.argmax(x)
    ind_min=np.argmin(x)
    distance = ind_max-ind_min
    return distance

zaehler= 0
for i in wave_form:
     spike_mean = np.append(spike_mean,np.mean(i))
     spike_std = np.append(spike_std,np.std(i))
     spike_max = np.append(spike_max,np.amax(i))
     spike_low = np.append(spike_low,np.amin(i))
     spike_dis = np.append(spike_dis,extremadistance(i))
     spike_number = np.append(spike_number,zaehler)
     zaehler =zaehler + 1

np.random.seed(10)

fig, ax = plt.subplots(figsize=(15, 5))

for i in range(100):
    spike = np.random.randint(0, wave_form.shape[0])
    ax.plot(wave_form[spike, :])

ax.set_xlim([0, 90])
ax.set_ylabel('Data', fontsize=20)
ax.set_title('spike waveforms', fontsize=23)
plt.show()
# ----------------- Dimensionality Reduction (PCA) ----------------------------
import sklearn as sk
from sklearn.decomposition import PCA
test_data = np.array([spike_mean,spike_std,spike_max,spike_low,spike_dis])
test_data = np.transpose(test_data) 
# Apply min-max scaling
scaler= sk.preprocessing.MinMaxScaler()
dataset_scaled = scaler.fit_transform(test_data)

# Finds explained variance
pca = PCA().fit(dataset_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Explained variance')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

pca = PCA(n_components=2)
pca_result = pca.fit_transform(dataset_scaled)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(pca_result[:, 0], pca_result[:, 1])
ax.set_xlabel('1st principal component', fontsize=20)
ax.set_ylabel('2nd principal component', fontsize=20)

fig.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

k_range = range(1,8)
sse = []                        # Sum of Squared Errors

for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(dataset_scaled)
    sse.append(km.inertia_)
    
plt.plot(k_range, sse)          # Elbow graph
plt.xlabel('K')
plt.ylabel('Sum of Squared Errors')
plt.show()

n_clusters=3

kmeanModel = KMeans(n_clusters)
kmeanModel.fit(pca_result)
data_predict = kmeanModel.predict(pca_result)
centers = kmeanModel.cluster_centers_
    
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data_predict,alpha=0.2)

cluster1 = []
cluster2 = []

for i in data_predict:
    if i == 0:
        cluster1.append(pca_result[i,:])
    if i == 1:
        cluster2.append(pca_result[i,:])

orig_centers =np.transpose(pca.inverse_transform(centers))



data =np.transpose(np.array([spike_mean,spike_std,spike_max,spike_low,spike_dis]))
spike_class = []
df = pd.DataFrame(data=wave_form)

kmeanModel = KMeans(3,np.transpose(orig_centers))
kmeanModel.fit(data)
orig_data_pred = kmeanModel.predict(data)
final_cluster1 = pd.DataFrame()
final_cluster2 = pd.DataFrame()
final_cluster3 = pd.DataFrame()
i = 0
while i<len(orig_data_pred):
    if orig_data_pred[i] == 0:
        final_cluster1 = final_cluster1.append(df[i:i+1])
    if orig_data_pred[i] == 1:
        final_cluster2 = final_cluster2.append(df[i:i+1])
    if orig_data_pred[i] == 2:
        final_cluster3 = final_cluster3.append(df[i:i+1])
    i=i+1

final_cluster1_mean = final_cluster1.mean(axis=0)
final_cluster1_std = final_cluster1.std(axis=0)

final_cluster2_mean = final_cluster2.mean(axis=0)
final_cluster2_std = final_cluster2.std(axis=0)

final_cluster3_mean = final_cluster3.mean(axis=0)
final_cluster3_std = final_cluster3.std(axis=0)

final_cluster1_mean = final_cluster1_mean.to_numpy()
final_cluster2_mean = final_cluster2_mean.to_numpy()
final_cluster3_mean = final_cluster3_mean.to_numpy()
final_cluster1_std = final_cluster1_std.to_numpy()
final_cluster2_std = final_cluster2_std.to_numpy()
final_cluster3_std = final_cluster3_std.to_numpy()


i = np.arange(0,100,1)
plt.plot(i,final_cluster1_mean,"b",label="Cluster 1")
plt.plot(i,final_cluster2_mean,"g",label="Cluster 2")
plt.plot(i,final_cluster3_mean,"r",label="Cluster 3")
plt.fill_between(i,final_cluster1_mean-final_cluster1_std,final_cluster1_mean+final_cluster1_std,alpha=0.2,color="blue")
plt.fill_between(i,final_cluster2_mean-final_cluster2_std,final_cluster2_mean+final_cluster2_std,alpha=0.2,color="green")
plt.fill_between(i,final_cluster3_mean-final_cluster3_std,final_cluster3_mean+final_cluster3_std,alpha=0.2,color="red")
plt.grid()
plt.legend()
plt.show()



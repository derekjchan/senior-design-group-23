#%%
import numpy as np
from pylab import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, datasets
from scipy.integrate import simps
from mne.connectivity import spectral_connectivity
from scipy import signal
import os
import glob
from tqdm.notebook import tqdm
# %%
def get_node_features(data_seg):
    # set the sampling frequency
    sf = 128
    #set the window size
    win = data_seg[1].size
    #find the psd using welch's periodogram
    freqs, psd = signal.welch(data_seg, sf, nperseg=win)
    freq_res = freqs[1] - freqs[0]
    #delta, theta, alpha, beta frequency bands
    band_freqs = [0.5, 4, 7.5, 13, 16, 30, 40]
    all_channel_feats = []
    count = 0
    #goes through each channel (node) and creates an length 4 array of the relative power in each
    #frequency band
    #Then it appends each array to a larger array consisting of the 30 node features
    for channel in psd:
        channel_feats = []
        for band in range(len(band_freqs) - 1):
            low, high = band_freqs[band], band_freqs[band + 1]
            curr_idx = np.logical_and(freqs >= low, freqs <= high)

            band_power = simps(channel[curr_idx], dx=freq_res)
            total_power = simps(channel, dx = freq_res)
            rel_power = band_power / total_power
            channel_feats.append(rel_power)
        all_channel_feats.append(channel_feats)
    #converts to list to np array and then returns a pytorch tensor
    all_channel_feats = np.asarray(all_channel_feats)
    return all_channel_feats


C = 1.0
# %%
raw_file_names = [os.path.basename(x) for x in glob.glob("data/valid_data/*")]
raw_event_filenames = [os.path.basename(x) for x in glob.glob("data/valid_event_data/*")]

X = []
y = []
for raw_file, event_file in zip(raw_file_names, raw_event_filenames):
    with open(os.path.join('data/valid_data/', raw_file)) as f:
        lines = (line for line in f if not line.startswith('T'))
        FH = np.loadtxt(lines)

    with open(os.path.join('data/valid_event_data/', event_file)) as f:
        lines = (line for line in f if not line.startswith('n'))
        ED = np.loadtxt(lines)

    rt_list = []
    init_i_list = []
    for idx in tqdm(range(len(ED)-1)):
        curr_row = ED[idx]
        next_row = ED[idx+1]
        if round(curr_row[2]) == 252 or round(curr_row[2]) == 251:
            dev_on = curr_row[5]
            response_on = next_row[5]
            rt = response_on - dev_on
            init_index = curr_row[3]
            rt_list.append(rt)
            init_i_list.append(init_index)
    alert_rt = np.percentile(rt_list, 5)
    alertness = alert_rt*1.5
    drowsiness = alert_rt*2.5

    start_idx = 0
    while init_i_list[start_idx] < 500*90:
        start_idx += 1

    global_rts = []

    for idx in range(start_idx + 2):
        temp_rts = []
        for rt in range(idx+1):
            temp_rts.append(rt_list[rt])
        global_rts.append(mean(temp_rts))

    for idx in range(start_idx + 2, len(init_i_list)):
        i = 0
        temp_rts = []
        while init_i_list[idx] - init_i_list[idx - i] < 500*90:
            temp_rts.append(rt_list[idx - i])
            i += 1
        global_rts.append(mean(temp_rts))
    FH_t = FH.transpose()
    FH_t = FH_t[1:, :]

    alert_list = []
    drowsy_list = []
    for local_rt, global_rt, init_i in zip(rt_list, global_rts, init_i_list):
        if local_rt <= alertness and global_rt <= alertness:
            alert_list.append((local_rt, global_rt))
            print("Init i: " ,init_i)
            data_seg = FH_t[:,int(init_i-128*3):int(init_i)]
            features = get_node_features(data_seg)
            X.append(features)
            y.append(1)
        elif local_rt >= drowsiness and global_rt >= drowsiness:
            drowsy_list.append((local_rt, global_rt))
            print("Init i: ", init_i)
            data_seg = FH_t[:,int(init_i-128*3):int(init_i)]
            feautures = get_node_features(data_seg)
            X.append(features)
            y.append(0)
    print(len(alert_list), len(drowsy_list))
#%%
X_train = []
for element in X:
    flattened = element.flatten()
    X_train.append(flattened)

# %%
svc = svm.SVC(kernel='linear', C=C).fit(X_train[:2500],y[:2500])
# %%
X_test = X_train[2500:]
y_test = y[2500:]

predicted = []
for x,y in zip(X_test, y_test):
    pred = svc.predict(x)
    predicted.append(pred)
# %%
a = svc.predict(X_test)

from sklearn.metrics import mean_absolute_percentage_error, accuracy_score, precision_score, \
            recall_score, roc_auc_score, f1_score, confusion_matrix

acc = accuracy_score(a, y_test)
# %%

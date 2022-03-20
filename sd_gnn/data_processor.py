#%%
import os
import glob
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
from statistics import mean
from tqdm.notebook import tqdm
from mne.connectivity import spectral_connectivity
#%%
raw_file_names = [os.path.basename(x) for x in glob.glob("data/valid_data/*")]
raw_event_filenames = [os.path.basename(x) for x in glob.glob("data/valid_event_data/*")]

# %%
all_alert = []
all_drowsy = []

def get_edge_features(data_seg):
        #need to do this step in order for the spectral_connectivity function to work
        transf_dat = np.expand_dims(data_seg, axis=0)
        #initilaize a numpy array of the final size of the matrix
        spec_coh_matrix = np.zeros((30,29))
        for channel_idx in range(30):
            final_nodes = list(range(30))
            final_nodes.pop(channel_idx)
            #not including self loops
            spec_conn, freqs, times, n_epoch, n_tapers = spectral_connectivity(data=transf_dat, method='coh', 
                indices = ([channel_idx]*29, final_nodes), 
                sfreq = 500, fmin=5, fmax=40.0, faverage=True, verbose=False)
            spec_coh_values = np.squeeze(spec_conn)

            spec_coh_matrix[channel_idx, :] = spec_coh_values
        edge_attr = spec_coh_matrix.flatten()
        #edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_attr = edge_attr.reshape(870,1)
        return edge_attr

#%%

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
    print(FH_t.shape)
    FH_t = FH_t[1:, :]

    print("Global rt length: ", len(global_rts))
    print("Local rt length:", len(rt_list))
    print("Init_i_list length: ", len(init_i_list))
    alert_list = []
    drowsy_list = []
    for local_rt, global_rt, init_i in zip(rt_list, global_rts, init_i_list):
        if local_rt <= alertness and global_rt <= alertness:
            alert_list.append((local_rt, global_rt))
            print("Init i: " ,init_i)
            data_seg = FH_t[:,int(init_i-1500):int(init_i)]
            #a = get_edge_features(data_seg)
        elif local_rt >= drowsiness and global_rt >= drowsiness:
            drowsy_list.append((local_rt, global_rt))
            print("Init i: ", init_i)
            data_seg = FH_t[:,int(init_i-1500):int(init_i)]
            #b = get_edge_features(data_seg)
    print(len(alert_list), len(drowsy_list))

    all_alert.append(alert_list)
    all_drowsy.append(drowsy_list)
# %%
a = np.loadtxt('data/raw/1.txt')
# %%
raw_file_names = [os.path.basename(x) for x in glob.glob("data/good_data_ds/raw/raw_*")]
raw_event_filenames = [os.path.basename(x) for x in glob.glob("data/good_data_ds/raw/events_*")]
# %%

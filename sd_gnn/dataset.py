#%%
from cgi import test
from fileinput import filename
from http.client import NOT_IMPLEMENTED
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy import sparse
import networkx as nx
from scipy.integrate import simps
from mne.connectivity import spectral_connectivity
from tqdm.notebook import tqdm
import warnings
import random
from statistics import mean

class MyOwnDataset(Dataset):
    def __init__(self, root, ts_files, rt_files, feats=False, test=False, transform=None, pre_transform=None, classification=False, cross=False, good_data=False):
        #self.test = test
        self.ts_files = ts_files
        self.rt_files = rt_files
        self.feats = feats
        self.test = test
        self.test_cross = cross
        self.filenames = self.ts_files + self.rt_files
        self.mode = classification
        self.good_data = good_data
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filenames

    @property
    def processed_file_names(self):
        
        #if self.test:
        #    return [f'data_test_{i}.pt' for i in range(643)]
        #elif self.test_cross:
        #    return [f'data_test_{i}.pt' for i in range(1171)]
        #else:
        #    return [f'data_{i}.pt' for i in range(2242)]
            
        return "NOT_IMPLEMENTED"

    def download(self):
        pass

    def process(self):
        #warnings.filterwarnings("ignore")
        if self.good_data:
            name_idx = 0
            self.total_len = 0
            for raw_file, event_file in tqdm(zip(self.ts_files, self.rt_files), desc="Files",total=len(self.ts_files)):
                with open(os.path.join(self.raw_dir, raw_file)) as f:
                    lines = (line for line in f if not line.startswith('T'))
                    FH = np.loadtxt(lines)
                with open(os.path.join(self.raw_dir, event_file)) as f:
                    lines = (line for line in f if not line.startswith('n'))
                    ED = np.loadtxt(lines)
                
                rt_list = []
                init_i_list = []
                for idx in range(len(ED)-1):
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

                #print("Global rt length: ", len(global_rts))
                #print("Local rt length:", len(rt_list))
                FH_t = FH.transpose()
                FH_t = FH_t[1:, :]
                alert_list = []
                drowsy_list = []
                for local_rt, global_rt, init_i in tqdm(zip(rt_list, global_rts, init_i_list), desc="Within",total=len(rt_list)):
                    if local_rt <= alertness and global_rt <= alertness:
                        alert_list.append((local_rt, global_rt))
                        data_seg = FH_t[:,int(init_i - 128*3):int(init_i)]
                        edge_feats = self._get_edge_features(data_seg)
                        edge_index = self._get_adjacency_info(30)
                        if self.feats:
                            node_feats = self._get_raw_feat(data_seg)
                        else:
                            node_feats = self._get_node_features(data_seg)
                        label = np.asarray([1])
                        label = torch.tensor(label, dtype=torch.int64)
                        data = Data(x=node_feats,
                                edge_index=edge_index,
                                edge_attr=edge_feats,
                                y=label)
                        if self.test:
                            torch.save(data,
                                    os.path.join(self.processed_dir,
                                        f'data_test_{name_idx}.pt'))
                        else:
                            torch.save(data,
                                    os.path.join(self.processed_dir,
                                        f'data_{name_idx}.pt'))
                        name_idx += 1

                    elif local_rt >= drowsiness and global_rt >= drowsiness:
                        drowsy_list.append((local_rt, global_rt))
                        data_seg = FH_t[:,int(init_i - 128*3):int(init_i)]
                        edge_feats = self._get_edge_features(data_seg)
                        edge_index = self._get_adjacency_info(30)
                        if self.feats:
                            node_feats = self._get_raw_feat(data_seg)
                        else:
                            node_feats = self._get_node_features(data_seg)
                        label = np.asarray([0])
                        label = torch.tensor(label, dtype=torch.int64)
                        data = Data(x=node_feats,
                                edge_index=edge_index,
                                edge_attr=edge_feats,
                                y=label)
                        if self.test:
                            torch.save(data,
                                    os.path.join(self.processed_dir,
                                        f'data_test_{name_idx}.pt'))
                        else:
                            torch.save(data,
                                    os.path.join(self.processed_dir,
                                        f'data_{name_idx}.pt'))
                        name_idx += 1
            self.total_len = name_idx
        else:
            name_idx = 0
            self.total_len = 0
            rand_list = [0,1]
            for ts_file, rt_file in tqdm(zip(self.ts_files, self.rt_files), desc="Files", total=len(self.ts_files)):
                if self.mode:
                    idx_lim = 2
                    count = 2
                else:
                    idx_lim = 0
                    count = 0
                ts_path = os.path.join(self.raw_dir, ts_file)
                rt_path = os.path.join(self.raw_dir, rt_file)
                self.data = np.loadtxt(ts_path)
                all_labels = self._get_labels(rt_path)
                event_num = 0
                max_idx = len(self.data[0])/1501
                for index in tqdm(range(1, len(self.data[0])), desc="Within file", leave=False):
                    if index%1501 == 0 and event_num >= idx_lim and event_num <= max_idx-idx_lim:
                        data_seg = self.data[:, index-1500:index]
                        edge_feats = self._get_edge_features(data_seg)
                        edge_index = self._get_adjacency_info(30)
                        if self.feats:
                            node_feats = self._get_raw_feat(data_seg)
                        else:
                            node_feats = self._get_node_features(data_seg)

                        if self.mode:
                            label = self._get_class(idx=count, all_labels=all_labels)
                            rand_num = random.choice(rand_list)
                            if label is not None:
                                if label.item() == 1:
                                    data = Data(x=node_feats,
                                                edge_index=edge_index,
                                                edge_attr=edge_feats,
                                                y=label)
                                    if self.test or self.test_cross:
                                        torch.save(data,
                                                os.path.join(self.processed_dir,
                                                    f'data_test_{name_idx}.pt'))
                                    else:
                                        torch.save(data,
                                                os.path.join(self.processed_dir,
                                                    f'data_{name_idx}.pt'))
                                    name_idx += 1
                                elif rand_num:
                                    data = Data(x=node_feats,
                                                edge_index=edge_index,
                                                edge_attr=edge_feats,
                                                y=label)
                                    if self.test or self.test_cross:
                                        torch.save(data,
                                                os.path.join(self.processed_dir,
                                                    f'data_test_{name_idx}.pt'))
                                    else:
                                        torch.save(data,
                                                os.path.join(self.processed_dir,
                                                    f'data_{name_idx}.pt'))
                                    name_idx += 1
                        else:
                            label = all_labels[count]
                            data = Data(x=node_feats,
                                    edge_index=edge_index,
                                    edge_attr=edge_feats,
                                    y=label)
                            if self.test or self.test_cross:
                                torch.save(data,
                                        os.path.join(self.processed_dir,
                                            f'data_test_{name_idx}.pt'))
                            else:
                                torch.save(data,
                                        os.path.join(self.processed_dir,
                                            f'data_{name_idx}.pt'))
                            name_idx += 1

                        count += 1

                    if index%1501 == 0:
                        event_num += 1
            self.total_len = name_idx

    def _get_node_features(self, data_seg):
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
        all_channel_feats = torch.tensor(all_channel_feats, dtype=torch.float)
        return all_channel_feats

    def _get_raw_feat(self, data_seg):
        two_epochs = data_seg[:, 0:128]
        return torch.tensor(two_epochs, dtype=torch.float)

    def _get_edge_features(self, data_seg):
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
                sfreq = 128, fmin=5, fmax=40.0, faverage=True, verbose=False)
            spec_coh_values = np.squeeze(spec_conn)

            spec_coh_matrix[channel_idx, :] = spec_coh_values
        edge_attr = spec_coh_matrix.flatten()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_attr = edge_attr.reshape(870,1)
        return edge_attr

    def _get_adjacency_info(self, num_nodes):
        # Initialize edge index matrix
        E = torch.zeros((2, num_nodes * (num_nodes - 1)), dtype=torch.long)

        # Populate 1st row
        for node in range(num_nodes):
            for neighbor in range(num_nodes - 1):
                E[0, node * (num_nodes - 1) + neighbor] = node

        # Populate 2nd row
        neighbors = []
        for node in range(num_nodes):
            neighbors.append(list(np.arange(node)) + list(np.arange(node+1, num_nodes)))
        E[1, :] = torch.tensor([item for sublist in neighbors for item in sublist], dtype=torch.long)

        return E
        """
        g = nx.complete_graph(num_nodes)
        p = nx.to_pandas_edgelist(g)
        s = list(p.source)
        t = list(p.target)
        tens = torch.tensor([s, t], dtype=torch.long)

        return tens
        """

    def _get_labels(self, path):
        label = np.loadtxt(path)
        label = torch.tensor(label, dtype=torch.float)
        return label

    def _get_class(self, idx, all_labels):
        global_rt = sum(all_labels[idx-2:idx+3])/5
        local_rt = all_labels[idx]
        if global_rt >= 1.5 and local_rt >= 1.5:
            alertness = np.asarray([0])
            #print("class 1")
            return torch.tensor(alertness, dtype=torch.int64)
        elif global_rt <= 0.62 and local_rt <= 0.62:
            alertness = np.asarray([1])
            #print("class 2")
            return torch.tensor(alertness, dtype=torch.int64)
        else:
            return None

    def len(self):
        #col_len = self._get_labels(1).shape
        """
        total_len = 0
        for file in self.rt_files:
            curr_file = os.path.join(self.raw_dir, file)
            label = np.loadtxt(curr_file)
            curr_len = label.shape[0]
            total_len = total_len + curr_len - 4
            """
        #if self.test:
        #    return 643
        #elif self.test_cross:
        #    return 1171
        #else:
        #    return 2242
        return self.total_len#total_len
 
    def get(self, idx):
        if self.test or self.test_cross:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data
# %%

#%%
from cgi import test
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy import sparse
from scipy.integrate import simps
from mne.connectivity import spectral_connectivity



class MyOwnDataset(Dataset):
    def __init__(self, root, test=False, transform=None, pre_transform=None):
        #self.test = test
        #self.filename = filename
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["2.txt", "3.txt", "4.txt", "5.txt", "rt_2.txt"]

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        #if self.test:
        #    return [f'data_test_{i}.pt' for i in list(self.data.index)]
        #else:
        #    return [f'data_{i}.pt' for i in list(self.data.index)]
        return "not_implemented.pt"

    def download(self):
        pass

    def process(self):
        self.data = np.loadtxt(self.raw_paths[0])
        all_labels = self._get_labels(4)
        count = 0
        for index in range(1, len(self.data[0])):
            if index % 1500 == 0:
                data_seg = self.data[:, index-1500:index]
                node_feats = self._get_node_features(data_seg)
                edge_feats = self._get_edge_features(data_seg)
                edge_index = self._get_adjacency_info(30)
                label = all_labels[count]
                count += 1

                data = Data(x=node_feats,
                            edge_index=edge_index,
                            edge_attr=edge_feats,
                            y=label)
                
                torch.save(data,
                        os.path.join(self.processed_dir,
                        f'data_{count}.pt'))

    def _get_node_features(self, data_seg):
        # set the sampling frequency
        sf = 500
        #set the window size
        win = data_seg[1].size
        #find the psd using welch's periodogram
        freqs, psd = signal.welch(data_seg, sf, nperseg=win)
        freq_res = freqs[1] - freqs[0]
        #delta, theta, alpha, beta frequency bands
        band_freqs = [0.5, 4, 8, 12, 30]
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
        return torch.tensor(all_channel_feats, dtype=torch.float)

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
                sfreq = 500, fmin=1.0, fmax=40.0, faverage=True, verbose=False)
            spec_coh_values = np.squeeze(spec_conn)

            spec_coh_matrix[channel_idx, :] = spec_coh_values
        edge_attr = spec_coh_matrix.flatten()
        return torch.tensor(edge_attr, dtype=torch.float)

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
        E[1, :] = torch.Tensor([item for sublist in neighbors for item in sublist])

        return E

    def _get_labels(self, num):
        label = np.loadtxt(self.raw_paths[num])
        return torch.tensor(label, dtype=torch.float64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    
#%%
dataset = MyOwnDataset(root="data/")
# %%
dataset[1]
# %%
print(dataset[1].edge_index)
print(dataset[1].x)
print(dataset[1].edge_attr)
print(dataset[1].y)
# %%
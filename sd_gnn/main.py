from cgi import test
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
from scipy import signal
from scipy import sparse
from scipy.integrate import simps
from mne.connectivity import spectral_connectivity


if __name__ == '__main__':
    class MyOwnDataset(Dataset):
        def __init__(self, root, test=False, transform=None, pre_transform=None):
            #self.test = test
            #self.filename = filename
            super(MyOwnDataset, self).__init__(root, transform, pre_transform)

        @property
        def raw_file_names(self):
            return ["2.txt", "3.txt", "4.txt", "5.txt"]

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
            for index in range(1, len(self.data[0])):
                if index % 1500 == 0:
                    data_seg = self.data[:, index-1500:index]
                    node_feats = self._get_node_features(data_seg)
                    edge_feats = self._get_edge_features(data_seg)



        def _get_node_features(data_seg):
            sf = 500
            win = data_seg[1].size
            freqs, psd = signal.welch(data_seg, sf, nperseg=win)
            freq_res = freqs[1] - freqs[0]
            band_freqs = [0.5, 4, 8, 12, 30]
            all_channel_feats = []

            count = 0

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
            
            all_channel_feats = np.asarray(all_channel_feats)
            return torch.tensor(all_channel_feats, dtype=torch.float)

        def _get_edge_features(data_seg):
            transf_dat = np.expand_dims(data_seg, axis=0)
            spec_coh_matrix = np.zeros((30,29))
            for channel_idx in range(30):
                final_nodes = range(30)
                spec_conn, freqs, times, n_epoch, n_tapers = spectral_connectivity(data=transf_dat, method='coh', indices = ([channel_idx]*29, final_nodes.pop(channel_idx)), 
                sfreq = 500, fmin=1.0, fmax=40.0, faverage=True, verbose=False)
                spec_coh_values = np.squeeze(spec_conn)

                spec_coh_matrix[channel_idx, :] = spec_coh_values
            edge_attr = spec_coh_matrix.flatten()
            return torch.tensor(edge_attr, dtype=torch.float)

        def _get_adjacency_info(self):
            ones_mat = np.ones([30,30])
            id_mat = np.identity(30)
            adjacency_mat = ones_mat - id_mat


            return "not_implemented"
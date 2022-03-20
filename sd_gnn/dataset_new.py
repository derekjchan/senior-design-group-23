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
from math import sqrt
import math


class MyOwnDataset_new(Dataset):
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
        super(MyOwnDataset_new, self).__init__(root, transform, pre_transform)

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
        a = 10
        b = 10
        c = 10

        channel_coords = [(72,18), (72,342), (72,54), (54, 36), (36,0), (54,324), (72, 306), (72,72), (45,63), (18,0), (45,297), (72,288), (72,90), (36,90), (0,0), (36,270), (72,270), (72,118), (45, 117), (18,180), (45, 243), (72,252), (72,126), (54,144), (36,180), (54,216), (72, 234), (72, 162), (72,180), (72, 198)]
        x_coords = []
        y_coords = []
        z_coords = []
        for coord in channel_coords:
            theta, phi = coord
            theta = math.radians(theta)
            phi = math.radians(phi)
            term1 = b**2 * c**2 * math.cos(theta)**2 * math.cos(phi)**2
            term2 = a**2 * c**2 * math.sin(theta)**2 * math.cos(phi)**2
            term3 = a**2 * b**2 * math.sin(phi)**2
            rho = (a*b*c) / sqrt(term1 + term2 + term3)
            x = rho*math.sin(theta)*math.cos(phi)
            y = rho*math.sin(theta)*math.sin(phi)
            z = rho*math.cos(theta)
            #x_coord = (x,y,z)
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

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
                while init_i_list[start_idx] < 128*90:
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
                    while init_i_list[idx] - init_i_list[idx - i] < 128*90:
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
                        unique, edge_index = self._get_adjacency_info(data_seg)

                        # getting the edge indeces right
                        d = np.asarray(edge_index[0])
                        min_el = np.amin(d)
                        d = d - min_el
                        orig = d

                        idx = 0
                        while idx < len(d) - 1:
                            gap_width = 0
                            d_sort = np.sort(d)
                            smallest_el = 0
                            second_smallest_el = 0
                            while gap_width <= 1 and idx < len(d_sort)-1:
                                gap_width = d_sort[idx + 1] - d_sort[idx]
                                idx = idx + 1
                                smallest_el = d_sort[idx-1]
                                second_smallest_el = d_sort[idx]
                                diff = gap_width-1

                            new_list = [a - diff if a > smallest_el else a for a in d]
                            d = new_list

                        e = np.asarray(edge_index[1])
                        min_el = np.amin(e)
                        e = e - min_el
                        orig = e

                        idx = 0
                        while idx < len(e) - 1:
                            gap_width = 0
                            e_sort = np.sort(e)
                            smallest_el = 0
                            second_smallest_el = 0
                            while gap_width <= 1 and idx < len(e_sort)-1:
                                gap_width = e_sort[idx + 1] - e_sort[idx]
                                idx = idx + 1
                                smallest_el = e_sort[idx-1]
                                second_smallest_el = e_sort[idx]
                                diff = gap_width-1

                            new_list = [a - diff if a > smallest_el else a for a in e]
                            e = new_list

                        edge_index = torch.tensor([d, e], dtype=torch.long)

                        
                        edge_feats = self._get_edge_features(edge_index, x_coords, y_coords, z_coords)
                        if self.feats:
                            node_feats = self._get_raw_feat(data_seg)
                        else:
                            node_feats = self._get_node_features(data_seg, unique)
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
                        unique, edge_index = self._get_adjacency_info(data_seg)

                        #getting edge indices right
                        d = np.asarray(edge_index[0])
                        min_el = np.amin(d)
                        d = d - min_el
                        orig = d

                        idx = 0
                        while idx < len(d) - 1:
                            gap_width = 0
                            d_sort = np.sort(d)
                            smallest_el = 0
                            second_smallest_el = 0
                            while gap_width <= 1 and idx < len(d_sort)-1:
                                gap_width = d_sort[idx + 1] - d_sort[idx]
                                idx = idx + 1
                                smallest_el = d_sort[idx-1]
                                second_smallest_el = d_sort[idx]
                                diff = gap_width-1

                            new_list = [a - diff if a > smallest_el else a for a in d]
                            d = new_list

                        e = np.asarray(edge_index[1])
                        min_el = np.amin(e)
                        e = e - min_el
                        orig = e

                        idx = 0
                        while idx < len(e) - 1:
                            gap_width = 0
                            e_sort = np.sort(e)
                            smallest_el = 0
                            second_smallest_el = 0
                            while gap_width <= 1 and idx < len(e_sort)-1:
                                gap_width = e_sort[idx + 1] - e_sort[idx]
                                idx = idx + 1
                                smallest_el = e_sort[idx-1]
                                second_smallest_el = e_sort[idx]
                                diff = gap_width-1

                            new_list = [a - diff if a > smallest_el else a for a in e]
                            e = new_list

                        edge_index = torch.tensor([d, e], dtype=torch.long)

                        edge_feats = self._get_edge_features(edge_index, x_coords, y_coords, z_coords)
                        if self.feats:
                            node_feats = self._get_raw_feat(data_seg)
                        else:
                            node_feats = self._get_node_features(data_seg, unique)
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

    def _get_node_features(self, data_seg, unique):
        new_data = []
        for element in unique:
            new_data.append(data_seg[element])
        new_data = np.asarray(new_data)
        # set the sampling frequency
        sf = 128
        #set the window size
        win = data_seg[1].size
        #find the psd using welch's periodogram
        freqs, psd = signal.welch(new_data, sf, nperseg=win)
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

    def _get_edge_features(self, edge_index, x_coords, y_coords, z_coords):
        first_node = edge_index[0]
        second_node = edge_index[1]

        all_deltas = []

        for node1, node2 in zip(first_node, second_node):
            x,y,z = x_coords[node1], y_coords[node1], z_coords[node1]
            x, y, z = self._get_normal(x,y,z)

            x1, y1, z1 = x_coords[node2], y_coords[node2], z_coords[node2]
            x1, y1, z1 = self._get_normal(x1, y1, z1)
            
            delta = math.acos(np.dot([x,y,z], [x1,y1,z1]))
            all_deltas.append(delta)
        
        length = len(all_deltas)
        all_deltas = torch.tensor(all_deltas, dtype=torch.float)
        all_deltas = all_deltas.reshape(length, 1)
        return all_deltas

    def _get_adjacency_info(self, data_seg):
        #need to do this step in order for the spectral_connectivity function to work
        transf_dat = np.expand_dims(data_seg, axis=0)
        spec_conn, freqs, times, n_epoch, n_tapers = spectral_connectivity(data=transf_dat, method='coh', 
                sfreq = 128, fmin=5, fmax=40.0, faverage=True, verbose=False)
        
        spec_coh_values = np.squeeze(spec_conn)
        spec_coh_mat = np.asarray(spec_coh_values)

        b,c = np.where(spec_coh_mat >= 0.9)
        d = np.concatenate((b,c))
        unique = np.unique(d)

        in_nodes = np.concatenate((b,c))
        out_nodes = np.concatenate((c,b))

        edge_index = torch.tensor([in_nodes, out_nodes], dtype=torch.long)

        return unique, edge_index

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

    def _get_normal(self, x,y,z):
        x_normal = x/10
        y_normal = y/10
        z_normal = z/10
        return x_normal, y_normal, z_normal
# %%
#a = np.loadtxt('data/raw/1.txt')
#data_seg = a[:,:1500]
#unique, edge_index = get_adjacency_info(data_seg)
# %%
"""
d = np.asarray(edge_index[0])
min_el = np.amin(d)
d = d - min_el
orig = d

idx = 0
while idx < len(d) - 1:
    gap_width = 0
    d_sort = np.sort(d)
    smallest_el = 0
    second_smallest_el = 0
    while gap_width <= 1 and idx < len(d_sort)-1:
        gap_width = d_sort[idx + 1] - d_sort[idx]
        idx = idx + 1
        smallest_el = d_sort[idx-1]
        second_smallest_el = d_sort[idx]
        diff = gap_width-1

    new_list = [a - diff if a > smallest_el else a for a in d]
    d = new_list

e = np.asarray(edge_index[1])
min_el = np.amin(e)
e = e - min_el
orig = e

idx = 0
while idx < len(e) - 1:
    gap_width = 0
    e_sort = np.sort(e)
    smallest_el = 0
    second_smallest_el = 0
    while gap_width <= 1 and idx < len(e_sort)-1:
        gap_width = e_sort[idx + 1] - e_sort[idx]
        idx = idx + 1
        smallest_el = e_sort[idx-1]
        second_smallest_el = e_sort[idx]
        diff = gap_width-1

    new_list = [a - diff if a > smallest_el else a for a in e]
    e = new_list

edge_index = torch.tensor([d, e], dtype=torch.long)
"""
# %%

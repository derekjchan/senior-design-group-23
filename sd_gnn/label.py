# %%
import numpy as np
import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy import sparse
from scipy.integrate import simps
from mne.connectivity import spectral_connectivity
# %%
a = np.loadtxt("data/raw/2.txt")

b = a[:, 0:15000]
# %%
def get_node_features(data_seg):
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
        return all_channel_feats
# %%
c = get_node_features(b)
# %%

#%%
import numpy as np

data = np.loadtxt('data/2.txt')
print('done')

#%%

import pandas as pd

data_pd = pd.read_csv('data/2.txt', sep='\t')
print('done')

#%%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)

sf = 500
time = np.arange(data[1].size) / sf

fig, ax = plt.subplots(1, 1, figsize=(12,4))
plt.plot(time, data[1], lw=1.5, color = 'k')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage')
plt.xlim([time.min(), 200])
plt.ylim([-70, 70])
plt.title('EEG')
sns.despine

# %%
from scipy import signal

win = 3 * sf

freqs, psd = signal.welch(data, sf, nperseg=win)

sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8,4))
plt.plot(freqs, psd[6], color='k', lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
plt.xlim([0, 50])
sns.despine()

# %%
import matplotlib.pyplot as plt

sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8,4))

for channel in range(psd.shape[0]):
    plt.plot(freqs, psd[channel])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.xlim([0, 15])
    
plt.show()


# %%
low, high = 0.5, 4

idx_delta = np.logical_and(freqs >= low, freqs <= high)

plt.figure(figsize=(7,4))
plt.plot(freqs, psd, lw=2, color='k')
plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (uV^2 / Hz)')
plt.xlim([0, 10])
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
sns.despine()
# %%
from scipy.integrate import simps

freq_res = freqs[1] - freqs[0]

delta_power = simps(psd[idx_delta], dx=freq_res)
print('Absolute delta power: %.3f uV^2' % delta_power)
# %%
total_power = simps(psd, dx = freq_res)
delta_rel_power = delta_power / total_power
print('Relative delta power: %.3f' % delta_rel_power)
# %%
def get_node_features(data_seg):
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
            return all_channel_feats

a = get_node_features(data)
# %%

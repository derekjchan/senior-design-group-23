# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

data = np.loadtxt('data/raw/2.txt')
print('done')

# %%
a = data[1, 0:1500]
b = data[2, 0:1500]
plt.cohere(a, b)
# %%

from mne.connectivity import spectral_connectivity

data = data[:, 0:1500]
# %%

indices = (np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]),    # row indices
           np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])) 
# %%
transf_dat = np.expand_dims(data, axis=0)
spec_coh_matrix = np.zeros((30,29))
for channel_idx in range(30):
    final_nodes = list(range(30))
    final_nodes.pop(channel_idx)
    spec_conn, freqs, times, n_epoch, n_tapers = spectral_connectivity(data=transf_dat, method='coh', 
    indices = ([channel_idx]*29, final_nodes), sfreq = 500, fmin=1.0, fmax=40.0, faverage=True, verbose=False)
    spec_coh_values = np.squeeze(spec_conn)

    spec_coh_matrix[channel_idx, :] = spec_coh_values
print('done')
# %%
i_mat = np.identity(30)
t_spec_coh_values = np.matrix.transpose(spec_coh_values)
t_spec_coh_values = t_spec_coh_values + i_mat

# %%
np.savetxt('test.csv', t_spec_coh_values, delimiter=',')
# %%
ones_mat = np.ones([30,30])
id_mat = np.identity(30)
adjacency_mat = ones_mat - id_mat
# %%
a = sparse.coo_matrix(adjacency_mat)
# %%
b = a.toarray()
# %%

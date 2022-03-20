#%%
from cgi import test
from fileinput import filename
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy import sparse
from scipy.integrate import simps
from mne.connectivity import spectral_connectivity

#%%
def get_adjacency_info(num_nodes):
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

a = get_adjacency_info(30)
# %%
import networkx as nx
g = nx.complete_graph(30)
p = nx.to_pandas_edgelist(g)
s = list(p.source)
t = list(p.target)
tens = torch.tensor([s, t], dtype=torch.long)


# %%
c = list(g)
# %%
a = [1,2,3,4]
# %%
a = np.loadtxt("data/raw/2.txt")
# %%
idx = 1
if idx:
    print("idx not zero")
# %%
if True:
    label = 1
else:
    label = 2
# %%
import matplotlib.pyplot as plt
label_1 = np.loadtxt("data/raw/rt_2.txt")
label_2 = np.loadtxt("data/raw/rt_3.txt")
label_3 = np.loadtxt("data/raw/rt_4.txt")
label_4 = np.loadtxt("data/raw/rt_5.txt")
label_5 = np.loadtxt("data/raw/rt_1.txt")
label_6 = np.loadtxt("data/raw/rt_6.txt")
label_7 = np.loadtxt("data/raw/rt_7.txt")
label_8 = np.loadtxt("data/raw/rt_8.txt")
label_9 = np.loadtxt("data/raw/rt_9.txt")
label_10 = np.loadtxt("data/raw/rt_10.txt")
label_11 = np.loadtxt("data/raw/rt_11.txt")
label_12 = np.loadtxt("data/raw/rt_12.txt")
label_13 = np.loadtxt("data/raw/rt_13.txt")
label_14 = np.loadtxt("data/raw/rt_14.txt")
label_15 = np.loadtxt("data/raw/rt_15.txt")
label_16 = np.loadtxt("data/raw/rt_16.txt")
label_17 = np.loadtxt("data/raw/rt_17.txt")
label_18 = np.loadtxt("data/raw/rt_18.txt")
label_19 = np.loadtxt("data/raw/rt_19.txt")
label_20 = np.loadtxt("data/raw/rt_20.txt")
label_21 = np.loadtxt("data/raw/rt_21.txt")
label_22 = np.loadtxt("data/raw/rt_22.txt")
label_23 = np.loadtxt("data/raw/rt_23.txt")
label_24 = np.loadtxt("data/raw/rt_24.txt")
label_25 = np.loadtxt("data/raw/rt_25.txt")
label_26 = np.loadtxt("data/raw/rt_26.txt")
label_27 = np.loadtxt("data/raw/rt_27.txt")
label_28 = np.loadtxt("data/raw/rt_28.txt")
label_29 = np.loadtxt("data/raw/rt_29.txt")
label_30 = np.loadtxt("data/raw/rt_30.txt")
label_31 = np.loadtxt("data/raw/rt_31.txt")
label_32 = np.loadtxt("data/raw/rt_32.txt")
label_33 = np.loadtxt("data/raw/rt_33.txt")
label_34 = np.loadtxt("data/raw/rt_34.txt")
label_35 = np.loadtxt("data/raw/rt_35.txt")
label_36 = np.loadtxt("data/raw/rt_36.txt")
label_37 = np.loadtxt("data/raw/rt_37.txt")
label_38 = np.loadtxt("data/raw/rt_38.txt")
label_39 = np.loadtxt("data/raw/rt_39.txt")

all_labels = np.concatenate((label_1, label_2, label_3, label_4, label_5, 
                                label_6, label_7, label_8, label_9, label_10,
                                label_11, label_12, label_13, label_14, label_15,
                                label_16, label_17, label_18, label_19, label_20,
                                label_21, label_22, label_23, label_24, label_25,
                                label_26, label_27, label_28, label_29, label_30,
                                label_31, label_32, label_33, label_34, label_35,
                                label_36, label_37, label_38, label_39))
all_labels = label_1
drowsy_count = 0
alert_count = 0
for idx in range(2, len(all_labels)-1):
    local_rt = all_labels[idx]
    global_rt = sum(all_labels[idx-2:idx+3])/5
    if global_rt >= 1.5 and local_rt >= 1.5:
        print(global_rt, local_rt)
        drowsy_count += 1
    elif global_rt <= 0.62 and local_rt <= 0.62:
        print(global_rt, local_rt)
        alert_count += 1
print("total count = ", drowsy_count + alert_count)
print("drowsy count = ", drowsy_count)
print("alert count = ", alert_count)
print("drowsy/total = ", drowsy_count/(drowsy_count + alert_count))
# %%
#plt.plot(local_rts, global_rts, 'ro')
# %%
for i in range(3):
    a = i
print(a)
# %%
import os
import glob

rt_filenames = [os.path.basename(x) for x in glob.glob("data/raw/rt_*.txt")]

ts_filenames_path = os.listdir("data/raw/")
ts_filenames = [file for file in ts_filenames_path if 'rt' not in file]
# %%
from time import sleep
from tqdm.notebook import tqdm

for i in tqdm(range(10), desc="first loop"):
    sleep(0.5)
    for n in tqdm(range(10), desc="second loop", leave=False):
        sleep(0.5)
# %%
import random
rand_list = [0,1]
rand_num = random.choice(rand_list)
# %%
a = 1
b = torch.tensor([1], dtype = torch.int64)
print(a==b)
# %%
label_1 = np.loadtxt("data/raw/2.txt")
# %%
label_1 = label_1[:, 0:500]
a = [row for row in label_1]
a = np.asmatrix(a)
b = torch.tensor(label_1, dtype=torch.float)
# %%
print(a[1] == a1)
# %%
def get_edge_features(data_seg):
        #need to do this step in order for the spectral_connectivity function to work
        transf_dat = np.expand_dims(data_seg, axis=0)
        spec_conn, freqs, times, n_epoch, n_tapers = spectral_connectivity(data=transf_dat, method='coh', 
                sfreq = 128, fmin=5, fmax=40.0, faverage=True, verbose=False)
        
        spec_coh_values = np.squeeze(spec_conn)

        return spec_coh_values
# %%

a = np.loadtxt('data/raw/1.txt')
data_seg = a[:, 0:1500]
spec_coh_mat = get_edge_features(data_seg)
spec_coh_mat = np.asarray(spec_coh_mat)

b,c = np.where(spec_coh_mat >= 0.9)
d = np.concatenate((b,c))
unique = np.unique(d)

spec_coh_mat[spec_coh_mat >= 0.9] = 1
spec_coh_mat[spec_coh_mat < 0.9] = 0


#np.savetxt("spec_coh_mat_1.txt",spec_coh_mat)
nx_mat = nx.from_numpy_array(spec_coh_mat)
pd_edge_list = nx.to_pandas_edgelist(nx_mat)
print(pd_edge_list)

# %%
a = 20 # length of skull
b = 15 # width of skull
c = 8.5 # height of skull

new_data = []
for element in unique:
    new_data.append(data_seg[element])
new_data = np.asarray(new_data)
# %%
with open(os.path.join('data/good_data/raw', 'raw_s01_061102n.txt')) as f:
    lines = (line for line in f if not line.startswith('T'))
    FH = np.loadtxt(lines)
with open(os.path.join('data/good_data/raw', 'events_s01_061102n.txt')) as f:
    lines = (line for line in f if not line.startswith('n'))
    ED = np.loadtxt(lines)

FH_t = FH.transpose()
FH_t = FH_t[1:, :]
                
# %%
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
#%%
def get_normal(x,y,z):
    x_normal = x/10
    y_normal = y/10
    z_normal = z/10
    return x_normal, y_normal, z_normal

x = x_coords[0]
y = y_coords[0]
z = z_coords[0]
x,y,z = get_normal(x,y,z)

x1 = x_coords[1]
y1 = y_coords[1]
z1 = z_coords[1]

x1,y1,z1 = get_normal(x1,y1,z1)

delta = math.acos(np.dot([x,y,z], [x1,y1,z1]))
print(delta)
# %%
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

ax = plt.axes(projection='3d')

ax.scatter(x_coords, y_coords, z_coords)
ax.set_title('3D')
plt.show()
# %%
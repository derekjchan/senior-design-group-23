#%%
from cgi import test
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
from tqdm import tqdm
from model import GNN
from main import MyOwnDataset
import mlflow.pytorch

#%%

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
train_dataset = MyOwnDataset(root='data/', ts_filename="2.txt", rt_filename="rt_2.txt")
test_dataset = MyOwnDataset(root='data/', ts_filename="3.txt", rt_filename="rt_3.txt", test=True)

#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = GNN(feature_size=train_dataset[1].x.shape[1])
#model = model.to(device)
# %%
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# %%
NUM_GRAPHS_PER_BATCH = 1
train_loader = DataLoader(train_dataset,
        batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(test_dataset,
        batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
#%%
loss_list = []
all_preds = []
all_labels = []
for epoch in range(100):
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        print(i)
        pred = model.forward(batch.x.float(), batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        print(pred.cpu().detach().numpy())
        print(batch.y.cpu().detach().numpy())
        print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
#%%
from cgi import test
from statistics import mean
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_percentage_error, accuracy_score, precision_score, \
            recall_score, roc_auc_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
from model import GNN, GNN_C
from main import MyOwnDataset
from statistics import mean
import mlflow.pytorch
import os
import glob
import tqdm.notebook as tqdm
#%%

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
train_dataset = MyOwnDataset(root='data/', ts_files=["1.txt"], rt_files=["rt_1.txt"], classification=True)

#%%
test_dataset = MyOwnDataset(root='data/', ts_files=["5.txt"], rt_files=["rt_5.txt"], test=True)

#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = GNN(feature_size=train_dataset[1].x.shape[1])
#model = model.to(device)
# %%
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# %%
NUM_GRAPHS_PER_BATCH = 1
train_loader = DataLoader(train_dataset,
        batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
#test_loader = DataLoader(test_dataset,
#        batch_size=1, shuffle=True)
#%%
loss_list = []
all_preds = []
all_labels = []
for epoch in range(300):
    pred_list = []
    label_list = []
    losses = []
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        #print(i)
        pred = model(batch.x.float(), batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(torch.squeeze(pred), batch.y)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        #print(pred.cpu().detach().numpy())
        #print(batch.y.cpu().detach().numpy())

        mape = mean_absolute_percentage_error(pred.cpu().detach().numpy(), batch.y.cpu().detach().numpy())
        #print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}, MAE: {mae}")

        losses.append(loss.item())
        pred_list.extend(pred.cpu().detach().numpy())
        label_list.extend(batch.y.cpu().detach().numpy())

        #print(len(pred_list))
        #print(len(label_list))
    mape = mean_absolute_percentage_error(pred_list, label_list)
    print(f"Epoch: {epoch}, Loss: {mean(losses)}, MAPE: {mape}")
# %%
rt_filenames = [os.path.basename(x) for x in glob.glob("data/raw/rt_*.txt")]
ts_filenames_path = os.listdir("data/raw/")
ts_filenames = [file for file in ts_filenames_path if 'rt' not in file]

# %%
train_dataset = MyOwnDataset(root='data/', ts_files=ts_filenames, rt_files=rt_filenames, classification=True)
# %%
model_2 = GNN_C(feature_size=train_dataset[1].x.shape[1])

loss_fun = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.01)

NUM_GRAPHS_PER_BATCH = 256
train_loader = DataLoader(train_dataset,
        batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)

# %%
model_2.train()
epochs = 100
for i in range(epochs):
    pred_list = []
    label_list = []
    acc_list = []
    losses = []
    for j, batch in enumerate(train_loader):
        optimizer.zero_grad()
        #print(i)
        pred = model_2(batch.x.float(), batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y
        target = target.unsqueeze(1)
        target = target.to(torch.float)
        loss = loss_fun(pred, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        #print(pred.cpu().detach().numpy())
        #print(batch.y.cpu().detach().numpy())

        acc = accuracy_score(np.rint(pred.cpu().detach().numpy()), batch.y.cpu().detach().numpy())
        #print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}, MAE: {mae}")
        acc_list.append(acc)

        #print(acc)
        losses.append(loss.item())
        pred_list.extend(np.rint(pred.cpu().detach().numpy()))
        label_list.extend(batch.y.cpu().detach().numpy())

        #print(len(pred_list))
        #print(len(label_list))
    #acc = accuracy_score(pred_list, label_list)
    print(f"Epoch: {i}, Loss: {mean(losses)}, Acc: {mean(acc_list)}")
    print(f"Confusion matrix: \n {confusion_matrix(pred_list, label_list)}")
    print(f"F1 Score: {f1_score(pred_list, label_list)}")
    print(f"Accuracy: {accuracy_score(pred_list, label_list)}")
    print(f"Precision: {precision_score(pred_list, label_list)}")
    print(f"Recall: {recall_score(pred_list, label_list)}")
# %%
ts_filenames = ["44.txt", "45.txt", "46.txt", "47.txt", "48.txt", "49.txt", "50.txt", "51.txt", "52.txt", "53.txt", "54.txt", "55.txt", "56.txt", "57.txt", "58.txt", "59.txt", "60.txt", "61.txt", "62.txt"]
rt_filenames = ["rt_44.txt", "rt_45.txt", "rt_46.txt", "rt_47.txt", "rt_48.txt", "rt_49.txt", "rt_50.txt", "rt_51.txt", "rt_52.txt", "rt_53.txt", "rt_54.txt", "rt_55.txt", "rt_56.txt", "rt_57.txt", "rt_58.txt", "rt_59.txt", "rt_60.txt", "rt_61.txt", "rt_62.txt"]
test_dataset = MyOwnDataset(root='data/', ts_files=ts_filenames, rt_files=rt_filenames, classification=True)
# %%
test_loader = DataLoader(train_dataset,
        batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
# %%
# Testing
model_2.eval()
epochs = 5
for i in range(epochs):
    pred_list = []
    label_list = []
    acc_list = []
    losses = []
    for j, batch in enumerate(test_loader):
        pred = model_2(batch.x.float(), batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y
        target = target.unsqueeze(1)
        target = target.to(torch.float)
        loss = loss_fun(pred, target)

        losses.append(loss.item())
        #print(pred.cpu().detach().numpy())
        #print(batch.y.cpu().detach().numpy())

        acc = accuracy_score(np.rint(pred.cpu().detach().numpy()), batch.y.cpu().detach().numpy())
        #print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}, MAE: {mae}")
        acc_list.append(acc)

        #print(acc)
        losses.append(loss.item())
        pred_list.extend(pred.cpu().detach().numpy())
        label_list.extend(batch.y.cpu().detach().numpy())

        #print(len(pred_list))
        #print(len(label_list))
    #acc = accuracy_score(pred_list, label_list)
    print(f"Epoch: {i}, Loss: {mean(losses)}, Acc: {mean(acc_list)}")

# %%
for i, batch in enumerate(train_loader):
    pred = model_2(batch.x.float(), batch.edge_index, batch.edge_attr, batch.batch)
    a = batch.y
    a = a.to(torch.long)
    b = a.unsqueeze(1)
    print(b)
    print(b.shape)
# %%

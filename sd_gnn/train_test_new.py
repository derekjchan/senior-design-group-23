# %%
from cgi import test
from statistics import mean
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_percentage_error, accuracy_score, precision_score, \
            recall_score, roc_auc_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
from model import GNN, GNN_C, GAT_C, GTrans_C
from dataset_new import MyOwnDataset_new
from statistics import mean
import mlflow.pytorch
import os
import glob
import tqdm.notebook as tqdm
import matplotlib.pyplot as plt
import mlflow.pytorch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
mlflow.set_tracking_uri("http://localhost:5000")
#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def train(epoch, model, train_loader, optimizer, loss_fun):
    pred_list = []
    label_list = []
    losses = []
    for i, batch in enumerate(train_loader):
        #batch.to(device)
        optimizer.zero_grad()
        pred = model(x = batch.x.float(), edge_index=batch.edge_index, edge_weight=batch.edge_attr, batch=batch.batch)
        target = batch.y
        target = target.unsqueeze(1)
        target = target.to(torch.float)
        loss = loss_fun(pred, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pred_list.extend(np.rint(pred.cpu().detach().numpy()))
        label_list.extend(batch.y.cpu().detach().numpy())
    log_metrics(pred_list, label_list, epoch, "train")
    return mean(losses)

def test(epoch, model, train_loader, loss_fun, type):
    pred_list = []
    label_list = []
    losses = []
    for i, batch in enumerate(train_loader):
        #batch.to(device)
        pred = model(x = batch.x.float(), edge_index=batch.edge_index, edge_weight=batch.edge_attr, batch=batch.batch)
        target = batch.y
        target = target.unsqueeze(1)
        target = target.to(torch.float)
        loss = loss_fun(pred, target)

        losses.append(loss.item())
        pred_list.extend(np.rint(pred.cpu().detach().numpy()))
        label_list.extend(batch.y.cpu().detach().numpy())
    log_metrics(pred_list, label_list, epoch, type)
    return mean(losses)
    
def log_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion Matrix \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_pred, y_true)}")
    print(f"Accuracy: {accuracy_score(y_pred, y_true)}")
    prec = precision_score(y_pred, y_true)
    rec = recall_score(y_pred, y_true)
    acc = accuracy_score(y_pred, y_true)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    mlflow.log_metric(key=f"Accuracy-{type}", value=float(acc), step=epoch)

# %%
raw_file_names = [os.path.basename(x) for x in glob.glob("data/good_data_ds/raw/raw_*")]
raw_event_filenames = [os.path.basename(x) for x in glob.glob("data/good_data_ds/raw/events_*")]

#train_dataset = MyOwnDataset(root='data/good_data/', ts_files=raw_file_names[1:], rt_files=raw_event_filenames[1:], feats=True, classification=True, good_data=True)
#test_dataset = MyOwnDataset(root='data/good_data/', ts_files=raw_file_names[:1], rt_files=raw_event_filenames[:1], feats=True, classification=True, good_data=True, test=True)

# %%
"""
rt_filenames = [os.path.basename(x) for x in glob.glob("data/raw/rt_*.txt")]
ts_filenames_path = os.listdir("data/raw/")
ts_filenames = [file for file in ts_filenames_path if 'rt' not in file]

rt_filenames1 = [os.path.basename(x) for x in glob.glob("data/test/raw/rt_*.txt")]
ts_filenames_path1 = os.listdir("data/test/raw/")
ts_filenames1 = [file for file in ts_filenames_path1 if 'rt' not in file]

rt_filenames2 = [os.path.basename(x) for x in glob.glob("data/test_cross/raw/rt_*.txt")]
ts_filenames_path2 = os.listdir("data/test_cross/raw/")
ts_filenames2 = [file for file in ts_filenames_path2 if 'rt' not in file]

# %%

train_dataset = MyOwnDataset(root='data/', ts_files=ts_filenames, rt_files=rt_filenames, classification=True)

test_dataset = MyOwnDataset(root='data/test/', ts_files=ts_filenames1, rt_files=rt_filenames1, test=True, classification=True)

#test_cross_dataset = MyOwnDataset(root='data/test_cross/', ts_files=ts_filenames2, rt_files=rt_filenames2, cross=True, classification=True)
"""
#%%
#train_dataset = MyOwnDataset(root='data/good_data/', ts_files=[raw_file_names[2]], rt_files=[raw_event_filenames[2]], classification=True, good_data=True)
#%%
#test_dataset = MyOwnDataset(root='data/good_data_ds/', ts_files='raw_s22_090825n.txt', rt_files='events_s22_090825n.txt', classification=True, good_data=True, test=True)

# %%
for i in range(11):
    new_raw_filenames = list(raw_file_names)
    del(new_raw_filenames[i])
    new_event_filenames = list(raw_event_filenames)
    del(new_event_filenames[i])

    train_dataset = MyOwnDataset_new(root='data/good_data_ds/', ts_files=new_raw_filenames, rt_files=new_event_filenames, classification=True, good_data=True)
    test_dataset = MyOwnDataset_new(root='data/good_data_ds/', ts_files=[raw_file_names[i]], rt_files=[raw_event_filenames[i]], classification=True, good_data=True, test=True)
    
    model = GNN_C(feature_size=train_dataset[1].x.shape[1])#, edge_dim=train_dataset[1].edge_attr.shape[1])
    #model = model.to(device)
    model_name = "GNN_C"

    loss_fun = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    val_loss_same = []
    val_loss_cross = []
    train_loss = []

    val_acc_same = []
    val_acc_cross = []

    NUM_GRAPHS_PER_BATCH = 64
    train_loader = DataLoader(train_dataset,
            batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, 
            batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
    #test_cross_loader = DataLoader(test_cross_dataset, 
    #        batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
        
    model.train()
    epochs = 75
    best_loss = 1000
    early_stopping_counter = 0

    with mlflow.start_run() as run:
        for epoch in range(epochs):
            if early_stopping_counter <= 50:
                model.train()
                loss = train(epoch, model, train_loader, optimizer, loss_fun)
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)
                train_loss.append(loss)

                model.eval()
                #loss = test(epoch, model, test_cross_loader, loss_fun, "test cross")
                #print(f"Epoch {epoch} | Test Cross Loss {loss}")
                #mlflow.log_metric(key="Test Cross Loss", value=float(loss), step=epoch)
                #val_loss_cross.append(loss)

                model.eval()
                loss = test(epoch, model, test_loader, loss_fun, f"test {model_name}")
                print(f"Epoch {epoch} | Test Loss {loss} {model_name}")
                mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)

                if float(loss) < best_loss:
                    best_loss = loss
                else:
                    early_stopping_counter += 0#1


for i in range(11):
    new_raw_filenames = list(raw_file_names)
    del(new_raw_filenames[i])
    new_event_filenames = list(raw_event_filenames)
    del(new_event_filenames[i])

    train_dataset = MyOwnDataset_new(root='data/good_data_ds/', ts_files=new_raw_filenames, rt_files=new_event_filenames, classification=True, good_data=True)
    test_dataset = MyOwnDataset_new(root='data/good_data_ds/', ts_files=[raw_file_names[i]], rt_files=[raw_event_filenames[i]], classification=True, good_data=True, test=True)
    
    model = GAT_C(feature_size=train_dataset[1].x.shape[1])#, edge_dim=train_dataset[1].edge_attr.shape[1])
    #model = model.to(device)
    model_name = "GAT_C"

    loss_fun = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    val_loss_same = []
    val_loss_cross = []
    train_loss = []

    val_acc_same = []
    val_acc_cross = []

    NUM_GRAPHS_PER_BATCH = 64
    train_loader = DataLoader(train_dataset,
            batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, 
            batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
    #test_cross_loader = DataLoader(test_cross_dataset, 
    #        batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
        
    model.train()
    epochs = 75
    best_loss = 1000
    early_stopping_counter = 0

    with mlflow.start_run() as run:
        for epoch in range(epochs):
            if early_stopping_counter <= 50:
                model.train()
                loss = train(epoch, model, train_loader, optimizer, loss_fun)
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)
                train_loss.append(loss)

                model.eval()
                #loss = test(epoch, model, test_cross_loader, loss_fun, "test cross")
                #print(f"Epoch {epoch} | Test Cross Loss {loss}")
                #mlflow.log_metric(key="Test Cross Loss", value=float(loss), step=epoch)
                #val_loss_cross.append(loss)

                model.eval()
                loss = test(epoch, model, test_loader, loss_fun, f"test {model_name}")
                print(f"Epoch {epoch} | Test Loss {loss} {model_name}")
                mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)

                if float(loss) < best_loss:
                    best_loss = loss
                else:
                    early_stopping_counter += 0#1

for i in range(11):
    new_raw_filenames = list(raw_file_names)
    del(new_raw_filenames[i])
    new_event_filenames = list(raw_event_filenames)
    del(new_event_filenames[i])

    train_dataset = MyOwnDataset_new(root='data/good_data_ds/', ts_files=new_raw_filenames, rt_files=new_event_filenames, classification=True, good_data=True)
    test_dataset = MyOwnDataset_new(root='data/good_data_ds/', ts_files=[raw_file_names[i]], rt_files=[raw_event_filenames[i]], classification=True, good_data=True, test=True)
    
    model = GTrans_C(feature_size=train_dataset[1].x.shape[1], edge_dim=train_dataset[1].edge_attr.shape[1])
    #model = model.to(device)
    model_name = "GTrans_C"

    loss_fun = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    val_loss_same = []
    val_loss_cross = []
    train_loss = []

    val_acc_same = []
    val_acc_cross = []

    NUM_GRAPHS_PER_BATCH = 64
    train_loader = DataLoader(train_dataset,
            batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, 
            batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
    #test_cross_loader = DataLoader(test_cross_dataset, 
    #        batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
        
    model.train()
    epochs = 75
    best_loss = 1000
    early_stopping_counter = 0

    with mlflow.start_run() as run:
        for epoch in range(epochs):
            if early_stopping_counter <= 50:
                model.train()
                loss = train(epoch, model, train_loader, optimizer, loss_fun)
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)
                train_loss.append(loss)

                model.eval()
                #loss = test(epoch, model, test_cross_loader, loss_fun, "test cross")
                #print(f"Epoch {epoch} | Test Cross Loss {loss}")
                #mlflow.log_metric(key="Test Cross Loss", value=float(loss), step=epoch)
                #val_loss_cross.append(loss)

                model.eval()
                loss = test(epoch, model, test_loader, loss_fun, f"test {model_name}")
                print(f"Epoch {epoch} | Test Loss {loss} {model_name}")
                mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)

                if float(loss) < best_loss:
                    best_loss = loss
                else:
                    early_stopping_counter += 0#1
# %%
for i, batch in enumerate(train_loader):
    print(batch)
# %%

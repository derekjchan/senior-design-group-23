# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, ModuleList
from torch_geometric.nn import TransformerConv, GCNConv, GATConv, TopKPooling, BatchNorm
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap

# %%
class GNN(torch.nn.Module):
    def __init__(self, feature_size):
        super(GNN, self).__init__()

        self.conv1 = GCNConv(feature_size, 16, improved=True, cached=True, normalize=False)
        self.conv2 = GCNConv(16, 32, improved=True, cached=True, normalize=False)
        self.conv3 = GCNConv(32, 64, improved=True, cached=True, normalize=False)
        self.conv4 = GCNConv(64, 50, improved=True, cached=True, normalize=False)
        self.conv4_bn = BatchNorm(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.fc_block1 = nn.Linear(50, 30)
        self.fc_block2 = nn.Linear(30, 20)
        self.fc_block3 = nn.Linear(20, 1)

        self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.fc_block3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)

        self.linear = Linear(1,1)

    def forward(self, x, edge_index, edge_weight, batch, return_graph_embedding=False):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weight))

        x = F.leaky_relu(self.conv2(x, edge_index, edge_weight))

        x = F.leaky_relu(self.conv3(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x, edge_index, edge_weight)))
        out = gap(x, batch=batch)

        if return_graph_embedding:
            return out

        out = F.leaky_relu(self.fc_block1(out), negative_slope=0.01)
        out = F.dropout(out, p = 0.2, training=self.training)
        out = F.leaky_relu(self.fc_block2(out), negative_slope=0.01)
        out = self.fc_block3(out)
        
        return out

# %%
class GNN_C(torch.nn.Module):
    def __init__(self, feature_size):
        super(GNN_C, self).__init__()

        self.conv1 = GCNConv(feature_size, 16, improved=True, cached=True, normalize=False)
        self.conv2 = GCNConv(16, 32, improved=True, cached=True, normalize=False)
        self.conv3 = GCNConv(32, 64, improved=True, cached=True, normalize=False)
        self.conv4 = GCNConv(64, 50, improved=True, cached=True, normalize=False)
        self.conv4_bn = BatchNorm(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.fc_block1 = nn.Linear(50, 256)
        self.fc_block2 = nn.Linear(256, 128)
        self.fc_block3 = nn.Linear(128, 1)

        self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.fc_block3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)

        self.linear = Linear(1,1)

    def forward(self, x, edge_index, edge_weight, batch, return_graph_embedding=False):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weight))

        x = F.leaky_relu(self.conv2(x, edge_index, edge_weight))

        x = F.leaky_relu(self.conv3(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x, edge_index, edge_weight)))
        out = gap(x, batch=batch)

        if return_graph_embedding:
            return out

        out = F.leaky_relu(self.fc_block1(out), negative_slope=0.01)
        out = F.dropout(out, p = 0.5, training=self.training)
        out = F.leaky_relu(self.fc_block2(out), negative_slope=0.01)
        out = F.dropout(out, p = 0.5, training=self.training)
        out = torch.sigmoid(self.fc_block3(out))
        
        return out

# %%
class GAT_C(torch.nn.Module):
    def __init__(self, feature_size):
        super(GAT_C, self).__init__()
        embedding_size = 256

        self.conv1 = GATConv(feature_size, embedding_size, heads=3, dropout=0.3)
        self.head_transfrom1 = Linear(embedding_size*3, embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)

        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transfrom2 = Linear(embedding_size*3, embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)

        self.conv3 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transfrom3 = Linear(embedding_size*3, embedding_size)
        self.pool3 = TopKPooling(embedding_size, ratio=0.2)

        self.linear1 = Linear(embedding_size*2, 256)
        self.linear2 = Linear(256, 1)

    def forward(self, x, edge_weight, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.head_transfrom1(x)

        x, edge_index, edge_weight, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x = self.head_transfrom2(x)

        x, edge_index, edge_weight, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv3(x, edge_index)
        x = self.head_transfrom3(x)

        x, edge_index, edge_weight, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = torch.sigmoid(x)

        return x

# %%
class GTrans_C(torch.nn.Module):
    def __init__(self, feature_size, edge_dim):
        super(GTrans_C, self).__init__()
        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        embedding_size = 64
        self.n_layers = 3
        self.conv1 = TransformerConv(feature_size, embedding_size, heads=3, dropout=0.2, edge_dim=edge_dim, beta=True)

        self.transf1 = Linear(embedding_size*3, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size, embedding_size, heads=3, dropout=0.2, edge_dim=edge_dim, beta=True))
            self.transf_layers.append(Linear(embedding_size*3, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            self.pooling_layers.append(TopKPooling(embedding_size, ratio=0.5))

        self.linear1 = Linear(embedding_size*2, 256)
        self.linear2 = Linear(256, 128)
        self.linear3 = Linear(128, 64)
        self.linear4 = Linear(64, 1)

    def forward(self, x, edge_weight, edge_index, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_weight)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            x, edge_index, edge_weight, batch, _, _ = self.pooling_layers[i](x, edge_index, edge_weight, batch)

            global_representation.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))

        x = sum(global_representation)

        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)
        x = torch.sigmoid(self.linear4(x))

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import TransformerConv, GCNConv, GATConv, TopKPooling, BatchNorm
from torch_geometric.nn import global_mean_pool as gap, global_add_pool as gap

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
        out = self.linear(out)
        
        return out

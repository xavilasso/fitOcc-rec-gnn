# rgcn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()

        # Primera capa R-GCN
        self.conv1 = RGCNConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            #num_nodes=num_nodes,
            num_relations=num_relations
        )

        # Segunda capa R-GCN
        self.conv2 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=out_dim,
            #num_nodes=num_nodes,
            num_relations=num_relations
        )

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x

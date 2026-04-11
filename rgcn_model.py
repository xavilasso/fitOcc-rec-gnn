# rgcn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

# class RGCN(torch.nn.Module):
#     def __init__(self, num_nodes, in_dim, hidden_dim, out_dim, num_relations):
#         super().__init__()

#         # Primera capa R-GCN
#         self.conv1 = RGCNConv(
#             in_channels=in_dim,
#             out_channels=hidden_dim,
#             #num_nodes=num_nodes,
#             num_relations=num_relations
#         )

#         # Segunda capa R-GCN
#         self.conv2 = RGCNConv(
#             in_channels=hidden_dim,
#             out_channels=out_dim,
#             #num_nodes=num_nodes,
#             num_relations=num_relations
#         )

#     def forward(self, x, edge_index, edge_type):
#         x = self.conv1(x, edge_index, edge_type)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index, edge_type)
#         return x

class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_bases=None, dropout=0.3):
        super().__init__()

        self.conv1 = RGCNConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases
        )

        self.conv2 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=out_dim,
            num_relations=num_relations,
            num_bases=num_bases
        )

        self.dropout = nn.Dropout(dropout)

        # Relation-aware decoder weights
        self.rel_weights = nn.Parameter(
            torch.randn(num_relations, out_dim)
        )

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_type)

        # Normalize embeddings (mejora coherencia funcional)
        x = F.normalize(x, p=2, dim=1)

        return x

    def decode(self, z, edge_index, edge_type):
        src, dst = edge_index
        rel_w = self.rel_weights[edge_type]
        return (z[src] * rel_w * z[dst]).sum(dim=1)
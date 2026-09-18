import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """2-3 layer GraphSAGE, mirroring GCN's structure for a fair ablation."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, num_classes))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, num_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

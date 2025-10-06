# models/gin.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        # First layer
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.layers.append(GINConv(nn1))

        # Hidden layers
        for _ in range(num_layers - 2):
            nnk = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
            self.layers.append(GINConv(nnk))

        # Final layer
        nn_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels),
        )
        self.layers.append(GINConv(nn_out))

        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)

        # If batching multiple graphs → pool
        if batch is not None:
            x = global_mean_pool(x, batch)

        return x

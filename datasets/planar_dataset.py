import networkx as nx
from torch_geometric.data import Dataset, Data
import torch
from pathlib import Path

from datasets.layouts.positional_encodings import (
    tutte_embedding,
    spectral_embedding,
    force_embedding,
    random_embedding,
)
import math


class PlanarDataset(Dataset):
    def __init__(self, root, split="train", seed=0, pe_type=None,
                 pe_mode="concat",   # {"concat", "replace", "polar"}
                 transform=None, pre_transform=None):
        """
        Args:
            root (str or Path): dataset folder (e.g., "data/grids")
            split (str): one of {"train", "val", "test"}
            seed (int): which split to use
            pe_type (str or None): {"tutte", "spectral", "force", "random", None}
            pe_mode (str): 
                - "concat": stack PE alongside baseline features
                - "replace": use PE only
                - "polar": convert Tutte PE into polar coords [r, θ]
        """
        super().__init__(root, transform, pre_transform)
        self.root = Path(root)
        meta = torch.load(self.root / "meta.pt")
        self.split_indices = meta["splits"][seed][split]
        self.graph_files = [self.root / f"graph_{i}.pt" for i in self.split_indices]
        self.pe_type = pe_type
        self.pe_mode = pe_mode

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        d = torch.load(self.graph_files[idx])
        x = d["x"]

        if self.pe_type is not None:
            G = nx.Graph()
            G.add_edges_from(d["edge_index"].T.tolist())

            if self.pe_type == "tutte":
                pos_enc = tutte_embedding(G)
            elif self.pe_type == "spectral":
                pos_enc = spectral_embedding(G)
            elif self.pe_type == "force":
                pos_enc = force_embedding(G)
            elif self.pe_type == "random":
                pos_enc = random_embedding(G)
            else:
                raise ValueError(f"Unknown pe_type: {self.pe_type}")

            # --- Handle feature modes ---
            if self.pe_mode == "concat":
                x = torch.cat([x, pos_enc], dim=1)
            elif self.pe_mode == "replace":
                x = pos_enc
            elif self.pe_mode == "polar":
                # pos_enc is (N, 2) with (x, y) Tutte coords
                xs, ys = pos_enc[:, 0], pos_enc[:, 1]
                r = torch.sqrt(xs**2 + ys**2).unsqueeze(1)
                theta = torch.atan2(ys, xs).unsqueeze(1)
                # normalize θ to [-1, 1]
                theta = theta / math.pi
                polar = torch.cat([r, theta], dim=1)
                x = torch.cat([x, polar], dim=1)  # concat baseline + polar
            else:
                raise ValueError(f"Unknown pe_mode: {self.pe_mode}")

        return Data(x=x, y=d["y"], edge_index=d["edge_index"])

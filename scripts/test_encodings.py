import torch
from pathlib import Path
from datasets.layouts.positional_encodings import tutte_embedding, spectral_embedding, force_embedding, random_embedding
import networkx as nx

d = torch.load("data/triangulated/graph_0.pt")
x, edge_index = d['x'], d['edge_index']
G = nx.Graph()
G.add_nodes_from(range(len(x)))
G.add_edges_from(edge_index.numpy().T)

coords_tutte = tutte_embedding(G)
coords_spec = spectral_embedding(G)
coords_force = force_embedding(G)
coords_rand = random_embedding(G)

print(coords_tutte.shape, coords_spec.shape, coords_force.shape, coords_rand.shape)

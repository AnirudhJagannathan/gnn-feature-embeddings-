import torch
import networkx as nx
import matplotlib.pyplot as plt

from datasets.layouts.positional_encodings import tutte_embedding

# --- Load one graph ---
d = torch.load("data/triangulated/graph_0.pt")

edge_index = d["edge_index"]
y = d["y"]

# Convert edge_index to list of edges
edges = edge_index.t().tolist()
G = nx.Graph()
G.add_edges_from(edges)

# --- Compute Tutte embedding ---
pos = tutte_embedding(G)   # [num_nodes, 2]

# --- Scatter plot ---
plt.figure(figsize=(6, 6))
plt.scatter(pos[:, 0], pos[:, 1], c=y, cmap="tab10", s=50, edgecolors="k")
plt.title("Tutte Embedding of Graph_0 (colored by labels)")
plt.axis("equal")
plt.show()

print("Labels:", y.unique())
print("Number of nodes:", pos.shape[0])

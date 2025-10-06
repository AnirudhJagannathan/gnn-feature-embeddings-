# datasets/preview_pe.py
import torch
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import numpy as np

from datasets.layouts.positional_encodings import (
    tutte_embedding,
    spectral_embedding,
    force_embedding,
    random_embedding,
)

def visualize_embeddings(graph_path, seed=0):
    data = torch.load(graph_path)
    x = data['x'].numpy()
    edges = data['edge_index'].numpy().T
    labels = data['y'].numpy() if 'y' in data else None

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(len(x)))
    G.add_edges_from(edges)

    embeddings = {
        "Tutte": tutte_embedding(G, seed=seed).numpy(),
        "Spectral": spectral_embedding(G).numpy(),
        "Force-directed": force_embedding(G, seed=seed).numpy(),
        "Random": random_embedding(G, seed=seed).numpy(),
    }

    # Plot
        # Plot
    fig, axes = plt.subplots(1, len(embeddings), figsize=(16, 4))
    for ax, (name, coords) in zip(axes, embeddings.items()):
        # --- Draw edges first ---
        for u, v in edges:
            ax.plot([coords[u, 0], coords[v, 0]],
                    [coords[u, 1], coords[v, 1]],
                    color="lightgray", linewidth=0.5, zorder=1)

        # --- Draw nodes on top ---
        if labels is not None and np.issubdtype(labels.dtype, np.integer):
            cmap = plt.cm.tab10
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=labels, cmap=cmap, s=30, zorder=2)
        else:
            cmap = plt.cm.viridis
            if labels is not None:
                sc = ax.scatter(coords[:, 0], coords[:, 1],
                                c=labels, cmap=cmap, s=30, zorder=2)
                plt.colorbar(sc, ax=ax, shrink=0.7)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], s=30, zorder=2)

        ax.set_aspect("equal")
        ax.set_title(name)
        ax.axis("off")


    plt.suptitle(f"Embeddings for {graph_path}")
    plt.tight_layout()
    plt.show()


def preview_dataset(folder="data/triangulated", max_graphs=1, seed=0):
    folder = Path(folder)
    graph_files = sorted(folder.glob("graph_*.pt"))

    if not graph_files:
        print(f"No graphs found in {folder}")
        return

    for graph_file in graph_files[:max_graphs]:
        visualize_embeddings(graph_file, seed=seed)


if __name__ == "__main__":
    # Change folder as needed (grids, delaunay, random_planar, triangulated)
    preview_dataset("data/triangulated", max_graphs=1, seed=0)

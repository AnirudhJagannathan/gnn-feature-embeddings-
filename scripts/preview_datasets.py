# preview_datasets.py
import torch
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def scatter_plot(data, show_labels=True, title=""):
    """
    Quick scatter plot of coordinates only (ignores edges).
    Useful for checking embeddings directly.
    """
    x = data['x'].numpy()
    labels = data['y'].numpy() if 'y' in data else None

    plt.figure(figsize=(5, 5))
    if labels is not None and show_labels:
        if np.issubdtype(labels.dtype, np.integer):
            cmap = plt.cm.tab10   # discrete colormap for classes
            sc = plt.scatter(x[:,0], x[:,1], c=labels, cmap=cmap, s=20)
        else:
            cmap = plt.cm.viridis # continuous colormap for regression
            sc = plt.scatter(x[:,0], x[:,1], c=labels, cmap=cmap, s=20)
            plt.colorbar(sc, shrink=0.7, label="Regression target")
    else:
        plt.scatter(x[:,0], x[:,1], s=20)

    plt.gca().set_aspect('equal')
    plt.title(f"Scatter: {title}")
    plt.show()


def visualize_graph(graph_path, show_labels=True, use_scatter=False):
    data = torch.load(graph_path)

    if use_scatter:
        scatter_plot(data, show_labels=show_labels, title=graph_path)
        return

    x = data['x'].numpy()
    edges = data['edge_index'].numpy().T
    labels = data['y'].numpy() if 'y' in data else None

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(len(x)))
    G.add_edges_from(edges)

    # Always use stored coordinates
    pos = {i: (x[i, 0], x[i, 1]) for i in range(len(x))}

    plt.figure(figsize=(5, 5))
    if labels is not None and show_labels:
        if np.issubdtype(labels.dtype, np.integer):
            # Classification: discrete colormap
            cmap = plt.cm.tab10
            nx.draw(G, pos, node_color=labels, cmap=cmap, node_size=40, with_labels=False)
        else:
            # Regression: continuous colormap + colorbar
            cmap = plt.cm.viridis
            nodes = nx.draw_networkx_nodes(G, pos, node_color=labels, cmap=cmap, node_size=40)
            nx.draw_networkx_edges(G, pos, alpha=0.3)
            plt.colorbar(nodes, shrink=0.7, label="Regression target")
    else:
        nx.draw(G, pos, node_size=40, with_labels=False)

    plt.gca().set_aspect('equal')  # ensure geometry isn’t distorted
    plt.title(f"Graph: {graph_path}")
    plt.show()

def preview_dataset(folder="data/triangulated", max_graphs=3, use_scatter=False):
    """
    Preview the first few graphs in a dataset folder.
    """
    folder = Path(folder)
    graph_files = sorted(folder.glob("graph_*.pt"))

    if not graph_files:
        print(f"No graphs found in {folder}")
        return  

    for graph_file in graph_files[:max_graphs]:
        visualize_graph(graph_file, use_scatter=use_scatter)


if __name__ == "__main__":
    # Change folder to grids, delaunay, etc.
    preview_dataset(folder="data/random_planar", max_graphs=5, use_scatter=False)

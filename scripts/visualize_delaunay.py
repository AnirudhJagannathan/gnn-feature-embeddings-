# datasets/visualize_delaunay.py
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.tri as mtri

def visualize_delaunay(graph_path):
    data = torch.load(graph_path)
    x = data['x'].numpy()
    labels = data['y'].numpy()

    # Recompute Delaunay triangulation from coordinates
    tri = Delaunay(x)
    triang = mtri.Triangulation(x[:, 0], x[:, 1], tri.simplices)

    plt.figure(figsize=(5, 5))
    if np.issubdtype(labels.dtype, np.integer):
        # Classification labels
        sc = plt.scatter(x[:, 0], x[:, 1], c=labels, cmap=plt.cm.tab10, s=30, edgecolors="k")
    else:
        # Regression labels
        sc = plt.scatter(x[:, 0], x[:, 1], c=labels, cmap=plt.cm.viridis, s=30, edgecolors="k")
        plt.colorbar(sc, shrink=0.7, label="Regression target")

    # Draw triangulation edges
    plt.triplot(triang, color="black", alpha=0.3, linewidth=0.8)

    plt.gca().set_aspect("equal")
    plt.title(f"Delaunay Triangulation: {graph_path}")
    plt.show()


def preview_delaunay(folder="data/delaunay", max_graphs=3):
    folder = Path(folder)
    graph_files = sorted(folder.glob("graph_*.pt"))
    if not graph_files:
        print(f"No graphs found in {folder}")
        return

    for graph_file in graph_files[:max_graphs]:
        visualize_delaunay(graph_file)


if __name__ == "__main__":
    preview_delaunay("data/delaunay", max_graphs=2)

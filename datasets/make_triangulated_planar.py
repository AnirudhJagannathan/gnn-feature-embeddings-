# datasets/make_triangulated_planar.py
import networkx as nx
import numpy as np
import torch
import shutil
from pathlib import Path
from scipy.spatial import Delaunay

from datasets.utils import save_graph_list, split_indices, ensure_planar_or_die

def make_triangulated_planar(n_boundary=8, n_interior=20, seed=0):
    rng = np.random.default_rng(seed)

    # Boundary points on a unit circle
    angles = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    boundary_pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Interior points sampled in unit disk
    r = np.sqrt(rng.random(n_interior))   # sqrt for uniform density in disk
    theta = 2 * np.pi * rng.random(n_interior)
    interior_pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    # Combine points
    pts = np.vstack([boundary_pts, interior_pts])

    # Delaunay triangulation
    tri = Delaunay(pts)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            u, v = simplex[i], simplex[(i + 1) % 3]
            if u != v:
                edges.add(tuple(sorted((u, v))))

    G = nx.Graph()
    for i, (x, y) in enumerate(pts):
        G.add_node(i, pos_x=float(x), pos_y=float(y),
                   boundary=(i < n_boundary))  # tag boundary nodes
    G.add_edges_from(edges)

    ensure_planar_or_die(G)
    return G


def main():
    out = Path("data/triangulated")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    graphs = []
    # Generate a few sizes with different seeds
    configs = [(6, 20), (8, 40), (10, 60), (26, 80), (43, 100)]
    for (nb, ni) in configs:
        for seed in [0, 1, 2, 3, 4]:
            graphs.append(make_triangulated_planar(n_boundary=nb, n_interior=ni, seed=seed))

    # Example labels: boundary vs. interior
    for G in graphs:
        labels = {i: int(G.nodes[i]['boundary']) for i in G.nodes()}
        nx.set_node_attributes(G, {i: {'label': labels[i]} for i in G.nodes()})

    splits = split_indices(len(graphs), seeds=[0, 1, 2, 3, 4])
    save_graph_list(graphs, out, splits)

    d = torch.load(out / "graph_0.pt")
    print("Graph 0:", d['x'].shape, "nodes; edge_index shape:", d['edge_index'].shape)
    print("First 5 coords:", d['x'][:5])
    print("First 5 labels:", d['y'][:5])


if __name__ == "__main__":
    main()

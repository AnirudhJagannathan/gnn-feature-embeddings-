# datasets/make_delaunay.py
import networkx as nx
import numpy as np
import torch
import shutil
from pathlib import Path
from scipy.spatial import Delaunay

from datasets.utils import save_graph_list, split_indices, ensure_planar_or_die

def make_delaunay(n_points, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.random((n_points, 2))  # uniform in unit square

    # Delaunay triangulation
    tri = Delaunay(pts)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            u, v = simplex[i], simplex[(i+1) % 3]
            if u != v:
                edges.add(tuple(sorted((u, v))))

    G = nx.Graph()
    for i, (x, y) in enumerate(pts):
        G.add_node(i, pos_x=float(x), pos_y=float(y))
    G.add_edges_from(edges)

    ensure_planar_or_die(G)
    return G

def main():
    out = Path("data/delaunay")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Make a few sizes
    graphs = [make_delaunay(n, seed) for (n, seed) in [(50, 0), (100, 1), (200, 2)]]

    # Example labels: harmonic function proxy (x+y > 1? else 0)
    for G in graphs:
        labels = {i: int(G.nodes[i]['pos_x'] + G.nodes[i]['pos_y'] > 1.0) for i in G.nodes()}
        nx.set_node_attributes(G, {i: {'label': labels[i]} for i in G.nodes()})

    splits = split_indices(len(graphs), seeds=[0, 1, 2])
    save_graph_list(graphs, out, splits)

    d = torch.load(out / "graph_0.pt")
    print("Graph 0:", d['x'].shape, "nodes; edge_index shape:", d['edge_index'].shape)
    print("First 5 node coords:", d['x'][:5])
    print("First 5 labels:", d['y'][:5])

if __name__ == "__main__":
    main()

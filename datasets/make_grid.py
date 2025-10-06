# datasets/make_grid.py
import networkx as nx
import numpy as np
from pathlib import Path
import shutil
import torch
from datasets.utils import save_graph_list, split_indices, ensure_planar_or_die

def make_grid(m, n):
    # Create a 2D grid graph
    G = nx.grid_2d_graph(m, n)  # nodes are (i, j)

    mapping = {}
    idx = 0
    for i in range(m):
        for j in range(n):
            mapping[(i, j)] = idx
            idx += 1

    # Assign coordinates BEFORE relabeling
    for (i, j) in G.nodes():
        G.nodes[(i, j)]['pos_x'] = float(j)
        G.nodes[(i, j)]['pos_y'] = float(i)

    # Relabel to integers
    G = nx.relabel_nodes(G, mapping)

    ensure_planar_or_die(G)
    return G

def main():
    out = Path("data/grids")
    if out.exists():
        shutil.rmtree(out)   # wipe old graphs
    out.mkdir(parents=True, exist_ok=True)

    # make a few grid sizes
    sizes = [(10,10), (15,15), (20,20), (25,25), (30,30)]
    graphs = [make_grid(m, n) for (m, n) in sizes]

    # checkerboard labels (stored under "label")
    for G in graphs:
        labels = {i: ((G.nodes[i]['pos_x'] + G.nodes[i]['pos_y']) % 2 == 0) for i in G.nodes()}
        nx.set_node_attributes(G, {i: {'label': int(labels[i])} for i in G.nodes()})

    splits = split_indices(len(graphs), seeds=[0,1,2])
    save_graph_list(graphs, out, splits)

    d = torch.load("data/grids/graph_0.pt")
    print("Shape:", d['x'].shape)
    print("Unique x values:", sorted(set(d['x'][:,0].tolist())))
    print("Unique y values:", sorted(set(d['x'][:,1].tolist())))

if __name__ == "__main__":
    main()

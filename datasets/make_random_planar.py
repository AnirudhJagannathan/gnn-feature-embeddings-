# datasets/make_random_planar.py
import networkx as nx
import numpy as np
import torch
import shutil
from pathlib import Path

from datasets.utils import save_graph_list, split_indices, ensure_planar_or_die

def make_random_planar(n_nodes, n_extra_edges=0, seed=0):
    rng = np.random.default_rng(seed)

    # Start with a random spanning tree from the complete graph
    G_complete = nx.complete_graph(n_nodes)
    T = nx.random_spanning_tree(G_complete, seed=seed)
    G = nx.Graph()
    G.add_nodes_from(T.nodes())
    G.add_edges_from(T.edges())

    # Candidate edges = all non-tree edges
    possible_edges = [(u, v) for u in range(n_nodes) for v in range(u + 1, n_nodes)
                      if not G.has_edge(u, v)]
    rng.shuffle(possible_edges)

    added = 0
    for u, v in possible_edges:
        G.add_edge(u, v)
        if nx.check_planarity(G)[0]:
            added += 1
            if added >= n_extra_edges:
                break
        else:
            G.remove_edge(u, v)

    # Planar embedding coordinates
    is_planar, embedding = nx.check_planarity(G)
    assert is_planar
    pos = nx.combinatorial_embedding_to_pos(embedding)
    for i in G.nodes():
        G.nodes[i]['pos_x'] = float(pos[i][0])
        G.nodes[i]['pos_y'] = float(pos[i][1])

    ensure_planar_or_die(G)
    return G


def main():
    out = Path("data/random_planar")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    graphs = []
    for n in [30, 50, 80]:
        for seed in [0, 1, 2, 3, 4]:
            # Add ~ n/2 extra edges
            graphs.append(make_random_planar(n, n_extra_edges=n//2, seed=seed))

    # Example labels: degree parity
    for G in graphs:
        labels = {i: G.degree[i] % 2 for i in G.nodes()}
        nx.set_node_attributes(G, {i: {'label': labels[i]} for i in G.nodes()})

    splits = split_indices(len(graphs), seeds=[0, 1, 2, 3, 4])
    save_graph_list(graphs, out, splits)

    d = torch.load(out / "graph_0.pt")
    print("Graph 0:", d['x'].shape, "nodes; edge_index shape:", d['edge_index'].shape)
    print("First 5 coords:", d['x'][:5])
    print("First 5 labels:", d['y'][:5])

if __name__ == "__main__":
    main()

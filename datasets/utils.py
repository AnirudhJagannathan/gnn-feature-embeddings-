import torch, numpy as np, networkx as nx
from pathlib import Path

def ensure_planar_or_die(G):
    is_planar, _ = nx.check_planarity(G)
    if not is_planar:
        raise ValueError("Graph is not planar")

def split_indices(N, seeds=[0], train=0.7, val=0.15):
    out = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = np.arange(N); rng.shuffle(idx)
        t = int(train*N); v = int((train+val)*N)
        out[s] = {'train': idx[:t].tolist(),
                  'val': idx[t:v].tolist(),
                  'test': idx[v:].tolist()}
    return out

def nx_to_tensors(G):
    nodes = list(range(len(G.nodes())))
    # coordinates (always stored as 'pos_x' and 'pos_y')
    xs = np.array([[G.nodes[i]['pos_x'], G.nodes[i]['pos_y']] for i in nodes], dtype=np.float32)

    # labels: stored under 'label' if present
    ys = np.array([G.nodes[i].get('label', -1) for i in nodes])

    # auto-detect type: int labels → int64, float labels → float32
    if np.issubdtype(ys.dtype, np.floating):
        ys = ys.astype(np.float32)
    else:
        ys = ys.astype(np.int64)

    edges = np.array(list(G.edges()), dtype=np.int64).T
    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(edges)


def save_graph_list(graphs, out_dir: Path, splits):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {'num_graphs': len(graphs), 'splits': splits}
    torch.save(meta, out_dir / "meta.pt")
    for i,G in enumerate(graphs):
        x, y, edge_index = nx_to_tensors(G)
        torch.save({'x': x, 'y': y, 'edge_index': edge_index}, out_dir / f"graph_{i}.pt")

def list_faces_from_embedding(G, embedding):
    """
    Extract all faces (cycles) from a PlanarEmbedding.
    Each face is returned as a list of node indices (cycle order).
    """
    seen = set()   # track directed edges ("darts")
    faces = []
    for u, v in G.edges():
        for a, b in [(u, v), (v, u)]:
            if (a, b) in seen:
                continue
            face = embedding.traverse_face(a, b)
            faces.append(face[:-1])  # drop duplicate last node
            for i in range(len(face)-1):
                seen.add((face[i], face[i+1]))
    return faces

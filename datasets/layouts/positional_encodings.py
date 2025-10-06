# datasets/layouts/positional_encodings.py
import numpy as np
import torch
import networkx as nx
import scipy.spatial
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, eigsh

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import torch
import networkx as nx

from datasets.utils import list_faces_from_embedding  # we define this in utils.py

def tutte_embedding(G, seed=0, boundary_rule="longest"):
    # --- Step 1: get planar embedding & faces ---
    is_planar, embedding = nx.check_planarity(G, counterexample=False)
    if not is_planar:
        raise ValueError("Graph is not planar")

    faces = list_faces_from_embedding(G, embedding)

    # --- Step 2: choose boundary face ---
    if boundary_rule == "longest":
        boundary = max(faces, key=len)
    elif boundary_rule == "random":
        rng = np.random.default_rng(seed)
        boundary = faces[rng.integers(len(faces))]
    else:
        raise ValueError(f"Unknown boundary_rule: {boundary_rule}")

    n = G.number_of_nodes()
    coords = np.zeros((n, 2), dtype=np.float64)

    # print("Running Tutte embedding, nodes:", G.number_of_nodes())

    # --- Step 3: place boundary nodes on unit circle ---
    m = len(boundary)
    angles = np.linspace(0, 2*np.pi, m, endpoint=False)
    for i, v in enumerate(boundary):
        coords[v] = [np.cos(angles[i]), np.sin(angles[i])]

    # --- Step 4: Laplacian system for interior nodes ---
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight=None, format='csr')
    deg = np.array(A.sum(axis=1)).flatten()
    L = csr_matrix(np.diag(deg)) - A

    interior = [i for i in range(n) if i not in boundary]
    Li = L[interior, :][:, interior]
    Lb = L[interior, :][:, boundary]

    xb = coords[boundary, 0]
    yb = coords[boundary, 1]

    xi = spsolve(Li, -Lb @ xb)
    yi = spsolve(Li, -Lb @ yb)

    for k, v in enumerate(interior):
        coords[v, 0] = xi[k]
        coords[v, 1] = yi[k]

    return torch.tensor(coords, dtype=torch.float32)

def spectral_embedding(G, dim=2):
    L = nx.laplacian_matrix(G).astype(float)
    # Compute smallest nonzero eigenvalues
    vals, vecs = eigsh(L, k=dim+1, which='SM')
    coords = vecs[:, 1:dim+1]  # skip first (all-ones)
    return torch.tensor(coords, dtype=torch.float32)


def force_embedding(G, seed=0):
    pos = nx.spring_layout(G, dim=2, seed=seed)
    coords = np.array([pos[i] for i in range(len(G.nodes()))])
    return torch.tensor(coords, dtype=torch.float32)


def random_embedding(G, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((len(G.nodes()), 2))
    return torch.tensor(coords, dtype=torch.float32)

import torch, networkx as nx, numpy as np
import matplotlib.pyplot as plt
from datasets.layouts.positional_encodings import tutte_embedding

# --- Load a graph (change the index to poke around) ---
d = torch.load("data/triangulated/graph_0.pt")
edge_index = d["edge_index"]; y = d["y"].numpy()
edges = edge_index.t().tolist()
G = nx.Graph(); G.add_edges_from(edges)

# --- Tutte embedding ---
pos = tutte_embedding(G).numpy()        # (N, 2)
r = np.linalg.norm(pos, axis=1)         # radius from origin

# --- 1) Basic stats ---
n = len(y)
n1 = int(y.sum()); n0 = n - n1
print(f"Nodes: {n} | class0: {n0} | class1: {n1} (pos rate={n1/n:.3f})")
print(f"Mean radius class0: {r[y==0].mean():.3f} | class1: {r[y==1].mean():.3f}")

# --- 2) Quick separability with radius ---
# Threshold that best splits classes by radius
best_thr = None; best_acc = -1
for thr in np.linspace(r.min(), r.max(), 200):
    pred = (r > thr).astype(int)        # class1 ≈ boundary ≈ large radius
    acc = (pred == y).mean()
    if acc > best_acc:
        best_acc = acc; best_thr = thr
print(f"Best radius threshold={best_thr:.3f} | acc={best_acc:.3f}")

# --- 3) Plots with better contrast and edges ---
fig, ax = plt.subplots(1,2, figsize=(10,5))

# Nodes + edges
ax[0].set_title("Tutte + edges (boundary vs interior)")
ax[0].axis("equal"); ax[0].axis("off")
# faint edges
for u,v in edges:
    ax[0].plot([pos[u,0], pos[v,0]], [pos[u,1], pos[v,1]], lw=0.7, alpha=0.25, color="gray")
# class colors with strong contrast
colors = np.where(y==1, "#E64A19", "#1976D2")  # orange vs blue
ax[0].scatter(pos[:,0], pos[:,1], c=colors, s=45, edgecolors="k", linewidths=0.4)
# draw unit circle for reference
t = np.linspace(0, 2*np.pi, 400)
ax[0].plot(np.cos(t), np.sin(t), "--", alpha=0.4, color="black")

# Radius histograms by class
ax[1].set_title("Radius distribution by class")
ax[1].hist(r[y==0], bins=15, alpha=0.7, label="class 0", density=True)
ax[1].hist(r[y==1], bins=15, alpha=0.7, label="class 1", density=True)
ax[1].axvline(best_thr, ls="--", color="k", label=f"best thr={best_thr:.2f}")
ax[1].legend()

plt.tight_layout()
plt.show()
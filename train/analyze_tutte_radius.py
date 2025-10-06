import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from datasets.planar_dataset import PlanarDataset
from datasets.layouts.positional_encodings import tutte_embedding

def evaluate_radius(dataset, split="train"):
    radii, labels = [], []
    for i in range(len(dataset)):
        d = torch.load(dataset.graph_files[i])
        G = nx.Graph()
        G.add_edges_from(d["edge_index"].T.tolist())
        pos = tutte_embedding(G)  # [N,2]

        r = torch.linalg.norm(pos, dim=1).numpy()
        y = d["y"].numpy()
        radii.extend(r)
        labels.extend(y)

    radii, labels = np.array(radii), np.array(labels)

    # grid search threshold
    best_thr, best_acc = None, 0
    for thr in np.linspace(0, 1, 200):
        preds = (radii > thr).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    # plot
    plt.figure(figsize=(6,5))
    for c in np.unique(labels):
        plt.hist(radii[labels==c], bins=20, alpha=0.6, label=f"class {c}")
    plt.axvline(best_thr, color="k", linestyle="--", label=f"best thr={best_thr:.2f}")
    plt.title(f"Tutte radius distribution ({dataset.root.stem}, {split})")
    plt.xlabel("radius"); plt.ylabel("count")
    plt.legend()
    plt.show()

    print(f"Best radius threshold: {best_thr:.2f}, Accuracy: {best_acc:.3f}")

if __name__ == "__main__":
    # Try on multiple datasets
    for name in ["triangulated", "random_planar", "delaunay"]:
        print(f"\n=== Dataset: {name} ===")
        ds = PlanarDataset(f"data/{name}", split="train", seed=0, pe_type=None)
        evaluate_radius(ds, split="train")

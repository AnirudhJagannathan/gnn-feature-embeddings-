import torch
import networkx as nx
import matplotlib.pyplot as plt
from datasets.planar_dataset import PlanarDataset
from datasets.layouts.positional_encodings import tutte_embedding

def radius_threshold_test(dataset_name="triangulated", split="train", seed=0):
    # Load dataset
    dataset = PlanarDataset(f"data/{dataset_name}", split=split, seed=seed, pe_type=None)
    correct, total = 0, 0
    all_r, all_labels = [], []

    for i in range(len(dataset)):
        d = dataset[i]
        G = nx.Graph()
        G.add_edges_from(d.edge_index.T.tolist())
        pos_enc = tutte_embedding(G)

        # radius of each node
        r = torch.norm(pos_enc, dim=1)

        # labels
        labels = d.y

        # store for distribution plot
        all_r.extend(r.tolist())
        all_labels.extend(labels.tolist())

    all_r = torch.tensor(all_r)
    all_labels = torch.tensor(all_labels)

    # Try different thresholds
    best_acc, best_thr = 0, None
    for thr in torch.linspace(0.1, 1.0, steps=50):
        preds = (all_r > thr).long()
        acc = (preds == all_labels).float().mean().item()
        if acc > best_acc:
            best_acc, best_thr = acc, thr.item()

    print(f"Best radius threshold: {best_thr:.2f}, Accuracy: {best_acc:.3f}")

    # Plot histogram
    plt.hist(all_r[all_labels==0], bins=20, alpha=0.6, label="class 0")
    plt.hist(all_r[all_labels==1], bins=20, alpha=0.6, label="class 1")
    plt.axvline(best_thr, color="k", linestyle="--", label=f"best thr={best_thr:.2f}")
    plt.legend()
    plt.xlabel("radius")
    plt.ylabel("count")
    plt.title(f"Tutte radius distribution ({dataset_name}, {split})")
    plt.show()

if __name__ == "__main__":
    radius_threshold_test("triangulated", split="train", seed=0)

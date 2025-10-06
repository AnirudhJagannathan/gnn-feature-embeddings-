import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from datasets.planar_dataset import PlanarDataset
from models.gcn import GCN
from models.gin import GIN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_radius_feature(data: Data):
    # Tutte embedding radius = sqrt(x^2 + y^2)
    if data.x.size(1) >= 2:  # need at least 2D coords
        radius = torch.norm(data.x[:, :2], dim=1, keepdim=True)
        data.radius = radius
    else:
        raise ValueError("Input features do not have XY coords for radius.")
    return data


def train(model_type="gcn", dataset_name="grids", features="baseline",
          seed=0, epochs=50, device="cpu"):
    # --- Seed ---
    set_seed(seed)

    # --- Load datasets ---
    train_dataset = PlanarDataset(f"data/{dataset_name}", split="train", seed=seed, pe_type="tutte", pe_mode="replace")
    val_dataset   = PlanarDataset(f"data/{dataset_name}", split="val",   seed=seed, pe_type="tutte", pe_mode="replace")
    test_dataset  = PlanarDataset(f"data/{dataset_name}", split="test",  seed=seed, pe_type="tutte", pe_mode="replace")

    def process_dataset(dataset):
        new_list = []
        for d in dataset:
            d = add_radius_feature(d)
            if features == "baseline":
                d.x = d.x  # already baseline (whatever PlanarDataset loaded)
            elif features == "radius":
                d.x = d.radius
            elif features == "concat":
                d.x = torch.cat([d.x, d.radius], dim=1)
            else:
                raise ValueError(f"Unknown features mode: {features}")
            new_list.append(d)
        return new_list

    train_dataset = process_dataset(train_dataset)
    val_dataset   = process_dataset(val_dataset)
    test_dataset  = process_dataset(test_dataset)

    print(f"Dataset: {dataset_name}, Features: {features}, Device: {device}")
    print("Input dim:", train_dataset[0].x.shape[1])
    print("Sample features:", train_dataset[0].x[:5])

    # --- DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16)
    test_loader  = DataLoader(test_dataset, batch_size=16)

    # --- Model ---
    in_dim = train_dataset[0].x.shape[1]
    out_dim = int(train_dataset[0].y.max().item()) + 1
    if model_type == "gcn":
        model = GCN(in_channels=in_dim, hidden_channels=64, out_channels=out_dim)
    elif model_type == "gin":
        model = GIN(in_channels=in_dim, hidden_channels=64, out_channels=out_dim)
    else:
        raise ValueError(f"Unknown model {model_type}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # --- Evaluation ---
    def evaluate(loader):
        model.eval()
        correct, total, loss_sum = 0, 0, 0
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch=getattr(batch, "batch", None))
            loss = F.cross_entropy(out, batch.y)
            loss_sum += loss.item() * batch.num_nodes
            preds = out.argmax(dim=-1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_nodes
        if total == 0:
            return 0.0, 0.0
        return correct / total, loss_sum / total

    # --- Training loop ---
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch=getattr(batch, "batch", None))
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes

        train_acc, _ = evaluate(train_loader)
        val_acc, val_loss = evaluate(val_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    test_acc, test_loss = evaluate(test_loader)
    print(f"\nTest Acc: {test_acc:.3f}, Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN with different feature modes")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gin"], help="Model type")
    parser.add_argument("--dataset", type=str, default="grids",
                        choices=["grids", "triangulated", "random", "delaunay"], help="Dataset name")
    parser.add_argument("--features", type=str, default="baseline",
                        choices=["baseline", "radius", "concat"], help="Which features to use")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    train(
        model_type=args.model,
        dataset_name=args.dataset,
        features=args.features,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
    )

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from datasets.planar_dataset import PlanarDataset
from models.gcn import GCN
from models.gin import GIN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model_type="gcn", dataset_name="grids", pe_type=None, pe_mode="concat", 
          seed=0, epochs=50, device="cpu"):
    # --- Set seed ---
    set_seed(seed)

    # --- Load datasets ---
    train_dataset = PlanarDataset(f"data/{dataset_name}", split="train", seed=seed, pe_type=pe_type, pe_mode=pe_mode)
    val_dataset   = PlanarDataset(f"data/{dataset_name}", split="val", seed=seed, pe_type=pe_type, pe_mode=pe_mode)
    test_dataset  = PlanarDataset(f"data/{dataset_name}", split="test", seed=seed, pe_type=pe_type, pe_mode=pe_mode)

    print(f"Dataset: {dataset_name}, PE: {pe_type}, Mode: {pe_mode}, Device: {device}")
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

    # --- Evaluation function ---
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

        train_loss = total_loss / len(train_dataset)
        train_acc, _ = evaluate(train_loader)
        val_acc, val_loss = evaluate(val_loader)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")

    # --- Final test evaluation ---
    test_acc, test_loss = evaluate(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN on planar graphs")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gin"], help="Model type")
    parser.add_argument("--dataset", type=str, default="grids", 
                        choices=["grids", "triangulated", "random_planar", "delaunay"], help="Dataset name")
    parser.add_argument("--pe", type=str, default=None, choices=[None, "tutte", "spectral", "force", "random"], help="Positional encoding type")
    parser.add_argument("--pe_mode", type=str, default="concat", choices=["concat", "replace", "polar"], help="PE mode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device: 'cpu' or 'cuda'")

    args = parser.parse_args()

    train(
        model_type=args.model,
        dataset_name=args.dataset,
        pe_type=args.pe,
        pe_mode=args.pe_mode,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
    )
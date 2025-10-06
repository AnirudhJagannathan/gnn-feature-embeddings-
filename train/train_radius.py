import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from datasets.planar_dataset import PlanarDataset

# --- MLP on radius only ---
class RadiusMLP(nn.Module):
    def __init__(self, hidden=32, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, r):
        x = F.relu(self.fc1(r))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def run_radius_baseline(dataset_name="triangulated", seed=0, epochs=50, device="cpu"):
    # --- Load Tutte embeddings, replace mode ---
    train_dataset = PlanarDataset(f"data/{dataset_name}", split="train", seed=seed, pe_type="tutte", pe_mode="replace")
    val_dataset   = PlanarDataset(f"data/{dataset_name}", split="val",   seed=seed, pe_type="tutte", pe_mode="replace")
    test_dataset  = PlanarDataset(f"data/{dataset_name}", split="test",  seed=seed, pe_type="tutte", pe_mode="replace")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16)
    test_loader  = DataLoader(test_dataset, batch_size=16)

    model = RadiusMLP(out_dim=int(train_dataset[0].y.max())+1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def evaluate(loader):
        model.eval()
        correct, total, loss_sum = 0, 0, 0
        for batch in loader:
            batch = batch.to(device)
            # compute radius = sqrt(x^2 + y^2)
            r = torch.norm(batch.x, dim=1, keepdim=True)
            out = model(r)
            loss = F.cross_entropy(out, batch.y)
            loss_sum += loss.item() * batch.num_nodes
            preds = out.argmax(dim=-1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_nodes

        if total == 0: 
            return 0.0, 0.0
            
        return correct / total, loss_sum / total

    # --- Train ---
    for epoch in range(1, epochs+1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            r = torch.norm(batch.x, dim=1, keepdim=True)
            optimizer.zero_grad()
            out = model(r)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            train_acc, _ = evaluate(train_loader)
            val_acc, val_loss = evaluate(val_loader)
            print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    test_acc, test_loss = evaluate(test_loader)
    print(f"\nTest Acc: {test_acc:.3f}, Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    run_radius_baseline(dataset_name="delaunay", seed=0, epochs=50, device="cpu")

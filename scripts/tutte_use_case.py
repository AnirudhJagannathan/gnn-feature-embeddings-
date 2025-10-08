import random
from dataclasses import dataclass
from typing import Optional

# numpy is optional – fall back gracefully if it is not installed.
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None
import torch
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from datasets.planar_dataset import PlanarDataset
from models.gcn import GCN


@dataclass
class ExperimentConfig:
    label: str
    pe_type: Optional[str]
    pe_mode: str = "concat"
    constant_features: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConstantFeatureDataset(Dataset):
    """Wrap a PlanarDataset and overwrite node features with a constant."""

    def __init__(self, base: PlanarDataset, value: float = 1.0):
        self.base = base
        self.value = value

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.base)

    def __getitem__(self, idx: int):  # type: ignore[override]
        data = self.base[idx].clone()
        data.x = torch.full((data.num_nodes, 1), self.value, dtype=data.x.dtype)
        return data


def build_dataloaders(dataset_name: str, cfg: ExperimentConfig, seed: int = 0):
    def make_split(split: str):
        return PlanarDataset(
            f"data/{dataset_name}",
            split=split,
            seed=seed,
            pe_type=cfg.pe_type,
            pe_mode=cfg.pe_mode,
        )

    train_base = make_split("train")
    val_base = make_split("val")
    test_base = make_split("test")

    if cfg.constant_features:
        train_base = ConstantFeatureDataset(train_base)
        val_base = ConstantFeatureDataset(val_base)
        test_base = ConstantFeatureDataset(test_base)

    train_loader = DataLoader(train_base, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_base, batch_size=16)
    test_loader = DataLoader(test_base, batch_size=16)
    sample = next(iter(train_loader))
    in_channels = sample.x.size(1)
    num_classes = int(sample.y.max().item()) + 1
    class_weights = compute_class_weights(train_loader, num_classes)
    class_distribution = summarize_distribution(train_loader, num_classes)
    return (
        train_loader,
        val_loader,
        test_loader,
        in_channels,
        num_classes,
        sample.x[:5],
        class_weights,
        class_distribution,
    )


def compute_class_weights(loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights to counter the heavy boundary imbalance."""

    counts = torch.zeros(num_classes, dtype=torch.float)
    for batch in loader:
        counts += torch.bincount(batch.y, minlength=num_classes).float()
    counts = counts.clamp_min_(1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights


def summarize_distribution(loader: DataLoader, num_classes: int) -> Counter:
    counts: Counter = Counter()
    for batch in loader:
        counts.update(batch.y.tolist())
    for cls in range(num_classes):
        counts.setdefault(cls, 0)
    return counts


def run_experiment(cfg: ExperimentConfig, epochs: int = 30, device: str = "cpu"):
    (
        train_loader,
        val_loader,
        test_loader,
        in_channels,
        num_classes,
        sample_feats,
        class_weights,
        class_distribution,
    ) = build_dataloaders(
        dataset_name="triangulated", cfg=cfg
    )
    model = GCN(in_channels=in_channels, hidden_channels=64, out_channels=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    weight_tensor = class_weights.to(device)
    boundary_class = min(class_distribution, key=class_distribution.get)

    print(f"\n== {cfg.label} ==")
    print(f"Input dim: {in_channels}")
    print("Sample features:\n", sample_feats)
    print("Class distribution (train split):", dict(class_distribution))
    print(f"Minority class treated as boundary: {boundary_class}")

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_nodes = 0
        correct = 0
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch=batch.batch if hasattr(batch, "batch") else None)
                loss = F.cross_entropy(out, batch.y, weight=weight_tensor)
                total_loss += loss.item() * batch.num_nodes
                total_nodes += batch.num_nodes
                preds = out.argmax(dim=-1)
                correct += (preds == batch.y).sum().item()
                true_labels.append(batch.y.cpu())
                pred_labels.append(preds.cpu())
        if total_nodes == 0:
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "boundary_recall": 0.0,
                "loss": 0.0,
            }
        y_true = torch.cat(true_labels)
        y_pred = torch.cat(pred_labels)
        metrics = compute_metrics(y_true, y_pred, num_classes, boundary_class)
        metrics["loss"] = total_loss / total_nodes
        return metrics

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_nodes = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch if hasattr(batch, "batch") else None)
            loss = F.cross_entropy(out, batch.y, weight=weight_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes
        train_loss = total_loss / max(total_nodes, 1)
        val_metrics = evaluate(val_loader)
        print(
            "Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, "
            "Val Acc: {acc:.3f}, Balanced Acc: {bal:.3f}, Boundary Recall: {rec:.3f}".format(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                acc=val_metrics["accuracy"],
                bal=val_metrics["balanced_accuracy"],
                rec=val_metrics["boundary_recall"],
            )
        )

    test_metrics = evaluate(test_loader)
    print(
        "Test Loss: {loss:.4f}, Test Acc: {acc:.3f}, Balanced Acc: {bal:.3f}, Boundary Recall: {rec:.3f}".format(
            loss=test_metrics["loss"],
            acc=test_metrics["accuracy"],
            bal=test_metrics["balanced_accuracy"],
            rec=test_metrics["boundary_recall"],
        )
    )


def compute_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int, boundary_class: int
):
    total = y_true.numel()
    accuracy = (y_true == y_pred).float().mean().item()
    recalls = []
    boundary_recall = 0.0
    for cls in range(num_classes):
        mask = y_true == cls
        if mask.sum() == 0:
            recalls.append(1.0)
            continue
        recall = (y_pred[mask] == cls).float().mean().item()
        recalls.append(recall)
        if cls == boundary_class:
            boundary_recall = recall
    balanced_accuracy = sum(recalls) / len(recalls)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "boundary_recall": boundary_recall,
    }


if __name__ == "__main__":
    set_seed(0)

    # 1) Remove all geometric signal — the model sees identical node features.
    no_geometry_cfg = ExperimentConfig(
        label="No positional encoding (features collapsed to constant)",
        pe_type=None,
        pe_mode="concat",
        constant_features=True,
    )
    run_experiment(no_geometry_cfg)

    # 2) Replace features with Tutte embeddings, recovering geometry from topology.
    tutte_cfg = ExperimentConfig(
        label="Tutte embedding replaces missing coordinates",
        pe_type="tutte",
        pe_mode="replace",
        constant_features=False,
    )
    run_experiment(tutte_cfg)
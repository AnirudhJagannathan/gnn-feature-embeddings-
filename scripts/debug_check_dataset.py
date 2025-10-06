from datasets.planar_dataset import PlanarDataset
from torch_geometric.loader import DataLoader

# Load splits
train_dataset = PlanarDataset("data/grids", split="train", seed=0)
val_dataset   = PlanarDataset("data/grids", split="val", seed=0)
test_dataset  = PlanarDataset("data/grids", split="test", seed=0)

# Make DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16)
test_loader  = DataLoader(test_dataset, batch_size=16)

# Inspect one batch
batch = next(iter(train_loader))
print(batch)
print("x:", batch.x.shape)
print("edges:", batch.edge_index.shape)
print("y:", batch.y.shape)

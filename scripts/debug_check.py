# debug_check.py
import torch

data = torch.load("data/grids/graph_0.pt")

print("Node features (coordinates):")
print(data['x'][:20])   # should look like grid points (0..9)

print("Node labels (checkerboard parity):")
print(data['y'][:20])   # will only be 0 or 1

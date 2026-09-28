import torch
import torch.nn as nn

# Example batch:
# 4 samples, each sample has 3 features
X = torch.tensor([
    [10.0, 100.0, 1000.0],
    [20.0, 200.0, 2000.0],
    [30.0, 300.0, 3000.0],
    [40.0, 400.0, 4000.0]
])

# Batch Normalization for 3 features
batch_norm = nn.BatchNorm1d(3)

# Normalize the batch
output = batch_norm(X)

print("Before Batch Normalization:")
print(X)

print("\nAfter Batch Normalization:")
print(output)
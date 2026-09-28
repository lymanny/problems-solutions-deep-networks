import torch
import torch.nn as nn

# Example data:
# 2 samples, each sample has 3 features
X = torch.tensor([
    [10.0, 100.0, 1000.0],
    [20.0, 200.0, 2000.0]
])

# Layer Normalization for 3 features
layer_norm = nn.LayerNorm(3)

# Normalize each sample
output = layer_norm(X)

print("Before Layer Normalization:")
print(X)

print("\nAfter Layer Normalization:")
print(output)
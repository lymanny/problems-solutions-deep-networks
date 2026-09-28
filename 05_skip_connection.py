import torch
import torch.nn as nn


# Simple block with a Skip Connection
class SkipBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(3, 3)

    def forward(self, x):
        # Normal layer output
        fx = self.layer(x)

        # Skip Connection
        output = fx + x

        return output


# Example input
x = torch.tensor([[1.0, 2.0, 3.0]])

# Create model
model = SkipBlock()

# Forward pass
output = model(x)

print("Input:")
print(x)

print("\nOutput:")
print(output)
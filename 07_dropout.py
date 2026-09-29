import torch
import torch.nn as nn

# Example neuron outputs
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Dropout = 0.4
# 40% may be randomly turned off during training
dropout = nn.Dropout(0.4)

# Training mode
dropout.train()

print("Training:")
print(dropout(x))
print(dropout(x))
print(dropout(x))

# Evaluation mode
dropout.eval()

print("\nTesting / Evaluation:")
print(dropout(x))
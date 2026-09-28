import torch
import torch.nn as nn

# ReLU activation function
relu = nn.ReLU()

# Example values
values = torch.tensor([-5.0, -3.0, -1.0, 0.0, 2.0, 5.0])

# Apply ReLU
output = relu(values)

print("Input :", values)
print("Output:", output)


# Dying ReLU example
negative_values = torch.tensor([-5.0, -4.0, -3.0, -2.0])

dying_output = relu(negative_values)

print("\nDying ReLU Example")
print("Before ReLU:", negative_values)
print("After ReLU :", dying_output)
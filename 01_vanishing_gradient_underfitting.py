import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ==================================================
# 1. Vanishing Gradient Example
# ==================================================

print("=== Vanishing Gradient ===")

gradient = 1.0

print("Start Gradient:", gradient)

for layer in range(1, 8):

    # Simulate the gradient getting smaller
    gradient = gradient * 0.2

    print(f"Layer {layer}: {gradient}")


# ==================================================
# 2. Underfitting Example
# ==================================================

print("\n=== Underfitting ===")

# Curved data
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Real pattern
y = np.array([2, 5, 10, 17, 26, 37])

# Simple Linear Regression model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
prediction = model.predict(X)

# Calculate error
mse = mean_squared_error(y, prediction)

print("MSE:", mse)

# Show actual data
plt.scatter(X, y, label="Actual Data")

# Show model prediction
plt.plot(X, prediction, label="Simple Linear Model")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Underfitting Example")
plt.legend()

plt.show()
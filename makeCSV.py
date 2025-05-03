import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

num_points_per_class = 100

# Generate class 0: circular cluster
theta0 = np.random.uniform(0, 2 * np.pi, num_points_per_class)
r0 = np.random.uniform(10, 30, num_points_per_class)
x0 = 50 + r0 * np.cos(theta0)
y0 = 50 + r0 * np.sin(theta0)
class0 = np.column_stack((x0, y0, np.zeros(num_points_per_class)))

# Generate class 1: sine wave pattern
x1 = np.linspace(0, 100, num_points_per_class)
y1 = 50 + 20 * np.sin(0.1 * x1) + np.random.normal(0, 2, num_points_per_class)
class1 = np.column_stack((x1, y1, np.ones(num_points_per_class)))

# Generate class 2: diagonal band with noise
x2 = np.random.uniform(0, 100, num_points_per_class)
y2 = x2 + np.random.uniform(-10, 10, num_points_per_class)
class2 = np.column_stack((x2, y2, np.full(num_points_per_class, 2)))

# Combine all classes
data = np.vstack((class0, class1, class2))

# Clip values to stay within 0–100 range
data[:, 0] = np.clip(data[:, 0], 0, 100)
data[:, 1] = np.clip(data[:, 1], 0, 100)

# Create DataFrame
df = pd.DataFrame(data, columns=['feature1', 'feature2', 'class'])

# Save to CSV
df.to_csv('knn_data.csv', index=False)

print("CSV file 'knn_data.csv' generated with 3-class nonlinear data.")

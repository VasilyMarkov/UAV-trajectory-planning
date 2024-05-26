import matplotlib.pyplot as plt
import numpy as np

# Number of vertices
n = 7

# Minimum diameter
min_diameter = 0.3

# Generate random vertices in 2D space
vertices = np.random.rand(n, 2)

# Ensure the minimum diameter
while True:
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(vertices[i] - vertices[j]) < min_diameter:
                vertices = np.random.rand(n, 2)
                break
        else:
            break
    else:
        break

# Ensure the shape is convex by sorting the vertices in angular order
vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]

# Plot the convex polygon
plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.5)
plt.scatter(vertices[:, 0], vertices[:, 1], s=50, alpha=1)
plt.show()
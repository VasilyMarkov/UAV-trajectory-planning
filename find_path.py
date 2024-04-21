import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_random_convex_polygon(min_x, max_x, min_y, max_y):
    points = np.random.uniform(min_x, max_x, (4, 2))
    points[:, 1] = np.random.uniform(min_y, max_y, 4)

    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    indices = np.argsort(angles)
    vertices = points[indices]

    return vertices

def draw_polygon(vertices, borders):
    fig, ax = plt.subplots()
    polygon = patches.Polygon(vertices, fill=True)
    ax.add_patch(polygon)
    ax.set_xlim(-5, borders[0]+5)
    ax.set_ylim(-5, borders[1]+5)
    plt.show()


H = 300
W = 300

polygon = generate_random_convex_polygon(0, H, 0, W)


draw_polygon(polygon, (H, W))
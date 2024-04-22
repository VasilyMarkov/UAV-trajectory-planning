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


H = 355
W = 355
uav = 2
polygon = [[0,0], [20,300], [300,350], [320,0]]
n = 320/10
lines = np.zeros([2,2,int(n)])
step = 10
# lines[:, 0, 0] = np.arange(lines.shape[0]) * step
# for i in range(lines.shape[2]):
#     lines[i, :, 0]
# draw_polygon(polygon, (H, W))



for i in range(lines.shape[2]):
    lines[:, :, i][:, 0] = i*step
    lines[1, 1, i] = 300

fig, ax = plt.subplots()
polygon = patches.Polygon(polygon, fill=True)
ax.add_patch(polygon)
ax.set_xlim(-5, W+5)
ax.set_ylim(-5, H+5)

for i in range(lines.shape[2]):
    line = lines[:, :, i]
    x = [point[0] for point in line]
    y = [point[1] for point in line]
    print(line)
    ax.plot(x, y, color = 'r')

plt.show()

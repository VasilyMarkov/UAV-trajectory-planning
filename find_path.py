import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon

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


def intersectLineWithPolygon(polygon, line):
    # polygon = Polygon([(0, 0), (20, 300), (300, 350), (320, 0)])
    polygon = Polygon(polygon)
    intersection_points = []
    intersection = polygon.intersection(line)
    # print(intersection.coords)
    intersection_points.append(intersection)
    points = list(intersection.coords)
    # if intersection.geom_type == 'Point':
    #     intersection_points.append(intersection)
    # elif intersection.geom_type == 'MultiPoint':
    #     intersection_points.extend(list(intersection.geoms))
    return points 

H = 355
W = 355
uav = 2
polygon = [[0,0], [20,300], [300,350], [320,0]]
a_pol = [[100,100], [95,120], [100,140], [105,120]]
n = 320/10
lines = np.zeros([2,2,int(n)])
step = 10

for i in range(lines.shape[2]):
    lines[:, :, i][:, 0] = i*step
    lines[1, 1, i] = 400

line_str = []

for i in range(lines.shape[2]):
    line_str.append(LineString(lines[:, :, i]))

intersects = []
for line in line_str:
    intersects.append(intersectLineWithPolygon(polygon, line))

fig, ax = plt.subplots()
polygon = patches.Polygon(polygon, fill=False)
a_pol = patches.Polygon(a_pol, fill=True, color = 'g')

ax.add_patch(polygon)
ax.add_patch(a_pol)
ax.set_xlim(-5, W+5)
ax.set_ylim(-5, H+5)

for i in range(lines.shape[2]):
    line = lines[:, :, i]
    x = [point[0] for point in line]
    y = [point[1] for point in line]
    ax.plot(x, y, color = 'b')

for point in intersects:
    ax.scatter(*zip(*point), color = 'r', s = 15)
    
plt.show()

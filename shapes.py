import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon

def generates_rays(polygon, step, traversal = 'vertical'):
    height = np.max(polygon[:, 1])-np.min(polygon[:, 1])
    width = np.max(polygon[:, 0])-np.min(polygon[:, 0])

    amount = {
        "horizontal": int(height/step),
        "vertical": int(width/step),
    }
    lines = np.zeros([2,2, amount[traversal]])
    
    for i in range(lines.shape[2]):
        if traversal == 'vertical':
            lines[:, :, i][:, 0] = i*step
            lines[1, 1, i] = height
            lines[:, 0, i] += int(step/2)
        elif traversal == 'horizontal':
            lines[:, :, i][:, 1] = i*step
            lines[1, 0, i] = width
            lines[:, 1, i] += int(step/2)


    return lines


def intersection(polygon, line):
    line_str = LineString(line)
    polygon = Polygon(polygon)
    intersection_points = []
    intersection = polygon.intersection(line_str)
    intersection_points.append(intersection)
    points = list(intersection.coords)
    return points 


def plot_lines(plot, lines):
    for i in range(lines.shape[2]):
        line = lines[:, :, i]
        x = [point[0] for point in line]
        y = [point[1] for point in line]
        plot.plot(x, y, color = 'b')


def plot_intersection(plot, intersects):
    for point in intersects:
        ax.scatter(*zip(*point), color = 'r', s = 15)

global_pol = [[0,0], [20,300], [300,350], [320,0]]
global_pol =np.array(global_pol)
a_pol = [[105,100], [102,120], [105,140], [108,120]]
a_pol =np.array(a_pol)

polygons = {
  "glob_pol": global_pol,
  "a_pol": a_pol,
}

lines = generates_rays(global_pol, 10, traversal='horizontal')

intersect_points = []
for i in range(lines.shape[2]):
    intersect_points.append(intersection(polygons['glob_pol'], lines[:, :, i]))

polygon = patches.Polygon(polygons['glob_pol'], fill=False)

fig, ax = plt.subplots()
ax.add_patch(polygon)
plot_lines(ax, lines)
plot_intersection(ax, intersect_points)
plt.show()

# print(polygons['a_pol'])
# print(polygons['a_pol'][:, 1])
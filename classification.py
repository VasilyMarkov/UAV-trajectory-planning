import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon

class MyPolygon:
    def __init__(self, points):
        self.points = np.array(points)
        self._update()

    def print(self):
        print(self.points)

    def _update(self):   
        self.min_x = np.min(self.points[:, 0])
        self.max_x = np.max(self.points[:, 0])
        self.min_y = np.min(self.points[:, 1])
        self.max_y = np.max(self.points[:, 1])
        self.height = self.max_y-self.min_y
        self.width = self.max_x-self.min_x

    def move_x(self, x):
        self.points[:, 0] += x
        self._update()

    def move_y(self, y):
        self.points[:, 1] += y
        self._update()


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


def classification(global_poly, lines, poly, step, traversal='horizontal'):
    treshold = 0
    if traversal == 'vertical':
        diffs_min = np.abs(poly.min_x - lines[0,0,:])
        diffs_max = np.abs(poly.max_x - lines[0,0,:])
        closest_line_left = lines[:, :, np.argmin(diffs_min)]
        closest_line_right= lines[:, :, np.argmin(diffs_max)]
        is_crossing_left = poly.min_x - closest_line_left[0][0] <= treshold 
        is_crossing_right = closest_line_right[0][0] - poly.max_x <= treshold 
        if is_crossing_left == False and is_crossing_right == False:
            return 'A class'
    if traversal == 'horizontal':
        ...


def plot_lines(plot, lines):
    for i in range(lines.shape[2]):
        line = lines[:, :, i]
        x = [point[0] for point in line]
        y = [point[1] for point in line]
        plot.plot(x, y, color = 'b')


def plot_intersection(plot, intersects):
    for point in intersects:
        ax.scatter(*zip(*point), color = 'r', s = 15)


a_poly = MyPolygon([[105,100], 
                    [102,120], 
                    [105,140], 
                    [108,120]])

glob_poly = MyPolygon([[0,0], 
                       [20,300], 
                       [300,350], 
                       [320,0]])

polygons = {
  "glob": glob_poly,
  "a": a_poly,
}

lines = generates_rays(polygons['glob'].points, 10, traversal='vertical')

intersect_points = []
for i in range(lines.shape[2]):
    intersect_points.append(intersection(polygons['glob'].points, lines[:, :, i]))


polygons['a'].move_x(5)

print(classification(polygons['glob'], lines, polygons['a'], step=10, traversal='vertical'))

fig, ax = plt.subplots()
ax.add_patch(patches.Polygon(polygons['glob'].points, fill=False))
ax.add_patch(patches.Polygon(polygons['a'].points, color = 'black', fill=True))

plot_lines(ax, lines)
plot_intersection(ax, intersect_points)
plt.show()

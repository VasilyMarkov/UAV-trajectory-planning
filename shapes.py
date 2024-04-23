import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from shapely.geometry import LineString, Polygon
from scipy.spatial import ConvexHull

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
        self.mass_center = (np.mean(self.points[:, 0]), np.mean(self.points[:, 1]))

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

def min_distance(poly1, poly2):
    dist = []
    for i in range(poly1.points.shape[0]):
        for j in range(poly2.points.shape[0]):
            dist.append(math.hypot(poly1.points[i, 0]-poly2.points[j, 0], 
                              poly1.points[i, 1]-poly2.points[j, 1]))
    min_dist = np.min(np.array(dist))
    return min_dist


def combine(poly1, poly2):
    points = np.vstack((poly1.points, poly2.points)) 
    hull = ConvexHull(points)
    vertices = hull.vertices
    bounding_polygon = MyPolygon(points[vertices])
    return bounding_polygon 

# def find_and_combine_close_polygons(polygons, treshold):
#     for i in range(polygons.shape[0]):
#         for j in range(1, polygons.shape[0]):
#             if min_distance(polygons[i], polygons[j]) < treshold:


def classification(global_poly, lines, polygons, step, traversal='horizontal'):
    polygons = polygons.copy()
    treshold = 0
    classified = {'a': [], 'b': [], 'c': [], 'd': []}
    index = 0
    for poly in polygons:
        if traversal == 'vertical':
            diffs_min = np.abs(poly.min_x - lines[0,0,:])
            diffs_max = np.abs(poly.max_x - lines[0,0,:])
            closest_line_left = lines[:, :, np.argmin(diffs_min)]
            closest_line_right= lines[:, :, np.argmin(diffs_max)]
            is_crossing_left = poly.min_x - closest_line_left[0][0] <= treshold 
            is_crossing_right = closest_line_right[0][0] - poly.max_x <= treshold 
            if is_crossing_left == False and is_crossing_right == False:
                classified['a'].append(polygons.pop(index))
        if traversal == 'horizontal':
            ...
        index += 1
    return classified


def plot_lines(plot, lines):
    for i in range(lines.shape[2]):
        line = lines[:, :, i]
        x = [point[0] for point in line]
        y = [point[1] for point in line]
        plot.plot(x, y, color = 'b')


def plot_intersection(plot, intersects):
    for point in intersects:
        ax.scatter(*zip(*point), color = 'r', s = 15)


glob_poly = MyPolygon([[0,0], 
                       [20,300], 
                       [300,350], 
                       [320,0]])

a_poly = MyPolygon([[105,100], 
                    [102,120], 
                    [105,140], 
                    [108,120]])

b_poly = MyPolygon([[200,200], 
                    [200,225], 
                    [225,225], 
                    [225,200]])

c_poly = MyPolygon([[233,200], 
                    [233,225], 
                    [258,225], 
                    [258,200]])


c_poly.move_y(20)

polygons = [a_poly, b_poly, c_poly]

lines = generates_rays(glob_poly.points, 10, traversal='vertical')

intersect_points = []
for i in range(lines.shape[2]):
    intersect_points.append(intersection(glob_poly.points, lines[:, :, i]))


print(classification(glob_poly, lines, polygons, step=10, traversal='vertical'))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
ax.add_patch(patches.Polygon(glob_poly.points, fill=False))
for poly in polygons:
    ax.add_patch(patches.Polygon(poly.points, color = 'black', fill=True))
    
ax.add_patch(patches.Polygon(combine(b_poly, c_poly).points, color = 'black', fill=True))

plot_lines(ax, lines)
plot_intersection(ax, intersect_points)

ax.set_xlim([0, 360])
ax.set_ylim([0, 360])
plt.show()
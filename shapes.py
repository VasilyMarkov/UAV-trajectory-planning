import numpy as np
from itertools import chain
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
    dist_x = []
    dist_y = []
    for i in range(poly1.points.shape[0]):
        for j in range(poly2.points.shape[0]):
            dist.append(math.hypot(poly1.points[i, 0]-poly2.points[j, 0], 
                              poly1.points[i, 1]-poly2.points[j, 1]))
            dist_x.append(np.abs(poly1.points[i, 0]-poly2.points[j, 0]))
            dist_y.append(np.abs(poly1.points[i, 1]-poly2.points[j, 1]))
    min_dist_x = np.min(np.array(dist_x))
    min_dist_y = np.min(np.array(dist_y))
    return min_dist_x, min_dist_y


'''
An obstacle that due to size and conﬁguration in relation 
to the driving direction does not affect the coverage plan generation.
Returns a new list with no polygons of this class.
'''
def a_class(polygons, lines, threshold, traversal='vertical'):
    index = 0
    output = []
    threshold = 2
    for poly in polygons:
        if traversal == 'vertical':
            diffs_min = np.abs(poly.min_x - lines[0,0,:])
            diffs_max = np.abs(poly.max_x - lines[0,0,:])
            closest_line_left = lines[:, :, np.argmin(diffs_min)]
            closest_line_right= lines[:, :, np.argmin(diffs_max)]
            is_crossing_left = poly.min_x - closest_line_left[0][0] <= threshold 
            is_crossing_right = closest_line_right[0][0] - poly.max_x <= threshold 
            if is_crossing_left == False and is_crossing_right == False:
                output.append(polygons.pop(index))
        if traversal == 'horizontal':
            ...
        index += 1
    return output


def point_filling(polygon):
    size =  polygon.points.shape[0]
    new_points = []
    for i in range(size):
        p2 = polygon.points[(i+1) % size]
        p0 = polygon.points[i % size]
        new_points.append((p2 + p0)/2)
    new_points = np.array(new_points)
    result = np.zeros([len(polygon.points) + len(new_points), 2])
    result[::2] = polygon.points
    result[1::2] = new_points
    return MyPolygon(result)


def combine(poly1, poly2):
    points = np.vstack((poly1.points, poly2.points)) 
    hull = ConvexHull(points)
    vertices = hull.vertices
    bounding_polygon = MyPolygon(points[vertices])
    return bounding_polygon 


def find_and_combine_close_polygons(polygons, threshold):
    polygons = [point_filling(i) for i in polygons]

    used = []
    output = []
    uav_size = 10
    
    for i in range(len(polygons)):
        if (i in used) == False:
            for j in range(1, len(polygons)):
                if i != j:
                    dist_x, dist_y = min_distance(polygons[i], polygons[j])
                    if dist_x <= threshold and dist_y <= 1.5*uav_size:
                        output.append(combine(polygons[i], polygons[j]))
                        used.append(i)
                        used.append(j)

    residue = [el for i, el in enumerate(polygons) if i not in used]
    return residue+output


'''                 
Obstacles where the minimum distance between another obstacle is less than the operating width,
w. In this case both obstacles are classiﬁed as of type C and a subroutine is used to ﬁnd the minimal bounding polygon (MBP) to
enclose these obstacles.
Returns a new list with combined polygons of this class.
'''
def c_class(polygons, threshold, traversal='vertical'):
    return find_and_combine_close_polygons(polygons, threshold)

'''                 
Obstacles where their boundary intersects with the inner boundary of the ﬁeld.
Removes matching polygons from the source list.
'''
def b_class(global_poly, polygons, threshold):
    # used = []
    # output = []

    # for i in range(len(polygons)):
    #     dist_x, dist_y = min_distance(global_poly, polygons[i])
    #     if dist_x <= threshold and dist_y <= threshold:
    #         output.append(combine(polygons[i], polygons[j]))
    #         used.append(i)
    # for index in reversed(used):
    #     del polygons[index]
    # return output 
    ...   

def classification(global_poly, lines, polygons, step, traversal='vertical'):
    polygons = polygons.copy()
    threshold = 10
    classified = {'a': [], 'b': [], 'c': [], 'd': []}
    
    classified['a'] = a_class(polygons, lines, threshold, traversal)
    classified['c'] = c_class(polygons, threshold, traversal) 
    # classified['b'] = b_class(polygons, lines, traversal)
    # classified['d'][0] = polygons
    
    return classified


def plot_lines(plot, lines):
    for i in range(lines.shape[2]):
        line = lines[:, :, i]
        x = [point[0] for point in line]
        y = [point[1] for point in line]
        plot.plot(x, y, color = 'black')


def plot_intersection(plot, intersects):
    for point in intersects:
        ax.scatter(*zip(*point), color = 'r', s = 15)


glob_poly = MyPolygon([[0,0], 
                       [20,300], 
                       [300,350], 
                       [320,0]])

a_poly = MyPolygon([[105,100], 
                    [103,120], 
                    [105,140], 
                    [107,120]])

c1_poly = MyPolygon([[200,200], 
                     [200,225], 
                     [225,225], 
                     [225,200]])

c2_poly = MyPolygon([[233,200], 
                     [233,225], 
                     [258,225], 
                     [258,200]])
                    
b1_poly = MyPolygon([[50,5], 
                     [50,30], 
                     [75,30], 
                     [75,5]])

b2_poly = MyPolygon([[53,5], 
                     [53,30], 
                     [78,30], 
                     [78,5]])
                    
# b2_poly = MyPolygon([[233,200], 
#                      [233,225], 
#                      [258,225], 
#                      [258,200]])

d_poly = MyPolygon([[200,100], 
                    [200,125], 
                    [225,125], 
                    [225,100]])


c1_poly.move_y(12)
c2_poly.move_x(-6)
a_poly.move_x(5)
polygons = [a_poly, c1_poly, c2_poly, b1_poly, b2_poly, d_poly]

lines = generates_rays(glob_poly.points, 10, traversal='vertical')

intersect_points = []
for i in range(lines.shape[2]):
    intersect_points.append(intersection(glob_poly.points, lines[:, :, i]))


marked_obstacles = classification(glob_poly, lines, polygons, step=10, traversal='vertical')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
ax.add_patch(patches.Polygon(glob_poly.points, fill=False))

for elem in marked_obstacles['a']:
    ax.add_patch(patches.Polygon(elem.points, color = 'g', fill=True))

for elem in marked_obstacles['b']:
    ax.add_patch(patches.Polygon(elem.points, color = 'r', fill=True))

for elem in marked_obstacles['c']:
    ax.add_patch(patches.Polygon(elem.points, color = 'b', fill=True))

for elem in marked_obstacles['d']:
    ax.add_patch(patches.Polygon(elem.points, color = 'purple', fill=True))

plot_lines(ax, lines)
plot_intersection(ax, intersect_points)

ax.set_xlim([0, 360])
ax.set_ylim([0, 360])
plt.show()

import numpy as np
from shapely.geometry import LineString, Polygon
import math
from scipy.spatial import ConvexHull

class MyPolygon:
    def __init__(self, points):
        self.points = np.array(points)
        self._update()

    def print(self):
        print(f'min x:{self.min_x}, max x:{self.max_x},  min y:{self.min_y},  max y:{self.max_y}')
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


def distance_point_to_line(x, y, x1, y1, x2, y2):
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    distance = abs(a*x + b*y + c) / math.sqrt(a**2 + b**2)
    return distance

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


def closedPolygons(global_poly, poly, threshold):
    min_distances = []
    global_poly_size = len(global_poly.points)
    for i in range(len(poly.points)):
        for j in range(global_poly_size):
            line_point1 = global_poly.points[(j+1) % global_poly_size]
            line_point0 = global_poly.points[j % global_poly_size]
            point = poly.points[i]
            dist = distance_point_to_line(point[0], 
                                          point[1], 
                                          line_point0[0], 
                                          line_point0[1], 
                                          line_point1[0], 
                                          line_point1[1]) 
            min_distances.append(dist)
    min_dist = min(min_distances)
    return min_dist <= threshold

'''                 
Obstacles where their boundary intersects with the inner boundary of the ﬁeld.
Returns a new list with matching polygons and removes ones from the source list.
'''
def b_class(global_poly, polygons, threshold):
    output = []
    for i in range(len(polygons)):
        if closedPolygons(global_poly, polygons[i], 5):
            output.append(polygons.pop(i))
    return output


def classification(global_poly, lines, polygons, step, traversal='vertical'):
    polygons = polygons.copy()
    threshold = 10
    classified = {'a': [], 'b': [], 'c': [], 'd': []}
    
    classified['a'] = a_class(polygons, lines, threshold, traversal)
    polygons = find_and_combine_close_polygons(polygons, threshold)
    classified['b'] = b_class(global_poly, polygons, threshold)
    classified['d'] = polygons
    
    return classified


def plot_lines(plot, lines, in_color, width):
    for i in range(lines.shape[2]):
        line = lines[:, :, i]
        x = [point[0] for point in line]
        y = [point[1] for point in line]
        plot.plot(x, y, color = in_color, linewidth = width)



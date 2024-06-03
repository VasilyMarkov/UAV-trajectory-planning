from classification import *
from functools import cmp_to_key
from shapely.geometry import Polygon, MultiPolygon

class Point:    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def move(self, x, y):
        self.x += x
        self.y += y
    def print(self):
        print(f'x: {self.x}, y: {self.y}')


class Line:
    def  __init__(self, Point1, Point2):
        self.Point1 = Point1
        self.Point2 = Point2
        self.vector = np.array([self.Point2.x - self.Point1.x, self.Point2.y - self.Point1.y])
    def move(self, x, y):
        self.Point1.move(x,y)
        self.Point2.move(x,y)
        self.vector = np.array([self.Point2.x - self.Point1.x, self.Point2.y - self.Point1.y])
    def print(self):
        print(f'Line: p1: ({self.Point1.x}, {self.Point1.y}), p2: ({self.Point2.x}, {self.Point2.y})')


class Slice:
    def  __init__(self, Line, Points, sl_id):
        self.Line = Line
        self.Points = Points
        self.id = sl_id
    def print(self):
        print(f'Slice: id {self.id}')
        self.Line.print()
        print('Points: ')
        for p in self.Points:
            p.print()


def intersect(polygon, line):
    line_str = LineString([[line.Point1.x, line.Point1.y], [line.Point2.x, line.Point2.y]] )
    polygon = Polygon(polygon)
    intersection_points = []
    intersection = polygon.intersection(line_str)
    intersection_points.append(intersection)
    points = list(intersection.coords)
    return points 


def intersect1(polygon, line):
    line_str = LineString([[line[0][0], line[0][1]], [line[1][0], line[1][1]]] )
    polygon = Polygon(polygon)
    intersection_points = []
    intersection = polygon.intersection(line_str)
    intersection_points.append(intersection)
    points = list(intersection.coords)
    return points 


def create_slices(glob, obstacles):
    min_x = list(map(lambda i, o: (i, o.min_x), range(len(obstacles)), obstacles))
    max_x = list(map(lambda i, o: (i, o.max_x), range(len(obstacles)), obstacles))
    slices = []
    for x in min_x:
        line = Line(Point(x[1], glob.min_y), Point(x[1],glob.max_y))
        points = intersect(glob.points, line)
        line = Line(Point(points[0][0], points[0][1]), Point(points[1][0], points[1][1]))
        points = intersect(obstacles[x[0]].points, line)
        p_list = []
        for p in points:
            p_list.append(Point(p[0], p[1]))
        slices.append(Slice(line, p_list, x[0]))

    for x in max_x:
        line = Line(Point(x[1], glob.min_y), Point(x[1],glob.max_y))
        points = intersect(glob.points, line)
        line = Line(Point(points[0][0], points[0][1]), Point(points[1][0], points[1][1]))
        points = intersect(obstacles[x[0]].points, line)
        p_list = []
        for p in points:
            p_list.append(Point(p[0], p[1]))
        slices.append(Slice(line, p_list, x[0]))
    # boundaries = sorted(set(min_x+max_x))
    # lines = np.zeros([2,2, len(boundaries)])
    # for i in range(lines.shape[2]):
    #     lines[:, :, i] = np.array([[boundaries[i], glob.min_y], [boundaries[i], glob.max_y]])

    return slices


def line_poly_intersect_merge(polygons, lines):
    # print(lines[0])
    # print(polygons[0])
    points = intersection(polygons[3].points, lines[:,:,0])


    for i in range(polygons):
        other_polygons = polygons
        other_polygons.pop(i)
        for j in other_polygons:
            points_l = intersection(polygons[j].points, polygons[i].slice_l)
            points_r = intersection(polygons[j].points, polygons[i].slice_r)
    #     for line in lines:
    #         points = intersection(poly, line)


def intersect_slices_with_polygons(polygons, slices):
    intersect_slices = []
    p = intersection(polygons[0].points, slices[:,:,0])

    for slc in slices:
        intr_points = []
        for poly in polygons:
            intr_points.append(intersection(poly.points, slc[:,:,0]))
        
    # return intersect_slices


def create_sub_poly(glob, polygons, slices):
    sub_polyes = []
    for i in range(slices.shape[2]):
        if i == 0:
            sub_polyes.append(MyPolygon([list(glob.points[0]), list(glob.points[1]), list(slices[1,:,i]), list(slices[0,:,i])]))
        elif i == slices.shape[2]-1:
            sub_polyes.append(MyPolygon([list(slices[0,:,i]), list(slices[1,:,i]), list(glob.points[2]), list(glob.points[3])]))
        else:
            ...
    # sub_polyes[1].print()


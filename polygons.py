import numpy as np
import shapely.geometry as sg
import sys

def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))

class MyPolygon:
    def __init__(self, points):
        self.points = np.array(points)
        self.slice_l = np.array([2,2])
        self.slice_r = np.array([2,2])
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
        self.radius = np.max([distance(point, self.mass_center) for point in self.points])
        self.slice_l = np.array([[self.min_x, 0],[self.min_x, 0]])
        self.slice_r = np.array([[self.max_x, 0],[self.max_x, 0]])

    def move_x(self, x):
        self.points[:, 0] += x
        self._update()

    def move_y(self, y):
        self.points[:, 1] += y
        self._update()

    @property
    def edges(self):
        ''' Returns a list of tuples that each contain 2 points of an edge '''
        edge_list = []
        for i,p in enumerate(self.points):
            p1 = p
            p2 = self.points[(i+1) % len(self.points)]
            edge_list.append((p1,p2))

        return edge_list
    
    def contains(self, point: list):
        # _huge is used to act as infinity if we divide by 0
        _huge = sys.float_info.max
        # _eps is used to make sure points are not on the same line as vertexes
        _eps = 0.00001

        # We start on the outside of the polygon
        inside = False
        for edge in self.edges:
            # Make sure A is the lower point of the edge
            A, B = edge[0], edge[1]
            if A[1] > B[1]:
                A, B = B, A

            # Make sure point is not at same height as vertex
            if point[1] == A[1] or point[1] == B[1]:
                point[1] += _eps

            if (point[1] > B[1] or point[1] < A[1] or point[0] > max(A[0], B[0])):
                # The horizontal ray does not intersect with the edge
                continue

            if point[0] < min(A[0], B[0]): # The ray intersects with the edge
                inside = not inside
                continue

            try:
                m_edge = (B[1] - A[1]) / (B[0] - A[0])
            except ZeroDivisionError:
                m_edge = _huge

            try:
                m_point = (point[1] - A[1]) / (point[0] - A[0])
            except ZeroDivisionError:
                m_point = _huge

            if m_point >= m_edge:
                # The ray intersects with the edge
                inside = not inside
                continue

        return inside
    
glob_poly = MyPolygon([[0,0],
                       [0,150],  
                       [20,300], 
                       [150,330], 
                       [300,300], 
                       [310,150], 
                       [320,0]])

glob_poly1 = MyPolygon([[40,0],
                        [0,100],  
                        [50,300], 
                        [150,330], 
                        [250,200], 
                        [330,200], 
                        [300,0]])

glob_poly2 = MyPolygon([[50,0],
                        [0,100],  
                        [150,300], 
                        [250,300], 
                        [300,200], 
                        [100,0]])

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

c3_poly = MyPolygon([[133,200], 
                     [133,225], 
                     [158,225], 
                     [158,200]])          

b1_poly = MyPolygon([[50,5], 
                     [50,30], 
                     [75,30], 
                     [75,5]])

b2_poly = MyPolygon([[78,5], 
                     [78,30], 
                     [103,30], 
                     [103,5]])
                    
d_poly = MyPolygon([[200,100], 
                    [200,125], 
                    [225,125], 
                    [225,100]])


c1_poly.move_y(12)
c2_poly.move_x(-6)
a_poly.move_x(-45)
a_poly.move_y(40)
d_poly.move_x(-10)

polygons23 = [a_poly, c1_poly, c2_poly, c3_poly, b1_poly, b2_poly, d_poly]

poly1 = MyPolygon([[50,100], 
                    [50,150], 
                    [100,150], 
                    [100,100]])

poly2 = MyPolygon([[150,100], 
                    [150,150], 
                    [200,150], 
                    [200,100]])

poly3 = MyPolygon([[250,100], 
                    [250,150], 
                    [300,150], 
                    [300,100]])

polygons1 = [poly2] 

def boundary(poly, width):
    poly = sg.Polygon(poly)
    boundary = poly.buffer(width, join_style=2)
    boundary_coords = list(boundary.boundary.coords)
    return MyPolygon(boundary_coords[:-1])



poly1 = [[50,100], 
        [50,150], 
        [100,150], 
        [100,100]]

poly2 = [[150,100], 
        [150,150], 
        [200,150], 
        [200,100]]

poly3 = [[250,100], 
        [250,150], 
        [300,150], 
        [300,100]]


gl = [[0,0],
    [0,150],  
    [20,300], 
    [150,330], 
    [300,300], 
    [310,150], 
    [320,0]]

gl1 = [[40,0],
    [0,100],  
    [50,300], 
    [150,330], 
    [250,200], 
    [330,200], 
    [300,0]]

gl2 = [[50,0],
    [0,100],  
    [150,300], 
    [250,300], 
    [300,200], 
    [100,0]]

test_polygons = [poly1, poly3, poly2] 
# test_polygons = [poly2] 
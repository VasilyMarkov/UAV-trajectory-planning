from classification import *
from functools import cmp_to_key

def comparator(lhs_poly, rhs_poly):
    return rhs_poly.min_x - lhs_poly.min_x 

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


obstacles = [a_poly, c1_poly, c2_poly, b1_poly, b2_poly, d_poly]
sorted_obstacles = sorted(obstacles, key=lambda p: p.min_x)
list(map(lambda x:x.print(),sorted_obstacles))

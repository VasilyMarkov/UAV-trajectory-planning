import matplotlib.pyplot as plt
import matplotlib.patches as patches
from classification import *

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
polygons = [a_poly, c1_poly, c2_poly, b1_poly, b2_poly, d_poly]

lines = generates_rays(glob_poly.points, 10, traversal='vertical')

intersect_points = []
for i in range(lines.shape[2]):
    intersect_points.append(intersection(glob_poly.points, lines[:, :, i]))

for i in range(lines.shape[2]):
    lines[:, :, i][0] = intersect_points[i][0]
    lines[:, :, i][1] = intersect_points[i][1]

marked_obstacles = classification(glob_poly, lines, polygons, step=10, traversal='vertical')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
ax.add_patch(patches.Polygon(glob_poly.points, fill=False))

# for poly in polygons:
#     ax.add_patch(patches.Polygon(poly.points, color = 'b', fill=False))

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
plt.legend()

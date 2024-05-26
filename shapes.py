import matplotlib.pyplot as plt
import matplotlib.patches as patches
from classification import *
from decomposition import *
from polygons import *
from adj_graph import *
import rotating_calipers as rc
import copy
import math
import time
from optimizer import *

def shift_line(line, distance):
    # Find the directional vector of the line
    dx = line.Point2.x - line.Point1.x
    dy = line.Point2.y - line.Point1.y
    # Rotate it 90 degrees counterclockwise to get the normal vector
    nx = dy
    ny = -dx
    # Normalize the vector
    length = math.sqrt(nx*nx + ny*ny)
    nx /= length
    ny /= length
    # Multiply the normal vector by the distance
    nx *= distance
    ny *= distance
    line.Point1.x += nx
    line.Point1.y += ny
    line.Point2.x += nx
    line.Point2.y += ny

    return line

def plot_intersection(plot, intersects):
    for point in intersects:
        ax.scatter(*zip(*point), color = 'r', s = 15)


def plot_line(plot, line, i_color, width):
    x = [line.Point1.x, line.Point2.x]
    y = [line.Point1.y, line.Point2.y]
    plot.plot(x, y, color = i_color, linewidth = width)

def flatten_list(i_list):
    output = []
    for item in i_list:
        if isinstance(item, tuple):
            output.append(item)
        elif isinstance(item, list):
            output.extend([tup for tup in item])
    return output


glob = glob_poly1

# lines = generates_rays(glob_poly.points, 10, traversal='vertical')

# intersect_points = []
# for i in range(lines.shape[2]):
#     intersect_points.append(intersection(glob_poly.points, lines[:, :, i]))

# for i in range(lines.shape[2]):
#     lines[:, :, i][0] = intersect_points[i][0]
#     lines[:, :, i][1] = intersect_points[i][1]

# marked_obstacles = classification(glob_poly, lines, polygons1, step=10, traversal='vertical')

# marked_obstacles['d'] = create_slices(glob_poly, marked_obstacles['d'])
# slices = np.array(create_slices(glob_poly, marked_obstacles['d']))
slices = np.array(create_slices(glob_poly, polygons1))
result, vertices = rc.smallest_rectangle(list(glob.points), rc.compare_area)
mbr = MyPolygon(vertices)

ref_line = Line(Point(vertices[3][0], vertices[3][1]), Point(vertices[2][0], vertices[2][1]))
unit_ref_line = ref_line.vector/np.linalg.norm(ref_line.vector)
basis_vector = np.array([1,0])
dot = np.dot(unit_ref_line, basis_vector)
angle = np.arccos(dot)
w = 10
l = w/np.cos(np.pi/2 - angle)

lines = []
offset = 0
ver_x = np.array(vertices)[:, :1]

while(True):
    new_line = copy.deepcopy(ref_line)
    new_line = shift_line(new_line, offset)
    global_points = intersect(glob.points, new_line)
    if len(global_points) == 1:
        offset += l
        continue
    if len(global_points) == 0:
        break
    points = []
    points.append(global_points[0])
    
    intersect_p = []
    for poly in polygons1:
        point = intersect(poly.points, new_line)
        if len(point) != 0:
            points.append(point)
    points.append(global_points[1])
    points = flatten_list(points)
    for i in range(len(points)-1):
        if i % 2 == 0:
            line = Line(Point(points[i][0], points[i][1]), Point(points[i+1][0], points[i+1][1])) 
            lines.append(line)
    
    offset += l

line1 = Line(Point(0, 0), Point(0, 10))
line2 = Line(Point(10, 0), Point(10, 10))
line3 = Line(Point(20, 0), Point(20, 10))
graph = create_graph(lines)
# graph = create_graph([line1, line2, line3])
cost = nx.get_edge_attributes(graph, "cost")
line_points = []
for line in lines:
    line_points.append((line.Point1.x, line.Point1.y))
    line_points.append((line.Point2.x, line.Point2.y))
line_points = np.array(line_points)  

graph_points = line_points.copy()

graph_points -= graph_points[0]

start = time.time()
cost_matrix = cost_matrix_with_start(graph_points[1:], graph_points[0], distance)
best_route, best_route_cost  = ant_colony(cost_matrix, graph_points[0], n_ants=2)
print(best_route, best_route_cost)
end = time.time()
print(end-start)

output_lines = []
for i in best_route:
    output_lines.append(line_points[i])
# intersect_slices_with_polygons(polygons1, slices)
# create_sub_poly(glob_poly, polygons1, slices)


# intersect_slices = []
# for i in range(slices.shape[2]):
#     intersect_slices.append(intersection(glob_poly.points, slices[:, :, i]))

# for i in range(slices.shape[2]):
#     slices[:, :, i][0] = intersect_slices[i][0]
#     slices[:, :, i][1] = intersect_slices[i][1]

# line_poly_intersect_merge(polygons, slices)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

ax.add_patch(patches.Polygon(glob.points, fill=False))
ax.add_patch(patches.Polygon(mbr.points, color = 'b', fill=False))
plot_line(ax, ref_line, i_color ='r', width=3)
for line in lines:
    plot_line(ax, line, i_color ='g', width=1)

for i in range(len(output_lines)-1):
    x1 = output_lines[i][0]
    y1 = output_lines[i][1]
    x2 = output_lines[i+1][0]
    y2 = output_lines[i+1][1]
    ax.plot([x1,x2], [y1,y2], color = 'b', linewidth = 3)
# fig = plt.figure(figsize=(8, 8))
# ax1 = fig.add_subplot()
# plot_graph(graph)

for elem in polygons1:
    ax.add_patch(patches.Polygon(elem.points, color = 'purple', fill=True))

# for elem in marked_obstacles['a']:
#     ax.add_patch(patches.Polygon(elem.points, color = 'g', fill=True))

# for elem in marked_obstacles['b']:
#     ax.add_patch(patches.Polygon(elem.points, color = 'r', fill=True))

# for elem in marked_obstacles['c']:
#     ax.add_patch(patches.Polygon(elem.points, color = 'b', fill=True))

# for elem in marked_obstacles['d']:
#     ax.add_patch(patches.Polygon(elem.points, color = 'purple', fill=True))

# plot_lines(ax, lines, 'black', 1)
# plot_lines(ax, slices, 'blue', 3)
# plot_lines(ax, print_lines, 'blue', 3)
# print(slices[:,:,5])
# for obs in marked_obstacles['d']:
#     x_l = [point[0] for point in obs.slice_l]
#     y_l = [point[1] for point in obs.slice_l]
#     x_r = [point[0] for point in obs.slice_r]
#     y_r = [point[1] for point in obs.slice_r]
#     ax.plot(x_l, y_l, color = 'blue', linewidth = '3')
#     ax.plot(x_r, y_r, color = 'blue', linewidth = '3')

# plot_intersection(ax, intersect_points)

ax.set_xlim([-20, 360])
ax.set_ylim([-40, 360])
plt.show()
plt.legend()

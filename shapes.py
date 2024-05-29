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
from bcd import *
import dubins
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


glob = glob_poly
polygons = boundaries

# marked_obstacles = classification(glob_poly, lines, polygons1, step=10, traversal='vertical')

# marked_obstacles['d'] = create_slices(glob_poly, marked_obstacles['d'])
# slices = np.array(create_slices(glob_poly, marked_obstacles['d']))
slices = np.array(create_slices(glob_poly, polygons))
result, vertices = rc.smallest_rectangle(list(glob.points), rc.compare_area)
mbr = MyPolygon(vertices)

ref_line = Line(Point(vertices[3][0], vertices[3][1]), Point(vertices[2][0], vertices[2][1]))

ref_line = Line(Point(-100, -100), Point(-100, 400))

# ref_line.print()

unit_ref_line = ref_line.vector/np.linalg.norm(ref_line.vector)
basis_vector = np.array([1,0])
dot = np.dot(unit_ref_line, basis_vector)
angle = np.arccos(dot)
w = 30
l = w/np.cos(np.pi/2 - angle)

lines = []
offset = 0
ver_x = np.array(vertices)[:, :1]
point_cnt = 0
for i in range(100):
    new_line = copy.deepcopy(ref_line)
    new_line = shift_line(new_line, offset)
    global_points = intersect(glob.points, new_line)
    # if len(global_points) == 1:
    #     offset += l
    #     continue
    # if len(global_points) == 0:
    #     break
    if len(global_points) == 2:
        points = []
        points.append(global_points[0])
        
        intersect_p = []
        for poly in polygons:
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

print(point_cnt)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def create_intersect_lines_and_attributes():
    delta = -100
    center = (175,175)
    grid = []
    for i in range((int(350/w)+30)):
        lns = [[delta, -200],[delta, 700]]
        delta += w
        grid.append(lns)

    for i in range(1):
        rotate_grid = []
        for line in grid:
            p1x, p1y = rotate(center, (line[0][0], line[0][1]), np.radians(i))
            p2x, p2y = rotate(center, (line[1][0], line[1][1]), np.radians(i))
            line = [[p1x, p1y],[p2x, p2y]]
            rotate_grid.append(line)

        inter_line = []
        attr = []
        cnt = 0
        for rot_line in rotate_grid:
            points = intersect1(glob.points, rot_line)
            if len(points) == 2:
                tmp = []
                tmp.append(points[0])
                attr.append((cnt, True, None))
                cnt += 1
                for i in range(len(polygons)):
                    poly_points = intersect1(polygons[i].points, rot_line)
                    if len(poly_points) != 0:
                        for j in range(len(poly_points)):
                            tmp.append(poly_points[j])
                            attr.append((cnt, False, i))
                            cnt += 1
                tmp.append(points[1])   
                attr.append((cnt, True, None))
                cnt += 1
                tmp = flatten_list(tmp)
                for i in range(len(tmp)-1):
                    if i % 2 == 0:
                        line = [[tmp[i][0], tmp[i][1]],[tmp[i+1][0], tmp[i+1][1]]]
                        inter_line.append(line)
    return inter_line, attr


points_cnt = []
inter_lines, attr = create_intersect_lines_and_attributes()
print(attr[:15])
points_cnt.append(len(inter_lines)*2)
# print(len(inter_lines))
# print(f'Arg: {np.argmax(points_cnt)}, Max: {np.max(points_cnt)}')
# print(f'Arg: {np.argmin(points_cnt)}, Max: {np.min(points_cnt)}')



# p1 = (lines[1].Point2.x, lines[1].Point2.y, np.radians(90))
# p2 = (lines[3].Point2.x, lines[3].Point2.y, np.radians(-90))

# turning_radius = w/2
# step_size = 0.5

# path, _ = dubins.path_sample(p1, p2, turning_radius, 1)
# path = np.array(path)
# print(path[:, 0])

# configurations, _ = path.sample_many(step_size)

# path_iterator = dubins_path(p1, p2, 15)
# for mode, path, curvature in path_iterator:
#     # Print the mode, t, and q values for each iteration
#     print('Mode:', mode)
#     print('t:', path)
#     print('q:', curvature)
#     print()
# lines = np.array(lines)
# lines = np.roll(lines, 0)

# route_cost = []
# route = []
# start = time.time()
# for i in range(1):
#     lines = np.roll(lines, 0)
#     line_points = []
#     for line in lines:
#         line_points.append((line.Point1.x, line.Point1.y))
#         line_points.append((line.Point2.x, line.Point2.y))
#     line_points = np.array(line_points)  
#     graph_points = line_points.copy()
#     graph_points -= graph_points[0]
#     cost_matrix = cost_matrix_lines1(graph_points, distance)
#     best_route, best_route_cost  = ant_colony(cost_matrix, graph_points[0], n_ants=2)
#     route.append(best_route)
#     route_cost.append(best_route_cost)
# end = time.time()

# print(f'Time: {end-start} s')
# best_route = route[np.argmin(route_cost)]
# output_lines = []
# for i in best_route:
#     output_lines.append(line_points[i])

# p = []
# for i in range(len(test_polygons)):
#     points = [np.array(i) for i in test_polygons[0]]
#     p.append(points)

# print(test_polygons)
# print(list(glob_poly.points))
# cells=BCD(p, gl2, 10)
# cellplot=[]
# for i in range(len(cells)):
#     cell=Polygon(np.array(cells[i]))
#     cellplot.append(cell)

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
# ax.plot(path[:,0], path[:,1], color = 'b', linewidth = 2)
# for elem in cellplot:
#     ax.add_patch(patches.Polygon(elem, color = 'purple', fill=False))

for line in inter_lines:
    x = [line[0][0],line[1][0]]
    y = [line[0][1],line[1][1]]
    ax.plot(x, y, color = 'b', linewidth = 2)

ax.add_patch(patches.Polygon(glob.points, fill=False))

# ax.add_patch(patches.Polygon(mbr.points, color = 'b', fill=False))
# plot_line(ax, ref_line, i_color ='r', width=3)
# for line in lines:
#     plot_line(ax, line, i_color ='g', width=1)

# for i in range(len(output_lines)-1):
#     x1 = output_lines[i][0]
#     y1 = output_lines[i][1]
#     x2 = output_lines[i+1][0]
#     y2 = output_lines[i+1][1]
#     ax.plot([x1,x2], [y1,y2], color = 'b', linewidth = 2)
# print(len(output_lines))
# ax.scatter(output_lines[0][0], output_lines[0][1], color = 'red', linewidths=5)
# ax.scatter(output_lines[len(output_lines)-1][0], output_lines[len(output_lines)-1][1], color = 'g', linewidths=5)

    # ax.scatter(x2, y2, color = 'black', linewidths=1)
# for i in range(len(line_points)-1):
#     x1 = line_points[i][0]
#     y1 = line_points[i][1]
#     x2 = line_points[i+1][0]
#     y2 = line_points[i+1][1]
#     ax.scatter(x1, y1, color = 'black', linewidths=1)
#     ax.scatter(x2, y2, color = 'black', linewidths=1)
#     ax.annotate(str(i), (x1, y1))
#     ax.annotate(str(i+1), (x2, y2))
# fig = plt.figure(figsize=(8, 8))
# ax1 = fig.add_subplot()

# ax1.plot(points_cnt, color = 'b', linewidth = 1)
# ax1.set_xlabel('Angle, deg')
# ax1.set_ylabel('Points')
# plot_graph(graph)

for elem in polygons1:
    ax.add_patch(patches.Polygon(elem.points, color = 'purple', fill=True))

for elem in boundaries:
    ax.add_patch(patches.Polygon(elem.points, color = 'r', fill=False))

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

ax.set_xlim([0, 350])
ax.set_ylim([0, 350])
plt.show()
plt.legend()

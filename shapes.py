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
import dubins
from dataclasses import dataclass
from a_star import *
import sys
import os
custom_path = os.path.abspath("dubins")
sys.path.append(custom_path)
from hybrid_astar import *

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


def create_grid(area: MyPolygon, obstacle: MyPolygon, step):
    segment_x = area.max_x - area.min_x
    segment_y = area.max_y - area.min_y
    delta = step
    x = area.min_x
    y = area.min_y
    output_list = []
    for i in range(int(segment_x/delta)):
        for j in range(int(segment_y/delta)):
            p = [x, y]
            y += delta
            if obstacle.contains(p):
                output_list.append(p)
        y = area.min_y
        x += delta
    return output_list

def create_walls(area: MyPolygon, obstacle: MyPolygon, step):
    segment_x = area.max_x - area.min_x
    segment_y = area.max_y - area.min_y
    delta = step
    x = area.min_x
    y = area.min_y
    output_list = []
    for i in range(int(segment_x/delta)):
        for j in range(int(segment_y/delta)):
            p = (x, y)
            y += delta
            if obstacle.contains([p[0], p[1]]):
                output_list.append(p)
        y = area.min_y
        x += delta
    return output_list


def create_weights(grid, weight):
    weights = {}
    for i in range(len(grid)):
            weights.update({(grid[i][0], grid[i][1]): weight})
    return weights

glob = glob_poly

boundaries = [boundary(poly.points, 2) for poly in polygons1]
boundaries1 = [boundary(poly.points, 30) for poly in polygons1]
boundaries2 = [boundary(poly.points, 50) for poly in polygons1]
polygons = boundaries1


result, vertices = rc.smallest_rectangle(list(glob.points), rc.compare_area)
mbr = MyPolygon(vertices)
ref_line = Line(Point(vertices[3][0], vertices[3][1]), Point(vertices[2][0], vertices[2][1]))
ref_line = Line(Point(-100, -100), Point(-100, 400))

unit_ref_line = ref_line.vector/np.linalg.norm(ref_line.vector)
basis_vector = np.array([1,0])
dot = np.dot(unit_ref_line, basis_vector)
angle = np.arccos(dot)
w = 25

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

@dataclass
class Attrubute:
    index: int
    global_intersect: bool
    polygon_index: int


angle = 1
def create_intersect_lines_and_attributes():
    delta = -100
    center = (175,175)
    grid = []
    for i in range((int(350/w)+30)):
        lns = [[delta, -200],[delta, 700]]
        delta += w
        grid.append(lns)

    for i in range(angle,angle+1):
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
                attr.append(Attrubute(cnt, True, None))
                cnt += 1
                for i in range(len(polygons)):
                    poly_points = intersect1(polygons[i].points, rot_line)
                    if len(poly_points) != 0:
                        for j in range(len(poly_points)):
                            tmp.append(poly_points[j])
                            attr.append(Attrubute(cnt, False, i))
                            cnt += 1
                tmp.append(points[1])   
                attr.append(Attrubute(cnt, True, None))
                cnt += 1
                tmp = flatten_list(tmp)
                for i in range(len(tmp)-1):
                    if i % 2 == 0:
                        line = [[tmp[i][0], tmp[i][1]],[tmp[i+1][0], tmp[i+1][1]]]
                        inter_line.append(line)
    return inter_line, attr

def calc_dubins(in_point, out_point, raduis, step_size):
    path, _ = dubins.path_sample(in_point, out_point, raduis, step_size)
    path = np.array(path)
    l = 0
    for i in range(len(path)-1):
        x1, x2 = path[i][0], path[i+1][0]
        y1, y2 = path[i][1], path[i+1][1]
        l += np.hypot(x2-x1, y2-y1)
    return np.array(path), l

def nearest_point(point, polygon):
    poly_points = polygon.points
    distances = []
    for p in poly_points:
        distances.append(distance(point, p))
    return copy.copy(poly_points[np.argmin(distances)])


def avoid_obs(start_position, end_position, angle):
    start_pos_obs_x = boundaries2[0].min_x
    start_pos_obs_y = boundaries2[0].min_y
    w = boundaries2[0].max_x - boundaries2[0].min_x
    h = boundaries2[0].max_y - boundaries2[0].min_y
    obs = [
        [boundaries[0].min_x-start_pos_obs_x, 
        boundaries[0].min_y-start_pos_obs_y,
        boundaries[0].max_x-boundaries[0].min_x, 
        boundaries[0].max_y-boundaries[0].min_y
        ]
        ]

    factor = (h/9)+1

# obs = [
#     [1.5, 1.5, 5, 5]
#     ]

    obs[0][0] /= factor        
    obs[0][1] /= factor        
    obs[0][2] /= factor        
    obs[0][3] /= factor        
    start_pos = [start_position[0], start_position[1], angle]
    end_pos = [end_position[0], end_position[1], angle]
    # print(start_pos)
    # print(end_pos)
    start_pos[0] -= start_pos_obs_x
    start_pos[1] -= start_pos_obs_y
    start_pos[0] /= factor
    start_pos[1] /= factor
    end_pos[0] -= start_pos_obs_x
    end_pos[1] -= start_pos_obs_y
    end_pos[0] /= factor
    end_pos[1] /= factor
    
    # if start_pos[1] == 0.0:
    #     start_pos[1] = 1
    # if end_pos[1] == 0.0:
    #     end_pos[1] = 1


    rec_path, cost = get_path(start_pos, end_pos, obs)
    new_path = [[i.pos[0]*factor + start_pos_obs_x, i.pos[1]*factor + start_pos_obs_y, i.pos[2]] for i in rec_path]
    return np.array(new_path), cost*factor


def cost_matr(points, attr, radius, cost_function: Callable = distance) -> np.ndarray:

    n = len(points)
    step_size = 3
    cost_matrix = np.zeros((n, n))
    paths = {}
    # test_path = []
    for i in range(n):
        if attr[i].global_intersect == True:
            for j in range(i, n):
                if i != j: 
                    if attr[j].global_intersect == True: 
                        if i % 2 == 0 and j % 2 == 0:
                            path, length = calc_dubins((points[i][0], points[i][1], np.radians(270+angle)), 
                                                    (points[j][0], points[j][1], np.radians(90+angle)), radius, step_size)
                            paths.update({(i,j): path})
                            cost_matrix[i, j] = length
                            cost_matrix[j, i] = length
                        elif i % 2 != 0 and j % 2 != 0:
                            path, length = calc_dubins((points[i][0], points[i][1], np.radians(90+angle)), 
                                                    (points[j][0], points[j][1], np.radians(270+angle)), radius, step_size)
                            paths.update({(i,j): path})
                            cost_matrix[i, j] = length
                            cost_matrix[j, i] = length  
                        else:
                            cost_matrix[i, j] = 1000
                            cost_matrix[j, i] = 1000  
                    else:
                        cost_matrix[i, j] = 1000
                        cost_matrix[j, i] = 1000  
                    if i % 2 == 0 and j == i + 1:
                        cost_matrix[i, j] = 0.01
                        cost_matrix[j, i] = 0.01

        else:
            for j in range(i, n):
                if i != j: 
                    if attr[j].global_intersect == False:
                        if i % 2 == 0 and j % 2 == 0:
                            path, length = calc_dubins((points[i][0], points[i][1], np.radians(270+angle)), 
                                                    (points[j][0], points[j][1], np.radians(90+angle)), radius, step_size)
                            paths.update({(i,j): path})
                            
                            cost_matrix[i, j] = length
                            cost_matrix[j, i] = length
                        elif i % 2 != 0 and j % 2 != 0:
                            path, length = calc_dubins((points[i][0], points[i][1], np.radians(90+angle)), 
                                                    (points[j][0], points[j][1], np.radians(270+angle)), radius, step_size)
                            paths.update({(i,j): path})
                            cost_matrix[i, j] = length
                            cost_matrix[j, i] = length 
                        elif i % 2 != 0:
                            path, cost = avoid_obs(points[i], points[j], np.radians(90+angle))
                            paths.update({(i,j): path})
                            cost_matrix[i, j] = cost
                            cost_matrix[j, i] = cost  
                        #     nearest1 = nearest_point(points[i], boundaries1[attr[j].polygon_index])
                        #     nearest2 = nearest_point(points[j], boundaries1[attr[j].polygon_index])
                            
                        #     middle = (nearest1+nearest2)/2
                        #     path1, length1 = calc_dubins((points[i][0], points[i][1], np.radians(90+angle)), 
                        #                     (middle[0], middle[1], np.radians(90+angle)), 5, step_size)
                        #     second_start_point = [path1[len(path1)-1][0], path1[len(path1)-1][1]]
                        #     path2, length2 = calc_dubins((second_start_point[0], second_start_point[1], np.radians(90+angle)), 
                        #                     (points[j][0], points[j][1], np.radians(90+angle)), 5, step_size)
                        #     test_path = np.concatenate([path1, path2], axis=0)
                        #     paths.update({(i,j): test_path})
                        #     cost_matrix[i, j] = length1+length2
                        #     cost_matrix[j, i] = length1+length2
                        elif i % 2 == 0:
                            print(i, j)
                            path, cost = avoid_obs(points[i], points[j], np.radians(270+angle))
                            paths.update({(i,j): path})
                            cost_matrix[i, j] = cost
                            cost_matrix[j, i] = cost 
                        

                    if i % 2 == 0 and j == i + 1:
                        cost_matrix[i, j] = 0.01
                        cost_matrix[j, i] = 0.01
                    else:
                        if attr[j].global_intersect == False:
                            cost_matrix[i, j] = 1000
                            cost_matrix[j, i] = 1000
                        else:
                            cost_matrix[i, j] = 1000
                            cost_matrix[j, i] = 1000  
    return cost_matrix, paths


points_cnt = []
inter_lines, attr = create_intersect_lines_and_attributes()

points_cnt.append(len(inter_lines)*2)

points = []

for line in inter_lines:
    points.append((line[0][0], line[0][1]))
    points.append((line[1][0], line[1][1]))
points = np.array(points) 

graph_points = points.copy()
offset = copy.copy(graph_points[0])

cost, paths = cost_matr(graph_points, attr, w/2, distance)

# print(cost[4][7])
# paths[(0,2)][:,0] += offset[0]
# paths[(0,2)][:,1] += offset[1]
# paths[(1,5)][:,0] += offset[0]
# paths[(1,5)][:,1] += offset[1]

# for key, value in paths.items():
#     value[:, 0] += offset[0]
#     value[:, 1] += offset[1]
graph_points -= offset
best_route, best_route_cost  = ant_colony(cost, graph_points[0], n_ants=2)



# test_poly = MyPolygon([[0,0], 
#                         [0,50], 
#                         [50,50], 
#                         [50,0]])

# obs = MyPolygon([[20,20], 
#                 [20,30], 
#                 [30,30], 
#                 [30,20]])
# step = 1
# walls = create_walls(polygons[0], boundaries[0], step)
# grid = create_grid(polygons[0], polygons[0], step)
# weights = create_weights(grid, step)

# start_position = (polygons[0].min_x, polygons[0].min_y)
# w = polygons[0].max_x - polygons[0].min_x
# h = polygons[0].max_y - polygons[0].min_y

# weight_grid = GridWithWeights(start_position, w, h)

# weight_grid.walls = walls
# weight_grid.weights = weights

# start, goal = start_position, (polygons[0].max_x, polygons[0].max_y)
# came_from, cost_so_far = a_star_search(weight_grid, start, goal, euclidean)

# rec_path = reconstruct_path(came_from, start=start, goal=(polygons[0].max_x-2, polygons[0].max_y-2))



# print(came_from)
# test_path[:, 0] += offset[0] 
# test_path[:, 1] += offset[1] 
# route_cost = []
# route = []
# route.append(best_route)
# route_cost.append(best_route_cost)
# print(paths)
# print(len(inter_lines))
# print(f'Arg: {np.argmax(points_cnt)}, Max: {np.max(points_cnt)}')
# print(f'Arg: {np.argmin(points_cnt)}, Max: {np.min(points_cnt)}')


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
output_points = []

for i in best_route:
    output_points.append(points[i])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

# for elem in cellplot:
#     ax.add_patch(patches.Polygon(elem, color = 'purple', fill=False))

for line in inter_lines:
    x = [line[0][0],line[1][0]]
    y = [line[0][1],line[1][1]]
    ax.plot(x, y, color = 'g', linewidth = 1)

ax.add_patch(patches.Polygon(glob.points, fill=False))

# ax.add_patch(patches.Polygon(mbr.points, color = 'b', fill=False))
# plot_line(ax, ref_line, i_color ='r', width=3)
# for line in lines:
#     plot_line(ax, line, i_color ='g', width=1)
print(best_route)
for i in range(len(best_route)-1):
    curr_index = best_route[i]
    next_index = best_route[i+1]
    if (curr_index, next_index) in paths:
        ax.plot(paths[(curr_index, next_index)][:,0], paths[(curr_index, next_index)][:,1], color = 'b', linewidth = 2)
    elif (next_index, curr_index) in paths:
        ax.plot(paths[(next_index, curr_index)][:,0], paths[(next_index, curr_index)][:,1], color = 'b', linewidth = 2)
    else:
        x1 = points[i][0]
        y1 = points[i][1]
        x2 = points[i+1][0]
        y2 = points[i+1][1]
        ax.plot([x1,x2], [y1,y2], color = 'b', linewidth = 2)

ax.scatter(points[best_route[0]][0], output_points[best_route[0]][1], color = 'g', linewidths=5)
ax.scatter(points[best_route[len(best_route)-1]][0], points[best_route[len(best_route)-1]][1], color = 'r', linewidths=5)

# ax.plot(test_path[:,0], test_path[:,1], color = 'r', linewidth = 2)
# ax.plot(test_path[:,0], test_path[:,1], color = 'r', linewidth = 2)
# for path in test_path:
# ax.plot(test_path[:,0], test_path[:,1], color = 'r', linewidth = 2)
for i in range(len(points)-1):
    x1 = points[i][0]
    y1 = points[i][1]
    x2 = points[i+1][0]
    y2 = points[i+1][1]
    ax.scatter(x1, y1, color = 'black', linewidths=1)
    ax.scatter(x2, y2, color = 'black', linewidths=1)
    ax.annotate(str(i), (x1, y1))
    ax.annotate(str(i+1), (x2, y2))

# for point in grid:
#     ax.scatter(point[0], point[1], color = 'r', linewidths=1)

# for point in walls:
#     ax.scatter(point[0], point[1], color = 'b', linewidths=1)

# for i in range(len(new_path)-1):
#     x1 = new_path[i][0]
#     y1 = new_path[i][1]
#     x2 = new_path[i+1][0]
#     y2 = new_path[i+1][1]
#     ax.plot([x1,x2], [y1,y2], color = 'b', linewidth = 2)

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

for elem in boundaries1:
    ax.add_patch(patches.Polygon(elem.points, color = 'black', fill=False))



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

ax.set_xlim([-100, 450])
ax.set_ylim([-100, 450])
plt.show()
plt.legend()

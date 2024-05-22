import matplotlib.pyplot as plt
import matplotlib.patches as patches
from classification import *
from decomposition import *
from polygons import *

def plot_intersection(plot, intersects):
    for point in intersects:
        ax.scatter(*zip(*point), color = 'r', s = 15)

lines = generates_rays(glob_poly.points, 10, traversal='vertical')

intersect_points = []
for i in range(lines.shape[2]):
    intersect_points.append(intersection(glob_poly.points, lines[:, :, i]))

for i in range(lines.shape[2]):
    lines[:, :, i][0] = intersect_points[i][0]
    lines[:, :, i][1] = intersect_points[i][1]

marked_obstacles = classification(glob_poly, lines, polygons1, step=10, traversal='vertical')

# marked_obstacles['d'] = create_slices(glob_poly, marked_obstacles['d'])
slices = np.array(create_slices(glob_poly, marked_obstacles['d']))

# intersect_slices_with_polygons(polygons1, slices)
# create_sub_poly(glob_poly, slices)

test_intersect(glob_poly, polygons1, slices)

intersect_slices = []
for i in range(slices.shape[2]):
    intersect_slices.append(intersection(glob_poly.points, slices[:, :, i]))

for i in range(slices.shape[2]):
    slices[:, :, i][0] = intersect_slices[i][0]
    slices[:, :, i][1] = intersect_slices[i][1]

# line_poly_intersect_merge(polygons, slices)


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

plot_lines(ax, lines, 'black', 1)
plot_lines(ax, slices, 'blue', 3)
# plot_lines(ax, print_lines, 'blue', 3)
# print(slices[:,:,5])
for obs in marked_obstacles['d']:
    x_l = [point[0] for point in obs.slice_l]
    y_l = [point[1] for point in obs.slice_l]
    x_r = [point[0] for point in obs.slice_r]
    y_r = [point[1] for point in obs.slice_r]
    ax.plot(x_l, y_l, color = 'blue', linewidth = '3')
    ax.plot(x_r, y_r, color = 'blue', linewidth = '3')

# plot_intersection(ax, intersect_points)

ax.set_xlim([0, 360])
ax.set_ylim([0, 360])
plt.show()
plt.legend()

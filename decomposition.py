from classification import *
from functools import cmp_to_key

def create_slices(glob, obstacles):
    # for obs in obstacles:
    #     obs.slice_l[0][1] = glob.min_y
    #     obs.slice_l[1][1] = glob.max_y
    #     obs.slice_r[0][1] = glob.min_y
    #     obs.slice_r[1][1] = glob.max_y
    # return obstacles
    min_x = [obs.min_x for obs in obstacles]
    max_x = [obs.max_x for obs in obstacles]
    boundaries = sorted(set(min_x+max_x))
    lines = np.zeros([2,2, len(boundaries)])
    for i in range(lines.shape[2]):
        lines[:, :, i] = np.array([[boundaries[i], glob.min_y], [boundaries[i], glob.max_y]])

    return lines


def line_poly_intersect_merge(polygons, lines):
    # print(lines[0])
    # print(polygons[0])
    points = intersection(polygons[3].points, lines[:,:,0])
    print(points)

    for i in range(polygons):
        other_polygons = polygons
        other_polygons.pop(i)
        for j in other_polygons:
            points_l = intersection(polygons[j].points, polygons[i].slice_l)
            points_r = intersection(polygons[j].points, polygons[i].slice_r)
    #     for line in lines:
    #         points = intersection(poly, line)
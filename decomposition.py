from classification import *
from functools import cmp_to_key
from shapely.geometry import Polygon, MultiPolygon


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


def intersect_slices_with_polygons(polygons, slices):
    intersect_slices = []
    p = intersection(polygons[0].points, slices[:,:,0])
    print(p)
    for slc in slices:
        intr_points = []
        for poly in polygons:
            intr_points.append(intersection(poly.points, slc[:,:,0]))
        
    # return intersect_slices


def create_sub_poly(glob, slices):
    sub_polyes = []
    for i in range(slices.shape[2]):
        if i == 0:
            sub_polyes.append(MyPolygon([list(glob.points[0]), list(glob.points[1]), list(slices[1,:,i]), list(slices[0,:,i])]))
        elif i == slices.shape[2]-1:
            sub_polyes.append(MyPolygon([list(slices[0,:,i]), list(slices[1,:,i]), list(glob.points[2]), list(glob.points[3])]))
        else:
            sub_polyes.append(MyPolygon([list(slices[0,:,i-1]), list(slices[1,:,i-1]), list(slices[1,:,i]), list(slices[0,:,i])]))
    sub_polyes[1].print()


def test_intersect(glob, polygons, lines): 
    large_polygon = Polygon(glob.points)
    smaller_polygons = [Polygon(poly.points) for poly in polygons]
    shared_lines = []
    for smaller_polygon in smaller_polygons:
        smaller_polygon_boundary = smaller_polygon.boundary
        shared_lines_temp = large_polygon.boundary.intersection(smaller_polygon_boundary)
        print(shared_lines_temp)
        # shared_lines.extend(shared_lines_temp)
    print(shared_lines)
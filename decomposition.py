from classification import *
from functools import cmp_to_key

def create_slices(glob, obstacles):
    min_x = [obs.min_x for obs in obstacles]
    max_x = [obs.max_x for obs in obstacles]
    boundaries = sorted(set(min_x+max_x))
    lines  = [[[point, glob.min_y], [point, glob.max_y]] for point in boundaries]
    return lines

from classification import *
from functools import cmp_to_key

def create_slices(glob, obstacles):
    min_x = [obs.min_x for obs in obstacles]
    max_x = [obs.max_x for obs in obstacles]
    boundaries = sorted(set(min_x+max_x))

    # lines  = [[[point, glob.min_y], [point, glob.max_y]] for point in boundaries]

    lines = np.zeros([2,2, len(boundaries)])
    # lines[:,:,0] = np.array([[1, 2], [3, 4]])
    # print(lines[:,:,0])
    
    for i in range(lines.shape[2]):
        lines[:, :, i] = np.array([[boundaries[i], glob.min_y], [boundaries[i], glob.max_y]])

    return lines

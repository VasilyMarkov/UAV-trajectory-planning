import matplotlib.pyplot as plt
import numpy as np

def coords(mode, length, radius):
    sector_angle = length/radius
    x = []
    y = []
    f = 20
    ds = 0
    if mode == 'L':
        for i in range(f+1):
            x.append(np.cos(ds)*radius)
            y.append(np.sin(ds)*radius)
            ds += sector_angle/f
        x = np.array(x) - radius
        y = np.array(y)
    if mode == 'R':
        for i in range(f+1):
            x.append(np.cos(np.pi-ds)*radius)
            y.append(np.sin(ds)*radius)
            ds += sector_angle/f
        x = np.array(x) + radius
        y = np.array(y)
    print(ds)
    return x, y
mode, length, radius = 'R', 3, 15
full_l = 2*radius*np.pi/2
x, y = coords(mode, length, radius)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
ax.plot(x,y)
ax.set_xlim((-30, 30))
ax.set_ylim((-30, 30))
plt.show()
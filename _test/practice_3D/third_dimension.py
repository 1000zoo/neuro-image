import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BLANK = "□"
LINE = "■"
ZERO = 0.5

def get_cube(x, y, z):
    x_min, x_max = get_minmax(x)
    y_min, y_max = get_minmax(y)
    z_min, z_max = get_minmax(z)
    x_len = int(x_max - x_min + 1)
    y_len = int(y_max - y_min + 1)
    z_len = int(z_max - z_min + 1)


    print(x_len, y_len, z_len)

    cube = [[[BLANK for _ in range(y_len)] for _ in range(x_len)] for _ in range(z_len)]

    for _x, _y, _z in zip(x, y, z):
        nx = normalization(_x, x_min)
        ny = normalization(_y, y_min)
        nz = normalization(_z, z_min)
        cube[nz][nx][ny] = LINE

    return cube

def normalization(ori, ori_min):
    return int(ori - ori_min)

def _print_cube(cube):
    for plane in cube:
        print("="*101)
        for line in plane:
            for dot in line:
                print(dot, end="")
            print()
    print()

def get_minmax(v):
    return np.min(v), np.max(v)

def _flatten(v):
    temp = []
    for r in v:
        for c in r:
            temp.append(c)

    return temp

if __name__ == "__main__":
    t = np.linspace(0, 2*np.pi, 30)
    r = np.linspace(-4, 3, 30)
    t, r = np.meshgrid(t, r)

    # x = r*np.cos(t)
    # y = r*np.sin(t)
    # z = np.sqrt(x**2 + y**2)

    x = (r**2 + 1)*np.cos(t)
    y = (r**2 + 1)*np.sin(t)
    z = r

    _x = _flatten(x)
    _y = _flatten(y)
    _z = _flatten(z)

    cube = get_cube(_x,_y,_z)
    _print_cube(cube)

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.plot_surface(x, y, z, linewidth=0)
    ax2.scatter(_x, _y, _z, linewidth=0)
    # ax.set_zlim(-1, 3)
    plt.show()
    
"""
matrix로 곡선을 표현
"""

import numpy as np
import math

BLANK = "□"
LINE = "■"
ZERO = 0.5

def get_matrix(x, y1, y2):
    y_max = max(np.max(y1(x)), np.max(y2(x)))
    y_min = min(np.min(y1(x)), np.min(y2(x)))

    y_len = int(y_max - y_min + 1)

    matrix = [[BLANK for _ in x] for _ in range(y_len)]

    for i, _x in enumerate(x):
        matrix[int(y1(_x) - y_min)][i] = LINE
        matrix[int(y2(_x) - y_min)][i] = LINE

    return matrix


def add_line(matrix, line):
    for c in line:
        matrix[c[1]][c[0]] = LINE
    return matrix


def _print_matrix(matrix):
    for t1 in matrix:
        for t in t1:
            print(t, end="")
        print()
    print("")


def get_medianline(curve1, curve2):
    medianline = []
    dx = 2
    xi = 0

    len1 = len(curve1)
    len2 = len(curve2)
    
    for i in range(len1 // dx):
        xi = i * dx
        xf = (i + 1) * dx

        x1 = (xi + xf) // 2
        y1 = curve1[x1]
        line = get_vertical_line(curve1, xi, xf)
        min_point = float('inf')
        xmin = 0

        for x2, y2 in enumerate(curve2):
            value = abs(line(x2, y2))

            if value < min_point:
                min_point = value
                xmin = x2

        medianline.append(((x1 + xmin) // 2, (y1 + curve2[xmin]) // 2))


    return medianline


def get_ylist(x, y1, y2):
    y_max = max(np.max(y1(x)), np.max(y2(x)))
    y_min = min(np.min(y1(x)), np.min(y2(x)))

    _y1 = [int(y1(_x) - y_min) for _x in x]
    _y2 = [int(y2(_x) - y_min) for _x in x]

    return _y1, _y2


def get_slope(curve, x1, x2):
    return (curve[x2] - curve[x1]) / (x2 - x1)


def get_vertical(slope):
    return -1 / slope if slope != 0 else 1 / ZERO

def get_vertical_line(curve, xi, xf):
    slope = get_slope(curve, xi, xf)
    slope = get_vertical(slope)

    x1 = (xi + xf) // 2
    y1 = curve[x1]
    c = -slope * x1 + y1
    return lambda _x, _y : slope*_x - _y + c

from matplotlib import pyplot as plt


if __name__ == "__main__":
    T = 1
    xi = 3
    xf = 5
    total = 101

    x = np.linspace(xi, xf, total)
    y1 = lambda x : np.exp(x) * T
    y2 = lambda x : (0.3 * np.exp(x)) * T


    matrix1 = get_matrix(x, y1, y2)

    curve1, curve2 = get_ylist(x, y1, y2)
    m = get_medianline(curve1, curve2,)
    add_line(matrix1, m)
    # _print_matrix(matrix1)

    plt.plot(x, y1(x))
    plt.plot(x, y2(x))
    plt.plot(x, m)
    plt.show()




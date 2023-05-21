"""
matrix로 곡선을 표현
"""

import numpy as np

def get_matrix(x, y1, y2):
    y_max = max(np.max(y1(x)), np.max(y2(x)))
    y_min = min(np.min(y1(x)), np.min(y2(x)))

    y_len = int(y_max - y_min + 1)

    matrix = [[0 for _ in x] for _ in range(y_len)]

    for i, _x in enumerate(x):
        matrix[int(y1(_x) - y_min)][i] = 1
        matrix[int(y2(_x) - y_min)][i] = 1

    return matrix

def add_centerline(matrix, centerline):
    for c in centerline:
        matrix[c[0]][c[1]] = 1
    return matrix


def sketch_matrix(x, y1, y2):
    y_max = max(np.max(y1(x)), np.max(y2(x)))
    y_min = min(np.min(y1(x)), np.min(y2(x)))

    y_len = int(y_max - y_min + 1)

    matrix = [[0 for _ in x] for _ in range(y_len)]

    for i, _x in enumerate(x):
        _y1 = int(y1(_x) - y_min)
        _y2 = int(y2(_x) - y_min)
        yi = min(_y1, _y2)
        yf = max(_y1, _y2)

        for _y in range(yi, yf + 1):
            matrix[_y][i] = 1

    return matrix

def _print_matrix(matrix):
    for t1 in matrix:
        for t in t1:
            print(t, end="")
        print()
    print("")

def get_centerline(matrix):
    centerline = []

    for j in range(len(matrix[0])):
        _sum = 0
        count = 0
        for i in range(len(matrix)):
            if matrix[i][j] == 1:
                _sum += i
                count += 1
        
        centerline.append((int(_sum / count), j))

    return centerline


if __name__ == "__main__":
    T = 1

    x = np.linspace(0, 5, 101)
    y1 = lambda x : (x ** 2) * T
    y2 = lambda x : (-10 + x ** 2) * T

    matrix1 = get_matrix(x, y1, y2)
    _print_matrix(matrix1)
    matrix2 = sketch_matrix(x, y1, y2)
    t = get_centerline(matrix2)
    _print_matrix(add_centerline(matrix1, t))


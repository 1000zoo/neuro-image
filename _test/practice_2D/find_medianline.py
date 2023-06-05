"""
matrix로 곡선을 표현
"""

import numpy as np
import math

BLANK = "□"
LINE = "■"

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
        # breakpoint()
        matrix[c[1]][c[0]] = LINE
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
        
        centerline.append((j, int(_sum / count)))

    return centerline

def get_medianline(curve1, curve2):
    medianline = []

    n1 = len(curve1)
    n2 = len(curve2)


    for x, y in enumerate(curve1):
        left = 0
        right = n2 - 1
        jmin = 0
        dmin = float('inf')

        while left <= right:
            mid = (left + right) // 2
            d = get_distance((x, y), (mid, curve2[mid]))
            if d <= dmin:
                dmin = d
                jmin = mid
            if x < mid:
                right = mid - 1
            else:
                left = mid + 1

        medianline.append((int(x + jmin) // 2, int(y + curve2[jmin]) // 2))

    return medianline

def chat_medianline(curve1, curve2):
    medianline = []

    n1 = len(curve1)
    n2 = len(curve2)

    for i in range(n1):
        left = 0
        right = n2 - 1

        while left < right:
            mid = (left + right) // 2
            if curve2[mid] < curve1[i]:
                left = mid + 1
            else:
                right = mid

        if left == 0:
            medianline.append((curve2[0], curve1[i]))
        elif left == n2:
            medianline.append((curve2[n2 - 1], curve1[i]))
        else:
            d1 = get_distance((left, curve2[left]), (i, curve1[i]))
            d2 = get_distance((left - 1, curve2[left - 1]), (i, curve1[i]))
            if d1 < d2:
                medianline.append((curve2[left], curve1[i]))
            else:
                medianline.append((curve2[left - 1], curve1[i]))

    return medianline


def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

def get_ylist(x, y1, y2):
    y_max = max(np.max(y1(x)), np.max(y2(x)))
    y_min = min(np.min(y1(x)), np.min(y2(x)))

    _y1 = [int(y1(_x) - y_min) for _x in x]
    _y2 = [int(y2(_x) - y_min) for _x in x]

    return _y1, _y2

if __name__ == "__main__":
    T = 1
    
    x = np.linspace(3, 5, 101)
    y1 = lambda x : np.exp(x) * T
    y2 = lambda x : (0.3 * np.exp(x)) * T

    # matrix1 = get_matrix(x, y1, y2)
    # _print_matrix(matrix1)
    # matrix2 = sketch_matrix(x, y1, y2)
    # t = get_centerline(matrix2)
    # _print_matrix(add_line(matrix1, t))

    curve1, curve2 = get_ylist(x, y1, y2)


    # # _print_matrix(m)
    # print(curve1)
    medianline = get_medianline(curve1, curve2)
    matrix1 = get_matrix(x, y1, y2)
    matrix3 = add_line(matrix1, medianline)
    # print(len(matrix3))
    _print_matrix(matrix3)


    


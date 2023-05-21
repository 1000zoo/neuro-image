"""
matrix로 곡선을 표현
"""

import numpy as np
import math

T = 1

x = np.linspace(-5, 5, 151)
y1 = lambda x : (x ** 2)
y2 = lambda x : (-5 + x ** 2)
y_max = max(np.max(y1(x)), np.max(y2(x)))
y_min = min(np.min(y1(x)), np.min(y2(x)))


y_len = int(y_max - y_min + 1)

matrix = [[0 for _ in x] for _ in range(y_len)]

for i, _x in enumerate(x):
    matrix[int(y1(_x) - y_min)][i] = 1
    matrix[int(y2(_x) - y_min)][i] = 1


for t1 in matrix:
    for t in t1:
        print(t, end="")
    print()
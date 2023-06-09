import numpy as np
import matplotlib.pyplot as plt
import random
import math

from collections import defaultdict, deque
from random import shuffle
from copy import copy


class Node:
    def __init__(self, *args, **kargs) -> None:
        if args:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
            self.v = (self.x, self.y, self.z)

        elif kargs:
            self.v = kargs["v"]
            self.x = self.v[0]
            self.y = self.v[1]
            self.z = self.v[2]

        else:
            print(args)
            print(kargs)
            assert False

        self.neighbors = []

    def __str__(self) -> str:
        return f"{self.x}, {self.y}, {self.z}"

    def set_neighbor(self, neighbors):
        self.neighbors = neighbors


class Tensor:
    def __init__(self, nii) -> None:
        self.nii = nii
        self.tensor = []
        self.graph = []
        self.vnmap = defaultdict(Node)
        self.max_xyz = None
        self.min_xyz = None

        self.X, self.Y, self.Z = self.get_XYZ()
        self.set_connection()  # 각 Node들의 인근 Node연결

        # 출력을 위해 필요한 변수들
        self.plimit = self.get_index_limit()
        self.com = self.get_COM(self.X, self.Y, self.Z)  # x, y, z들의 중심

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

    def __str__(self):
        res = ""
        for node in self.graph:
            res += f"{node}\n"

        return res

    @staticmethod
    def get_COM(x, y, z):
        return sum(x) // len(x), sum(y) // len(y), sum(z) // len(z)

    def get_index_limit(self):
        return max(max(self.X) - min(self.X), max(self.Y) - min(self.Y), max(self.Z) - min(self.Z)) + 1

    def get_XYZ(self):
        _min = float('inf')
        _max = -float('inf')
        for i, x in enumerate(self.nii):
            for j, y in enumerate(x):
                for k, z in enumerate(y):
                    if z != 0.0:
                        self.tensor.append((i, j, k))
                        temp = Node(i, j, k)
                        self.graph.append(temp)
                        self.vnmap[(i, j, k)] = temp

                        if _max < i + j + k:
                            _max = i + j + k
                            self.max_xyz = (i, j, k)
                        if _min > i + j + k:
                            _min = i + j + k
                            self.min_xyz = (i, j, k)

        x, y, z = [], [], []
        for ten in self.tensor:
            _x, _y, _z = ten
            x.append(_x)
            y.append(_y)
            z.append(_z)

        return x, y, z


    def set_connection(self):
        assert len(self.tensor) != 0

        __u = [-1, 0, 1]
        uv = [(i, j, k) for i in __u for j in __u for k in __u if not (i == 0 and j == 0 and k == 0)]

        for x, y, z in self.tensor:
            node = self.vnmap[(x, y, z)]
            neighbors = []
            for i, j, k in uv:
                dx, dy, dz = x + i, y + j, z + k
                if (dx, dy, dz) in self.tensor:
                    neighbor = self.vnmap[(dx, dy, dz)]
                    neighbors.append(neighbor)

            node.set_neighbor(neighbors)


    def plot(self, title="0", p=True):
        _max = self.plimit / 2
        xcom, ycom, zcom = self.com

        self.ax.scatter(self.X, self.Y, self.Z, linewidth=0)
        self.ax.set_xlim([xcom - _max, xcom + _max])
        self.ax.set_ylim([ycom - _max, ycom + _max])
        self.ax.set_zlim([zcom - _max, zcom + _max])

        if p:
            plt.title(f"tract {title}")
            plt.show()
            plt.close()


    def plot_with_centerline(self, short=True, p=True, color="r"):
        _max = self.plimit / 2
        xcom, ycom, zcom = self.com
        xc, yc, zc = self.get_centerline(short, True)

        self.ax.scatter(xc, yc, zc, c=color, linewidth=2)

        self.ax.set_xlim([xcom - _max, xcom + _max])
        self.ax.set_ylim([ycom - _max, ycom + _max])
        self.ax.set_zlim([zcom - _max, zcom + _max])

        if p:
            plt.show()
            plt.close()


    def get_centerline(self, short=True, getXYZ=False):
        X, Y, Z = self.short_cut(getXYZ=True) if short else self.get_random_points_alongXYZ()
        # self.plot_etc(X, Y, Z, size=2)
        centers = []

        points = []

        for x, y, z in zip(X, Y, Z):
            points.append((x, y, z))

        vectors = get_vectors(points)

        for i, xyz in enumerate(zip(X, Y, Z)):
            if i == len(X) - 1:
                break
            x, y, z = xyz
            vector = vectors[i]
            centers.append(self.min_area_plane_center(x, y, z, vector))

        if getXYZ:
            return [center[0] for center in centers], [center[1] for center in centers], [center[2] for center in centers]
        else:
            return centers


    def min_area_plane_center(self, x0, y0, z0, vector):
        lin = linspace(-0.5, 0.5, 5)
        uv = [add_vector(norm(vector), (i, j, k)) for i in lin for j in lin for k in lin]

        uv = list(set(uv))
        xin, yin, zin = [], [], []
        min_area = float('inf')
        min_nv = None

        for i, j, k in uv:
            area = 0
            d = -1 * (i * x0 + j * y0 + k * z0)
            equation = lambda _x, _y, _z: abs(i * _x + j * _y + k * _z + d)
            distance = lambda _x, _y, _z: math.sqrt(((_x - x0) ** 2) + (_y - y0) ** 2 + (_z - z0) ** 2)
            tx, ty, tz = [], [], []

            for x, y, z in self.tensor:
                res = equation(x, y, z)
                if res < 0.7 and distance(x, y, z) <= 12:
                    area += 1
                    tx.append(x)
                    ty.append(y)
                    tz.append(z)

            if area < min_area:
                min_area = area
                min_nv = (i, j, k)
                xin = tx
                yin = ty
                zin = tz

        return self.get_COM(xin, yin, zin)


    def short_cut(self, p=False, getXYZ=False, interval=3):
        start, end = self.vnmap[self.min_xyz], self.vnmap[self.max_xyz]

        q = deque()
        visited = set()
        history = defaultdict(list)
        last_node = None
        q.append(start)

        while q:
            curr = q.popleft()
            curr_history = history[curr]
            curr_history.append(curr)
            if curr == end:
                break
            for node in curr.neighbors:
                if not node in visited:
                    q.append(node)
                    visited.add(node)
                    history[node] = curr_history + [node]
                    last_node = node

        if end not in visited:
            history[end] = history[last_node] + [end]

        sx, sy, sz = [], [], []

        short_cut = []

        for i in range(0, len(history[end]), interval):
            short_cut.append(history[end][i])

        for n in short_cut:
            sx.append(n.x)
            sy.append(n.y)
            sz.append(n.z)

        if p:
            self.plot_etc(sx, sy, sz, color="y")


        if getXYZ:
            return sx, sy, sz

        else:
            return [(node.x, node.y, node.z) for node in short_cut]


    def get_random_points_alongXYZ(self, interval=3):
        xmin, xmax = min(self.X), max(self.X)
        ymin, ymax = min(self.Y), max(self.Y)
        zmin, zmax = min(self.Z), max(self.Z)
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        dmax = max(dx, dy, dz)

        T = self.X if dx == dmax else self.Y if dy == dmax else self.Z
        T = list(set(T))
        ind = 0 if dx == dmax else 1 if dy == dmax else 2
        _min = xmin if dx == dmax else ymin if dy == dmax else zmin
        _max = xmax if dx == dmax else ymax if dy == dmax else zmax

        random_points = []
        i = 0

        while i < len(T):
            temp = []
            v = T[i]
            i += interval

            for point in self.tensor:
                if point[ind] == v:
                    temp.append(point)

            random_points.append(random.choice(temp))

        return [x[0] for x in random_points], [x[1] for x in random_points], [x[2] for x in random_points]



    def plot_etc(self, x, y, z, p=False, color="black", size=0):
        _max = max(max(x) - min(x), max(y) - min(y), max(z) - min(z)) + 1
        xcom, ycom, zcom = sum(x) / len(x), sum(y) / len(y), sum(z) / len(z)

        self.ax.scatter(x, y, z, c=color, linewidth=size)
        self.ax.set_xlim([xcom - _max, xcom + _max])
        self.ax.set_ylim([ycom - _max, ycom + _max])
        self.ax.set_zlim([zcom - _max, zcom + _max])

        if p:
            plt.show()
            plt.close()


def linspace(start, end, step):
    lin = np.linspace(start, end, step)
    return [x for x in lin]

def get_vectors(points):
    assert len(points) > 1
    vectors = []

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        vectors.append(sub_vector(p2, p1))

    return vectors


def add_vector(t1, t2):
    assert len(t1) == 3 and len(t2) == 3 ## (x,y,z) 형식만
    temp = []

    for o1, o2 in zip(t1, t2):
        temp.append(o1+o2)

    return tuple(temp)

def sub_vector(t1, t2):
    assert len(t1) == 3 and len(t2) == 3
    t2 = scalar_mul(t2, -1)
    return add_vector(t1, t2)
def norm(v: tuple):
    assert len(v) == 3
    i, j, k = v
    roots = math.sqrt(i ** 2 + j ** 2 + k ** 2)
    if roots == 0:
        return None
    return i / roots, j / roots, k / roots


def scalar_mul(v: tuple, k):
    assert len(v) == 3
    return k * v[0], k * v[1], k * v[2]

if __name__ == "__main__":
    n = Node(v=(1, 2, 3))
    print(n)

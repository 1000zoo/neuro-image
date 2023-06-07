import numpy as np
import matplotlib.pyplot as plt
import random

from collections import defaultdict, deque
from random import shuffle
from copy import copy

__u = [-1, 0, 1]
UNIT_VECTORS = [(i, j, k) for i in __u for j in __u for k in __u if not (i == 0 and j == 0 and k == 0)]


class PointTree:
    def __init__(self, val=-1) -> None:
        self.val = val
        self.children = defaultdict(Node)

    def node_start_with(self, x):
        if x not in self.children.keys():
            return None
        xn = self.children[x]
        yn = (random.choice(list(xn.children.values())))
        zn = (random.choice(list(yn.children.values())))

        return xn.val, yn.val, zn.val
    


class Node:
    def __init__(self, *args, **kargs) -> None:
        if args:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
            self.vector = (self.x, self.y, self.z)
        
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
    def __init__(self, nii, direction_max=5, p=0.1) -> None:
        self.nii = nii
        self.tensor = []
        self.graph = []
        self.vnmap = defaultdict(Node)
        self.max_xyz = None
        self.min_xyz = None

        self.X, self.Y, self.Z = self.get_XYZ()
        # 출력을 위해 필요한 변수들
        self.plimit = self.get_index_limit()
        self.com = self.get_COM(self.X, self.Y, self.Z)           # x, y, z들의 중심
        self.set_connection()               # 각 Node들의 인근 Node연결
        self.direction_max = direction_max

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.p = p

        # x 좌표로 랜덤한 포인트를 생성하기 위한 변수들
        self.root = PointTree(val=-1)   ## root (dummy node)
        self.set_point_tree()


    def __str__(self):
        res = ""
        for node in self.graph:
            res +=  f"{node}\n"
        
        return res
    
    @staticmethod
    def get_COM(x, y, z):
        return sum(x) / len(x), sum(y) / len(y), sum(z) / len(z)


    def get_index_limit(self):
        return max(max(self.X) - min(self.X), max(self.Y) - min(self.Y), max(self.Z) - min(self.Z)) + 1
    

    def get_XYZ(self):
        _min = float('inf')
        _max = -float('inf')
        for i, x in enumerate(self.nii):
            for j, y in enumerate(x):
                for k, z in enumerate(y):
                    if z > 0.0:
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


    def set_point_tree(self):
        for x, y, z in self.tensor:
            curr = self.root
            if not x in curr.children.keys():
                curr.children[x] = PointTree(x)
            curr = curr.children[x]
            if not y in curr.children.keys():
                curr.children[y] = PointTree(y)
            curr = curr.children[y]
            if not z in curr.children.keys():
                curr.children[z] = PointTree(z)

    def set_connection(self):
        assert len(self.tensor) != 0

        for x, y, z in self.tensor:
            node = self.vnmap[(x, y, z)]
            neighbors = []
            for i, j, k in UNIT_VECTORS:
                dx, dy, dz = x + i, y + j, z + k
                if (dx, dy, dz) in self.tensor:
                    neighbor = self.vnmap[(dx, dy, dz)]
                    neighbors.append(neighbor)

            node.set_neighbor(neighbors)


    def plot(self, p=True):
        _max = self.plimit / 2
        xcom, ycom, zcom = self.com

        
        self.ax.scatter(self.X, self.Y, self.Z, linewidth=0)
        self.ax.set_xlim([xcom - _max, xcom + _max])
        self.ax.set_ylim([ycom - _max, ycom + _max])
        self.ax.set_zlim([zcom - _max, zcom + _max])

        if p:
            plt.show()
            plt.close()

    def plot_with_centerline(self, xyz=None, p=True, color="r"):
        _max = self.plimit / 2
        xcom, ycom, zcom = self.com
        xc, yc, zc = self.get_centerline(xyz, True)

        self.ax.scatter(xc, yc, zc, c=color, linewidth=0)

        self.ax.set_xlim([xcom - _max, xcom + _max])
        self.ax.set_ylim([ycom - _max, ycom + _max])
        self.ax.set_zlim([zcom - _max, zcom + _max])

        if p:
            plt.show()
            plt.close()
    
    def get_centerline(self, xyz=None, getXYZ=False):
        X, Y, Z = xyz if xyz else self.get_random_points()
        self.plot_etc(X, Y, Z)
        centers = []

        for x, y, z in zip(X, Y, Z):
            centers.append(self.min_area_plane_center(x, y, z))

        if getXYZ:
            return [center[0] for center in centers], [center[1] for center in centers], [center[2] for center in centers]
        else:
            return centers


    def get_random_points_withX(self, interval=3):
        xmin, xmax = min(self.X), max(self.X)
        dx = xmax - xmin
        ind = xmin

        temp = []

        while ind <= xmax:
            if self.root.node_start_with(ind):
                temp.append(self.root.node_start_with(ind))
            ind += interval


        return [x[0] for x in temp], [x[1] for x in temp], [x[2] for x in temp]


    def get_random_points(self, minimum=5):
        temp = copy(self.graph)
        shuffle(temp)
        cut = max(int(self.p * len(temp)), minimum)
        temp = temp[:cut]
        return [node.x for node in temp], [node.y for node in temp], [node.z for node in temp]
    

    def min_area_plane_center(self, x0, y0, z0):
        I = (0, 1)
        JK = (-1, 0, 1)
        ed = range(self.direction_max)
        uv = [(i * p, j * q, k * r) for i in I for j in JK for k in JK
              for p in ed for q in ed for r in ed if not (i == 0 and j == 0 and k == 0)]
        uv = list(set(uv))
        xin, yin, zin = [], [], []
        min_area = float('inf')

        for i, j, k in uv:
            area = 0
            d = -1 * (i * x0 + j * y0 + k * z0)
            equation = lambda x, y, z: abs(i * x + j * y + k * z + d)

            for x, y, z in self.tensor:
                res = equation(x, y, z)
                if res < 1:
                    area += 1
                    xin.append(x)
                    yin.append(y)
                    zin.append(z)
            
            min_area = min(area, min_area)

        return self.get_COM(xin, yin, zin)

    def short_cut(self, p=False):
        print(self.min_xyz)
        print(self.max_xyz)
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
        print(history[end])
        for n in history[end]:
            sx.append(n.x)
            sy.append(n.y)
            sz.append(n.z)

        self.plot_etc(sx, sy, sz, color="y")

        return sx, sy, sz



    def plot_etc(self, x, y, z, p=False, color="black"):
        _max = max(max(x) - min(x), max(y) - min(y), max(z) - min(z)) + 1
        xcom, ycom, zcom = sum(x) / len(x), sum(y) / len(y), sum(z) / len(z)
        
        
        self.ax.scatter(x, y, z, c=color, linewidth=0)
        self.ax.set_xlim([xcom - _max, xcom + _max])
        self.ax.set_ylim([ycom - _max, ycom + _max])
        self.ax.set_zlim([zcom - _max, zcom + _max])

        if p:
            plt.show()
            plt.close()

if __name__ == "__main__":

    n = Node(v=(1,2,3))
    print(n)
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from collections import defaultdict, deque

__u = [-1, 0, 1]
UNIT_VECTORS = [(i, j, k) for i in __u for j in __u for k in __u if not (i == 0 and j == 0 and k == 0)]


class Node:
    def __init__(self, x=-1, y=-1, z=-1):
        self.x = x
        self.y = y
        self.z = z
        self.neighbors = []

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def set_neighbor(self, neighbors):
        self.neighbors = neighbors

    def is_surface(self):
        return len(self.neighbors) < 23
    
    def to_xyz(self):
        return (self.x, self.y, self.z)

class Tensor:
    def __init__(self, nii):
        self.nii = nii      # origin        : 0,1 로 구성된 Tensor
        self.tensor = []    # sparse matrix : (x, y, z)들의 집합
        self.graph = []     # nodes map     : 점이 저장된 Node들의 집합
        self.vnmap = defaultdict(Node)  # key: (x, y, z) val: Node(x, y, z)
        self.max_xyz = None
        self.min_xyz = None
        self.X, self.Y, self.Z = self.get_XYZ()
        self.plimit = self.get_index_limit()
        self.com = self.get_COM()           # x, y, z들의 중심
        self.set_connection()               # 각 Node들의 인근 Node연결
        self.surface = self.get_surface()   # 도형의 표면 집합, -> list[Node]
        self.surf_xyz = [n.to_xyz() for n in self.surface]  # 도형의 표면 집합 -> list[(x, y, z)]

    def __str__(self):
        res = ""
        for node in self.graph:
            res +=  f"{node}\n"
        
        return res
    
    def plot(self):
        _max = self.plimit / 2
        xcom, ycom, zcom = self.com

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.X, self.Y, self.Z, linewidth=0)
        ax.set_xlim([xcom - _max, xcom + _max])
        ax.set_ylim([ycom - _max, ycom + _max])
        ax.set_zlim([zcom - _max, zcom + _max])

        plt.show()
        plt.close()

    def surface_plot(self):
        _X, _Y, _Z = [], [], []

        for _x, _y, _z in self.surf_xyz:
            _X.append(_x)
            _Y.append(_y)
            _Z.append(_z)

        _max = self.plimit / 2
        xcom, ycom, zcom = self.com
        maxx, maxy, maxz = self.max_xyz
        minx, miny, minz = self.min_xyz
        sx, sy, sz = self.short_cut()
        _X = np.arange(min(_X), max(_X), 1)
        _Y = np.arange(min(_Y), max(_Y), 1)



        fig = plt.figure()
        ax = fig.subplots(subplot_kw={"projection": "3d"})
        ax.contour(_X, _Y, _Z, linewidth=0)
        # ax.scatter(maxx, maxy, maxz, c='r', linewidth=0)
        # ax.scatter(minx, miny, minz, c='r', linewidth=0)
        # ax.scatter(sx, sy, sz, c='black', linewidth=0)
        ax.set_xlim([xcom - _max, xcom + _max])
        ax.set_ylim([ycom - _max, ycom + _max])
        ax.set_zlim([zcom - _max, zcom + _max])

        plt.show()
        plt.close()

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
    
    def get_COM(self):
        return sum(self.X) / len(self.X), sum(self.Y) / len(self.Y), sum(self.Z) / len(self.Z)
    
    def get_index_limit(self):
        return max(max(self.X) - min(self.X), max(self.Y) - min(self.Y), max(self.Z) - min(self.Z)) + 1
    
    def get_surface(self):
        return [node for node in self.graph if node.is_surface()]

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


    

    # 시작점과 끝점을 알 때
    # 표면의 최단경로를 담은 list(x), list(y), list(z) 리턴
    def short_cut(self):
        start, end = self.vnmap[self.min_xyz], self.vnmap[self.max_xyz]
        print(start, end)

        short = []
        q = deque()
        visited = set()
        history = defaultdict(list)

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

        sx, sy, sz = [], [], []
        for n in history[end]:
            sx.append(n.x)
            sy.append(n.y)
            sz.append(n.z)

        return sx, sy, sz


if __name__=="__main__":
    print(UNIT_VECTORS)
    print(len(UNIT_VECTORS))
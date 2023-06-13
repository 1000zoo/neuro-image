import numpy as np
import matplotlib.pyplot as plt
import random
import math

from collections import defaultdict, deque
from random import shuffle
from copy import copy

RES = 1.3
DISMAX = 12

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


class SubTract:
    def __init__(self, tract, sub_tract, overlap) -> None:
        self.tract = Tensor(tract)
        self.common_centerline = self.tract.centerline
        self.sub_tract = Tensor(sub_tract, self.common_centerline)
        self.area_ratio = defaultdict()
        self.set_area_ratio()
        self.min_center = self.get_min_center()
        self.min_plane = self.get_min_area_plane()
        self.min_tract_plane, self.min_sub_plane = self.get_min_area_plane()
        self.overlap = Tensor(overlap, self.common_centerline)

    
    def get_min_area_plane(self):
        next_point = self.common_centerline[self.common_centerline.index(self.min_center) + 1]
        i, j, k = vector = sub_vector(self.min_center, next_point)
        x0, y0, z0 = point = self.min_center

        d = -1 * (i * x0 + j * y0 + k * z0)
        equation = lambda _x, _y, _z: abs(i * _x + j * _y + k * _z + d)
        distance = lambda _x, _y, _z: math.sqrt(((_x - x0) ** 2) + (_y - y0) ** 2 + (_z - z0) ** 2)

        tx, ty, tz = [], [], []
        sx, sy, sz = [], [], []
        for x, y, z in self.tract.tensor:
            res = equation(x, y, z)
            if res < RES and distance(x, y, z) < DISMAX:
                tx.append(x)
                ty.append(y)
                tz.append(z)


        for x, y, z in self.sub_tract.tensor:
            res = equation(x, y, z)
            if res < RES and distance(x, y, z) < DISMAX:
                sx.append(x)
                sy.append(y)
                sz.append(z)

        return (tx, ty, tz), (sx, sy, sz)


    def get_min_center(self):
        _min = float('inf')
        min_center = None

        for center in self.area_ratio:
            if _min > self.area_ratio[center]:
                _min = self.area_ratio[center]
                min_center = center
        
        return min_center


    def set_area_ratio(self):
        ori_areas = self.tract.cross_section_area
        sub_areas = self.sub_tract.cross_section_area
        
        for center in self.common_centerline:
            if center in sub_areas:
                if ori_areas[center] == 0:
                    self.area_ratio[center] = -float('inf')
                else:
                    self.area_ratio[center] = sub_areas[center] / ori_areas[center]
        
    
    def surface(self, title):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = np.array(self.tract.X)
        y = np.array(self.tract.Y)
        z = np.array(self.tract.Z)

        ax.plot_trisurf(x, y, z)
        plt.show()

    def get_voxel_configs(self, dtype, color='#FFD65DC0'):
        if dtype == 0:      ## origin tract
            t = self.tract        
            voxels = self.rollback(t.X, t.Y, t.Z)
        elif dtype == 1:    ## subtract
            t = self.sub_tract
            voxels = self.rollback(t.X, t.Y, t.Z)
        elif dtype == 2:               ## overlap
            t = self.overlap
            voxels = self.rollback(t.X, t.Y, t.Z)
        elif dtype == 3:                ## centerline
            x, y, z = pointlist_to_xyz(self.common_centerline)
            voxels = self.rollback(x, y, z)
        else:                           ## cross plane
            x, y, z = self.min_sub_plane
            voxels = self.rollback(x, y, z)

        facecolors = np.where(voxels, color, "#7A88CCC0")
        
        filled_2 = explode(voxels)
        fcolors_2 = explode(facecolors)

        x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
        x[0::2, :, :] += 0.05
        y[:, 0::2, :] += 0.05
        z[:, :, 0::2] += 0.05
        x[1::2, :, :] += 0.95
        y[:, 1::2, :] += 0.95
        z[:, :, 1::2] += 0.95

        return (x, y, z), filled_2, fcolors_2

    def voxel_plot(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ## sub tract
        points, filled_2, fcolors_2 = self.get_voxel_configs(dtype=1, color=("#46AAFF")) # 푸른색  
        x, y, z = points

        ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, alpha=0.05)
        
        ## over tract
        points, filled_2, fcolors_2 = self.get_voxel_configs(dtype=2, color=("#FFAF0A")) # 노랑 주황 사이
        x, y, z = points

        ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, alpha=0.05)
        
        # cross area
        points, filled_2, fcolors_2 = self.get_voxel_configs(dtype=3, color=("#CD0000")) # 빨간색
        x, y, z = points

        ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, alpha=0.65)
        x, y, z = self.min_sub_plane
        ax.scatter(x, y, z, linewidths=4, c="r")
        xcom, ycom, zcom = self.tract.com

        _max = self.tract.plimit / 2
        ax.set_xlim([xcom - _max, xcom + _max])
        ax.set_ylim([ycom - _max, ycom + _max])
        ax.set_zlim([zcom - _max, zcom + _max])
        plt.show()


    def rollback(self, x, y, z):
        temp = np.array(self.tract.nii)
        nii = np.zeros(temp.shape, dtype=bool)

        for i, j, k in zip(x, y, z):
            nii[i][j][k] = True

        return nii


    def plot(self, title):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        _max = self.tract.plimit / 2
        xcom, ycom, zcom = self.tract.com

        # tx, ty, tz = self.min_tract_plane
        sx, sy, sz = self.min_sub_plane
        # sx = np.array(sx)
        # sy = np.array(sy)
        # sz = np.array(sz)
        print(sx, sy, sz)

        # ax.scatter(tx, ty, tz, c="red", linewidth=2)
        ax.scatter(sx, sy, sz, linewidth=3, alpha=0.5, color="red")
        ax.scatter(self.sub_tract.X, self.sub_tract.Y, self.sub_tract.Z, linewidth=0, alpha=0.2)
        # ax.scatter(self.tract.X, self.tract.Y, self.tract.Z, linewidth=0, alpha=0.15)
        ax.scatter(self.overlap.X, self.overlap.Y, self.overlap.Z, c="orange", linewidth=0, alpha=0.5)

        ax.set_xlim([xcom - _max, xcom + _max])
        ax.set_ylim([ycom - _max, ycom + _max])
        ax.set_zlim([zcom - _max, zcom + _max])
        plt.title(title)
        plt.show()
        plt.close()


class Tensor:
    def __init__(self, nii, centerline=None) -> None:
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
        self.centerline = self.get_centerline() if not centerline else centerline

        self.cross_section_area = defaultdict()
        self.set_cross_section_areas()



    def __str__(self):
        res = ""
        for node in self.graph:
            res += f"{node}\n"

        return res
    
    def origin_voxel_plot(self, title="title"):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        tracts = self.rollback(self.X, self.Y, self.Z)
        facecolors = np.where(tracts, '#FFD65DC0', '#7A88CCC0')
        
        filled_2 = explode(tracts)
        fcolors_2 = explode(facecolors)

        x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
        x[0::2, :, :] += 0.05
        y[:, 0::2, :] += 0.05
        z[:, :, 0::2] += 0.05
        x[1::2, :, :] += 0.95
        y[:, 1::2, :] += 0.95
        z[:, :, 1::2] += 0.95

        ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, alpha=0.35)
        
        xcom, ycom, zcom = self.tract.com

        _max = self.tract.plimit / 2
        ax.set_xlim([xcom - _max, xcom + _max])
        ax.set_ylim([ycom - _max, ycom + _max])
        ax.set_zlim([zcom - _max, zcom + _max])
        plt.show()


    def rollback(self, x, y, z):
        temp = np.array(self.tensor)
        nii = np.zeros(temp.shape, dtype=bool)

        for i, j, k in zip(x, y, z):
            nii[i][j][k] = True

        return nii

    @staticmethod
    def get_COM(x, y, z):
        return sum(x) // len(x), sum(y) // len(y), sum(z) // len(z)

    def get_index_limit(self):
        return max(max(self.X) - min(self.X), max(self.Y) - min(self.Y), max(self.Z) - min(self.Z)) + 1
    

    def set_cross_section_areas(self):
        for i in range(len(self.centerline) - 1):
            point = self.centerline[i]
            vector = sub_vector(self.centerline[i], self.centerline[i + 1])
            self.cross_section_area[point] = self.get_cross_section_area(point, vector)
            

    def get_XYZ(self):
        _min = float('inf')
        _max = -float('inf')
        for i, x in enumerate(self.nii):
            for j, y in enumerate(x):
                for k, z in enumerate(y):
                    if z == 1.0:
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


    def plot(self, title, p=True):
        _max = self.plimit / 2
        xcom, ycom, zcom = self.com

        self.ax.scatter(self.X, self.Y, self.Z, linewidth=0, alpha=0.2)
        self.ax.set_xlim([xcom - _max, xcom + _max])
        self.ax.set_ylim([ycom - _max, ycom + _max])
        self.ax.set_zlim([zcom - _max, zcom + _max])

        if p:
            plt.title(title)
            plt.show()
            plt.close()


    def plot_with_centerline(self,title="Superior longitudinal fasciculus L", short=True, p=True, plot_plane=False, color="r", plane_color="black", plane_interval=2):
        _max = self.plimit / 2
        xcom, ycom, zcom = self.com
        xc, yc, zc = self.get_centerline(short, True)
        centerline = [(_x, _y, _z) for _x, _y, _z in zip(xc, yc, zc)]

        self.ax.scatter(xc, yc, zc, c=color, linewidth=2)

        if plot_plane:
            for i in range(0, len(centerline) - 1):
                vector = sub_vector(centerline[i], centerline[i + 1])
                x, y, z = plane = self.get_nv_plane(centerline[i], vector)
                self.ax.scatter(x, y, z, c=plane_color, linewidth=2)


        self.ax.set_xlim([xcom - _max, xcom + _max])
        self.ax.set_ylim([ycom - _max, ycom + _max])
        self.ax.set_zlim([zcom - _max, zcom + _max])


        if p:
            plt.title("Superior longitudinal fasciculus L")
            plt.show()
            plt.close()


    def get_centerline(self, short=True, getXYZ=False, interval=2):
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

        rc = []
        for i in range(0, len(centers), interval):
            rc.append(centers[i])

        self.centerline = rc

        if getXYZ:
            return [center[0] for center in rc], [center[1] for center in rc], [center[2] for center in rc]
        else:
            return rc


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
                if res < RES and distance(x, y, z) < DISMAX:
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


    def get_nv_plane(self, point, vector):
        x0, y0, z0 = point
        iv, jv, kv = vector

        d = -1 * (iv * x0 + jv * y0 + kv * z0)
        equation = lambda _x, _y, _z: abs(iv * _x + jv * _y + kv * _z + d)
        distance = lambda _x, _y, _z: math.sqrt(((_x - x0) ** 2) + (_y - y0) ** 2 + (_z - z0) ** 2)
        tx, ty, tz = [], [], []

        for x, y, z in self.tensor:
            res = equation(x, y, z)
            if res < RES and distance(x, y, z) < DISMAX:
                tx.append(x)
                ty.append(y)
                tz.append(z)

        return tx, ty, tz



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


    def get_random_points_alongXYZ(self, interval=1):
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

    def get_cross_section_area(self, point, vector):
        x0, y0, z0 = point
        iv, jv, kv = vector
        d = -1 * (iv * x0 + jv * y0 + kv * z0)
        equation = lambda _x, _y, _z: abs(iv * _x + jv * _y + kv * _z + d)
        distance = lambda _x, _y, _z: math.sqrt(((_x - x0) ** 2) + (_y - y0) ** 2 + (_z - z0) ** 2)
        tx, ty, tz = [], [], []

        area = 0
        for x, y, z in self.tensor:
            if equation(x, y, z) < RES and distance(x, y, z) < DISMAX:
                area += 1

        return area


def pointlist_to_xyz(l):
    x, y, z = [], [], []
    for i, j, k in l:
        x.append(i)
        y.append(j)
        z.append(k)

    return x, y, z


def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


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


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

__u = [-1, 0, 1]
UNIT_VECTORS = [(i, j, k) for i in __u for j in __u for k in __u if not (i == 0 and j == 0 and k == 0)]


class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.neightbor = self.all_neighbor()
        self.n_n = len(self.neightbor)

    def all_neighbor():
        pass
    

class Tensor:
    def __init__(self, nii):
        self.nii = nii      # origin
        self.tensor = []    # sparse matrix
        self.graph = []     # nodes map
        self.XSUM, self.YSUM, self.ZSUM = 0, 0, 0
        self.X, self.Y, self.Z = self.get_XYZ()
        self.com = self.get_COM()

    def get_XYZ(self):
        for i, x in enumerate(self.nii):
            for j, y in enumerate(x):
                for k, z in enumerate(y):
                    if z > 0.0:
                        self.tensor.append((i, j, k))
                        temp = Node(i, j, k)
                        self.graph.append(temp)
                        self.XSUM += i
                        self.YSUM += j
                        self.ZSUM += k
        
        x, y, z = [], [], []
        for ten in self.tensor:
            _x, _y, _z = ten
            x.append(_x)
            y.append(_y)
            z.append(_z)
        
        return x, y, z
    
    def get_COM(self):
        return self.XSUM / len(self.X), self.YSUM / len(self.Y), self.ZSUM / len(self.Z)



if __name__=="__main__":
    print(UNIT_VECTORS)
    print(len(UNIT_VECTORS))
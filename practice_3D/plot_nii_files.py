import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

DIR_PATH = "/mnt/nvme1n1/JW/crps/rois_tracts_bin"

XSUM = 0
YSUM = 0
ZSUM = 0

def non_zero_elem(arr):
    return len(arr[arr>0.0])


def get_XYZ(nii):
    tensor = []
    global XSUM, YSUM, ZSUM

    for i, x in enumerate(nii):
        for j, y in enumerate(x):
            for k, z in enumerate(y):
                if z > 0.0:
                    tensor.append((i, j, k))
                    XSUM += i
                    YSUM += j
                    ZSUM += k
    
    x, y, z = [], [], []
    for ten in tensor:
        _x, _y, _z = ten
        x.append(_x)
        y.append(_y)
        z.append(_z)
    
    return x, y, z

def plot_nii(nii):
    global XSUM, YSUM, ZSUM
    x, y, z = get_XYZ(nii)
    X = XSUM / len(x)
    Y = YSUM / len(y)
    Z = ZSUM / len(z)
    XSUM, YSUM, ZSUM = 0, 0, 0
    _max = max(max(x) - min(x), max(y) - min(y), max(z) - min(z)) / 2
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, linewidth=0)
    ax.set_xlim([X - _max, X + _max])
    ax.set_ylim([Y - _max, Y + _max])
    ax.set_zlim([Z - _max, Z + _max])

    plt.show()
    plt.close()

if __name__ == "__main__":
    # plots()
    import os
    
    filenames = os.listdir(DIR_PATH)

    for file in filenames:
        file_path = os.path.join(DIR_PATH, file)
        print(file)
        nii = nib.load(file_path).get_fdata()
        plot_nii(nii)



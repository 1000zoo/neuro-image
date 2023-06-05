import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# LESION_PATH = "/mnt/nvme1n1/JW/JHU-ICBM-tracts-maxprob-thr25-2mm.nii.gz"
PATH = "/mnt/nvme1n1/JW/crps/rois_tracts_bin/tract_20.nii.gz"
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
    x, y, z = get_XYZ(nii)

    X = XSUM / len(x)
    Y = YSUM / len(y)
    Z = ZSUM / len(z)
    _max = max(max(x) - min(x), max(y) - min(y), max(z) - min(z))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, linewidth=0)
    ax.set_xlim([X - _max, X + _max])
    ax.set_ylim([Y - _max, Y + _max])
    ax.set_zlim([Z - _max, Z + _max])
    plt.show()


if __name__ == "__main__":
    # plots()
    track = nib.load(PATH).get_fdata()
    plot_nii(track)


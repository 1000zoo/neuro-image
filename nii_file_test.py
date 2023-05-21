import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

LESION_PATH = "/Users/1000zoo/Documents/crps_data/crps/lesion/ROI_C16_FLAIR.nii"
TRACK_PATH = "/Users/1000zoo/Documents/crps_data/crps/rois_tracts_bin/tract_1.nii.gz"
OVERLAP_PATH = "/Users/1000zoo/Documents/crps_data/crps/overlap_results/C02/overlap_tract_1.nii.gz"

def non_zero_elem(arr):
    return len(arr[arr>0.0])

def get_midline(nii):
    midline = []

    for i, x in enumerate(nii):
        for j, y in enumerate(x):
            for k, z in enumerate(y):
                if z > 0.0:
                    midline.append((i, j, k))


def get_XYZ(nii):
    tensor = []

    for i, x in enumerate(nii):
        for j, y in enumerate(x):
            for k, z in enumerate(y):
                if z > 0.0:
                    tensor.append((i, j, k))
    
    x, y, z = [], [], []
    for ten in tensor:
        _x, _y, _z = ten
        x.append(_x)
        y.append(_y)
        z.append(_z)
    
    return x, y, z

def plot_nii(nii):
    x, y, z = get_XYZ(nii)
    print(len(x))
    print(len(y))
    print(len(z))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, linewidth=0)

    from matplotlib import animation

    def animate(i):
        ax.view_init(elev=30., azim=i)
        return fig,

    ani = animation.FuncAnimation(fig, animate, frames=360, interval=20, blit=True)
    ani.save("c16_flair.gif", fps=30)
    # plt.show()


if __name__ == "__main__":
    # plots()
    # track = nib.load(TRACK_PATH).get_fdata()
    # plot_nii(track)
    lesion = nib.load(LESION_PATH).get_fdata()
    plot_nii(lesion)


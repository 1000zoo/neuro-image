from src.tract_centerline import Tensor
import nibabel as nib
import numpy as np
from random import shuffle


PATH = lambda x : f"/mnt/nvme1n1/JW/crps/rois_tracts_bin/tract_{x}.nii.gz"


if __name__ == "__main__":
    s = 1
    e = 21
    for i in range(s, e):
        track = nib.load(PATH(i)).get_fdata()
        t = Tensor(track, 5, p=0.01)

        t.plot_with_centerline(t.short_cut(), p=False)
        t.plot()
    
    
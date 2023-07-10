from src.tract_centerline import Tensor, SubTract
import nibabel as nib
import numpy as np
from random import shuffle


PATH = lambda x : f"/mnt/nvme1n1/JW/crps/rois_tracts_bin/tract_{x}.nii.gz"
SUB_PATH = lambda x : f"/mnt/nvme1n1/JW/36/subtract/sub_tract_{x}.nii.gz"
OVER = lambda x : f"/mnt/nvme1n1/JW/36/overlap/overlap_tract_{x}.nii.gz"
LESION = "/mnt/nvme1n1/JW/ROI_C36_CT.nii"
# LESION = "/mnt/nvme1n1/JW/crps/lesion/ROI_C36_CT.nii"

# 10 -> 9
tract_name = [
    "",
    "Anterior thalamic radiation L",
    "Anterior thalamic radiation R",
    "Corticospinal tract L",
    "Corticospinal tract R",
    "Cingulum (cingulate gyrus) L",
    "Cingulum (cingulate gyrus) R",
    "Cingulum (hippocampus) L",
    "Cingulum (hippocampus) R",
    "Forceps major",
    "Forceps minor",
    "Inferior fronto-occipital fasciculus L",
    "Inferior fronto-occipital fasciculus R",
    "Inferior longitudinal fasciculus L",
    "Inferior longitudinal fasciculus R",
    "Superior longitudinal fasciculus L",
    "Superior longitudinal fasciculus R",
    "Uncinate fasciculus L",
    "Uncinate fasciculus R",
    "Superior longitudinal fasciculus (temporal part) L",
    "Superior longitudinal fasciculus (temporal part) R"
]

if __name__ == "__main__":
    s = 15
    e = 16
    lesion = nib.load(LESION).get_fdata()
    
    for i in range(s, e):
        tract = nib.load(PATH(i)).get_fdata()
        st = nib.load(SUB_PATH(i)).get_fdata()
        overlap = nib.load(OVER(i)).get_fdata()

        # t = Tensor(tract, is_origin=True)
        # t.plot()

        # t1 = Tensor(tract)
        # t2 = Tensor(st)

        print("------")
        print(i)
        # t = Tensor(lesion)
        subtract = SubTract(tract, st, overlap, lesion)
        subtract.voxel_plot()
        # subtract.plot()
        # # print(t.tensor)
        # # subtract.surface(tract_name[i])


        # t.short_cut(p=True)
        # t.plot(tract_name[i])
        
        # print(len(subtract.tract.tensor), len(subtract.sub_tract.tensor), len(subtract.overlap.tensor))

        # subtract.plot(tract_name[i])
    
    
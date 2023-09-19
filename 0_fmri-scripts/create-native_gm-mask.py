import os
import glob
import numpy as np
import nibabel as nib
from nilearn.image import binarize_img, math_img, resample_to_img

sub_list = ["01", "02", "03", "04", "05", "06", "09", "10", "14", "15", "16", "17", "18", "19", "20"]
prob=.20

def gm_mask_prob(mask_path, func_ref_path, prob=.5):
    return resample_to_img(
        binarize_img(
            math_img(f"img>={prob}", img=mask_path), 
        threshold=prob), 
    func_ref_path, interpolation="nearest")

for sub in sub_list:
    print(f"segmenting subject-{sub}")
    anat_dir = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep_t1w/sub-{sub}/ses-movie/anat"
    func_dir = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep_t1w/sub-{sub}/ses-movie/func"
    sub_path = anat_dir + f"/sub-{sub}_ses-movie_label-GM_probseg.nii.gz"
    sub_func_path = func_dir + f"/sub-{sub}_ses-movie_task-movie_run-1_space-T1w_desc-preproc_bold.nii.gz"
    gm_mask = gm_mask_prob(mask_path=sub_path, prob=prob, func_ref_path=sub_func_path)
    nib.save(gm_mask, anat_dir + f"/sub-{sub}_ses-movie_label-GM_prob-{prob}.nii.gz")
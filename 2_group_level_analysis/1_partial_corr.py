import os
from itertools import product

import numpy as np
from scipy.stats import spearmanr
import nibabel as nib
from nilearn.image import new_img_like

sub_list = ["01", "02", "03", "04",  "06", "09", "10", "14", "15", "16", "17", "18", "19", "20"] # "05",
run_list = ["1", "2", "3", "4", "5", "6", "7", "8"]

hypothesis = "spatial-attent-imgsim"
scratch_dir = "/home/data/study_gaze_tracks/scratch"

# set paths to store output
output_folder = scratch_dir + f"/eucl-distance/partial-corr_hypothesis-{hypothesis}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

def upper_tri(RDM):
    """
    upper_tri returns the upper triangular index of an RDM

    Args:
        RDM 2Darray: squareform RDM

    Returns:
        1D array: upper triangular vector of the RDM
    """
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]

def fixat_img_sim(run, ssim=True):
    """
    calculate spearman correlation between fixation density RDMs
    and image similarity RDMs (ssim or euclidean distance).

    Returns:
        spearman r coefficient
    """
    if ssim == True:
        img_sim_path = os.path.join("/home", "data", "study_gaze_tracks", "derivatives", "eucl-distance-models", "spatial-attention-duration-4_img-sim", f"run-{run}", f"ssim_run-{run}.tsv")
    else: 
        img_sim_path = os.path.join("/home", "data", "study_gaze_tracks", "derivatives", "spatial-attention_ssim", f"run-{run}", f"ssim_run-{run}.tsv")   
    fixation_sim_path = os.path.join("/home", "data", "study_gaze_tracks", "derivatives", "eucl-distance-models", "spatial-attention_duration-4", "general_model", f"run-{run}_general-model-matrix.tsv")
    img_sim_model = upper_tri(np.loadtxt(img_sim_path, delimiter=","))
    fixation_sim_model = upper_tri(np.loadtxt(fixation_sim_path, delimiter=","))
    return spearmanr(img_sim_model, fixation_sim_model)[0]

def get_func(sub, run):
    """
    load functional data and return it as numpy array.
    """
    spa_att = os.path.join("/home", "data", "study_gaze_tracks", "scratch", "snpm_input_studyf", f"ssub-{sub}_run-{run}.nii")
    img_sim = os.path.join("/home", "data", "study_gaze_tracks", "scratch", "eucl-distance", "eucl-sim-maps_spatial-attention-duration-4_smooth-6_space-mni", f"sub-{sub}_run-{run}_mni.nii.gz")
    return nib.load(spa_att).get_fdata(), nib.load(img_sim).get_fdata()

def partial_corr(r_neuro_fix, r_neuro_sim, r_fix_sim):
    """
    calculate partial correlation between args.

    Args:
        r_neuro_fix: spearman corr. between sl-neuro value and fixation-density maps
        r_neuro_sim: spearman corr. between sl-neuro value and ssim / eucl. distance images
        r_fix_sim: spearman corr. between fixation-density maps and ssim / eucl. distance images

    Returns:
        partial correlation
    """
    ma = r_neuro_fix-(r_neuro_sim*r_fix_sim)
    ma_d = np.sqrt(1-r_neuro_sim**2) * np.sqrt(1-r_fix_sim**2)
    return ma / ma_d

for sub, run in product(sub_list, run_list):
    print(f"working on sub-{sub} and run-{run}")
    mask_path = "/home/data/study_gaze_tracks/scratch/studf_avg152T1_gray_prob50_bin_resampled.nii"
    mask_img = nib.load(mask_path)

    corr_eye_sim = fixat_img_sim(run=run, ssim=True)
    spa_att_data, img_sim_data = get_func(sub, run)

    # get 3D shape
    x, y, z = spa_att_data.shape

    spa_att_vec = np.reshape(spa_att_data, -1)
    img_sim_vec = np.reshape(img_sim_data, -1)

    partial_corr_list = list(map(partial_corr, spa_att_vec, img_sim_vec, np.repeat(corr_eye_sim, len(spa_att_vec))))
        
    corrected_values = np.reshape(partial_corr_list, [x, y, z])
    nib.save(new_img_like(mask_img, corrected_values), output_folder + f"/sub-{sub}_run-{run}_hypothesis-{hypothesis}")
import os
import sys
import glob

import numpy as np
import nibabel as nib
from rsatoolbox.rdm.rdms import RDMs, load_rdm
from rsatoolbox.util.searchlight import evaluate_models_searchlight
from rsatoolbox.model import ModelWeighted
from rsatoolbox.inference import eval_fixed

import matplotlib.pyplot as plt
import seaborn as sns

# I/O
sub, run = str(sys.argv[1]), str(sys.argv[2])
duration, fwhm = 4, 6

eval_metric = "spearman"
derivatives_dir, scratch_dir = "/home/data/study_gaze_tracks/derivatives/", "/home/data/study_gaze_tracks/scratch"
hypothesis_list = [f"spatial-attention-duration-{duration}"] # , f"spatial-attention_attention-mode_duration-{duration}"

def load_candidates(candidate_dir, sub, run, hypothesis, duration):
    models_rdm_list = []
    
    for model in glob.glob(candidate_dir + f"/general_model/run-{run}_general-model-matrix.tsv"):
        print("load general model:\t", model)
        model_rdm_general = np.loadtxt(model, delimiter=",")
        models_rdm_list.append(model_rdm_general)

    if hypothesis == f"spatial-attention-duration-{duration}":
        for model in glob.glob(candidate_dir + f"/sub-*/sub-*_run-{run}*.tsv"):
            print("load individual model:\t", model)
            model_rdm_general = np.loadtxt(model, delimiter=",")
            models_rdm_list.append(model_rdm_general)
    elif hypothesis == "spatial-attention_attention-mode_complete" or hypothesis == f"spatial-attention_attention-mode_duration-{duration}":
        for model in glob.glob(candidate_dir + f"/run-{run}/sub-{sub}*_sub-*_run-{run}*.tsv"):
            print("load individual model:\t", model)
            model_rdm_general = np.loadtxt(model, delimiter=",")
            models_rdm_list.append(model_rdm_general)
    return RDMs(
            dissimilarities=np.array(models_rdm_list),
            dissimilarity_measure="euclidean"
        )

def fisher_r_to_z(x):
    """    
    correct any rounding errors
    correlations cannot be greater than 1.
    """
    x = np.clip(x, -1, 1)

    return np.arctanh(x)

def plot_evaluation(sub, run, dataframe, save_folder):
    sns.distplot(dataframe)
    plt.title("Distributions of Correlations", size=18)
    plt.ylabel("Occurance", size=15)
    plt.xlabel("Spearmann Correlation", size=15)
    sns.despine()

    plot_name = save_folder + f"/sub-{sub}_run-{run}.png"

    plt.savefig(plot_name)
    plt.close()
    return None

def np2nii(img, scores, filename):
    """
    It saves data into a nifti file
    by leveraging on another image
    of the same size.
    Parameters
    ----------
    img : nifti file (e.g a mask)
    scores : numpy array containing decoding scores
    filename : string name of the new nifti file
    Returns
    -------
    nii_file : Nifti1Image
    """
    header = nib.Nifti1Header()
    affine = img.affine
    nii_file = nib.Nifti1Image(scores, affine, header)
    nib.save(nii_file, filename)
    return nii_file

for hypothesis in hypothesis_list:
    print(f"working on sub-{sub}, run-{run} and hypothesis:\t {hypothesis}")
    # load sl rdms for each voxel
    sl_rdm_dir = scratch_dir + f"/neural-rdms_spatial-attention-duration-{duration}_space-t1w_fwhm-{fwhm}_beta"
    save_folder = scratch_dir + f"/corr-maps_{hypothesis}_smooth-{fwhm}_space-t1w"
    
    filename = f"/sub-{sub}_run-{run}.nii.gz"
    if os.path.isfile(save_folder + filename):
        print("file already exists, abort mission ...")
        continue

    filename_hdf = f"/sub-{sub}_run-{run}_task-studyforrest"
    rdm = load_rdm(sl_rdm_dir + filename_hdf, file_type="hdf5")
    print("shape of target matrix:\t", rdm.dissimilarities[0].shape)

    weighted_model = ModelWeighted("weighted model", load_candidates(derivatives_dir + hypothesis, sub, run, hypothesis=hypothesis, duration=duration))

    # evaluate each voxel RDM with a fixed effects model
    eval_results = evaluate_models_searchlight(
        sl_RDM=rdm,
        models=weighted_model,
        eval_function=eval_fixed,
        method=eval_metric,
        n_jobs=4)

    eval_score = [np.float(e.evaluations) for e in eval_results]
    eval_score = fisher_r_to_z(eval_score)

    # load mask
    mask_path = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep/sub-{sub}/ses-movie/anat/sub-{sub}_ses-movie_label-GM_prob-0.2.nii.gz"
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata() 

    x, y, z = mask.shape
    RDM_brain = np.zeros([x * y * z])
    RDM_brain[list(rdm.rdm_descriptors["voxel_index"])] = list(eval_score)
    RDM_brain = RDM_brain.reshape([x, y, z])

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    plot_evaluation(sub, run, eval_score, save_folder)

    # save image
    print(f"saving file to: {save_folder + filename} ...")
    np2nii(mask_img, RDM_brain, save_folder + filename)
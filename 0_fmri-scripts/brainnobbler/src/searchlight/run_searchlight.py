#!/usr/bin/env python3

"""
"""
import os
import sys
import argparse
from dataset import GetData
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from searchlight import Searchlight

# parser = argparse.ArgumentParser(add_help=True)
# parser.add_argument("-d", "--data_dir", metavar="PATH", required=True)
# parser.add_argument("-t", "--tasks", type=str, nargs="+", required=True)
# parser.add_argument("-m", "--mask", type=str, required=True)
# parser.add_argument("-p", "--permutation", type=int, required=False)
# parser.add_argument("-e", "--estimator", type=str, required=True)
# parser.add_argument("-r", "--radius", type=float, required=True)
# parser.add_argument("-o", "--output_dir", type=str, required=False)
# args = parser.parse_args()

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
    img = nib.load(img)
    header = nib.Nifti1Header()
    affine = img.affine
    nii_file = nib.Nifti1Image(scores, affine, header)
    nib.save(nii_file, filename)
    return nii_file

SUB = str(sys.argv[1])

DATA_DIR = f"/home/data/NegativeCueing_RSA/NegCue_Random/derivatives/lss/sub-{SUB}/"
MASK = f"/home/data/NegativeCueing_RSA/NegCue_Random/derivatives/fmriprep/sub-{SUB}/func/sub-{SUB}_task-negativesearcheventrelated_run-1_space-T1w_desc-brain_mask.nii.gz"
RADIUS, ESTIMATOR = 5, "lsvc"
TASKS = ["negativecue", "neutralcue"]
OUTPUT = f"/home/data/NegativeCueing_RSA/NegCue_Random/derivatives/sl_svm/sub-{SUB}/"

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

dataset = GetData(
    tasks=TASKS, labels={TASKS[0]: 1, TASKS[1]: 2}, group_filter=TASKS
)
data = dataset(DATA_DIR)
print(data)
# instantiate searchlight
SL = Searchlight(RADIUS, MASK, estimator=ESTIMATOR, cv="rkfold")
accuracies = SL(data)
np2nii(MASK, accuracies, OUTPUT + f"sub-{SUB}_sl_accuracies.nii.gz")
np.save(OUTPUT + f"sub-{SUB}_sl.npy", accuracies)
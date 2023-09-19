#!/usr/bin/env python3
"""
"""
import argparse
from os.path import join
import nibabel as nib
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
import numpy as np
from sec_lev_design_matrix import SecLevDesignMat

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--data_dir", metavar="PATH", required=True)
parser.add_argument("-o", "--output", type=str, required=False)
parser.add_argument("-c", "--conditions", type=str, nargs="+", required=False)
parser.add_argument("-m", "--mask", type=str, required=True)
parser.add_argument("-n", "--num_subjects", type=int, required=True)
args = parser.parse_args()

if args.conditions is not None:
    COND = args.conditions
else:
    COND = ["f_col", "f_high", "f_low", "f_pl", "nf_col", "nf_high", "nf_low", "nf_pl"]

# Create a second level design matrix
des_mat = SecLevDesignMat(args.num_subjects, COND)
# in case your matrix is ranking deficient, make it full rank
des_mat.check_matrix_rank()
dm = des_mat.mk_full_rank("nf_pl")
# plot the design matrix
des_mat.plot_dm(dm)
# Set up the second level and fit the data
SL = SecondLevelModel(smoothing_fwhm=6.0)
second_level_model = SL.fit(
    second_level_input=[
        nib.load(
            join(
                args.data_dir,
                f"sub-{sub:02}",
                f"sub-{sub:02}_{cond}_fixed_effects.nii.gz",
            )
        )
        for sub in range(1, args.num_subjects + 1)
        for cond in COND
    ],
    design_matrix=dm,
)


def mk_contrasts(conditions, subjects):
    """ """
    contrast_matrix = np.eye(len(conditions))
    return {
        column: np.hstack((contrast_matrix[i], np.zeros(subjects)))
        for i, column in enumerate(conditions)
    }


CONTRASTS = mk_contrasts(dm.columns[: -args.num_subjects], args.num_subjects)
print(CONTRASTS)
ALL_CONTR = {
    "f_vs_nf": (
        CONTRASTS["f_col"]
        + CONTRASTS["f_high"]
        + CONTRASTS["f_low"]
        + CONTRASTS["f_pl"]
    )
    - (CONTRASTS["nf_col"] + CONTRASTS["nf_high"] + CONTRASTS["nf_low"])
}
# run each single contrast, threshold and save data
for contr_name, contr in ALL_CONTR.items():
    print(contr)
    print(f"running {contr_name} contrast")
    res = second_level_model.compute_contrast(contr, "stat", "t")
    # threshold the map and correct for multiple comparisons
    thresholded_map, threshold = threshold_stats_img(
        res,
        alpha=0.05,
        height_control="fpr",
        cluster_threshold=10,
        two_sided=True,
    )
    thresholded_map.to_filename(join(args.output, f"{contr_name}.nii.gz"))

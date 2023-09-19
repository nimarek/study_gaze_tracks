#!/usr/bin/env python3
import os
import sys
import pandas as pd
import argparse
from single_trial_glm import SingleTrialFl

SUB, RUN = str(sys.argv[1]), str(sys.argv[2])

FUNC_DIR = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep/sub-{SUB}/ses-movie/func"
EVENTS_DIR = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/sub-{SUB}/ses-movie/func/"
OUTPUT = f"/home/data/study_gaze_tracks/scratch/lss_spatial-attention-onsets-4_fwhm-6_beta/run-{RUN}"

def split_list(a_list):
    """
    From: https://stackoverflow.com/questions/752308/split-list-into-smaller-lists-split-in-half 
    """
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def select_confounds(sub, run, func_dir):
    """
    Select usual trans + rot parameters, but also select
    50% of the aCompCor noise components per subject and
    run.
    """
    realignment_para = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    tmp_df = pd.read_csv(func_dir + f"/sub-{sub}_ses-movie_task-movie_run-{run}_desc-confounds_regressors.tsv", sep="\t")
    acompcor_headers = [col for col in tmp_df.columns.values if "a_comp_cor" in col]
    acomcor_selectors, _ = split_list(acompcor_headers)
    return realignment_para # + acomcor_selectors[:3]

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

TR = 2
MOTION_REG = select_confounds(sub=SUB, run=RUN, func_dir=FUNC_DIR)
DUMMY_SCANS = 0

if RUN == "1":
    SCANS = 451
elif RUN == "2":
    SCANS = 441
elif RUN == "3":
    SCANS = 438
elif RUN == "4":
    SCANS = 488
elif RUN == "5":
    SCANS = 462
elif RUN == "6":
    SCANS = 439
elif RUN == "7":
    SCANS = 542
elif RUN == "8":
    SCANS = 338

ST = SingleTrialFl(SUB, RUN, FUNC_DIR, EVENTS_DIR, TR, MOTION_REG, DUMMY_SCANS, n_scans=SCANS)
DMs = list(ST.lss_design_matrix(ST.events[0], ST.confounds[0]))

# ST.plot_dm(DMs[45][0])
STATS = ST(output=OUTPUT)

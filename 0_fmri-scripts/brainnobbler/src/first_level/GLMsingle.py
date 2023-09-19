#!/usr/bin/env python3
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
from glmsingle.glmsingle import GLM_single

SUB, RUN = str(sys.argv[1]), str(sys.argv[2])

FUNC_DIR = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/sub-{SUB}/ses-movie/func/"
EVENTS_DIR = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/sub-{SUB}/ses-movie/func/"
OUTPUT = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/lss_spatial-attention_space-t1w_rev/run-{RUN}"

GLMsingle_instance = GLM_single()

def build_design(datafolder_event, data_bold):
    design_list = []
    eventfiles = np.sort([fn for fn in os.listdir(datafolder_event) if "events.tsv" in fn])

    for e in (range(len(eventfiles))):
        print("using event file:\t", eventfiles[e])
        print("shape of the bold data:\t", data_bold[e].shape)
        run_events = pd.read_csv(os.path.join(datafolder_event, eventfiles[e]), delimiter="\t")
        num_conds = len(run_events["trial_type"])
        conds = run_events["trial_type"]
        run_design = np.zeros((data_bold[e].shape[3], num_conds))
        
        for c, cond in enumerate(conds):
            cond_idx = np.argwhere(run_events["trial_type"].values == cond)
            cond_vols = run_events["onset"].divide(2)
            print(cond_vols)
            run_design[cond_vols, c] = 1
        design_list.append(run_design)
    return design_list


def get_volumes(datafolder_bold):
    data_list = []
    data_files = np.sort([fn for fn in os.listdir(datafolder_bold) if "bold.nii.gz" in fn])

    for d in tqdm(range(len(data_files))):
        tmp = nib.load(os.path.join(datafolder_bold, data_files[d]))
        data_list.append(tmp.get_data())
        xyz = data_list[0].shape[:3]
    return data_list


func_data = get_volumes(FUNC_DIR)
design_ma = build_design(datafolder_event=FUNC_DIR, data_bold=func_data)

results_glmsingle = GLMsingle_instance.fit(
    design=design_ma,
    data=func_data,
    stimdur=4,
    tr=2,
    outputdir=OUTPUT)
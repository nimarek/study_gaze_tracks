import os
import glob

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

root_dir = "/home/data/study_gaze_tracks/derivatives"

for run in range(1, 9):
    print("starting with run:\t", run)
    run_df_tmp, overall_df = [], []

    for model in glob.glob(root_dir + f"/spatial-attention_sim-struct/run-{run}/*run-{run}*.tsv"):# f"/ffa-paa_sim-struct/run-{run}/*run-{run}*.tsv"):
        model_rdm = np.loadtxt(model, delimiter=",")
        run_df_tmp.append(model_rdm)
        
        # average over all participants per run
        mean_value = np.mean(run_df_tmp)
        print(f"run-{run} mean: {mean_value}")
        run_df_tmp.clear()
        overall_df.append(mean_value)


print(f"overall mean: {np.mean(overall_df)}")
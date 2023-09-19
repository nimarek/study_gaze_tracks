import os
import sys
import glob

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

root_dir = "/home/data/study_gaze_tracks/derivatives"

def plot_heatmap_eucl(run, df, output_heat_maps):
    # masked_tria = np.triu(df)
    hm_fig_dissim = sns.heatmap(df,
                         cmap="RdBu_r") # , mask=masked_tria
    hm_fig_dissim.set_title(f"run-{run}", fontweight="bold")
    plt.savefig(output_heat_maps + f"/diff-model_run-{run}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return None

for run in range(1, 9):
    print("starting with run:\t", run)
    matrices_tmp = []

    for sub_matrix_a in glob.glob(f"/home/data/study_gaze_tracks/derivatives/attention_mode/run-{run}/*.tsv"):
        for sub_matrix_b in glob.glob(f"/home/data/study_gaze_tracks/derivatives/attention_mode/run-{run}/*.tsv"):
            matrix_a = np.loadtxt(sub_matrix_a, delimiter=",")
            matrix_b = np.loadtxt(sub_matrix_b, delimiter=",")
            matrices_tmp.append(euclidean_distances(matrix_a, matrix_b))

    # save difference model per run
    output_dir = root_dir + f"/attention_mode/difference_model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_heatmap_eucl(run, np.mean(matrices_tmp, axis=0), output_dir)
    pd.DataFrame(np.mean(matrices_tmp, axis=0)).to_csv(output_dir + f"/run-{run}_difference-model-matrix.tsv", header=False, index=False)
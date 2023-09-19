import os

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

root_dir = "/home/data/study_gaze_tracks/derivatives"

def plot_heatmap_eucl(run, df, output_heat_maps):
    # masked_tria = np.triu(df)
    hm_fig_dissim = sns.heatmap(df,
                         cmap="RdBu_r") # , mask=masked_tria
    hm_fig_dissim.set_title(f"run-{run}", fontweight="bold")
    plt.savefig(output_heat_maps + f"/general-model_run-{run}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return None

for run in range(1, 9):
    print("starting with run:\t", run)
    matrix_a = np.loadtxt(os.path.join(root_dir, "spatial-attention", "general_model", f"run-{run}_general-model-matrix.tsv"), delimiter=",")
    matrix_b = np.loadtxt(os.path.join(root_dir, "spatial-attention_attention-mode", "general_model", f"run-{run}_general-model-matrix.tsv"), delimiter=",")

    interaction_effect = np.multiply(matrix_a, matrix_b)

    # save interaction-effect matrix per run
    output_dir = root_dir + f"/spatial-attention_attention-mode_ie/ie_matrices"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_heatmap_eucl(run, interaction_effect, output_dir)
    pd.DataFrame(interaction_effect).to_csv(output_dir + f"/run-{run}_spatial-attention_attention-mode_ie-matrix.tsv", header=False, index=False)
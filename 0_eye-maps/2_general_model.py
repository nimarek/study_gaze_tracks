import os
import glob

import numpy as np
import pandas as pd 
from scipy.stats import zscore
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
    input_data_list = []

    for model in glob.glob(root_dir + f"/spatial-attention_attention-mode_onsets-2/*/*run-{run}*.tsv"):
        model_rdm = np.loadtxt(model, delimiter=",")
        input_data_list.append(model_rdm)
        #input_data_list.append(zscore(model_rdm, axis=None))

        # average over all participants per run
        general_model = np.array(input_data_list).mean(0)

        # save general model per run
        output_dir = root_dir + f"/spatial-attention_attention-mode_onsets-2/general_model"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    plot_heatmap_eucl(run, general_model, output_dir)
    pd.DataFrame(general_model).to_csv(output_dir + f"/run-{run}_general-model-matrix.tsv", header=False, index=False)
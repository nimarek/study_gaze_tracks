import os
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import distance

import matplotlib.pyplot as plt
import seaborn as sns

# chunk list for 4 sec (scene onset)
# chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]

# chunk list for 4 sec (complete)
chunk_list = [0, 253, 252, 242, 275, 249, 243, 306, 178]

def create_dirs(run, metric="sqeuclidean"):
    """
    Create output dirs for further analysis-
    
    Return:
        string of output path
    """
    output_dir= os.path.join(os.getcwd(), "hmax_output", f"rdms-{metric}", f"run-{run}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_heatmap(run, df, output_heat_maps):
    hm_fig_dissim = sns.heatmap(df, cmap="RdBu_r", yticklabels=False, xticklabels=False)
    
    plt.savefig(output_heat_maps + f"/hmax_run-{run}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return None

def start_analysis(run, chunk_max, metric="sqeucldiean"):
    """
    (1) Calculate average vector per chunk for all 
    subjects except the subject of interest (SOI).
    (2) Load heatmap for SOI and call flatten_img.

    pkl order: batch_size, num_orientations, height, width
    """
    output_dir = create_dirs(run=run, metric=metric)    
    corr_container = []

    outf_path = output_dir + f"\\hmax_run-{run}_matrix.tsv"
    if os.path.isfile(outf_path):
        raise ValueError(f"file already exists:\t {outf_path}")
    
    for chunk_a in range(1, chunk_max+1):
        in_path_a = os.path.join(os.getcwd(), "hmax_output", f"run-{run}", f"run-{run}_chunk-{chunk_a}_c1-activations.pkl")
        with open(in_path_a, "rb") as f:
            data_a = pickle.load(f)
        chunk_a_arr = [(data_a["c1"][:][0][:].mean(axis=0).flatten())]

        for chunk_b in range(1, chunk_max+1):
            in_path_b = os.path.join(os.getcwd(), "hmax_output", f"run-{run}", f"run-{run}_chunk-{chunk_b}_c1-activations.pkl")
            with open(in_path_b, "rb") as f:
                data_b = pickle.load(f)
            chunk_b_arr = [(data_b["c1"][:][0][:].mean(axis=0).flatten())]

            corr_container.append(
                np.around(
                    distance.cdist(chunk_a_arr, chunk_b_arr, metric="sqeuclidean")[0][0], 
                    decimals=3)
                    )

    hmax_df = np.reshape(corr_container, (chunk_max, chunk_max))
    plot_heatmap(run=run, df=hmax_df, output_heat_maps=output_dir)
    pd.DataFrame(hmax_df).to_csv(outf_path, header=False, index=False)
    return None

"""
expected data shape: 8 x (1, 4, 63, 63)
output of c1 is 8 pooling layers, with batch_size=1 (one img).

"""
for run in range(1, 9):
    print(f"starting run-{run} ...")
    chunk_max = chunk_list[int(run)]
    start_analysis(run=run, chunk_max=chunk_max)
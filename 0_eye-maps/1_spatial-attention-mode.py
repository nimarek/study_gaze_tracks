import sys
import os
import glob
import numpy as np
import pandas as pd

from PIL import Image
from scipy.stats import zscore, spearmanr
from scipy.spatial import distance

import matplotlib.pyplot as plt
import seaborn as sns

soi, run = str(sys.argv[1]), str(sys.argv[2])

duration, bin_list = 4, [8, 15, 40]
metric = "euclidean"

# chunk list for 4 sec
chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]

def flatten_img(file_path, bins):
    """
    Read plotted gaze path and convert it to a flattend
    numpy array. Zscore and order the array.
    
    Return:
        Digitized / z-scored numpy array.
    """
    file = Image.open(file_path).convert("L")
    file = np.stack((file,) * 3, axis=-1)
    file = np.array(file) / 255.0
    return np.digitize(zscore(file.flatten()), np.arange(bins))

def mean_vector(file_path_list, bins):
    """
    Calculate mean vector from given list of gaze
    path heatmaps.
    
    Return:
        numpy array containing average vector.
    """
    heatmap_container = []
    for path in file_path_list:
        heatmap_container.append(flatten_img(path, bins))
    return np.mean(np.array(heatmap_container), axis=0)

def create_dirs(sub, duration):
    """
    Create output dirs for further analysis-
    
    Return:
        string of output path
    """
    
    output_dir= os.path.join("/home", "data", "study_gaze_tracks", "derivatives", f"spatial-attention_attention-mode_duration-{duration}", f"sub-{sub}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_heatmap(sub, run, df, output_heat_maps, bins):
    #masked_tria = np.triu(df)
    hm_fig_dissim = sns.heatmap(df,
                                vmin=0., vmax=1.,
                                #mask=masked_tria, 
                                cmap="RdBu_r")
    
    hm_fig_dissim.set_title(f"sub-{sub} run-{run}", fontweight="bold")
    plt.savefig(output_heat_maps + f"/sub-{sub}_run-{run}_bin-{bins}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return None

def start_analysis(soi, run, chunk_max, bins, duration=2, metric="eucldiean"):
    """
    (1) Calculate average vector per chunk for all 
    subjects except the subject of interest (SOI).
    (2) Load heatmap for SOI and call flatten_img.
    """
    output_dir = create_dirs(soi, duration=2)    
    comp_container, correlation_container = [], []
    soi_ident = f"sub-{soi}"

    for chunk_target in range(1, chunk_max+1):
        soi_heatmap = os.path.join("/home", "data", "study_gaze_tracks", "scratch", 
                                   f"spatial-attention_heatmaps-duration-{duration}", 
                                   f"sub-{soi}_output_fixation-density-maps", 
                                   f"sub-{soi}_run-{run}_chunk-{chunk_target}.png")
        soi_arr = flatten_img(soi_heatmap, bins)

        for chunk_compare in range(1, chunk_max+1):
            comp_heatmaps = glob.glob(os.path.join("/home", "data", "study_gaze_tracks", "scratch", 
                                                   f"spatial-attention_heatmaps-duration-{duration}", 
                                                   "sub-*_output_fixation-density-maps", 
                                                   f"sub-*_run-{run}_chunk-{chunk_compare}.png"))

            # remove elements from list that contain soi
            comp_heatmaps = [x for x in comp_heatmaps if soi_ident not in x]
            
            mean_arr = mean_vector(comp_heatmaps, bins)
            # print(np.around(distance.cdist([soi_arr], [mean_arr], metric)[0][0], decimals=3))
            correlation_container.append(np.around(distance.cdist([soi_arr], [mean_arr], metric)[0][0], decimals=3))
    
    endo_exo_model = np.reshape(correlation_container, (chunk_max, chunk_max))
    plot_heatmap(sub=soi, run=run, df=endo_exo_model, output_heat_maps=output_dir, bins=bins)
    pd.DataFrame(endo_exo_model).to_csv(output_dir + f"/sub-{soi}_run-{run}_bins-{bins}-matrix.tsv", 
                                    header=False, index=False)
    return None

for bins in bin_list:
    print(f"starting soi-{soi}, run-{run} ...")
    chunk_max = chunk_list[int(run)]
    start_analysis(soi, run, chunk_max, bins, duration=4, metric=metric)
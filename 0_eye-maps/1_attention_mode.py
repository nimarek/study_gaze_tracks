import os
import sys

import numpy as np
import pandas as pd

from sklearn.utils import resample
from scipy.spatial import distance

import matplotlib.pyplot as plt
import seaborn as sns

sub, sub_comp = str(sys.argv[1]), str(sys.argv[2])
run = sys.argv[3]

metric = "euclidean"
root_dir = "/home/data/study_gaze_tracks/derivatives"

# chunk list for 4 sec
chunk_list = [0, 59, 68, 55, 71, 54, 68, 83, 51]

# chunk list for 4 sec (complete)
# chunk_list = [0, 253, 252, 242, 275, 249, 243, 306, 178]

# chunk lsit for ffa_ppa
# chunk_list = [0, 46, 38, 39, 45, 49, 56, 68, 39]

def create_dirs(run, root_dir):    
    output_dir= root_dir + f"/spatial-attention_attention-mode_onsets-2/run-{run}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def combine_data(chunks_run, data_list):
    """
    Convert numpy arrays back to pandas dataframe to
    leverage pandas nan friendly correlation function.
    """
    return pd.DataFrame(dict(zip(list(range(chunks_run)), data_list)))

def load_split(sub, root_dir):
    """
    Function to load preprocessed eye-tracking data provided
    by Hanke et al. (https://www.studyforrest.org/)
    """
    file_path = root_dir + f"/fix_maps/sub-{sub}_ses-movie_task-movie.npz"    
    
    with np.load(file_path, allow_pickle=True) as data:
        df_movie_frame = data["sa.movie_frame"]
        df_names = data["fa.name"].astype("str")
        df_samples = data["samples"]
        
    # create dataframe from arrays
    df_all = pd.DataFrame(data=df_samples, columns=df_names)
    df_all["frame"] = df_movie_frame
    return df_all

def load_events(run, add_sec):
    """
    Load precalculated event files and use them as 
    onsets and duration to slice the eye-movement 
    data. And convert from seconds back to frames.
    
    Return: list of tuples with frame onsets and durations
    """
    event_path = f"/home/data/study_gaze_tracks/code/reference_spatial-attention/ses-movie_task-movie_run-{run}_events.tsv"
    # event_path = f"/home/data/study_gaze_tracks/code/reference_spatial-attention/complete_ses-movie_task-movie_run-{run}_events.tsv"
    # event_path = f"/home/data/study_gaze_tracks/code/reference_ffa-ppa/run-{run}_task-ffa-ppa_events.tsv"
    tmp_df = pd.read_csv(event_path, index_col=None, delimiter="\t")
    tmp_df["onset"] = tmp_df["onset"] + add_sec
    tmp_df["onset"] = tmp_df["onset"].apply(lambda x: x * 25) 
    tmp_df["duration"] = tmp_df["duration"].apply(lambda x: x * 100) - 50
    tmp_df["offset"] = tmp_df["onset"] + tmp_df["duration"]
    return list(zip(tmp_df["onset"], tmp_df["offset"]))

def chunk_data(df_all, b_frame, e_frame):
    chunked_df = df_all.loc[(df_all["frame"] >= b_frame) & (df_all["frame"] <= e_frame)]
    return chunked_df

def downsample(chunked_df, resample_to=50):
    """
    Convert pandas df to numpy arrays and mask all nan values and remove from dataframe.
    """
    raw_x, raw_y = chunked_df["x"], chunked_df["y"]
    pros_x, pros_y = raw_x[~np.isnan(raw_x)], raw_y[~np.isnan(raw_y)]

    if pros_x.size == 0:
        res_x, res_y = [0], [0]
    else:
        res_x, res_y = resample(pros_x, replace=True, n_samples=resample_to, random_state=0), resample(pros_y, n_samples=resample_to, random_state=0)
    return np.concatenate((res_x, res_y), axis=None)

def calc_diff(chunk_list, chunk_max, sim_df, sim_comp_df, metric):
    """
    Calculate euclidean distance between pairs of image vectors
    from normalised fixation density maps. 
    """
    eucl_dist = []
    
    for chunk_a in chunk_list:
        for chunk_b in chunk_list:
            print(f"comparing chunk-{chunk_a} with chunk-{chunk_b}")
            df_a, df_b = sim_df[chunk_a].to_numpy(), sim_comp_df[chunk_b].to_numpy()
            dc = distance.cdist([df_a], [df_b], metric)[0]
            eucl_dist.append(dc)
    return np.reshape(eucl_dist, (chunk_max, chunk_max))

def create_df(steps_list_comp, soi=None):
    """
    Chunk data according to event files.
    """
    tmp_list = []
    for start_time_comp, end_time_comp in steps_list_comp:
        print(f"starting with onset frame: {start_time_comp} offset frame: {end_time_comp}")
        raw_input_df = load_split(sub=soi, root_dir=root_dir)
        chunked_df = chunk_data(df_all=raw_input_df, b_frame=start_time_comp, e_frame=end_time_comp)
        tmp_list.append(downsample(chunked_df, resample_to=50))
    return combine_data(chunk_list[int(run)], tmp_list)

def plot_heatmap_eucl(sub, sub_comp, run, labels, df, metric, output_heat_maps):
    hm_fig_dissim = sns.heatmap(df, xticklabels=labels, yticklabels=labels, 
                         cmap="RdBu_r") # , mask=masked_tria
    hm_fig_dissim.set_title(f"sub-{sub} run-{run}", fontweight="bold")
    plt.savefig(output_heat_maps + f"/sub-{sub}_sub-{sub_comp}_run-{run}.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return None

add_sec = - 1
corr_container_rest = []

# start the analysis
exa_df = load_split("01", root_dir)
start_frame, max_frame = np.min(exa_df["frame"]), np.max(exa_df["frame"])

output_dir = create_dirs(run, root_dir)

print(f"starting sub-{sub}, comparing to sub-{sub_comp} for run-{run}")
steps_list = load_events(run, add_sec=add_sec)
chunk_max = chunk_list[int(run)]

soi_df = create_df(steps_list, sub)
sub_comp_df = create_df(steps_list, sub_comp)

# calculate euclidean distance
output_df = calc_diff(list(range(chunk_list[int(run)])), chunk_list[int(run)], soi_df, sub_comp_df, metric)
plot_heatmap_eucl(sub, sub_comp, run, list(range(chunk_list[int(run)])), output_df, metric, output_dir)
pd.DataFrame(output_df).to_csv(output_dir + f"/sub-{sub}_sub-{sub_comp}_run-{run}_metric-{metric}.tsv", header=False, index=False)
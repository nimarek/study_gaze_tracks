import os
from eyegaze_utils import movie_dataset

sub_list = [22, 23, 24, 27, 28, 29, 30, 31, 33, 34, 36]

for sub in sub_list:
    print("working on sub %s" % (sub))
    base_path="/home/data/study_gaze_tracks/derivatives/fix_maps"
    fname_tmpl="/sub-%s_ses-movie_task-movie" % (sub)

    df = movie_dataset(sub)
    df.to_npz(base_path + fname_tmpl)
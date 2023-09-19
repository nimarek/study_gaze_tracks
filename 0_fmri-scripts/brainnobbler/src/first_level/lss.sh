#!/bin/bash

source /home/data/software/experimental/ipsy-env/activate

echo "calling python script with for sub-$1 run-$2"
python3 /home/data/study_gaze_tracks/code/0_fmri-scripts/brainnobbler/src/first_level/run_single_trials_glm.py $1 $2
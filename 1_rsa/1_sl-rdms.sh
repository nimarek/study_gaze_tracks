#!/bin/bash

source /home/data/software/spack-experimental_20230524/ipsy-env/activate

echo "calling python script for sub-$1 run-$2"
python3 /home/data/study_gaze_tracks/code/1_rsa/1_sl_rdms.py $1 $2
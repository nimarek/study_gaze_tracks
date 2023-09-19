#!/bin/bash

logs_dir=/home/data/study_gaze_tracks/code/logs_neural-rdm/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 1
request_memory = 6G
# Execution
initial_dir    = /home/data/study_gaze_tracks/code/1_rsa
executable     = 1_sl-rdms.sh
\n"

for sub in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do
    for run in {1..8}; do
        printf "arguments = ${sub} ${run}\n"
        printf "log       = ${logs_dir}/sub-${sub}_run-${run}\$(Cluster).\$(Process).log\n"
        printf "output    = ${logs_dir}/sub-${sub}_run-${run}\$(Cluster).\$(Process).out\n"
        printf "error     = ${logs_dir}/sub-${sub}_run-${run}\$(Cluster).\$(Process).err\n"
        printf "Queue\n\n"
    done
done
#!/bin/bash

source /home/data/software/experimental/ipsy-env/activate

echo "calling python script with for sub-$1"
python3 /home/data/NegativeCueing_RSA/NegCue_Random/code/brainnobbler/src/searchlight/run_searchlight.py $1
#!/bin/bash

export ANTSPATH=/home/nima/Downloads/install/bin/
export PATH=${ANTSPATH}:$PATH

reference_img=avg152T1_gray_prob50_bin_resampled.nii.gz

mkdir corr-maps_spatial-attention-onsets-2_smooth-6_space-mni/

for sub in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do 
for run in {1..8}; do
    echo "applying transform for sub-${sub} run-${run}"
    input_img=corr-maps_spatial-attention-onsets-2_smooth-6_space-t1w/sub-${sub}_run-${run}.nii.gz
    transform_file=sub-${sub}_ses-movie_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
    output_path=corr-maps_spatial-attention-onsets-2_smooth-6_space-mni/sub-${sub}_run-${run}_mni.nii.gz
    antsApplyTransforms -n GenericLabel[Linear] -i ${input_img} -r ${reference_img} -t ${transform_file} -o ${output_path} # GenericLabel[Linear]
done
done

echo "all done"
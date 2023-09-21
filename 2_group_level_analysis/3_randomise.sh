#!/bin/bash

sub_id=1
sigma=1 # 2.354 equals to FWHM of 5
threshold=0.95
permutations=5000
hypothesis="spatial-attention-onsets-2"
output_path="/home/nima/Downloads/norm_test/hypothesis-${hypothesis}"

echo creating folders ...
mkdir -p ${output_path}

echo apply smoothing kernel ...
for image in /home/nima/Downloads/norm_test/corr-maps_${hypothesis}_smooth-6_space-mni/*.nii.gz; do
    echo starting with image-${sub_id}
    # REMINDER: in fsl smoothing sigma is used instead of FWHM: FWHM = sigma*sqrt(8*ln(2)) = sigma*2.354
    fslmaths ${image} -s ${sigma} -mas /home/nima/Downloads/norm_test/avg152T1_gray_prob50_bin_resampled.nii.gz /home/nima/Downloads/norm_test/corr-maps_${hypothesis}_smooth-6_space-mni/sub-${sub_id}_smoothed.nii.gz
    let "sub_id++"
done

echo merging input files ...
fslmerge -t ${output_path}/output_complete `ls /home/nima/Downloads/norm_test/corr-maps_${hypothesis}_smooth-6_space-mni/*smoothed.nii.gz`

# echo calculating negative correlation values ...
# fslmaths ${output_path}/output_complete -mul -1 ${output_path}/output_complete 

echo starting permutation-test ...
randomise -i ${output_path}/output_complete -o ${output_path}/perm_hypothesis-${hypothesis} -1 -T -v 2 -n ${permutations} -m /home/nima/Downloads/norm_test/avg152T1_gray_prob50_bin_resampled.nii.gz

echo combining maps ...
fslmaths ${output_path}/perm_hypothesis-${hypothesis}_tfce_corrp_tstat1.nii.gz -thr ${threshold} -bin -mul ${output_path}/perm_hypothesis-${hypothesis}_tstat1.nii.gz ${output_path}/results-${hypothesis}_output_complete

echo extracting cluster information ...
cluster --in=${output_path}/results-${hypothesis}_output_complete --thresh=0.001 --oindex=${output_path}/results-${hypothesis}_output_complete_cluster_index --olmax=${output_path}/results-${hypothesis}_output_complete_lmax.txt --osize=${output_path}/results-${hypothesis}_output_complete_cluster_size

echo removing smoothed files and marged input file
# rm -r /home/nima/Downloads/norm_test/corr-maps_${hypothesis}_smooth-6_space-mni/*smoothed.nii.gz
rm ${output_path}/output_complete.nii.gz
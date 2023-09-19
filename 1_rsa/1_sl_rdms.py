import os
import re
import sys
import glob

import numpy as np
import nibabel as nib
from rsatoolbox.util.searchlight import (
    get_volume_searchlight,
    get_searchlight_RDMs,
)

sub, run = str(sys.argv[1]), str(sys.argv[2])

# analysis hyperparameters
radius, fwhm = 2, 6
duration = 4
distance = "euclidean"

print(f"working on sub-{sub} run-{run} with distance metric: {distance}...")
deriv_dir, scratch_dir = "/home/data/study_gaze_tracks/derivatives", "/home/data/study_gaze_tracks/scratch"
beta_dir = scratch_dir + f"/lss_spatial-attention-duration-{duration}_fwhm-{fwhm}_beta/run-{run}"

# set paths to store output
output_folder = scratch_dir + f"/neural-rdms_spatial-attention-duration-{duration}_space-t1w_fwhm-{fwhm}_beta"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# save output per run as .hdf5
filename = f"/sub-{sub}_run-{run}_task-studyforrest"

if os.path.isfile(output_folder + filename):
    raise Exception("file already exists, abort mission ...")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def set_paths(sub=None, run=None, beta_dir=None):
    """
    Function to create paths pointing to appropriate files for between- 
    or within-subject analysis. Defaults to between-subject analysis.
    Call function once for within and n*chunk times for between-subject
    analysis.
    """
    image_paths = glob.glob(beta_dir + f"/sub-{sub}_run-{run}_space-*_desc-chunk*.nii.gz")
    image_paths.sort(key=natural_keys)
    return image_paths
 
def create_dataset(image_paths):
    labels = []

    # load data with only one time point
    reference_img = nib.load(image_paths[0]).get_fdata()
    x, y, z = reference_img.shape

    # create dummy df according to coordinates
    data = np.zeros((len(image_paths), x, y, z))

    for x, im in enumerate(image_paths):
        print("contrast loaded:\t", im.split("/")[-1].split("_")[-1])
        labels.append(im.split("/")[-1].split("_")[-1])
        data[x] = nib.load(im).get_fdata()
    return data, labels

# load mask
mask_path = f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep/sub-{sub}/ses-movie/anat/sub-{sub}_ses-movie_label-GM_prob-0.2.nii.gz"
mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()
x, y, z = mask_data.shape

print(f"shape of the mask:\t {mask_data.shape}")

# create dataset
image_paths = set_paths(sub=sub, run=run, beta_dir=beta_dir)
data, labels = create_dataset(image_paths)

# reshape data so we have n_observastions x n_voxels
data_2d = data.reshape([data.shape[0], -1])
# data_2d[data_2d!=data_2d] = 0
data_2d = np.nan_to_num(data_2d)
print("shape of 2d data ...\t", data_2d.shape)

# mask based on all-zero patterns (https://github.com/rsagroup/rsatoolbox/issues/248)
mask_2d = ~np.all(data_2d==0, axis=0)
mask_3d = mask_2d.reshape(x, y, z)

centers, neighbors = get_volume_searchlight(mask_3d, radius=radius, threshold=0.5)
sl_rdms = get_searchlight_RDMs(data_2d, centers, neighbors, labels, method=distance)

# save output per run as .hdf5
filename = f"/sub-{sub}_run-{run}_task-studyforrest"
sl_rdms.save(output_folder + filename, file_type="hdf5", overwrite=True)
import os
import glob
import shutil

import pandas as pd
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def calc_mean(folder_path, chunk):
    """
    load first img as reference, crete np array to
    store rbg values. Calculate pixel intensities
    save it as img.

    adapted from: https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil
    """
    chunk_imgs = glob.glob(os.path.join(folder_path, "*.jpg"))
    width, hight = Image.open(chunk_imgs[0]).size
    N = len(chunk_imgs)
    arr = np.zeros((hight, width, 3), float)
    
    for im in chunk_imgs:
        imarr = np.array(Image.open(im), dtype=float)
        arr = arr + imarr / N
    arr = np.array(np.round(arr), dtype=np.uint8)
    out = Image.fromarray(arr,mode="RGB")
    out.save(os.path.join(folder_path, f"average-img_chunk-{chunk}.png"))
    return None

def arange_imgs(run=None):
    """
    create folders specific for each scene, copy
    corresponding jpgs into the folder.
    """
    reference_df = pd.read_csv(
        os.path.join(
            os.getcwd(), "reference_spatial-attention", f"complete_ses-movie_task-movie_run-{run}_events.tsv"),
        sep="\t")

    # subtract 1 from onsets, because of fmriprep
    reference_df["onset"] = reference_df["onset"] - 1

    for scene_onset, chunk in zip(reference_df["onset"], reference_df["trial_type"]):
        folder_path = os.path.join(
            os.getcwd(), "ssim_movie-frames", f"raw_run-{run}", chunk
        )
        os.makedirs(folder_path, exist_ok=True)

        # add 4
        img_identifier = list(range(int(scene_onset), int(scene_onset) + 4))
        for ident in img_identifier:
            src_path = os.path.join(
                os.getcwd(), "ssim_movie-frames", f"raw_run-{run}", f"studyf_run-{run}_onset-{ident}.0.jpg"
            )
            out_path = os.path.join(
                folder_path, f"studyf_run-{run}_onset-{ident}.0.jpg"
            )
            shutil.copyfile(src_path, out_path)
        
        # calculate average img
        calc_mean(folder_path, chunk)
    return None

for run in list(range(1, 9)):
    print(f"starting run-{run} ...")
    
    # sort images once
    arange_imgs(run=run)

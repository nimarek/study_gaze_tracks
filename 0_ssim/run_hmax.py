import os
import pickle
import shutil
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import hmax

# initialize the model with the universal patch set
model = hmax.HMAX(os.path.join(os.getcwd(), "universal_patch_set.mat"))
device = torch.device("cuda:0")
print("Running model on", device)

def arange_imgs(run=None):
    """
    create folders specific for each scene, copy
    corresponding jpgs into the folder.
    """
    reference_df = pd.read_csv(
        os.path.join(
            os.getcwd(), "reference_spatial-attention", f"ses-movie_task-movie_run-{run}_events.tsv"),
        sep="\t")

    # subtract 1 from onsets, because of fmriprep
    reference_df["onset"] = reference_df["onset"] - 1

    for scene_onset, chunk in zip(reference_df["onset"], reference_df["trial_type"]):
        folder_path = os.path.join(
            os.getcwd(), "spatial-attention_movie-frames", f"raw_run-{run}", chunk
        )
        os.makedirs(folder_path, exist_ok=True)

        img_identifier = list(range(int(scene_onset), int(scene_onset) + 4))
        for ident in img_identifier:
            src_path = os.path.join(
                os.getcwd(), "spatial-attention_movie-frames", f"raw_run-{run}", f"studyf_run-{run}_onset-{ident}.0.jpg"
            )
            out_path = os.path.join(
                folder_path, f"studyf_run-{run}_onset-{ident}.0.jpg"
            )
            shutil.copyfile(src_path, out_path)
    return None

def load_imgs(run=None):
    """
    function to load all imgs for a corresponding 4sec 
    scene. Since: 25 fps, should be 100 imgs in total.
    """
    scene_imgs = datasets.ImageFolder(
        os.path.join(os.getcwd(), "spatial-attention_movie-frames", f"raw_run-{run}"),
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])
    )
    return DataLoader(scene_imgs, batch_size=10)

for run in list(range(1, 9)):
    print(f"starting run-{run} ...")
    
    # sort images once
    # arange_imgs(run=run)

    # initialize dataset
    dataset = load_imgs(run=run)

    model = model.to(device)
    for X, y in dataset:
        s1, c1, s2, c2 = model.get_all_layers(X.to(device))

    print(f"saving output of all layers to: run-{run}_activations.pkl")
    with open(f"run-{run}_activations.pkl", "wb") as f:
        pickle.dump(dict(s1=s1, c1=c1, s2=s2, c2=c2), f)
    print("[done]")
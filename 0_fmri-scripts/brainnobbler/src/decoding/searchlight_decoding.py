#!/usr/bin/env python3
"""
TODO: save also permutations schemes 
"""
import argparse
from os.path import join
import nibabel as nib
from sklearn.model_selection import LeaveOneGroupOut
import nilearn.decoding
import numpy as np
from dataset import GetData
from permute_labels import PermLabels

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--data_dir", metavar="PATH", required=True)
parser.add_argument("-o", "--output_dir", type=str, required=False)
parser.add_argument("-t", "--tasks", type=str, nargs="+", required=True)
parser.add_argument("-m", "--mask", type=str, required=True)
parser.add_argument("-r", "--radius", type=float, required=True)
parser.add_argument("-s", "--subj", type=str, required=True)
parser.add_argument("-p", "--permutation", type=int, required=False)
parser.add_argument(
    "-e", "--estimator", const="svc", nargs="?", type=str, default="svc"
)
args = parser.parse_args()


def np2nii(img, scores, filename):
    """
    It saves data into a nifti file
    by leveraging on another image
    of the same size.
    Parameters
    ----------
    img : nifti file (e.g a mask)
    scores : numpy array containing decoding scores
    filename : string name of the new nifti file
    Returns
    -------
    nii_file : Nifti1Image
    """
    img = nib.load(img)
    header = nib.Nifti1Header()
    affine = img.affine
    nii_file = nib.Nifti1Image(scores, affine, header)
    nib.save(nii_file, filename)
    return nii_file


# get the dataset
dataset = GetData(
    tasks=args.tasks, labels={"_f_": 1, "_nf_": 2}, group_filter=args.tasks
)
data = dataset(args.data_dir)
# instantiate the cross validation
cv = LeaveOneGroupOut()
if args.permutation:
    print("you are now running the searchlight for the chance maps")
    cv = list(cv.split(data["data"], data["labels"], data["groups"]))
    # permute labels
    permutation = PermLabels(data["labels"], cv, data["groups"])
    for fold, ((train, test), perm) in enumerate(zip(cv, permutation())):
        print(f"fold number: {fold}")
        print(f"train indices: {train} - test indices: {test}")
        print(
            (
                f"non-permuted train labels: {data['labels'][train]}"
                f" permuted train labels: {perm}"
            )
        )

        labels = np.copy(data["labels"])
        print(data["labels"])
        labels[train] = perm
        print(labels)

        SL = nilearn.decoding.SearchLight(
            mask_img=args.mask,
            # process_mask_img=process_mask_img,
            radius=args.radius,
            estimator=args.estimator,
            n_jobs=1,
            verbose=1,
            cv=[(train, test)],
            scoring="accuracy",
        )
        SL.fit(data["data"], labels, data["groups"])
        scores = SL.scores_
        output = join(
            f"{args.output_dir}",
            (
                f"sub-{args.subj}_{args.tasks[0]}_{args.tasks[1]}_"
                f"radius_{int(args.radius)}_perm_"
                f"{str(args.permutation)}_fold_{fold+1}"
            ),
        )
        np.save(f"{output}.npy", scores)
else:
    SL = nilearn.decoding.SearchLight(
        mask_img=args.mask,
        # process_mask_img=process_mask_img,
        radius=args.radius,
        estimator=args.estimator,
        n_jobs=1,
        verbose=1,
        cv=cv,
        scoring="accuracy",
    )
    SL.fit(data["data"], data["labels"], data["groups"])
    scores = SL.scores_
    output = join(
        f"{args.output_dir}",
        (
            f"sub-{args.subj}_{args.tasks[0]}_"
            f"{args.tasks[1]}_radius_{int(args.radius)}"
        ),
    )
    print(scores[scores > 0])
    np.save(f"{output}.npy", scores)
    np2nii(args.mask, scores, f"{output}.nii.gz")

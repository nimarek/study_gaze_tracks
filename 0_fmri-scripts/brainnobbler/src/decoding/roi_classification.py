#!/usr/bin/env python3

"""
TODO: add an output directory as input to save data
save also permutations schemes 
"""
import argparse
import numpy as np
import pandas as pd
from dataset import GetData
from nilearn.maskers import NiftiMasker
from permute_labels import PermLabels
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--data_dir", metavar="PATH", required=True)
parser.add_argument("-t", "--tasks", type=str, nargs="+", required=True)
parser.add_argument("-m", "--mask", type=str, required=True)
parser.add_argument("-p", "--permutation", type=int, required=False)
parser.add_argument("-e", "--estimator", type=str, required=False)
parser.add_argument("-o", "--output_dir", type=str, required=False)
parser.add_argument("-s", "--scaler", required=False)
args = parser.parse_args()
# instantiate the dataset
dataset = GetData(
    tasks=args.tasks, labels={"_f_": 1, "_nf_": 2}, group_filter=args.tasks
)
data = dataset(args.data_dir)
# prepare bold data
masker = NiftiMasker(mask_img=args.mask, memory="nilearn_cache", memory_level=1)
fmri_masked = masker.fit_transform(data["data"])
# print(fmri_masked)
# cross validation
cv = LeaveOneGroupOut()
# classifier
# clf = SVC()
clf = LinearSVC(penalty="l1", dual=False, max_iter=1e4)

if args.permutation:
    # collect data in a dictionary
    clf_perm_scores = {}
    # split folds
    cv = list(cv.split(data["data"], data["labels"], data["groups"]))
    # get permutations
    permutation = PermLabels(data["labels"], cv, data["groups"])
    for fold, ((train, test), perm) in enumerate(zip(cv, permutation())):
        print(f"fold number: {fold}")
        print(f"train indices: {train} - test indices: {test}")
        print(
            f"non-permuted train labels: {data['labels'][train]} permuted train labels: {perm}"
        )
        clf.fit(fmri_masked[train], perm)
        prediction = clf.predict(fmri_masked[test])
        clf_perm_scores[f"fold_{fold + 1}"] = accuracy_score(
            prediction, data["labels"][test]
        )
    print(clf_perm_scores)
else:
    cv_scores = cross_val_score(
        clf,
        X=fmri_masked,
        y=data["labels"],
        cv=cv,
        scoring="accuracy",
        groups=data["groups"],
        n_jobs=-1,
    )
    print(cv_scores, f"average {np.mean(cv_scores)}")

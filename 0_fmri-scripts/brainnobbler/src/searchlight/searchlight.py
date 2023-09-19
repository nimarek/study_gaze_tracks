#!/usr/bin/env python3

"""
"""
from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, RepeatedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

ESTIMATOR = {
    "svc": SVC(kernel="linear"),
    "lsvc": LinearSVC(penalty="l1", dual=False, max_iter=1e4),
    "lda": LinearDiscriminantAnalysis(solver="svd", store_covariance=True),
}

CV = {
    "loo": LeaveOneGroupOut(),
    "rkfold": RepeatedKFold(n_splits=10),
}

pipeline = Pipeline([("scaler", StandardScaler()), ("estimator", LinearSVC(penalty="l1", dual=False, max_iter=1e4))])

class Searchlight:
    """
    Basic searchlight:
    provides spheres (and sphere indices)
    to be analyzed separately or
    it runs a specific estimator (e.g. a classifier)
    on each sphere and returns the chosen metric.
    """

    def __init__(self, radius, mask, estimator=None, cv=None):

        self.radius = radius
        self.mask = nib.load(mask).get_fdata()
        self.estimator = estimator
        self.cv = cv

    def sphere(self, center):
        """
        Computes sphere indices.
        Note: This spere implementation
        follows pymvpa method.

        Parameters
        ----------

        center : list/tuple of x,y,z coordinates
        mask_coord : 3D numpy array, mask array

        Returns
        -------

        sphere_idx : numpy array, single sphere indices
        sphere : numpy array of a boolean mask for a sphere

        """
        mask_coord = np.indices(self.mask.shape)
        bool_sphere = (
            np.sqrt(
                (mask_coord[0] - center[0]) ** 2
                + (mask_coord[1] - center[1]) ** 2
                + (mask_coord[2] - center[2]) ** 2
            )
            <= self.radius
        )
        sphere_idx = np.asarray(np.where(bool_sphere == True))
        return bool_sphere, sphere_idx

    def _get_data(self, data, bool_sphere):
        """ """
        return np.stack([chunk[bool_sphere] for chunk in data])

    def _set_model(self, dataset, center):
        """ """
        bool_sphere, _ = self.sphere(center)
        cv_scores = cross_val_score(
            pipeline,
            #ESTIMATOR[self.estimator],
            X=self._get_data(dataset["data"], bool_sphere),
            y=dataset["labels"],
            cv=CV[self.cv],
            scoring="accuracy",
            groups=dataset["groups"],
            n_jobs=-1,
        )
        return np.mean(cv_scores)

    def __call__(self, dataset):
        """ """
        # select only masked centers
        masked_centers = np.asarray(np.where(self.mask == 1.0)).T
        print("estimate model")
        dataset["data"] = [nib.load(data).get_fdata() for data in dataset["data"]]
        scores = Parallel(n_jobs=-1, verbose=0)(
            delayed(self._set_model)(dataset, c) for c in masked_centers
        )
        results = self.mask.copy()
        results[results == 1.0] = scores
        return results

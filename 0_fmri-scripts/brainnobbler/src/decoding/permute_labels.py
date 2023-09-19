"""
This class permutes labels 
for following permutation tests
"""
import numpy as np


class PermLabels:
    """
    Permutes labels and checks
    whether the permuted labels
    are accetable.
    """

    def __init__(self, labels, cv, groups=None, random_state=None, thresh=0.8):
        self.labels = labels
        self.cv = cv
        self.groups = groups
        self.random_state = random_state
        self.thresh = thresh

    def _check_permutation(self, orig_labels, perm_labels):
        """ """
        ratio = np.sum(orig_labels != perm_labels) / len(orig_labels)
        return ratio > self.thresh

    def permute(self, labels):
        """ """
        r_state = np.random.RandomState(self.random_state)
        rand = r_state.permutation(len(labels))
        perm = labels[rand]
        while not self._check_permutation(labels, perm):
            rand = r_state.permutation(len(labels))
            perm = labels[rand]
        return perm

    def __call__(self):
        """
        Permute labels taking into account
        group subdivisions if necessary
        """
        perm_labels = []
        for train, _ in self.cv:
            if self.groups is not None:
                group = self.groups[train]
                for group_id in np.unique(group):
                    grp_mask = group == group_id
                    sel_labels = self.labels[train[grp_mask]]
                    perm_labels.append(list(self.permute(sel_labels)))
            else:
                sel_labels = self.labels[train]
                perm_labels.append(self.permute(sel_labels))
        return perm_labels

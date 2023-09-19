from glob import glob
from os.path import join
import numpy as np


class GetData:
    """ """

    def __init__(self, tasks, labels, group_filter=None):
        self.tasks = tasks
        self.labels = labels
        self.group_filter = group_filter

    def _get_bold(self, data_dir):
        """ """
        files = []
        for task in self.tasks:
            files.extend(glob(join(data_dir, f"*{task}*.nii.gz")))
        return np.array(files)

    def _get_labels(self, files):
        """ """

        return np.array(
            [
                lab_value
                for img in files
                for label, lab_value in self.labels.items()
                if label in img
            ]
        )

    def _get_groups(self, files):
        """ """
        return np.array(
            [
                n + 1
                for img in files
                for n, filt in enumerate(self.group_filter)
                if filt in img
            ]
        )

    def __call__(self, data_dir):
        """ """
        files = self._get_bold(data_dir)
        labels = self._get_labels(files)
        groups = self._get_groups(files)
        return {"data": files, "labels": labels, "groups": groups}

import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from scipy.ndimage import label, sum_labels
from statsmodels.stats.multitest import multipletests


class ClusterPerm:

    """ """

    def __init__(self, data, chance_maps, n_bootstrap=100000, brain_shape=None):
        """
        chance_maps : dict containing subjects as keys
                      and a list of numpy
                      arrays (accuracies x 1) as values
        """
        # if not isinstance(data, [".nii", ".nii.gz"]):
        #    raise TypeError("data must be .nii or .nii.gz type of files")
        # self.data = np.mean([nib.load(img) for img in data]) # this needs a change
        self.data = data
        # if brain_shape is not None:
        #    self.brain_shape = nib.load(data[0]).shape
        self.brain_shape = brain_shape
        self.chance_maps = chance_maps
        self.n_bootstrap = n_bootstrap

    def _bootstrap_maps(self):
        """
        Steps:
        1. For each subject select a chance map
           average them
        2. Repeat step 1 10^5 times

        3. take the average of each bootstrapped group
        """
        for _ in range(self.n_bootstrap):
            yield np.mean(
                [
                    maps[int(np.random.choice(len(maps), 1))]
                    for _, maps in self.chance_maps.items()
                ],
                axis=0,
            )

    def _get_voxels(self, n_vox):
        """ """
        thresh_val = self.n_bootstrap * (len(self.chance_maps[0]) / self.n_bootstrap)
        return np.sort([float(boot[:, n_vox]) for boot in self._bootstrap_maps()])[
            -int(thresh_val)
        ]

    def _get_threshold(self):
        """ """
        # sort chance maps and select the
        # voxels that correspond to the 100/100000
        # position (0.001 tail)

        return np.hstack(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(self._get_voxels)(n_vox)
                for n_vox in range(np.prod(self.brain_shape))
            )
        )

    def _to_3D_shape(self, map):
        """
        It reshapes the chance maps to
        the original ND shape, by using
        the shape of the data or a shape
        provided by the user.
        """
        return map.reshape(self.brain_shape)

    def _get_clusters(self, thresh_map):
        """
        1. make map 3D
        2. calls scipy label to find the number of voxels
           which share a face with an other voxel
           (6-connections by default for 3D arrays,
           no need to input the structure)
        3. get cluster sizes for each map
        """
        thresh_map_3d = thresh_map  # self._to_3D_shape(thresh_map)
        clusters, n_features = label(thresh_map_3d)
        clust_size = sum_labels(
            thresh_map_3d,
            clusters,
            np.arange(1, n_features + 1),
        )
        return clust_size, clusters

    def _get_pvalues(self, clusters):
        """
        get probabilities for each cluster size
        """
        return {
            int(cl): clusters.count(cl) / len(clusters) for cl in np.unique(clusters)
        }

    def __str__(self):
        """ """

    def __call__(self):
        # make data into a contiguous flatten array
        data = np.ravel(self.data)  # here check flattening
        # get the threshold
        threshold = self._get_threshold()
        print(threshold)
        # threshold the data (binarize the data, data are turned into booleans)
        # and get the clusters (zip(*) to unpack the outputs)
        dist_clusters, _ = zip(
            *Parallel(n_jobs=-1, verbose=0)(
                delayed(self._get_clusters)(map > threshold)
                for map in self._bootstrap_maps()
            )
        )
        # now threshold real data and get clusters
        thresh_data = data > threshold
        data_clusters, clusters = self._get_clusters(thresh_data)
        # put together null-dist clusters and data clusters
        tot_clusters = np.hstack([np.hstack(dist_clusters), data_clusters]).tolist()
        clust_pvals = self._get_pvalues(tot_clusters)
        # get index relative to the data clusters and the cluster ID
        # needed later to get the significant clusters
        clust_idx = {
            clust: (list(clust_pvals.keys()).index(clust), clust_id)
            for clust, clust_id in zip(
                data_clusters,
                np.unique(clusters)[np.unique(clusters) > 0],  # WARNING CAMBIA!!!!!
            )
        }
        # now correct for multiple comparisons with fdr
        pvals = list(clust_pvals.values())
        reject, pvals_corrected, _, _ = multipletests(
            pvals, alpha=0.05, method="fdr_bh"
        )
        stats = {
            clust: [reject[index[0]], pvals_corrected[index[0]]]
            for clust, index in clust_idx.items()
        }
        return clust_pvals, reject, pvals_corrected, stats

import matplotlib.pyplot as plt
from nilearn.plotting import plot_design_matrix
import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd
from scipy.linalg import block_diag


class SecLevDesignMat:
    """
    Creates a simple second level design matrix
    """

    def __init__(self, n_subs, conds):
        """ """
        self.n_subs = n_subs
        self.conds = conds.copy()
        self.dm = self.mk_design_mat()

    def mk_design_mat(self):
        """
        it returns a design matrix as dataframe
        first conditions and then subjects
        """
        mat = block_diag(*(np.ones([self.n_subs, 1]) for n, _ in enumerate(self.conds)))
        mat = np.hstack([mat, np.tile(np.eye(self.n_subs), [len(self.conds), 1])])
        print(mat)
        self.conds.extend(list(range(1, self.n_subs + 1)))
        return pd.DataFrame(mat, columns=self.conds)

    def check_matrix_rank(self):
        """
        Checks whether the design matrix is full rank.
        If the matrix is rank deficient returns a "warning",
        otherwise it prints "matrix is full rank".
        """
        if isinstance(self.dm, pd.DataFrame):
            mat = self.dm.to_numpy()
        if matrix_rank(mat) != mat.shape[1]:
            print(
                "matrix is not full rank,\n\
            make it as full rank to avoid erratic results"
            )
        else:
            print("matrix is full rank")
        return self.dm

    def mk_full_rank(self, col):
        """
        simple method to make a matrix full
        rank by dropping redundant column.
        The dropped column becomes the baseline.

        Parameters
        ----------
        col : name (string) of pandas dataframe column

        Returns
        -------
        full rank design matrix as pandas dataframe
        """
        return self.dm.drop(columns=col)

    def plot_dm(self, dm=None):
        """
        Plots the design matrix.
        """
        if dm is not None:
            self.dm = dm
        plot_design_matrix(self.dm, rescale=True, ax=None, output_file=None)
        plt.show()

"""
"""
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class GroupPermTest:
    """
    It performs a one sample permutation test
    for classification measures.
    """

    def __init__(self, scores, perm_scores, alpha=0.01):
        """ """
        self.scores = np.mean(scores, axis=0)
        self.perm_scores = np.mean(perm_scores, axis=0)
        self.alpha = alpha
        if (1 / self.perm_scores.shape[0]) > alpha:
            warnings.warn(
                f"alpha threshold allowed given your \
            n. of permutations is {1 / (self.perm_scores.shape[0] + 1)}"
            )
            self.alpha = 1 / (self.perm_scores.shape[0] + 1)

    def __str__(self):
        """
        Handy method to directly print the results
        """
        return f"p value = {self.perm_test()}"

    def perm_test(self):
        """
        p value is computed according to the method suggested in:
        https://arxiv.org/pdf/1603.05766.pdf
        """
        return (np.sum(self.perm_scores >= self.scores) + 1) / (
            self.perm_scores.shape[0] + 1
        )

    def plot_dist(self):
        """
        plots null distribution and results
        - the red line indicates the alpha threshold possible
          or chosen by the user.
        - the blue line indicate the resulting p value
        """
        dist = np.hstack([self.scores, self.perm_scores])
        ax = sns.histplot(data=dist, kde=True, element="step")
        plt.xlabel("Accuracy", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        ymin, ymax = ax.get_ylim()
        alpha = np.sort(self.perm_scores)[
            int(self.perm_scores.shape[0] * (1 - self.alpha))
        ]
        plt.vlines(alpha, ymin, ymax, colors="red", ls="--", lw=2)
        plt.vlines(self.scores, ymin, ymax, colors="blue", ls="--", lw=2)
        plt.text(alpha, ymax - 0.3, f"alpha = {self.alpha}", fontsize=9)
        plt.show()


score = np.random.random([24, 1]) + 0.3
perm = np.random.random([24, 1000])
data = np.hstack([np.mean(score), np.mean(perm)])
stat = GroupPermTest(score, perm)
print(stat)
stat.plot_dist()

"""
first level analysis
"""
from glob import glob
from os.path import join
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import (
    make_first_level_design_matrix,
    FirstLevelModel,
)
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn.image import index_img
import numpy as np


class FirstLevel:
    """
    Wrapper for first level analysis
    """

    def __init__(
        self,
        sub,
        run, 
        func_dir,
        events_dir,
        tr,
        reg_names=None,
        n_dummy=None,
        hrf="spm + derivative",
        drift_model="polynomial",
        drift_order=5,
        high_pass=0.01,
        fir_delays=[0],
        n_scans=None,
    ):
        self.sub = sub
        self.run = run
        self.func_dir = func_dir
        self.events_dir = events_dir
        self.hrf = hrf
        self.tr = tr
        self.reg_names = reg_names
        self.n_dummy = n_dummy
        self.drift_model = drift_model
        self.drift_order = drift_order
        self.high_pass = high_pass
        self.fir_delays = fir_delays
        self.n_scans = n_scans
        self.events = self._get_events()
        self.confounds = self._get_confounds()
        self.func = sorted(glob(join(self.func_dir, f"*run-{self.run}*_bold.nii.gz")))
        self.mask = self._get_mask()

    def _get_mask(self):
        """
        Load all the masks present in the func directory
        """
        return f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep/sub-{self.sub}/ses-movie/anat/sub-{self.sub}_ses-movie_label-GM_prob-0.2.nii.gz" # self.mask[0], # here using only one mask


    def _get_events(self):
        """
        Load events according to BIDS nomenclature
        """
        rel_regr = ["onset", "duration", "trial_type"]
        return [
            pd.read_csv(ev, sep="\t")[rel_regr]
            for ev in sorted(glob(join(self.events_dir, f"*run-{self.run}*_events.tsv")))
        ]

    def _get_confounds(self):
        """
        Load fmriprep confounds, get rid of
        counfounds relative to the dummy scans
        """
        fmriprep_conf = sorted(glob(join(self.func_dir, f"*run-{self.run}*_regressors.tsv")))
        return [
            pd.read_csv(conf, sep="\t")[self.reg_names]
            .loc[self.n_dummy :]
            .reset_index(drop=True)
            for conf in fmriprep_conf
        ]

    def mk_design_mat(self, events, confounds):
        """
        Computes design matrix per run
        """
        frame_times = np.arange(self.n_scans - self.n_dummy) * self.tr
        return make_first_level_design_matrix(
            frame_times,
            events,
            hrf_model=self.hrf,
            add_regs=confounds,
            add_reg_names=self.reg_names,
            drift_order=self.drift_order,
            drift_model=self.drift_model,
            high_pass=self.high_pass,
            fir_delays=self.fir_delays,
        )

    def _remove_dummies(self, img):
        """
        This method removes superfluous dummy scans
        and checks whether the number of scans
        is what was expected.
        """
        bold_img = index_img(img, slice(self.n_dummy, self.n_scans))
        if bold_img.shape[-1] != (self.n_scans - self.n_dummy):
            raise ValueError(
                f"Original scans are {nib.load(img).shape[-1]},"
                f"there are {bold_img.shape[-1]} scans after dummy removal,"
                f"but according to your input you should have {(self.n_scans - self.n_dummy)}"
            )
        else:
            return bold_img

    def _fixed_effects(self, stats, output):
        """
        Put together the fixed effects runs.
        Parameters
        ----------
        stats : dictionary containing stats from the
                contrasts.
        Returns
        -------
        glm_stats : dictionary containing one Niimage
                    per contrast
        """
        glm_stats = {}
        for key, _ in stats.items():
            contrast_imgs = [i["effect_size"] for i in stats[key]]
            variance_imgs = [i["effect_variance"] for i in stats[key]]
            fx_contr, fx_var, fx_stat = compute_fixed_effects(
                contrast_imgs, variance_imgs
            )
            glm_stats[key] = fx_stat
            if output is not None:
                fx_stat.to_filename(f"{output}_{key}.nii.gz")
        return glm_stats

    def __call__(self, contrasts, noise_model="ar1", output=None):
        """
        Fit the first level over all the runs and saves data
        """
        fl_model = FirstLevelModel(
            self.tr, hrf_model=self.hrf, noise_model=noise_model, mask_img=self.mask[0], minimize_memory=False
        )  
        stats = {key: [] for key in contrasts.keys()}
        # loop over the runs
        for run, (func_img, ev, conf) in enumerate(
            zip(self.func, self.events, self.confounds)
        ):
            # this is required when contrasts are
            # different among runs
            if isinstance(contrasts, list):
                contrasts = contrasts[run]
            bold_img = self._remove_dummies(func_img)
            design_mat = self.mk_design_mat(ev, conf)
            glm = fl_model.fit(bold_img, design_matrices=design_mat)
            for contrast_id, contrast_value in contrasts.items():
                contr_stats = glm.compute_contrast(contrast_value, output_type="t")
                stats[contrast_id].append(contr_stats)
        return self._fixed_effects(stats, output)


def mk_contrasts(design_matrix):
    """
    Basic contrast dictionary.
    use this one to create more complex
    contrasts.
    """
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {
        column: contrast_matrix[i] for i, column in enumerate(design_matrix.columns)
    }
    return contrasts

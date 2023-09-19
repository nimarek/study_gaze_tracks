"""
Single trial GLM for MVPA analysis
using the lss approach.
"""
from glob import glob
from os.path import join
from utility import natural_keys
from first_level import FirstLevel
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.glm.first_level import (
    make_first_level_design_matrix,
    FirstLevelModel,
)
from nilearn.image import index_img
from nilearn.plotting import plot_design_matrix
import numpy as np
import pandas as pd


class SingleTrialFl(FirstLevel):
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
        drift_order=3,
        high_pass=0.01,
        fir_delays=[0],
        n_scans=None,
    ):
        super().__init__(
            sub,
            run,
            func_dir,
            events_dir,
            tr,
            reg_names,
            n_dummy,
            hrf,
            drift_model,
            drift_model,
            high_pass,
            fir_delays,
            n_scans,
        )
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
        self.func = sorted(glob(join(self.func_dir, f"*run-{self.run}*_bold.nii.gz")))
        self.events = self._get_events()
        self.confounds = self._get_confounds()
        self.mask = self._get_mask()

    def lss_design_matrix(self, events, confounds):
        """
        Computes design matrix for one run.
        Generator that yield a design matrix
        and the trial type pameters
        --------
        events : pandas DataFrame containing events
                 (BIDS compliant)
        confounds : pandas DataFrame of fMRIprep confounds

        Returns
        -------
        design_matrix : pandas DataFrame
        trial_type : string name given to the trial
        """
        frame_times = np.arange(self.n_scans - self.n_dummy) * self.tr
        for n, trial_type in enumerate(events.trial_type):
            ev_copy = events.copy()
            other_trials = ev_copy.index != n
            bool_mask = ev_copy.index[other_trials]
            ev_copy.loc[bool_mask, "trial_type"] = "other_trial_types"
            # since we have only unique runs, we don't need to count trial_type + n
            ev_copy.loc[n, "trial_type"] = f"{trial_type}" 
            ev_copy.reset_index(drop=True)
            trial_type = f"{trial_type}" 
            design_matrix = make_first_level_design_matrix(
                frame_times,
                ev_copy,
                hrf_model=self.hrf,
                add_regs=confounds,
                add_reg_names=self.reg_names,
                drift_order=self.drift_order,
                drift_model=self.drift_model,
                high_pass=self.high_pass,
                fir_delays=self.fir_delays,
            )
            yield design_matrix, trial_type

    def plot_dm(self, design_matrix):
        """
        Plots the design matrix.
        """
        plot_design_matrix(design_matrix, rescale=True, ax=None, output_file=None)
        plt.show()

    def __call__(self, noise_model="ar1", output=None):
        """
        This (honestly ugly) method runs the whole trial fitting
        over all the runs.
        """
        mask_list = ["VIS", "rFFA", "lFFA", "rLOC", "lLOC", "rEBA", "lEBA", "rPPA", "lPPA", "rOFA", "lOFA"]

        for run, (img, ev, conf) in enumerate(zip(self.func, self.events, self.confounds)):
            
            bold_img = self._remove_dummies(img)

            for dm, trial_type in self.lss_design_matrix(ev, conf):
                print(f"fitting run {self.run} trial and {trial_type} ...")
                first_level_model = FirstLevelModel(
                    self.tr,
                    hrf_model=self.hrf,
                    noise_model=noise_model,
                    smoothing_fwhm=6,
                    mask_img=f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep/sub-{self.sub}/ses-movie/anat/sub-{self.sub}_ses-movie_label-GM_prob-0.2.nii.gz" # self.mask[0], # here using only one mask
                )
                glm = first_level_model.fit(bold_img, design_matrices=dm)
                betas = glm.compute_contrast(trial_type, output_type="effect_size") # z_score
                betas.to_filename(output + f"/sub-{self.sub}_run-{self.run}_space-t1w_desc-{trial_type}.nii.gz")

                # for roi in mask_list:
                #     try:
                #         print(f"fitting run {self.run} trial {trial_type} and mask {roi}...")
                #         first_level_model = FirstLevelModel(
                #             self.tr,
                #             hrf_model=self.hrf,
                #             noise_model=noise_model,
                #             mask_img= f"/home/data/study_gaze_tracks/studyforrest-data-visualrois/sub-{self.sub}/sub-{self.sub}_mask-{roi}.nii"# f"/home/data/study_gaze_tracks/studyforrest-data-phase2/derivatives/fmriprep_t1w/sub-{self.sub}/ses-movie/anat/sub-{self.sub}_ses-movie_label-GM_prob-0.2.nii.gz" # self.mask[0], # here using only one mask
                #         )
                #         glm = first_level_model.fit(bold_img, design_matrices=dm)
                #         betas = glm.compute_contrast(trial_type, output_type="z_score")
                #         betas.to_filename(output + f"/sub-{self.sub}_run-{self.run}_space-{roi}_desc-{trial_type}.nii.gz")
                #         # resid = masker.fit_transform(glm.residuals[0])
                #     except:
                #         pass


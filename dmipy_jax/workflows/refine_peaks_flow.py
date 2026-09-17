"""
DIPY workflow: refine any peaks (CSD, MSMT-CSD, FORCE, …) with the PRISM-JAX
differentiable fit and return calibrated fixels (doc 008 §9.4).

    dipy_refine_peaks dwi.nii.gz dwi.bval dwi.bvec mask.nii.gz peaks.pam5 --out_dir derivatives/sbi4dwi/sub-01

Outputs (BIDS-derivative style, `<prefix>` = `--prefix`, default the DWI stem):
  <prefix>_desc-refined_peaks.pam5           refined PeaksAndMetrics (DIPY tracking)
  <prefix>_desc-refined_fintra.nii.gz        intra-axonal fraction (WM-internal, f_i)
  <prefix>_desc-refined_fwm.nii.gz           WM fraction Σ f_wm (so f_i·f_wm is the voxel intra fraction)
  <prefix>_desc-refined_sigmatheta.nii.gz    Laplace σ_θ of the main fixel (deg)
  <prefix>_desc-refined_fixels/              MRtrix fixel directory: index, directions, fraction, sigma_theta_deg, f_intra
  <prefix>_desc-refined_model.json           configuration + learned global diffusivities
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from dipy.workflows.workflow import Workflow


class RefinePeaksFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "refine_peaks"

    def run(self, input_files, bvalues_files, bvectors_files, mask_files, pam_files, n_fibres=3, n_iter=300,
            d_par_init=1.7, wm_only=False, peak_frac_min=0.1, no_laplace=False, prefix="", out_dir=""):
        """Refine DIPY peaks with the PRISM-JAX plus-x fit and Laplace posterior.

        Parameters
        ----------
        input_files : string
            DWI NIfTI.
        bvalues_files : string
        bvectors_files : string
        mask_files : string
        pam_files : string
            PeaksAndMetrics (.pam5) from any DIPY reconstruction.
        n_fibres : int, optional
        n_iter : int, optional
        d_par_init : float, optional
            Initial axial diffusivity in µm²/ms (learned; 1.7 in vivo, 0.6 ex vivo / DiSCo).
        wm_only : bool, optional
            Disable CSF/GM compartments (WM-only tissue or phantoms).
        peak_frac_min : float, optional
        no_laplace : bool, optional
            Skip the Laplace posterior (and the fixel directory / σ_θ map).
        prefix : string, optional
        out_dir : string, optional
        """
        from dipy.core.gradients import gradient_table
        from dipy.io.gradients import read_bvals_bvecs
        from dipy.io.image import load_nifti, save_nifti
        from dipy.io.peaks import load_pam, save_pam
        from dmipy_jax.validation import dipy_refine as dr
        from dmipy_jax.io.fixel import write_fixel_directory, fixels_from_posterior
        io_it = self.get_io_iterator()
        for dwi_f, bval_f, bvec_f, mask_f, pam_f in io_it:
            data, affine, img = load_nifti(dwi_f, return_img=True)
            bvals, bvecs = read_bvals_bvecs(bval_f, bvec_f); gtab = gradient_table(bvals, bvecs=bvecs)
            mask = load_nifti(mask_f)[0].astype(bool); pam = load_pam(pam_f)
            cfg = dr.default_config(n_fibres, n_iter=n_iter, d_par_init=d_par_init * 1e-9, wm_only=wm_only)
            pam_ref, post, fit = dr.refine_peaks(data.astype(np.float32), gtab, mask, pam, n_fibres=n_fibres, cfg=cfg,
                                                 peak_frac_min=peak_frac_min, affine=affine, laplace=not no_laplace)
            pre = prefix or Path(dwi_f).name.split(".")[0]
            out = Path(out_dir or "."); out.mkdir(parents=True, exist_ok=True); base = out / f"{pre}_desc-refined"
            save_pam(f"{base}_peaks.pam5", pam_ref)
            vol = np.zeros(mask.shape, np.float32); vol[mask] = fit.fintra; save_nifti(f"{base}_fintra.nii.gz", vol, affine)
            vol = np.zeros(mask.shape, np.float32); vol[mask] = fit.wm_fracs.sum(1); save_nifti(f"{base}_fwm.nii.gz", vol, affine)
            if post is not None:
                vol = np.zeros(mask.shape, np.float32); vol[mask] = post.sigma_deg[:, 0]; save_nifti(f"{base}_sigmatheta.nii.gz", vol, affine)
                dirs, vals = fixels_from_posterior(post, fit)
                vox = tuple(float(v) for v in img.header.get_zooms()[:3])
                write_fixel_directory(f"{base}_fixels", mask, dirs, vals, vox=vox, transform=affine, frac_min=peak_frac_min)
            json.dump({"n_fibres": n_fibres, "n_iter": n_iter, "wm_only": wm_only, "d_par_um2_ms": float(fit.d_par * 1e9),
                       "d_par_extra_um2_ms": float((fit.d_par_extra or fit.d_par) * 1e9), "sigma": None if fit.sigma is None else float(fit.sigma),
                       "source_pam": str(pam_f)}, open(f"{base}_model.json", "w"), indent=1)
            print(f"wrote {base}_* (D∥ = {fit.d_par*1e9:.2f} µm²/ms, {int(mask.sum()):,} voxels)")


def main():
    from dipy.workflows.flow_runner import run_flow
    run_flow(RefinePeaksFlow())


if __name__ == "__main__":
    main()

"""
MSMT-CSD baseline on DiSCo via dipy, on the shared tracker (doc 006 §4).

PRISM reports its DiSCo result *relative to* MSMT-CSD with oracle
response functions (r=.934 vs .920 at best tracking angle). Absolute r
is not comparable across papers (the connectivity normalisation is not
specified; see doc 004 §18–§19), so the fair target for "exceeding
PRISM" is our margin over MSMT-CSD on our own pipeline vs. their +1.6 pp.

Oracle responses, following PRISM: WM from the top-FA voxels in the
phantom; GM and CSF as isotropic tensors with MD 0.9 and 3.0 ×10⁻³ mm²/s
(DiSCo has neither tissue, so they cannot be estimated from data).
"""

from __future__ import annotations

import numpy as np


def msmt_csd_pam(
    data: np.ndarray,
    gtab,
    mask: np.ndarray,
    sphere,
    sh_order_max: int = 8,
    wm_fa_percentile: float = 80.0,
    relative_peak_threshold: float = 0.5,
    min_separation_angle: float = 25.0,
    npeaks: int = 5,
):
    """Fit MSMT-CSD and return a dipy PeaksAndMetrics (+ response info)."""
    from dipy.core.gradients import unique_bvals_tolerance
    from dipy.direction import peaks_from_model
    from dipy.reconst.dti import TensorModel
    from dipy.reconst.mcsd import (
        MultiShellDeconvModel, multi_shell_fiber_response, response_from_mask_msmt,
    )

    mask = np.asarray(mask, dtype=bool)
    # single-fibre WM mask from DTI FA (same rule as doc 004 §18.1 step 3)
    fa = TensorModel(gtab).fit(data, mask=mask).fa
    fa = np.nan_to_num(fa)
    thr = np.percentile(fa[mask], wm_fa_percentile)
    mask_wm = mask & (fa >= thr)

    wm_rf, _, _ = response_from_mask_msmt(gtab, data, mask_wm, mask_wm, mask_wm)
    ubv = unique_bvals_tolerance(gtab.bvals)
    s0 = float(np.mean(data[mask][:, gtab.b0s_mask]))
    n_shell = wm_rf.shape[0]
    gm_rf = np.tile([0.9e-3, 0.9e-3, 0.9e-3, s0], (n_shell, 1))
    csf_rf = np.tile([3.0e-3, 3.0e-3, 3.0e-3, s0], (n_shell, 1))

    response = multi_shell_fiber_response(sh_order_max, ubv, wm_rf, gm_rf, csf_rf)
    model = MultiShellDeconvModel(gtab, response, sh_order_max=sh_order_max)
    pam = peaks_from_model(
        model, data, sphere, relative_peak_threshold, min_separation_angle,
        mask=mask, npeaks=npeaks, return_sh=False, sh_order_max=sh_order_max,
    )
    pam.affine = np.eye(4)
    return pam, {"wm_rf": wm_rf, "n_wm_voxels": int(mask_wm.sum()), "fa_thr": float(thr)}

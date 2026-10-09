"""
FORCE → DiSCo connectivity matrix pipeline (reproduces FORCE paper §3.2).

Pipeline:
  1. Fit FORCE on a DiSCo subject's single-shell b=1900 data with the
     paper-protocol tuned library.
  2. Extract a PeaksAndMetrics object via force_peaks(fit).
  3. Run LocalTracking (Euler integration with a discrete peak direction
     getter — the EuDX-equivalent in current dipy).
  4. Filter streamlines to those connecting distinct ROI labels.
  5. Build a 16×16 connectivity matrix via
     dipy.tracking.utils.connectivity_matrix.
  6. Compute Pearson r against the ground-truth Connectivity_Matrix
     loaded from DiSCo1_Connectivity_Matrix_Cross-Sectional_Area.txt.

The paper §3.2 reports r=0.868 at SNR=10 and r=0.894 at SNR=50 on this
pipeline against the same ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


_DISCO_ROOT = Path.home() / ".dipy" / "disco"


def _disco_path(subject: int = 1) -> Path:
    p = _DISCO_ROOT / f"disco_{subject}"
    if not p.exists():
        raise FileNotFoundError(
            f"DiSCo subject {subject} not at {p}; fetch with "
            f"dipy.data.fetch_disco{subject}_dataset()."
        )
    return p


def load_gt_connectivity(subject: int = 1) -> np.ndarray:
    """Load DiSCo ground-truth Connectivity_Matrix_Cross-Sectional_Area.

    Returns a 16×16 float matrix of bundle cross-sectional areas in mm².
    The diagonal is zero (no self-connections); the matrix is symmetric.
    """
    base = _disco_path(subject)
    gt = np.loadtxt(base / f"DiSCo{subject}_Connectivity_Matrix_Cross-Sectional_Area.txt")
    assert gt.shape == (16, 16), f"Unexpected GT matrix shape: {gt.shape}"
    return gt.astype(np.float64)


def connectivity_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two N×N connectivity matrices, on the upper
    triangle (excluding the diagonal).

    Both matrices are assumed symmetric. Using the upper triangle avoids
    double-counting symmetric pairs and excludes the trivial-zero
    diagonal entries.
    """
    iu = np.triu_indices(a.shape[0], k=1)
    av = np.asarray(a)[iu].astype(np.float64)
    bv = np.asarray(b)[iu].astype(np.float64)
    valid = np.isfinite(av) & np.isfinite(bv)
    if valid.sum() < 3 or av[valid].std() < 1e-12 or bv[valid].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(av[valid], bv[valid])[0, 1])


def lin_ccc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Lin's Concordance Correlation Coefficient (CCC).

    Unlike Pearson r — which is shift- and scale-invariant — CCC penalises
    systematic bias: ``CCC = 2·cov(a,b) / (var(a) + var(b) + (mean(a) − mean(b))²)``.

    If ``mask`` is provided, the computation is restricted to that mask;
    otherwise the entire arrays are used. NaN/inf are dropped pairwise.
    Returns ``nan`` for fewer than 3 valid samples or zero variance.
    """
    a_f = np.asarray(a).astype(np.float64).ravel()
    b_f = np.asarray(b).astype(np.float64).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel().astype(bool)
        a_f = a_f[m]
        b_f = b_f[m]
    valid = np.isfinite(a_f) & np.isfinite(b_f)
    if valid.sum() < 3:
        return float("nan")
    a_v = a_f[valid]
    b_v = b_f[valid]
    mean_a, mean_b = a_v.mean(), b_v.mean()
    var_a, var_b = a_v.var(ddof=0), b_v.var(ddof=0)
    cov = ((a_v - mean_a) * (b_v - mean_b)).mean()
    denom = var_a + var_b + (mean_a - mean_b) ** 2
    if denom < 1e-12:
        return float("nan")
    return float(2.0 * cov / denom)


def run_force_connectivity(
    subject: int = 1,
    snr: int = 30,
    tuned: bool = True,
    seed_density: int = 2,
    step_size: float = 0.5,
    max_cross: Optional[int] = None,
    max_angle: float = 45.0,
    pmf_threshold: float = 0.1,
    rebalance_fibres: Optional[Dict[int, float]] = None,
    _smoke_roi_subset: Optional[Iterable[int]] = None,
):
    """Fit FORCE on DiSCo and produce a connectivity matrix.

    Uses the *current* dipy tractography API
    (:func:`dipy.tracking.tracker.eudx_tracking`), which is the
    paper-faithful equivalent of EuDX. The audit (doc 004 §17 follow-up)
    flagged the older :class:`LocalTracking` path as deprecated for
    :class:`PeaksAndMetrics` direction getters and missing the
    ``max_angle`` / ``pmf_threshold`` parameters paper §3.2 implicitly
    relies on.

    Stopping criterion now uses ``(rois > 0) | mask`` so streamlines can
    propagate into the ROI cylinders (which extend beyond the WM
    strand-bundle mask in DiSCo; ~50 % of each ROI cylinder is outside
    the brain mask). The previous ``mask``-only criterion terminated
    streamlines at ROI boundaries instead of inside ROI labels.

    Parameters
    ----------
    seed_density
        Per-dimension density of seeds inside the ROI mask.
    step_size
        Streamline integration step in mm.
    max_cross
        Maximum number of fibres to follow per voxel. ``None`` = all peaks.
    max_angle
        Maximum turning angle per step in degrees. eudx_tracking default
        is 60; we use 45 (more selective).
    pmf_threshold
        Minimum peak strength to consider tracking. eudx_tracking default
        is 0.0239; we raise to 0.1 to avoid following weak/spurious peaks
        in the FORCE ``fracs`` array (see audit finding #3).
    _smoke_roi_subset
        For tests only.
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    import nibabel as nib
    from dipy.reconst.force import FORCEModel, force_peaks, load_force_simulations
    from dipy.tracking.tracker import eudx_tracking
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.utils import connectivity_matrix, seeds_from_mask

    from validation.lib.force_disco import load_disco_subject

    out = load_disco_subject(subject=subject, snr=snr, single_shell_b=1900)
    data = out["data"]
    mask = out["mask"]
    gtab = out["gtab"]
    rois = out["rois"]
    affine = nib.load(_disco_path(subject) / f"highRes_DiSCo{subject}_DWI.nii.gz").affine

    # Library
    lib_cache = (Path.home() / ".cache" / "dipy_force" /
                 ("force_disco_500k_tuned.npz" if tuned
                  else "force_disco_500k.npz"))
    if not lib_cache.exists():
        raise FileNotFoundError(
            f"FORCE library not cached at {lib_cache}; "
            "run validate_force_disco_phantom.py {--tuned} first."
        )
    sims = load_force_simulations(str(lib_cache))
    if rebalance_fibres is not None:
        from validation.lib.force_library_rebalance import (
            rebalance_force_library,
        )
        sims = rebalance_force_library(sims, target_fractions=rebalance_fibres, seed=0)
    model = FORCEModel(gtab, simulations=sims, n_neighbors=50)

    # Fit
    if _smoke_roi_subset is not None:
        fit_mask = np.isin(rois, list(_smoke_roi_subset)) & mask
    else:
        fit_mask = mask
    fit = model.fit(data, mask=fit_mask)
    peaks = force_peaks(fit)

    # Stopping criterion: expanded to include ROI cylinders so streamlines
    # can enter and terminate on ROI labels (not at WM-mask edges).
    sc_mask = ((rois > 0) | mask).astype(np.uint8)
    sc = BinaryStoppingCriterion(sc_mask)

    # Seed from ROI mask (or smoke subset)
    if _smoke_roi_subset is not None:
        seed_mask = np.isin(rois, list(_smoke_roi_subset)) & mask
    else:
        seed_mask = (rois > 0) & mask
    seeds = seeds_from_mask(seed_mask, affine, density=seed_density)

    # Track via eudx_tracking — paper-faithful API with proper peak gating
    streamline_gen = eudx_tracking(
        seeds, sc, affine,
        pam=peaks,
        max_cross=max_cross,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        step_size=step_size,
        min_len=4, max_len=300,
        return_all=True,
        random_seed=0,
    )
    streamlines = Streamlines(streamline_gen)
    # Drop any zero-length streamlines that LocalTracking can occasionally
    # emit (they crash dipy.tracking.utils.connectivity_matrix).
    streamlines = Streamlines([s for s in streamlines if len(s) >= 2])

    # Connectivity matrix using ROI labels (skip background 0)
    if len(streamlines) == 0:
        cmat = np.zeros((17, 17), dtype=np.int64)
    else:
        cmat, _ = connectivity_matrix(
            streamlines, affine, rois.astype(np.int32),
            return_mapping=True, mapping_as_streamlines=False, symmetric=True,
        )
    # cmat is (n_labels+1) × (n_labels+1) including background label 0;
    # the paper's matrix is 16×16 (labels 1..16).
    cmat16 = cmat[1:17, 1:17].astype(np.float64)

    return {
        "streamlines_count": len(streamlines),
        "connectivity": cmat16,
        "fit_mask_voxels": int(fit_mask.sum()),
        "n_seeds": int(len(seeds)),
    }

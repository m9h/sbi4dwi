"""
Shared DiSCo tractography + connectivity helper.

Factored out of ``validation/validate_dmipy_disco_connectivity*.py`` so
that every method under comparison (dmipy-JAX dictionary, PRISM-JAX,
MSMT-CSD, FORCE) is pushed through the *identical* eudx_tracking →
connectivity_matrix pipeline of doc 004 §17.5 / §21.2. Only the
``PeaksAndMetrics`` differs between methods.
"""

from __future__ import annotations

import numpy as np


def peaks_to_pam(
    peak_dirs: np.ndarray,
    peak_values: np.ndarray,
    mask: np.ndarray,
    sphere,
    affine: np.ndarray | None = None,
):
    """Build a dipy ``PeaksAndMetrics`` from explicit per-voxel peaks.

    Parameters
    ----------
    peak_dirs : (X, Y, Z, 5, 3) unit vectors (zero rows = empty slot)
    peak_values : (X, Y, Z, 5) non-negative weights, descending per voxel
    mask : (X, Y, Z) bool
    sphere : dipy Sphere used to snap ``peak_indices`` (antipodal-aware)
    """
    from dipy.direction.peaks import PeaksAndMetrics

    shape = mask.shape
    peak_indices = -np.ones(shape + (5,), dtype=np.int32)
    verts = sphere.vertices
    idx = np.argwhere(mask)
    for i, j, k in idx:
        for s in range(5):
            if peak_values[i, j, k, s] > 0:
                dots = np.abs(verts @ peak_dirs[i, j, k, s])
                peak_indices[i, j, k, s] = int(np.argmax(dots))

    pam = PeaksAndMetrics()
    pam.peak_dirs = np.asarray(peak_dirs, dtype=np.float64)
    pam.peak_values = np.asarray(peak_values, dtype=np.float64)
    pam.peak_indices = peak_indices
    pam.sphere = sphere
    pam.affine = np.eye(4) if affine is None else affine
    pam.shm_coeff = None
    pam.B = None
    return pam


def track_connectivity(
    pam,
    mask: np.ndarray,
    rois: np.ndarray,
    affine: np.ndarray,
    seed_density: int = 2,
    step_size: float = 0.5,
    max_angle: float = 45.0,
    pmf_threshold: float = 0.1,
    random_seed: int = 0,
) -> dict:
    """§17.5 / §21.2 tracking protocol → 16×16 connectivity matrix."""
    from dipy.tracking.tracker import eudx_tracking
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.utils import connectivity_matrix, seeds_from_mask

    sc_mask = ((rois > 0) | mask).astype(np.uint8)
    sc = BinaryStoppingCriterion(sc_mask)
    seed_mask = (rois > 0) & mask
    seeds = seeds_from_mask(seed_mask, affine, density=seed_density)
    gen = eudx_tracking(
        seeds, sc, affine, pam=pam, max_cross=None, max_angle=max_angle,
        pmf_threshold=pmf_threshold, step_size=step_size,
        min_len=4, max_len=300, return_all=True, random_seed=random_seed,
    )
    streamlines = Streamlines([s for s in Streamlines(gen) if len(s) >= 2])

    if len(streamlines) == 0:
        cmat16 = np.zeros((16, 16), dtype=np.float64)
    else:
        cmat, _ = connectivity_matrix(
            streamlines, affine, rois.astype(np.int32),
            return_mapping=True, mapping_as_streamlines=False, symmetric=True,
        )
        cmat16 = cmat[1:17, 1:17].astype(np.float64)
    return {
        "connectivity": cmat16,
        "streamlines_count": int(len(streamlines)),
        "n_seeds": int(len(seeds)),
    }

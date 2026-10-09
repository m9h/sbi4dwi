"""
Reproduce dipy's Stanford HARDI FORCE example from
``dipy/doc/examples/reconst_force.py`` and the FORCE paper Figure 6
(NODDI parameter maps from a single-shell acquisition).

The aim is *confirmation*: show that dipy 1.12.1's `FORCEModel.fit` on
Stanford HARDI produces sensible NODDI-style maps before we make any
comparison claims about FORCE vs. dmipy-JAX.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np


CACHE_DIR = Path(os.environ.get(
    "FORCE_STANFORD_CACHE",
    str(Path.home() / ".cache" / "force_stanford"),
))


def load_stanford_hardi() -> Tuple[np.ndarray, np.ndarray, np.ndarray, "GradientTable"]:
    """Load Stanford HARDI as ``(data, affine, mask, gtab)``.

    Mirrors the dipy tutorial: median_otsu mask over volumes 10–50 with
    radius=4 / numpass=4. Caches the mask under ``FORCE_STANFORD_CACHE`` so
    tests don't repeat the (slow) median_otsu call.
    """
    from dipy.core.gradients import gradient_table
    from dipy.data import get_fnames
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti

    hardi_fname, bval_fname, bvec_fname = get_fnames(name="stanford_hardi")
    data, affine = load_nifti(hardi_fname)
    bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
    gtab = gradient_table(bvals, bvecs=bvecs)

    mask_path = CACHE_DIR / "stanford_mask.npy"
    if mask_path.exists():
        mask = np.load(mask_path)
    else:
        from dipy.segment.mask import median_otsu
        _, mask = median_otsu(
            data, vol_idx=range(10, 50), median_radius=4, numpass=4,
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, mask)
    return data, affine, mask.astype(bool), gtab


def fit_force_or_load_cached_maps(
    library_cache: str,
    n_neighbors: int = 50,
    n_jobs: int = -1,
    _smoke_n_voxels: int | None = None,
) -> dict:
    """Run dipy ``FORCEModel.fit`` on Stanford HARDI and return per-voxel maps.

    Caches the maps as a single ``.npz`` under ``FORCE_STANFORD_CACHE``. If
    the cache exists it is loaded instead of refitting.

    Parameters
    ----------
    library_cache
        Path to a cached FORCE simulation library (``.npz``) generated for
        the Stanford HARDI gradient table. ``validate_force_matched.py``
        produces one at ``~/.cache/dipy_force/force_matched_500k.npz``.
    n_jobs
        Passed through to :meth:`FORCEModel.fit`. Use -1 for all cores.
    _smoke_n_voxels
        For tests only: restrict the fit to the first N in-mask voxels.
        Skips caching when set.
    """
    import warnings as _w
    _w.filterwarnings("ignore")

    from dipy.reconst.force import FORCEModel, load_force_simulations

    smoke = _smoke_n_voxels is not None
    cache_file = CACHE_DIR / "force_stanford_maps.npz"
    if not smoke and cache_file.exists():
        d = np.load(cache_file)
        return {k: d[k] for k in d.files}

    data, affine, mask, gtab = load_stanford_hardi()

    if smoke:
        # Restrict mask to the first _smoke_n_voxels in-mask voxels for tests
        idx = np.argwhere(mask)[:_smoke_n_voxels]
        small_mask = np.zeros_like(mask)
        for i, j, k in idx:
            small_mask[i, j, k] = True
        mask = small_mask

    sims = load_force_simulations(library_cache)
    model = FORCEModel(gtab, simulations=sims, n_neighbors=n_neighbors)

    fit_kwargs = {} if smoke else {"n_jobs": n_jobs}
    fit = model.fit(data, mask=mask, **fit_kwargs)

    maps = {
        "fa": np.asarray(fit.fa, dtype=np.float32),
        "md": np.asarray(fit.md, dtype=np.float32),
        "rd": np.asarray(fit.rd, dtype=np.float32),
        "wm_fraction": np.asarray(fit.wm_fraction, dtype=np.float32),
        "gm_fraction": np.asarray(fit.gm_fraction, dtype=np.float32),
        "csf_fraction": np.asarray(fit.csf_fraction, dtype=np.float32),
        "num_fibers": np.asarray(fit.num_fibers, dtype=np.float32),
        "dispersion": np.asarray(fit.dispersion, dtype=np.float32),
        "nd": np.asarray(fit.nd, dtype=np.float32),
        "uncertainty": np.asarray(fit.uncertainty, dtype=np.float32),
        "ambiguity": np.asarray(fit.ambiguity, dtype=np.float32),
        "mask": mask,
        "affine": affine,
    }

    if not smoke:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, **maps)
    return maps

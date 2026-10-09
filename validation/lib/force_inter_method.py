"""
Stanford HARDI inter-method comparison helpers.

Compares FORCE-derived metrics against established baselines on the same
data — replicating the FORCE paper's Figure C1 (FORCE vs DTI). Adds:

  - fit_dti_baseline: dipy.reconst.dti TensorModel on a 4D volume.
  - masked_pearson: voxel-wise Pearson r within a mask.
  - load_force_maps: pull the cached Stanford HARDI FORCE maps.

The actual figure-rendering script lives at
``validation/validate_force_inter_method.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from validation.lib.force_stanford import CACHE_DIR


def fit_dti_baseline(
    data: np.ndarray,
    gtab,
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run dipy DTI on a 4D volume + mask. Returns FA, MD, RD, AD as
    full-volume float32 arrays (zero outside mask)."""
    from dipy.reconst.dti import TensorModel

    model = TensorModel(gtab)
    fit = model.fit(data, mask=mask)
    return {
        "fa": np.asarray(fit.fa, dtype=np.float32),
        "md": np.asarray(fit.md, dtype=np.float32),
        "rd": np.asarray(fit.rd, dtype=np.float32),
        "ad": np.asarray(fit.ad, dtype=np.float32),
    }


def masked_pearson(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Pearson correlation between two arrays restricted to in-mask voxels.

    Treats NaN/inf as missing and excludes them from the calculation.
    Returns ``nan`` if fewer than 3 valid voxels remain.
    """
    a_flat = a[mask].astype(np.float64)
    b_flat = b[mask].astype(np.float64)
    valid = np.isfinite(a_flat) & np.isfinite(b_flat)
    if valid.sum() < 3:
        return float("nan")
    a_v = a_flat[valid]
    b_v = b_flat[valid]
    if a_v.std() < 1e-12 or b_v.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a_v, b_v)[0, 1])


def load_force_maps() -> Dict[str, np.ndarray]:
    """Load the cached FORCE maps produced by validate_force_stanford_hardi.py."""
    cache = CACHE_DIR / "force_stanford_maps.npz"
    if not cache.exists():
        raise FileNotFoundError(
            f"FORCE maps not cached at {cache}. "
            "Run validation/validate_force_stanford_hardi.py first."
        )
    d = np.load(cache)
    out = {k: d[k] for k in d.files}
    out["mask"] = out["mask"].astype(bool)
    return out

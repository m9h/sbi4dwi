"""
DiSCo phantom loader + helpers for the FORCE-paper §3.2 benchmark.

The DiSCo dataset (DOI 10.17632/fgf86jdfg6) is fetched by
``dipy.data.fetch_disco1_dataset()`` (and analogous _2, _3) and unpacks
under ``~/.dipy/disco/disco_{N}/``. The FORCE paper §3.2 uses single-shell
b=2000 extraction; DiSCo's nearest shell is b=1900.

Shapes (subject 1, highRes):
  - DWI:                 (40, 40, 40, 364)  — 4 b=0 + 90 each at b∈{1k, 1.9k, 3.1k, 13.2k}
  - mask:                (40, 40, 40)
  - ROIs:                (40, 40, 40)       — bundle labels
  - Strand_Intra_VF:     (40, 40, 40)       — ground-truth NDI
  - Strand_Avg_Diameter: (40, 40, 40)       — ground-truth axon diameter
  - Strand_ODFs:         (40, 40, 40, 91)   — ground-truth ODFs (sym SH)

The mask file is stored as float; we threshold > 0 to get the brain-equivalent
boolean mask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


_DISCO_ROOT = Path.home() / ".dipy" / "disco"


def disco_subject_path(subject: int = 1) -> Path:
    p = _DISCO_ROOT / f"disco_{subject}"
    if not p.exists():
        raise FileNotFoundError(
            f"DiSCo subject {subject} not at {p}. Fetch with "
            f"`from dipy.data import fetch_disco{subject}_dataset; "
            f"fetch_disco{subject}_dataset()`."
        )
    return p


def filter_to_single_shell(
    bvals: np.ndarray,
    target_b: float,
    tol: float = 50.0,
) -> np.ndarray:
    """Return indices for b=0 and b≈target_b volumes (single-shell extraction)."""
    bvals = np.asarray(bvals)
    b0_mask = bvals < tol
    target_mask = np.abs(bvals - target_b) <= tol
    return np.where(b0_mask | target_mask)[0]


def load_disco_subject(
    subject: int = 1,
    snr: Optional[int] = None,
    single_shell_b: Optional[float] = 1900.0,
    res: str = "highRes",
):
    """Load a DiSCo subject's DWI, mask, gtab and ground-truth maps.

    Parameters
    ----------
    subject
        1, 2, or 3 (per DiSCo subjects).
    snr
        ``None`` for the clean (noise-free) DWI, or one of 10/20/30/40/50
        for the Rician-noised variant.
    single_shell_b
        If set, sub-select b=0 + b≈target volumes. Set ``None`` to keep all
        364 directions. FORCE paper §3.2 uses 2000; DiSCo's closest is 1900.
    res
        ``"highRes"`` (40^3) or ``"lowRes"`` (20^3).
    """
    import nibabel as nib
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs

    base = disco_subject_path(subject)
    snr_tag = f"_RicianNoise-snr{snr}" if snr is not None else ""
    dwi_path = base / f"{res}_DiSCo{subject}_DWI{snr_tag}.nii.gz"
    mask_path = base / f"{res}_DiSCo{subject}_mask.nii.gz"
    rois_path = base / f"{res}_DiSCo{subject}_ROIs.nii.gz"
    ndi_path = base / f"{res}_DiSCo{subject}_Strand_Intra_Volume_Fraction.nii.gz"
    diam_path = base / f"{res}_DiSCo{subject}_Strand_Average_Diameter.nii.gz"
    bvals_path = base / "DiSCo_gradients.bvals"
    bvecs_path = base / "DiSCo_gradients_dipy.bvecs"

    data = nib.load(dwi_path).get_fdata()
    mask = nib.load(mask_path).get_fdata() > 0
    rois = nib.load(rois_path).get_fdata().astype(np.int32)
    ndi = nib.load(ndi_path).get_fdata()
    diam = nib.load(diam_path).get_fdata()
    bvals, bvecs = read_bvals_bvecs(str(bvals_path), str(bvecs_path))

    if single_shell_b is not None:
        idx = filter_to_single_shell(bvals, target_b=single_shell_b)
        data = data[..., idx]
        bvals = bvals[idx]
        bvecs = bvecs[idx]

    gtab = gradient_table(bvals, bvecs=bvecs)

    return {
        "data": data.astype(np.float32),
        "mask": mask.astype(bool),
        "rois": rois,
        "gtab": gtab,
        "gt": {"ndi": ndi.astype(np.float32),
                "diameter": diam.astype(np.float32)},
    }

#!/usr/bin/env python3
"""Export datasets 2 (PRISM synthetic, in-model and dispersed) and the DiSCo
3-shell protocol data as plain npz/nii for external methods (FORCE on dipy
master, SBI_dMRI in Docker). Dataset 3 (substrates) is exported by
``validate_prism_substrate.py --save-signals``."""
import numpy as np
from pathlib import Path
from dmipy_jax.validation.prism_synthetic import make_benchmark

out = Path("validation/external"); out.mkdir(exist_ok=True)
for snr in (30, 10):
    for odi in (None, 0.2):
        b = make_benchmark(snr=snr, gt_odi=odi)
        tag = f"synthetic_snr{snr}" + ("" if odi is None else f"_odi{odi}")
        np.savez(out / f"{tag}.npz", data=b["data"], mask=b["mask"], bvals=b["bvals"] / 1e6,
                 bvecs=b["bvecs"], gt_dirs=b["gt_dirs"], angle=b["angle"], gt_fracs=b["gt_fracs"],
                 fintra=b["fintra"], gt_odi=-1.0 if odi is None else odi)
        print("wrote", tag)

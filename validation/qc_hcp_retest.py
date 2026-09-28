#!/usr/bin/env python3
"""Per-visit WM b0 SNR for every processed HCP retest subject → validation/hcp_retest/qc_snr.json.
Flags subjects whose worse visit is below --snr-min (data-quality outliers for the cohort table)."""
import json, sys
from pathlib import Path
import numpy as np, nibabel as nib
ROOT = Path("/data/datasets/hcp"); B1 = ROOT / "b1"
snr_min = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
out = {}
for sd in sorted(B1.glob("*/compare_results.json")):
    s = sd.parent.name; o = {}
    for v in ["test", "retest"]:
        d = ROOT / v / s / "T1w" / "Diffusion"; b = np.loadtxt(d / "bvals")
        data = np.asarray(nib.load(d / "data.nii.gz").dataobj, np.float32)
        g = np.load(B1 / s / v / "grid.npz"); wm = g["wm"] & g["mask"]
        X = data[wm][:, b < 50]; o[v] = float(np.median(X.mean(1) / X.std(1, ddof=1))); del data
    o["worse"] = min(o["test"], o["retest"]); o["flag"] = o["worse"] < snr_min
    out[s] = o; print(f"{s}: test {o['test']:.1f} retest {o['retest']:.1f}{'  FLAG' if o['flag'] else ''}", flush=True)
json.dump({"snr_min": snr_min, "subjects": out}, open("validation/hcp_retest/qc_snr.json", "w"), indent=1)
print("flagged:", [s for s, o in out.items() if o["flag"]])

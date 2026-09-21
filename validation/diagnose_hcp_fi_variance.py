#!/usr/bin/env python3
"""Where does the scan–rescan variance of the refine f_i come from, relative to NODDI? (doc 008 §10.4 follow-up)
Uses the cached per-visit arrays of validate_hcp_retest.py; retest resampled into the test grid with the stored rigid transform."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "validation"); import validate_hcp_retest as V
subj = sys.argv[1] if len(sys.argv) > 1 else "105923"; V.SUBJ = subj; V.OUT = V.ROOT / "b1" / subj
T = {k: np.load(V.OUT / "test" / k) for k in ["grid.npz", "refine.npz", "noddi.npz", "force.npz"]}; R = {k: np.load(V.OUT / "retest" / k) for k in ["grid.npz", "refine.npz", "noddi.npz", "force.npz"]}
reg = np.array(json.load(open(V.OUT / "compare_results.json"))["registration"]["affine"])
lin = V.resample_to_test(reg, T["grid.npz"], R["grid.npz"]); ok = lin >= 0
def pull(x): y = np.zeros((len(lin),) + x.shape[1:], x.dtype); y[ok] = x[lin[ok]]; return y
gt, gr = T["grid.npz"], R["grid.npz"]; wm_t = gt["wm"][gt["mask"]]; wm_r = pull(gr["wm"][gr["mask"]]); sel = ok & wm_t & wm_r
fa_t = np.load(V.OUT / "test" / "fa.npy"); fa_r = pull(np.load(V.OUT / "retest" / "fa.npy"))
fi_t, fi_r = T["refine.npz"]["fintra"], pull(R["refine.npz"]["fintra"])
nd_t, nd_r = T["noddi.npz"]["ndi"] * (1 - T["noddi.npz"]["fwf"]), pull(R["noddi.npz"]["ndi"] * (1 - R["noddi.npz"]["fwf"]))
nf_t = (T["refine.npz"]["wm_fracs"] >= 0.1).sum(1); nf_r = pull((R["refine.npz"]["wm_fracs"] >= 0.1).sum(1))
csf_t = T["refine.npz"]["fracs"][:, 0] + T["refine.npz"]["fracs"][:, 1]
print(f"{subj}: WM voxels {sel.sum():,}")
print(f"refine f_i vs NODDI NDI·(1−FWF), same visit: r={np.corrcoef(fi_t[sel], nd_t[sel])[0,1]:.3f}, mean {fi_t[sel].mean():.3f} vs {nd_t[sel].mean():.3f}")
def row(name, m):
    d1, d2 = (fi_t - fi_r)[m], (nd_t - nd_r)[m]
    print(f"  {name:28s} n={m.sum():7,}  refine |Δf_i| {np.abs(d1).mean():.3f} sd {d1.std():.3f} | NODDI |Δ| {np.abs(d2).mean():.3f} sd {d2.std():.3f} | ΔFA sd {(fa_t-fa_r)[m].std():.3f}")
print("by refine fixel count (test visit):")
for k in [1, 2, 3]: row(f"{k} fixel(s)", sel & (nf_t == k))
row("count differs between visits", sel & (nf_t != nf_r)); row("count agrees", sel & (nf_t == nf_r))
print("by FA (test visit):")
for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]: row(f"FA {lo}–{hi}", sel & (fa_t >= lo) & (fa_t < hi))
print("by refine isotropic fraction (test visit):")
for lo, hi in [(0, 0.05), (0.05, 0.2), (0.2, 1.0)]: row(f"f_csf+f_gm {lo}–{hi}", sel & (csf_t >= lo) & (csf_t < hi))
# is the refine variance in f_i traded against the isotropic fractions?  Δ(f_i) vs Δ(f_iso) correlation
csf_r = pull(R["refine.npz"]["fracs"][:, 0] + R["refine.npz"]["fracs"][:, 1])
print(f"corr(Δf_i, Δf_iso) refine: {np.corrcoef((fi_t-fi_r)[sel], (csf_t-csf_r)[sel])[0,1]:.3f};  corr(Δf_i, ΔD-free?) n/a;  refine Δf_i vs NODDI Δ: {np.corrcoef((fi_t-fi_r)[sel], (nd_t-nd_r)[sel])[0,1]:.3f}")

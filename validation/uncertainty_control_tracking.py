#!/usr/bin/env python3
"""Control for the posterior-mean connectome gain: is it the calibrated per-fixel σ, or just averaging many tracking runs?
Per subject/visit, 10 tracking runs of: (A) refine peaks, seed change only; (B) refine peaks + constant-σ jitter (σ=3.3°, isotropic);
the Laplace posterior samples (C) already exist in connectomes.npz. Saves control_connectomes.npz per visit.
    uv run python validation/uncertainty_control_tracking.py [--subjects a,b]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, jax
sys.path.insert(0, "validation"); import validate_hcp_retest as V
from dmipy_jax.prism import prism_uncertainty as pu
from dipy.data import default_sphere
ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--n", type=int, default=10); ap.add_argument("--sigma-deg", type=float, default=3.3)
a = ap.parse_args(); ROOT = V.ROOT / "b1"
subs = a.subjects.split(",") if a.subjects else json.load(open("validation/hcp_retest/cohort_groups.json"))["clean"]
for s in subs:
    for v in ["test", "retest"]:
        out = ROOT / s / v; f = out / "control_connectomes.npz"
        if f.exists(): continue
        g = np.load(out / "grid.npz"); R = np.load(out / "refine.npz"); mask = g["mask"]; aff = g["affine"]; keep = R["wm_fracs"] >= 0.10
        vals = np.where(keep, R["wm_fracs"], 0.0); nR = len(g["labels"]); t0 = time.time()
        A = np.stack([V.track_connectome(V.build_pam(R["dirs"], vals, mask, aff, default_sphere), g["track_mask"], g["wm"] & mask, g["rois"], aff, nR, random_seed=k + 1)[0] for k in range(a.n)])
        c = np.zeros_like(R["cov"]); sg = np.deg2rad(a.sigma_deg) ** 2; c[..., 0, 0] = sg; c[..., 1, 1] = sg
        post = pu.FixelPosterior(dirs=R["dirs"], cov=c, e1=R["e1"], e2=R["e2"], sigma_deg=np.full_like(R["sigma_deg"], a.sigma_deg), wm_fracs=R["wm_fracs"], mask=mask)
        S = post.sample_dirs(jax.random.key(0), a.n)
        B = np.stack([V.track_connectome(V.build_pam(S[k], vals, mask, aff, default_sphere), g["track_mask"], g["wm"] & mask, g["rois"], aff, nR, random_seed=k + 1)[0] for k in range(a.n)])
        np.savez_compressed(f, seed_only=A, const_sigma=B); V.log(f"{s} {v} controls {time.time()-t0:.0f}s")

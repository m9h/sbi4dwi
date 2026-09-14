#!/usr/bin/env python3
"""
Boundary check for the flow variants (doc 008 §7.1): the synthetic
benchmark's fibres lie in the z = 0 plane, which is the *boundary* of a
hemisphere prior (θ ≤ π/2, or the unit-disk map). A posterior mode on the
boundary spills half its mass to the antipode, so hemisphere variants
cannot look better than `base` there. Re-evaluate every trained variant
checkpoint on the same benchmark rotated by `--tilt` degrees about x
(fibres well inside the hemisphere), same noise, same metrics.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
sys.path.insert(0, str(Path(__file__).parent))
from diagnose_flow_variants import param_spec, make_forward, build_flow, evaluate, VARIANTS
from dmipy_jax.validation import prism_synthetic as ps


def rotated_benchmark(tilt_deg, snr, seed=0):
    b = ps.make_benchmark(snr=None)
    a = np.radians(tilt_deg); R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    gt = b["gt_dirs"] @ R.T
    b["gt_dirs"] = gt
    fwd = make_forward("base", b["bvals"], b["bvecs"])
    th = np.arccos(np.clip(gt[..., 2], -1, 1)); ph = np.arctan2(gt[..., 1], gt[..., 0]) % (2 * np.pi)
    fr = b["gt_fracs"]; single = b["angle"] == 0
    P = np.column_stack([th[:, 0], ph[:, 0], th[:, 1], ph[:, 1], fr[:, 2], np.where(single, 0.0, fr[:, 3]), np.full(len(gt), b["fintra"])])
    clean = np.asarray(fwd(jnp.asarray(P)))
    rng = np.random.default_rng(seed + 1); s = 1.0 / snr
    noisy = np.sqrt((clean + rng.normal(0, s, clean.shape)) ** 2 + rng.normal(0, s, clean.shape) ** 2)
    b["data"] = np.zeros(b["mask"].shape + (clean.shape[1],), np.float32); b["data"][b["mask"]] = noisy
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS); ap.add_argument("--tilt", type=float, default=45)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=400); ap.add_argument("--snr", type=float, default=30)
    a = ap.parse_args()
    b = rotated_benchmark(a.tilt, a.snr)
    print(f"benchmark tilted {a.tilt}° about x: GT z-components {np.round(np.unique(np.round(b['gt_dirs'][..., 2], 2)), 2).tolist()}")
    for v in a.variants:
        ck = Path(f"validation/flow_variant_{v}_{a.hidden}x{a.depth}.eqx")
        if not ck.exists():
            print(f"{v}: no checkpoint"); continue
        names, lo, hi = param_spec(v)
        flow = eqx.tree_deserialise_leaves(ck, build_flow(jax.random.key(0), 7, len(b["bvals"]), a.hidden, a.depth, embed=v.endswith("emb")))
        t0 = time.time(); ev = evaluate(flow, v, lo, hi, b, a.n_samples, jax.random.key(2))
        print(f"{v:11s} tilt {a.tilt:.0f}°: err={ev['err']:.2f}°  recall={100*ev['recall']:.1f}%  σ_θ={ev['sigma']:.1f}°  "
              f"antipodal={ev['anti']:.2f}  switch={ev['switch']:.2f}  ({time.time()-t0:.0f}s)")
        print("   angle    err   recall  σ_θ   anti  switch")
        for av, (e, r, sg, an, sw) in ev["rows"].items():
            if av in (15, 30, 45, 60, 90, 0):
                print(f"   {av:>4d}  {e:6.2f}° {100*r:6.1f}% {sg:5.1f}° {an:5.2f} {sw:5.2f}")


if __name__ == "__main__":
    main()

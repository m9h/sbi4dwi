#!/usr/bin/env python3
"""FORCE (dipy master: seeded, two_fiber_min_angle=0) on exported benchmarks —
pure numpy + dipy, runs in the dipy-master scratch venv.

  python run_force_external.py --npz validation/external/synthetic_snr30.npz \
      --kind synthetic --num-sims 500000 --num-cpus 16
  python run_force_external.py --npz validation/substrate_signals.npz --kind substrate

Scores best-match angular error / recall(<20°) exactly as prism_jax does,
plus FORCE's ND vs the geometric intra fraction on substrates.
"""
import argparse, time, json
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel, force_peaks


def angular_error_best_match(pred_dirs, pred_vals, gt_dirs, val_min=0.05):
    errs, hits, total = [], 0, 0
    for n in range(gt_dirs.shape[0]):
        for g in gt_dirs[n]:
            if np.linalg.norm(g) == 0:
                continue
            total += 1
            cand = [d for d, v in zip(pred_dirs[n], pred_vals[n]) if v >= val_min and np.linalg.norm(d) > 0]
            if not cand:
                continue
            cos = max(abs(float(np.dot(g, d))) for d in cand)
            e = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            errs.append(e); hits += int(e < 20.0)
    return (float(np.mean(errs)) if errs else float("nan")), hits / total if total else float("nan")


def fit_force(data, mask, bvals, bvecs, num_sims, num_cpus, seed, min_angle, dcfg, odi_range):
    gtab = gradient_table(bvals, bvecs=bvecs)
    model = FORCEModel(gtab, compute_odf=True)
    model.generate(num_simulations=num_sims, num_cpus=num_cpus, seed=seed, wm_threshold=0.0,
                   odi_range=odi_range, two_fiber_min_angle=min_angle, three_fiber_min_angle=min_angle,
                   diffusivity_config=dcfg, use_cache=True)
    fit = model.fit(data, mask=mask)
    pam = force_peaks(fit, mask=mask)
    nd = np.asarray(fit.nd) if hasattr(fit, "nd") else None
    return pam, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--kind", choices=["synthetic", "substrate"], required=True)
    ap.add_argument("--num-sims", type=int, default=500_000); ap.add_argument("--num-cpus", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2298); ap.add_argument("--min-angle", type=float, default=0.0)
    ap.add_argument("--dpar", type=float, nargs=2, default=None); ap.add_argument("--dperp", type=float, nargs=2, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    dcfg = None
    if args.dpar or args.dperp:
        dcfg = {}
        if args.dpar: dcfg["wm_d_par_range"] = tuple(args.dpar)
        if args.dperp: dcfg["wm_d_perp_range"] = tuple(args.dperp)
    results = {}
    if args.kind == "synthetic":
        data, mask = z["data"], z["mask"]; bvals = z["bvals"]; bvecs = z["bvecs"]
        t0 = time.time()
        pam, nd = fit_force(data, mask, bvals, bvecs, args.num_sims, args.num_cpus, args.seed, args.min_angle, dcfg, (0.01, 0.3))
        gt = z["gt_dirs"].copy(); ang = z["angle"]; gt[ang == 0, 1] = 0
        pd_, pv = pam.peak_dirs[mask], pam.peak_values[mask]
        pv = pv / np.maximum(pv.sum(1, keepdims=True), 1e-12)
        e, r = angular_error_best_match(pd_, pv, gt)
        print(f"FORCE {args.npz}: overall err={e:.2f}° recall={100*r:.1f}%  ({time.time()-t0:.0f}s)")
        results["overall"] = (e, r)
        for a in sorted(set(ang.tolist())):
            m = ang == a
            e, r = angular_error_best_match(pd_[m], pv[m], gt[m]); results[str(int(a))] = (e, r)
            print(f"  {int(a):>3d}  {e:6.2f}° / {100*r:5.1f}%")
    else:
        bvals = z["bvals"] / 1e6; bvecs = z["bvecs"]
        keys = sorted({k.rsplit("_", 1)[0] for k in z.files if k.endswith("_data")})
        for k in keys:
            data = z[f"{k}_data"]; mask = np.ones(data.shape[:3], bool); axes = z[f"{k}_axes"]; fi = float(z[f"{k}_f_intra"])
            t0 = time.time()
            pam, nd = fit_force(data, mask, bvals, bvecs, args.num_sims, args.num_cpus, args.seed, args.min_angle, dcfg, (0.01, 0.3))
            N = int(mask.sum()); gt = np.zeros((N, 2, 3)); gt[:, 0] = axes[0]
            if axes.shape[0] > 1: gt[:, 1] = axes[1]
            pd_, pv = pam.peak_dirs[mask], pam.peak_values[mask]; pv = pv / np.maximum(pv.sum(1, keepdims=True), 1e-12)
            e, r = angular_error_best_match(pd_, pv, gt)
            ndm = float(np.nanmean(nd[mask])) if nd is not None else float("nan")
            print(f"FORCE {k:14s} err={e:6.2f}° recall={100*r:5.1f}%  ND={ndm:.3f} (geom f_i {fi:.3f})  ({time.time()-t0:.0f}s)", flush=True)
            results[k] = (e, r, ndm, fi)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Where does FORCE's 7–17° crossing error come from? (doc 008 §7.3)
Runs in the dipy-master scratch venv (pure numpy + dipy).

FORCE's peaks are the sphere vertices labelled in the *single best-matching*
library entry (FORCEModel.fit: argmax cosine similarity over n_neighbors,
use_posterior=False by default). So its angular error decomposes into

  (a) sphere quantisation   — nearest 362-vertex to each GT direction
  (b) library coverage      — best achievable error over all library entries
                              with the right fibre count (the entry nearest
                              to the truth in *parameter* space)
  (c) matching              — what the cosine-similarity search actually picks
                              given noise and model mismatch (= FORCE's error)

(c) − (b) is the matching loss, (b) − (a) the coverage loss. Both are
measured on the same exported benchmarks as doc 008 §6.1, for library
sizes 500K and 2M, with and without posterior-weighted parameters,
and on noise-free signals (matching loss without noise).

  python diagnose_force_limits.py --npz validation/external/synthetic_snr30.npz --kind synthetic
"""
import argparse, time, json
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel, force_peaks
from dipy.sims.force import default_sphere


def best_match(pred_dirs, pred_vals, gt_dirs, val_min=0.05):
    errs, hits, total = [], 0, 0
    for n in range(gt_dirs.shape[0]):
        for g in gt_dirs[n]:
            if np.linalg.norm(g) == 0: continue
            total += 1
            cand = [d for d, v in zip(pred_dirs[n], pred_vals[n]) if v >= val_min and np.linalg.norm(d) > 0]
            if not cand: continue
            e = np.degrees(np.arccos(np.clip(max(abs(float(np.dot(g, d))) for d in cand), -1, 1)))
            errs.append(e); hits += int(e < 20.0)
    return (float(np.mean(errs)) if errs else float("nan")), hits / total if total else float("nan")


def entry_dirs(sims, sphere):
    """(N_lib, 5, 3) directions per library entry from its vertex labels (zero-padded)."""
    lab = np.asarray(sims["labels"]); V = sphere.vertices
    out = np.zeros((lab.shape[0], 5, 3), np.float32); cnt = np.zeros(lab.shape[0], int)
    for i in range(lab.shape[0]):
        idx = np.where(lab[i] == 1)[0][:5]; out[i, :len(idx)] = V[idx]; cnt[i] = len(idx)
    return out, cnt


def coverage_floor(gt_sets, lib_dirs, lib_cnt, n_gt):
    """For each GT configuration (list of unit vectors), the min over library
    entries with ≥ n_gt labelled directions of the mean best-match error."""
    res = []
    sel = lib_cnt >= n_gt; L = lib_dirs[sel]                    # (M,5,3)
    for gt in gt_sets:
        e_all = np.zeros(len(L))
        for g in gt:
            cos = np.abs(np.einsum("mkj,j->mk", L, g))
            cos[np.linalg.norm(L, axis=2) == 0] = 0
            e_all += np.degrees(np.arccos(np.clip(cos.max(1), -1, 1)))
        res.append(float(e_all.min() / len(gt)))
    return res


def quantisation_floor(gt_sets, sphere):
    V = sphere.vertices; out = []
    for gt in gt_sets:
        out.append(float(np.mean([np.degrees(np.arccos(np.clip(np.abs(V @ g).max(), -1, 1))) for g in gt])))
    return out


def run_fit(model, data, mask, use_posterior):
    model.use_posterior = use_posterior
    fit = model.fit(data, mask=mask); pam = force_peaks(fit, mask=mask)
    pv = pam.peak_values[mask]; pv = pv / np.maximum(pv.sum(1, keepdims=True), 1e-12)
    return pam.peak_dirs[mask], pv, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--clean-npz", default=None)
    ap.add_argument("--kind", choices=["synthetic", "substrate"], default="synthetic")
    ap.add_argument("--sizes", type=int, nargs="+", default=[500_000, 2_000_000])
    ap.add_argument("--num-cpus", type=int, default=16); ap.add_argument("--seed", type=int, default=2298)
    ap.add_argument("--out", default=None); ap.add_argument("--in-model", type=int, default=0); ap.add_argument("--in-model-easy", type=int, default=1)
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    if args.kind == "synthetic":
        sets = {"noisy": (z["data"], z["mask"])}
        if args.clean_npz:
            zc = np.load(args.clean_npz, allow_pickle=True); sets["clean"] = (zc["data"], zc["mask"])
        bvals, bvecs = z["bvals"], z["bvecs"]; ang = z["angle"]; gt = z["gt_dirs"].copy(); gt[ang == 0, 1] = 0
        angles = sorted(set(ang.tolist()))
        gt_sets = [[g for g in gt[np.where(ang == a)[0][0]] if np.linalg.norm(g) > 0] for a in angles]
    else:
        bvals = z["bvals"] / 1e6; bvecs = z["bvecs"]
        keys = sorted({k.rsplit("_", 1)[0] for k in z.files if k.endswith("_data")})
        sets = {k: (z[f"{k}_data"], np.ones(z[f"{k}_data"].shape[:3], bool)) for k in keys}
        gt_sets = [[a for a in z[f"{k}_axes"]] for k in keys]; angles = keys
    gtab = gradient_table(bvals, bvecs=bvecs)
    qf = quantisation_floor(gt_sets, default_sphere)
    print(f"sphere quantisation floor (362 vertices): mean {np.mean(qf):.2f}°  per condition {np.round(qf, 2).tolist()}")
    results = {"quantisation_floor": dict(zip(map(str, angles), qf))}
    for n_sims in args.sizes:
        t0 = time.time()
        model = FORCEModel(gtab, compute_odf=False)
        model.generate(num_simulations=n_sims, num_cpus=args.num_cpus, seed=args.seed, wm_threshold=0.0,
                       odi_range=(0.01, 0.3), two_fiber_min_angle=0.0, three_fiber_min_angle=0.0, use_cache=True)
        print(f"\n##### library {n_sims:,}: generated/loaded in {time.time()-t0:.0f}s", flush=True)
        lib_dirs, lib_cnt = entry_dirs(model.simulations, default_sphere)
        nf = np.asarray(model.simulations["num_fibers"])
        print(f"  fibre-count mix: " + "  ".join(f"{k}: {100*np.mean(nf==k):.1f}%" for k in (1, 2, 3)))
        _wm = np.asarray(model.simulations["wm_fraction"]); _fa = np.asarray(model.simulations["fraction_array"]); _od = np.asarray(model.simulations["dispersion"])
        _fmin = np.where(_fa[:, :2] > 0, _fa[:, :2], np.inf).min(1)
        _easy = (nf == 2) & (_wm >= 0.6) & (_fmin >= 0.15) & (_od <= 0.2)
        print(f"  library share that is a clean 2-fibre WM crossing (WM ≥ 0.6, each fibre ≥ 0.15, ODI ≤ 0.2): {100*np.mean(_easy):.2f}%  "
              f"({int(_easy.sum()):,} entries); WM fraction median {np.median(_wm):.2f}", flush=True)
        r = {"n_fibre_mix": {str(k): float(np.mean(nf == k)) for k in (1, 2, 3)}, "easy_share": float(np.mean(_easy))}
        for key, gts in (("2", [s for s in gt_sets if len(s) == 2]), ("1", [s for s in gt_sets if len(s) == 1])):
            if not gts: continue
            cf = coverage_floor(gts, lib_dirs, lib_cnt, int(key))
            cf3 = coverage_floor(gts, lib_dirs, lib_cnt, 3) if key == "2" else None
            print(f"  coverage floor for {key}-fibre GT: mean {np.mean(cf):.2f}° (entries with ≥{key} dirs)"
                  + (f", {np.mean(cf3):.2f}° (3-fibre entries only)" if cf3 else ""))
            r[f"coverage_floor_{key}"] = cf
        for sname, (data, mask) in sets.items():
            for post in (False, True):
                t0 = time.time(); pd_, pv, fit = run_fit(model, data, mask, post)
                if args.kind == "synthetic":
                    e, rc = best_match(pd_, pv, gt); rows = {}
                    for a in angles:
                        m = ang == a; rows[str(int(a))] = best_match(pd_[m], pv[m], gt[m])
                    print(f"  {sname:6s} posterior={post!s:5s}: err={e:.2f}° recall={100*rc:.1f}%  "
                          f"by angle: " + " ".join(f"{int(a)}:{rows[str(int(a))][0]:.1f}" for a in angles) + f"  ({time.time()-t0:.0f}s)", flush=True)
                else:
                    N = int(mask.sum()); axes = z[f"{sname}_axes"]; gts = np.zeros((N, 2, 3)); gts[:, 0] = axes[0]
                    if axes.shape[0] > 1: gts[:, 1] = axes[1]
                    e, rc = best_match(pd_, pv, gts); rows = {}
                    nd = float(np.nanmean(np.asarray(fit.nd)[mask]))
                    print(f"  {sname:14s} posterior={post!s:5s}: err={e:.2f}° recall={100*rc:.1f}%  ND={nd:.3f}  ({time.time()-t0:.0f}s)", flush=True)
                r[f"{sname}_post{int(post)}"] = {"err": e, "recall": rc, "rows": rows}
        if args.in_model > 0:
            # in-model held-out test: signals from FORCE's own generator (different
            # seed), 2-fibre entries only; error vs the entry's own labelled dirs.
            import tempfile
            from dipy.reconst.force import generate_force_simulations
            hold = generate_force_simulations(gtab, num_simulations=args.in_model, seed=args.seed + 1, num_cpus=args.num_cpus,
                                              wm_threshold=0.0, odi_range=(0.01, 0.3), two_fiber_min_angle=0.0,
                                              three_fiber_min_angle=0.0, output_dir=tempfile.mkdtemp(), verbose=False)
            hd, hc = entry_dirs(hold, default_sphere); nf_h = np.asarray(hold["num_fibers"])
            wmf = np.asarray(hold["wm_fraction"]); fa_ = np.asarray(hold["fraction_array"]); od = np.asarray(hold["dispersion"])
            fmin = np.where(fa_[:, :2] > 0, fa_[:, :2], np.inf).min(1) if fa_.ndim == 2 else np.ones(len(nf_h))
            easy = (wmf >= 0.6) & (fmin >= 0.15) & (od <= 0.2)
            print(f"  held-out 2-fibre sims: {int((nf_h == 2).sum())} total, {int(((nf_h == 2) & easy).sum())} 'easy' "
                  f"(WM ≥ 0.6, each fibre ≥ 0.15, ODI ≤ 0.2); fraction_array shape {fa_.shape}", flush=True)
            sel = np.where((nf_h == 2) & easy)[0] if args.in_model_easy else np.where(nf_h == 2)[0]
            sig = np.asarray(hold["signals"])[sel].astype(np.float64)
            b0 = gtab.bvals < 50; sig = sig / sig[:, b0].mean(1, keepdims=True)
            gts = hd[sel, :2]; cross = np.degrees(np.arccos(np.clip(np.abs(np.einsum("nj,nj->n", gts[:, 0], gts[:, 1])), 0, 1)))
            rng = np.random.default_rng(0); s = 1.0 / 30.0
            noisy = np.sqrt((sig + rng.normal(0, s, sig.shape)) ** 2 + rng.normal(0, s, sig.shape) ** 2)
            for nm, q in (("in-model clean", sig), ("in-model SNR30", noisy)):
                fit = model.fit(q.astype(np.float32)); pam = force_peaks(fit)
                pv = pam.peak_values; pv = pv / np.maximum(pv.sum(1, keepdims=True), 1e-12)
                e, rc = best_match(pam.peak_dirs, pv, gts)
                bins = [(0, 20), (20, 40), (40, 60), (60, 90)]; rows = {}
                for lo_, hi_ in bins:
                    m = (cross >= lo_) & (cross < hi_)
                    if m.sum() > 5: rows[f"{lo_}-{hi_}"] = best_match(pam.peak_dirs[m], pv[m], gts[m])
                print(f"  {nm:15s} ({len(sel)} 2-fibre held-out sims): err={e:.2f}° recall={100*rc:.1f}%  by crossing angle: "
                      + " ".join(f"{k}:{v[0]:.1f}°/{100*v[1]:.0f}%" for k, v in rows.items()), flush=True)
                r[nm] = {"err": e, "recall": rc, "rows": rows}
        results[str(n_sims)] = r
        if args.out: json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

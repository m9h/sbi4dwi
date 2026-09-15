#!/usr/bin/env python3
"""
Hybrid posterior (doc 008 §8 item 2 / §9): amortised flow proposal →
PRISM-JAX MAP refinement → Laplace fixel posterior.

Three initialisations of the same MAP + Laplace on the seam-free (tilted)
synthetic benchmark, K = 2, SNR 30:
  flow       mode-clustered posterior of the trained NPE (checkpoint from
             diagnose_flow_variants.py), read as (dirs, fracs) per voxel
  dict       the 300K-entry dictionary warm start used for DiSCo (doc 007 §8.3)
  none       PRISM's default init (uniform fractions, fixed seed directions)
and, for reference, the flow posterior alone. Reports angular error /
recall (best match), 90 % cone coverage of the Laplace posterior, median
σ_θ, and wall time per stage. Also runs the same three on the original
(seam) benchmark to show the refinement is seam-agnostic.
"""
import argparse, sys, time, importlib.util
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
sys.path.insert(0, str(Path(__file__).parent))
from diagnose_flow_variants import param_spec, build_flow, dirs_from_params_np
from diagnose_flow_rotated import rotated_benchmark
from validate_flow_sbi import summarise
from dmipy_jax.validation import prism_jax as pj, prism_uncertainty as pu
from dmipy_jax.validation import prism_synthetic as ps

spec = importlib.util.spec_from_file_location("vpdc", Path(__file__).with_name("validate_prism_disco_connectivity.py"))
vpdc = importlib.util.module_from_spec(spec); spec.loader.exec_module(vpdc)
TWO_PI = 2 * np.pi


def flow_proposal(flow, variant, lo, hi, y, n_samples, key):
    """(dirs (N,2,3), fracs (N,5), sigma (N,2)) from mode-clustered flow samples."""
    N = y.shape[0]; lo_j, span_j = jnp.asarray(lo), jnp.asarray(hi - lo)
    sample = jax.jit(lambda k, x: flow.sample(k, (n_samples,), condition=x) * span_j + lo_j)
    dirs = np.zeros((N, 2, 3)); fr = np.zeros((N, 5)); sig = np.zeros((N, 2)); fi = np.zeros(N)
    for i in range(N):
        key, sk = jax.random.split(key); s = np.asarray(sample(sk, jnp.asarray(y[i])))
        V = dirs_from_params_np(s, variant)
        th = np.arccos(np.clip(V[..., 2], -1, 1)); ph = np.arctan2(V[..., 1], V[..., 0]) % TWO_PI
        s2 = np.column_stack([th[:, 0], ph[:, 0], th[:, 1], ph[:, 1], s[:, 4], s[:, 5], s[:, 6]])
        d, f, sg, _, _ = summarise(s2); o = np.argsort(-f)
        dirs[i] = d[o]; f = np.clip(f[o], 0.02, 1.0); sig[i] = sg[o]; fi[i] = s[:, 6].mean()
        fr[i] = [0.02, 0.02, f[0], f[1], 0.02]; fr[i] /= fr[i].sum()
    return dirs, fr, sig, fi


def run_stage(name, b, init_dirs, init_fracs, cfg, bvals, bvecs, gt):
    t0 = time.time()
    fit = pj.fit_prism(b["data"], b["mask"], bvals, bvecs, cfg, init_dirs, init_fracs)
    t_map = time.time() - t0; t0 = time.time()
    post = pu.laplace_fixel_posterior(fit, b["data"], bvals, bvecs)
    t_lap = time.time() - t0
    err, rec = pj.angular_error_best_match(fit.dirs, fit.wm_fracs, gt)
    cov = pu.cone_coverage(post, gt, level=0.9)
    rows = {}
    for av in (15, 30, 45, 60, 90, 0):
        m = b["angle"] == av; e, r = pj.angular_error_best_match(fit.dirs[m], fit.wm_fracs[m], gt[m]); rows[av] = (e, r)
    print(f"  {name:12s} MAP+Laplace: err={err:.2f}° recall={100*rec:.1f}%  90% cone coverage={100*cov['coverage']:.1f}%  "
          f"median σ_θ={np.median(post.sigma_deg[:, 0]):.2f}°  f_i={fit.fintra.mean():.3f}  (MAP {t_map:.0f}s, Laplace {t_lap:.0f}s)  "
          + " ".join(f"{k}:{v[0]:.1f}" for k, v in rows.items()), flush=True)
    return {"err": err, "recall": rec, "coverage": cov["coverage"], "t_map": t_map, "t_lap": t_lap, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="hemi-2x"); ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=200); ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--tilts", type=float, nargs="+", default=[45, 0]); ap.add_argument("--snr", type=float, default=30)
    ap.add_argument("--out", default="validation/hybrid_posterior_results.npz")
    a = ap.parse_args()
    names, lo, hi = param_spec(a.variant)
    b0 = ps.make_benchmark(snr=None)
    flow = eqx.tree_deserialise_leaves(Path(f"validation/flow_variant_{a.variant}_{a.hidden}x{a.depth}.eqx"),
                                       build_flow(jax.random.key(0), 7, len(b0["bvals"]), a.hidden, a.depth, embed=a.variant.endswith("emb")))
    results = {}
    for tilt in a.tilts:
        b = rotated_benchmark(tilt, a.snr); bvals, bvecs = b["bvals"], b["bvecs"]
        gt = b["gt_dirs"].copy(); gt[b["angle"] == 0, 1] = 0
        y = b["data"][b["mask"]]; y = y / y[:, :1]
        print(f"\n=== tilt {tilt:.0f}°, SNR {a.snr:.0f}, {len(y)} voxels", flush=True)
        t0 = time.time(); fd, ff, fs, fi = flow_proposal(flow, a.variant, lo, hi, y, a.n_samples, jax.random.key(2)); t_flow = time.time() - t0
        e, r = pj.angular_error_best_match(fd, ff[:, 2:4], gt)
        print(f"  flow alone   : err={e:.2f}° recall={100*r:.1f}%  median σ_θ={np.median(fs[:,0]):.2f}°  ({t_flow:.0f}s)", flush=True)
        t0 = time.time(); dd, df = vpdc._warm_start({"data": b["data"], "mask": b["mask"]}, bvals, bvecs, 2, 300_000); t_dict = time.time() - t0
        e2, r2 = pj.angular_error_best_match(dd, df[:, 2:4], gt)
        print(f"  dict alone   : err={e2:.2f}° recall={100*r2:.1f}%  ({t_dict:.0f}s)", flush=True)
        cfg = pj.PrismConfig(n_fibres=2, n_iter=a.n_iter, loss="nll")
        res = {"flow_alone": (e, r, t_flow), "dict_alone": (e2, r2, t_dict)}
        res["flow"] = run_stage("flow-init", b, fd, ff, cfg, bvals, bvecs, gt)
        res["dict"] = run_stage("dict-init", b, dd, df, cfg, bvals, bvecs, gt)
        res["none"] = run_stage("no-init", b, None, None, cfg, bvals, bvecs, gt)
        results[tilt] = res
    np.savez(a.out, summary=np.array([[t, k, v["err"] if isinstance(v, dict) else v[0], v["recall"] if isinstance(v, dict) else v[1]]
                                      for t, r in results.items() for k, v in r.items()], dtype=object))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Posterior-sampled tractography on DiSCo (doc 007 §3.4): Laplace fixel
posterior from the best PRISM-plus fit → S direction samples → S connectomes
→ per-edge credible intervals. Reports r of the MAP and of the posterior
mean, CI width, and how many GT-zero edges have a CI that reaches zero."""
import argparse, time
from dataclasses import replace
import numpy as np, jax, nibabel as nib
from dipy.data import default_sphere
from dmipy_jax.validation.connectivity_metrics import summarize_method
from dmipy_jax.validation.force_disco import disco_subject_path, load_disco_subject
from dmipy_jax.validation.force_disco_connectivity import connectivity_pearson, load_gt_connectivity
from dmipy_jax.validation.prism_jax import PrismConfig, fit_prism
from dmipy_jax.validation import prism_uncertainty as pu
import importlib
drv = importlib.import_module("validate_prism_disco_connectivity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--n-fibres", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--peak-frac-min", type=float, default=0.10)
    ap.add_argument("--method", default="plus-tort-warm-nores")
    args = ap.parse_args()

    gt = load_gt_connectivity(subject=1)
    affine = nib.load(disco_subject_path(1) / "highRes_DiSCo1_DWI.nii.gz").affine
    out = load_disco_subject(subject=1, snr=args.snr, single_shell_b=None)
    data, mask, rois, gtab = out["data"], out["mask"], out["rois"], out["gtab"]
    bvals = np.asarray(gtab.bvals) * 1e6; bvecs = np.asarray(gtab.bvecs)
    cfg = drv._cfg_for(args.method, args.n_fibres, args.n_iter)
    init_dirs, init_fracs = drv._warm_start(out, bvals, bvecs, args.n_fibres, 500_000)
    t0 = time.time()
    fit = fit_prism(data, mask, bvals, bvecs, cfg, init_dirs, init_fracs)
    print(f"fit {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    post = pu.laplace_fixel_posterior(fit, data, bvals, bvecs)
    print(f"laplace {time.time()-t0:.0f}s; median σ_θ main fibre = {np.median(post.sigma_deg[:,0]):.2f}°, "
          f"90th pct = {np.percentile(post.sigma_deg[:,0], 90):.2f}°", flush=True)
    t0 = time.time()
    res = pu.posterior_connectivity(post, mask, rois, affine, default_sphere,
                                    jax.random.PRNGKey(0), n_samples=args.n_samples,
                                    peak_frac_min=args.peak_frac_min)
    print(f"tracking {args.n_samples+1} connectomes: {time.time()-t0:.0f}s", flush=True)
    iu = np.triu_indices(16, k=1)
    r_map = connectivity_pearson(res["map"], gt); r_mean = connectivity_pearson(res["mean"], gt)
    r_samples = [connectivity_pearson(c, gt) for c in res["samples"][1:]]
    gt_zero = gt[iu] == 0; gt_pos = ~gt_zero
    lo, hi, mean = res["lo"][iu], res["hi"][iu], res["mean"][iu]
    print(f"\nr(MAP) = {r_map:.4f}   r(posterior mean) = {r_mean:.4f}   "
          f"r over samples = {np.mean(r_samples):.4f} ± {np.std(r_samples):.4f} "
          f"[{np.min(r_samples):.4f}, {np.max(r_samples):.4f}]")
    rel_w = (hi - lo) / np.maximum(mean, 1)
    print(f"per-edge 90% CI relative width: median {np.median(rel_w[gt_pos]):.2f} on GT-positive edges, "
          f"{np.median(rel_w[gt_zero]):.2f} on GT-zero edges")
    print(f"GT-zero edges (n={gt_zero.sum()}): {np.mean(lo[gt_zero] == 0)*100:.0f}% have CI lower bound 0; "
          f"mean posterior-mean count {mean[gt_zero].mean():.1f} vs {mean[gt_pos].mean():.1f} on GT-positive")
    # edge-wise: does uncertainty separate FP from TP? rank GT-zero vs positive by CV
    cv = res["sd"][iu] / np.maximum(mean, 1e-6)
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(cv[gt_zero], cv[gt_pos], alternative="greater")
    print(f"coefficient of variation: median {np.median(cv[gt_zero]):.2f} (GT-zero) vs {np.median(cv[gt_pos]):.2f} (GT-positive); "
          f"Mann-Whitney p={u.pvalue:.2e} that FP edges are more uncertain")
    np.savez(f"validation/prism_posterior_disco_snr{args.snr}.npz", gt=gt, **{k: v for k, v in res.items() if v is not None},
             sigma_deg=post.sigma_deg, wm_fracs=post.wm_fracs, r_samples=np.array(r_samples), r_map=r_map, r_mean=r_mean)


if __name__ == "__main__":
    main()

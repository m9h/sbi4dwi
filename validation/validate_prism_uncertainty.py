#!/usr/bin/env python3
"""Calibration of the Laplace fixel posterior on PRISM's synthetic benchmark
(doc 007 §3.4): 90 % cone coverage per SNR and per crossing angle, and how
the angular uncertainty tracks the actual error."""
import argparse, time
import numpy as np
from dmipy_jax.validation import prism_jax as pj, prism_synthetic as ps, prism_uncertainty as pu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=float, nargs="+", default=[10, 30, 50])
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--level", type=float, default=0.9)
    args = ap.parse_args()
    out = {}
    for snr in args.snrs:
        b = ps.make_benchmark(snr=snr)
        cfg = pj.PrismConfig(n_fibres=2, n_iter=args.n_iter, loss="nll")
        t0 = time.time()
        fit = pj.fit_prism(b["data"], b["mask"], b["bvals"], b["bvecs"], cfg)
        post = pu.laplace_fixel_posterior(fit, b["data"], b["bvals"], b["bvecs"])
        gt = b["gt_dirs"].copy(); gt[b["angle"] == 0, 1] = 0.0
        cov = pu.cone_coverage(post, gt, level=args.level)
        # per-angle coverage + median sigma + median actual error
        rows = []
        for a in list(ps.ANGLES) + [0]:
            m = b["angle"] == a
            g = gt[m]
            sub = pu.FixelPosterior(post.dirs[m], post.cov[m], post.e1[m], post.e2[m],
                                    post.sigma_deg[m], post.wm_fracs[m], None)
            c = pu.cone_coverage(sub, g, level=args.level)
            err, _ = pj.angular_error_best_match(post.dirs[m], post.wm_fracs[m], g)
            rows.append((a, c["coverage"], float(np.median(post.sigma_deg[m][:, 0])), err))
        out[snr] = {"coverage": cov["coverage"], "rows": rows}
        print(f"\nSNR={snr:.0f}  overall {100*args.level:.0f}% cone coverage = {100*cov['coverage']:.1f}% "
              f"(n={cov['n']}, {time.time()-t0:.0f}s)")
        print("  angle  coverage  median σ_θ(main)  mean err")
        for a, c, s, e in rows:
            print(f"  {a:>5d}  {100*c:7.1f}%  {s:14.2f}°  {e:7.2f}°")
    np.savez("validation/prism_uncertainty_calibration.npz",
             **{f"snr{int(s)}_rows": np.array(v["rows"]) for s, v in out.items()},
             **{f"snr{int(s)}_coverage": v["coverage"] for s, v in out.items()})


if __name__ == "__main__":
    main()

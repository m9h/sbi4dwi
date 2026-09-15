#!/usr/bin/env python3
"""
Misspecification benchmark (doc 007 §3.5): PRISM-JAX vs PRISM-plus vs
MSMT-CSD on Monte Carlo signals from CATERPillar axon substrates.

Substrates: crossing angles × {straight, tortuous+beaded}. Scheme: PRISM's
3 shells × 64 dirs with DiSCo timing (δ=17.74 ms, Δ=35.78 ms). SNR 30,
8×8 voxels per condition with independent noise. Reports angular error,
recall, and f_i vs the measured intra-cellular fraction.
"""
import argparse, time
from dataclasses import replace
import numpy as np, jax
from dmipy_jax.validation import substrate_benchmark as sb
from dmipy_jax.validation import prism_jax as pj
from dmipy_jax.validation.prism_synthetic import prism_scheme


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 30, 45, 60, 90])
    ap.add_argument("--geoms", nargs="+", default=["straight", "tortuous"])
    ap.add_argument("--snr", type=float, default=30)
    ap.add_argument("--icvf", type=float, default=0.5)
    ap.add_argument("--box-um", type=float, default=10.0)
    ap.add_argument("--n-particles", type=int, default=4000)
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--methods", nargs="+",
                    default=["prism-nll", "plus-x", "plus-x-disp", "select", "msmt"])
    ap.add_argument("--save-signals", default=None,
                    help="write every condition's clean MC signal, noisy volume, GT axes and "
                         "f_intra to this .npz (for external methods: FORCE, SBI_dMRI)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bvals, bvecs = prism_scheme()
    results = {}
    saved = {"bvals": bvals, "bvecs": bvecs, "snr": args.snr}
    for geom in args.geoms:
        df = sb.caterpillar_bundle(icvf=args.icvf, box_um=args.box_um,
                                   tortuous=int(geom == "tortuous"),
                                   beading=0.3 if geom == "tortuous" else 0.0)
        print(f"\n### {geom}: {len(df)} spheres, {df['id'].nunique()} axons, "
              f"axis {np.round(sb.bundle_axis(df), 3)}", flush=True)
        sf_data = None          # single-bundle voxels → MSMT response function
        for ang in args.angles:
            sub = sb.make_crossing_substrate(df, ang, args.box_um)
            t0 = time.time()
            S, S_in, S_ex = sb.simulate_substrate_signal(sub, bvals, bvecs, n_particles=args.n_particles)
            print(f"  angle {ang:>4.0f}: f_intra={sub.f_intra:.3f}  MC {time.time()-t0:.0f}s  "
                  f"S(b3000) mean intra={S_in[-64:].mean():.3f} extra={S_ex[-64:].mean():.3f}", flush=True)
            data, mask = sb.build_volume(S, (8, 8, 1), args.snr, seed=args.seed)
            N = 64
            if ang == 0:
                sf_data = data
            saved[f"{geom}_{int(ang)}_signal"] = S
            saved[f"{geom}_{int(ang)}_data"] = data
            saved[f"{geom}_{int(ang)}_axes"] = sub.axes
            saved[f"{geom}_{int(ang)}_f_intra"] = sub.f_intra
            if args.save_signals:
                np.savez(args.save_signals, **saved)
            gt = np.zeros((N, 2, 3)); gt[:, 0] = sub.axes[0]
            if ang > 0:
                gt[:, 1] = sub.axes[1]
            for meth in args.methods:
                t0 = time.time()
                if meth == "msmt":
                    from dipy.core.gradients import gradient_table
                    from dipy.data import default_sphere
                    from dmipy_jax.validation.msmt_baseline import msmt_csd_pam
                    gtab = gradient_table(bvals / 1e6, bvecs=bvecs)
                    # oracle response: WM response estimated from the *single-bundle*
                    # voxels of the same geometry (appended as a second slice)
                    if sf_data is None:
                        raise RuntimeError("run angle 0 first so the MSMT response can be estimated")
                    both = np.concatenate([data, sf_data], axis=2)          # (8,8,2,M)
                    m_both = np.ones(both.shape[:3], bool)
                    wm = np.zeros_like(m_both); wm[:, :, 1] = True
                    pam, _ = msmt_csd_pam(both, gtab, m_both, default_sphere, wm_mask=wm)
                    sel = np.zeros_like(m_both); sel[:, :, 0] = True
                    dirs = pam.peak_dirs[sel]; v = pam.peak_values[sel]
                    fr = v / np.maximum(v.sum(1, keepdims=True), 1e-12); fi = np.full(N, np.nan)
                elif meth == "select":
                    # per-voxel Laplace-evidence selection between plus-x and plus-x-disp
                    from dmipy_jax.validation import prism_uncertainty as pu
                    cfgx = replace(pj.PrismConfig(n_fibres=2, n_iter=args.n_iter, loss="nll"),
                                   learn_diffusivities=True, tortuosity=True, use_restricted=False,
                                   separate_extra_dpar=True, lam_diffusivity_prior=1.0,
                                   diffusivity_prior_centre=2.0e-9, diffusivity_prior_sd=0.2, d_par=2.0e-9)
                    fx = pj.fit_prism(data, mask, bvals, bvecs, cfgx)
                    fd = pj.fit_prism(data, mask, bvals, bvecs, replace(cfgx, disperse=True))
                    out = pu.select_per_voxel([fx, fd], data, bvals, bvecs)
                    dirs, fr, fi = out["dirs"], out["wm_fracs"], out["fintra"]
                    cfg = cfgx; fit = fx
                    extra_sel = f"  chose disp in {100*np.mean(out['choice']==1):.0f}% of voxels"
                else:
                    cfg = pj.PrismConfig(n_fibres=2, n_iter=args.n_iter, loss="nll")
                    if meth.startswith("plus"):
                        cfg = replace(cfg, learn_diffusivities=True, tortuosity=True,
                                      use_restricted=False, d_par=1.7e-9,
                                      disperse=meth.endswith("disp") or meth.endswith("disp-noiso"),
                                      lam_diffusivity_prior=1.0 if meth == "plus-dprior" else 0.0)
                    if meth.endswith("-noiso"):
                        cfg = replace(cfg, use_isotropic=False)
                    if meth.startswith("plus-x"):
                        # decoupled extra-cellular D∥ + intra D∥ anchored at the
                        # intrinsic 2.0 µm²/ms (a physical bound, not a data-set fact)
                        cfg = replace(cfg, separate_extra_dpar=True, lam_diffusivity_prior=1.0,
                                      diffusivity_prior_centre=2.0e-9, diffusivity_prior_sd=0.2,
                                      d_par=2.0e-9)
                    fit = pj.fit_prism(data, mask, bvals, bvecs, cfg)
                    dirs, fr, fi = fit.dirs, fit.wm_fracs, fit.fintra
                err, rec = pj.angular_error_best_match(dirs, fr, gt)
                extra = "" if np.isnan(fi).all() else f"  f_i={np.nanmean(fi):.3f} (geom {sub.f_intra:.3f})"
                if meth == "select":
                    extra += extra_sel
                if meth not in ("msmt", "select") and cfg.learn_diffusivities:
                    extra += f"  D∥={fit.d_par*1e9:.2f}"
                    if cfg.separate_extra_dpar:
                        extra += f"  D∥ex={fit.d_par_extra*1e9:.2f}"
                print(f"    {meth:10s} err={err:6.2f}°  recall={100*rec:5.1f}%{extra}  ({time.time()-t0:.0f}s)", flush=True)
                results[(geom, ang, meth)] = (err, rec, float(np.nanmean(fi)) if not np.isnan(fi).all() else np.nan, sub.f_intra)
    np.savez("validation/prism_substrate_results.npz",
             keys=np.array([f"{g}|{a}|{m}" for (g, a, m) in results]),
             vals=np.array(list(results.values())))


if __name__ == "__main__":
    main()

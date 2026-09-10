#!/usr/bin/env python3
"""
PRISM-JAX vs MSMT-CSD vs dmipy-JAX §24 on DiSCo connectivity (doc 006 §4).

All methods go through the identical §21.2 tracker
(``dmipy_jax.validation.disco_tracking.track_connectivity``), so the only
variable is the per-voxel peaks. Reports Pearson r / CCC / Dice per SNR
and, crucially, each method's margin over MSMT-CSD — the quantity PRISM
reports (+1.6 pp at its best tracking angle) and the only one comparable
across papers, since absolute r depends on unstated connectivity
normalisation (doc 004 §18–§19).

Methods:
  msmt        dipy MSMT-CSD, oracle GM/CSF responses (PRISM's baseline)
  prism-mse   faithful PRISM, MSE loss, fixed D∥=1.7 D⊥=0.4, random init
  prism-nll   faithful PRISM, Rician NLL with learned σ
  plus-nll    PRISM-plus: NLL + learnable D∥/D⊥ (DiSCo needs ~0.6/0.35)
  plus-warm   PRISM-plus: NLL + learnable D + warm start from §22 library

Usage:
  uv run python validation/validate_prism_disco_connectivity.py \
      --methods msmt prism-nll plus-nll --snrs 10 30 50 --n-fibres 5
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from dmipy_jax.validation.connectivity_metrics import summarize_method
from dmipy_jax.validation.disco_tracking import track_connectivity
from dmipy_jax.validation.force_disco import disco_subject_path, load_disco_subject
from dmipy_jax.validation.force_disco_connectivity import (
    connectivity_pearson, load_gt_connectivity,
)
from dmipy_jax.validation.prism_jax import PrismConfig, fit_prism, prism_fit_to_pam

REF = {  # doc 004 §24.2 headline (§22 multi-shell) and PRISM paper
    "dmipy_s24": {10: 0.772, 30: 0.811, 50: 0.851},
    "prism_paper_disco": 0.934, "msmt_paper_disco": 0.920,
}


def _cfg_for(method: str, n_fibres: int, n_iter: int, seed: int = 0) -> PrismConfig:
    base = PrismConfig(n_fibres=n_fibres, n_iter=n_iter, seed=seed)
    return {
        "prism-mse": base,
        "prism-nll": replace(base, loss="nll"),
        # PRISM run the way the FORCE paper §3.2 retunes for DiSCo: fixed
        # diffusivities moved to the phantom's regime (D∥=0.6, D⊥=0.35).
        "prism-tuned-nll": replace(base, loss="nll", d_par=0.6e-9, d_perp=0.35e-9),
        "plus-nll": replace(base, loss="nll", learn_diffusivities=True,
                            d_par=0.6e-9, d_perp=0.35e-9),
        "plus-warm": replace(base, loss="nll", learn_diffusivities=True,
                             d_par=0.6e-9, d_perp=0.35e-9),
        "plus-tort": replace(base, loss="nll", learn_diffusivities=True,
                             d_par=0.6e-9, tortuosity=True),
        "plus-tort-warm": replace(base, loss="nll", learn_diffusivities=True,
                                  d_par=0.6e-9, tortuosity=True),
        # + Watson dispersion per fibre (learned ODI), tortuosity, warm start
        "plus-disp-warm": replace(base, loss="nll", learn_diffusivities=True,
                                  d_par=0.6e-9, tortuosity=True, disperse=True),
        # ablations: no restricted pool (PRISM: −9.4 pp on DiSCo)
        "prism-tuned-nll-nores": replace(base, loss="nll", d_par=0.6e-9, d_perp=0.35e-9,
                                         use_restricted=False),
        "prism-nll-nores": replace(base, loss="nll", use_restricted=False),
        "plus-tort-warm-nores": replace(base, loss="nll", learn_diffusivities=True,
                                        d_par=0.6e-9, tortuosity=True, use_restricted=False),
        # + weak prior on D∥ (λ=1, sd=0.3 log units) against low-SNR drift
        "plus-dprior-warm": replace(base, loss="nll", learn_diffusivities=True,
                                    d_par=0.6e-9, tortuosity=True,
                                    lam_diffusivity_prior=1.0),
    }[method]


def _warm_start(out, acq_bvals, acq_bvecs, n_fibres, library_size):
    """§22 DictionaryMatcher → (init_dirs, init_fracs) for PRISM-plus."""
    import jax, jax.numpy as jnp
    from dmipy_jax.acquisition import JaxAcquisition
    from dmipy_jax.library.generator import LibraryGenerator
    from dmipy_jax.library.matcher import DictionaryMatcher
    from dmipy_jax.library.storage import SimulationLibrary
    from dmipy_jax.validation.dmipy_disco_dict import (
        build_disco_tuned_3d_stick_zeppelin_simulator,
    )
    acq = JaxAcquisition(bvalues=jnp.asarray(acq_bvals),
                         gradient_directions=jnp.asarray(acq_bvecs))
    sim = build_disco_tuned_3d_stick_zeppelin_simulator(acq)
    params, signals = LibraryGenerator(sim, chunk_size=20_000).generate(
        library_size, key=jax.random.PRNGKey(1))
    lib = SimulationLibrary(params=params, signals=signals,
                            parameter_names=sim.parameter_names)
    maps = DictionaryMatcher(lib, k_best=10).match_volume(
        out["data"], mask=out["mask"], batch_size=4096)
    m = out["mask"]
    def sph(t, p):
        return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)
    N = int(m.sum())
    dirs = np.random.default_rng(0).normal(size=(N, n_fibres, 3))
    dirs[:, 0] = sph(maps["theta1"][m], maps["phi1"][m])
    if n_fibres > 1:
        dirs[:, 1] = sph(maps["theta2"][m], maps["phi2"][m])
    f1, fiso = maps["f1"][m], maps["f_iso"][m]
    f2 = np.clip(1 - f1 - fiso, 0, 1)
    fr = np.full((N, n_fibres + 3), 0.02)
    fr[:, 0] = 0.02; fr[:, 1] = 0.02; fr[:, -1] = fiso
    fr[:, 2] = f1
    if n_fibres > 1:
        fr[:, 3] = f2
    fr /= fr.sum(1, keepdims=True)
    return dirs, fr


def run_method(method, out, affine, n_fibres, n_iter, library_size,
               peak_frac_min=0.05, max_angle=45.0, seed=0):
    from dipy.data import default_sphere
    data, mask, rois, gtab = out["data"], out["mask"], out["rois"], out["gtab"]
    t0 = time.time()
    extra = {}
    if method == "msmt":
        from dmipy_jax.validation.msmt_baseline import msmt_csd_pam
        pam, info = msmt_csd_pam(data, gtab, mask, default_sphere)
        extra["n_wm_voxels"] = info["n_wm_voxels"]
    else:
        bvals = np.asarray(gtab.bvals) * 1e6
        bvecs = np.asarray(gtab.bvecs)
        cfg = _cfg_for(method, n_fibres, n_iter, seed)
        init_dirs = init_fracs = None
        if method in ("plus-warm", "plus-tort-warm", "plus-dprior-warm", "plus-disp-warm",
                      "plus-tort-warm-nores"):
            init_dirs, init_fracs = _warm_start(out, bvals, bvecs, n_fibres, library_size)
        fit = fit_prism(data, mask, bvals, bvecs, cfg, init_dirs, init_fracs)
        pam = prism_fit_to_pam(fit, default_sphere, peak_frac_min=peak_frac_min,
                               affine=affine)
        extra.update(d_par=fit.d_par, d_perp=fit.d_perp, sigma=fit.sigma,
                     fintra_mean=float(fit.fintra.mean()),
                     final_loss=float(fit.loss_history[-1]),
                     n_peaks_mean=float((fit.wm_fracs >= peak_frac_min).sum(1).mean()),
                     **({"odi_mean": float(fit.odi[:, 0].mean())} if fit.odi is not None else {}))
    t_fit = time.time() - t0
    res = track_connectivity(pam, mask, rois, affine, max_angle=max_angle,
                             random_seed=seed)
    res.update(extra, t_fit=t_fit)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["msmt", "prism-mse", "prism-nll", "prism-tuned-nll", "plus-nll"])
    ap.add_argument("--snrs", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--n-fibres", type=int, default=5, help="PRISM uses K=5 on DiSCo")
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--library-size", type=int, default=500_000)
    ap.add_argument("--single-shell", action="store_true")
    ap.add_argument("--peak-frac-min", type=float, default=0.05)
    ap.add_argument("--max-angle", type=float, default=45.0,
                    help="PRISM sweeps 15-30 and reports the best; §21 used 45")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="fit + tracking seeds; results keyed (method, snr) use the "
                         "mean r over seeds and store per-seed r")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import nibabel as nib
    gt = load_gt_connectivity(subject=1)
    affine = nib.load(disco_subject_path(1) / "highRes_DiSCo1_DWI.nii.gz").affine
    ssb = 1900 if args.single_shell else None

    results = {}
    for snr in args.snrs:
        out = load_disco_subject(subject=1, snr=snr, single_shell_b=ssb)
        for m in args.methods:
            print(f"\n=== {m} @ SNR={snr} ===", flush=True)
            per_seed = []
            for sd in args.seeds:
                res = run_method(m, out, affine, args.n_fibres, args.n_iter,
                                 args.library_size, args.peak_frac_min, args.max_angle, sd)
                res["r"] = connectivity_pearson(res["connectivity"], gt)
                per_seed.append(res)
                if len(args.seeds) > 1:
                    print(f"  seed {sd}: r={res['r']:.4f}", flush=True)
            res = per_seed[0]
            res["r_seeds"] = np.array([p["r"] for p in per_seed])
            res["r"] = float(res["r_seeds"].mean())
            res["r_sd"] = float(res["r_seeds"].std(ddof=1)) if len(per_seed) > 1 else 0.0
            res["connectivity"] = np.mean([p["connectivity"] for p in per_seed], axis=0)
            results[(m, snr)] = res
            print(f"  r={res['r']:.4f}±{res['r_sd']:.4f}  streamlines={res['streamlines_count']:,}  "
                  f"fit {res['t_fit']:.0f}s  " +
                  "  ".join(f"{k}={v:.3g}" for k, v in res.items()
                            if isinstance(v, float) and k not in ("r", "t_fit")),
                  flush=True)

    # summary table with margin over MSMT
    print("\n" + "=" * 78)
    print(f"{'method':<11}{'SNR':>4}{'r':>8}{'CCC':>8}{'Dice':>7}{'Δ vs msmt':>11}{'§24':>7}")
    print("=" * 78)
    summaries = {}
    for m in args.methods:
        cm = {s: results[(m, s)]["connectivity"] for s in args.snrs}
        summaries[m] = summarize_method(cm, gt, threshold=0.0)
    for s in args.snrs:
        for m in args.methods:
            sm = summaries[m][s]
            d = (sm["pearson_r"] - summaries["msmt"][s]["pearson_r"]
                 if "msmt" in summaries else float("nan"))
            print(f"{m:<11}{s:>4}{sm['pearson_r']:>8.4f}{sm['lin_ccc_sumnorm']:>8.3f}"
                  f"{sm['dice']:>7.3f}{d:>+11.4f}{REF['dmipy_s24'].get(s, float('nan')):>7.3f}")
    print(f"\nPRISM paper: r={REF['prism_paper_disco']} vs MSMT {REF['msmt_paper_disco']} "
          f"(+{100*(REF['prism_paper_disco']-REF['msmt_paper_disco']):.1f} pp) at SNR=50, K=5.")

    tag = f"_{args.tag}" if args.tag else ""
    outp = Path(f"validation/prism_disco_connectivity_results{tag}.npz")
    np.savez(outp, methods=np.array(args.methods), snrs=np.array(args.snrs), gt=gt,
             **{f"cmat_{m}_snr{s}": results[(m, s)]["connectivity"]
                for m in args.methods for s in args.snrs},
             **{f"r_{m}": np.array([results[(m, s)]["r"] for s in args.snrs])
                for m in args.methods},
             **{f"rseeds_{m}_snr{s}": results[(m, s)]["r_seeds"]
                for m in args.methods for s in args.snrs})
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DiSCo connectivity under the FORCE authors' *published* protocol
(Atharva-Shah-2298/FORCE experiments/force_disco.py, 2026-08-18; doc 007 §8).

Protocol, reproduced exactly:
  data      : all shells with b < 3100 s/mm² (b=13190 dropped) → 3 shells
  FORCE     : FORCEModel(gtab, compute_odf=True); generate(1M sims,
              wm_threshold=0, odi_range=(0.01, 0.15),
              diffusivity_config = ex-vivo ranges D∥ 0.4–0.9, D⊥ 0.1–0.4,
              GM 0.7–1.2, CSF 3.0 ×1e-3 mm²/s); force_peaks(fit, mask)
  seeds     : ROIs-mask ∩ brain mask, binary-eroded ×1, density 2
  stopping  : brain mask only
  tracking  : eudx_tracking(step 0.5, random_seed 42, eudx defaults:
              max_angle 60, pmf_threshold 0.0239), keep len > 1
  connectome: connectivity_matrix(streamlines, affine, labels)[1:, 1:]
  metric    : Pearson r on the lower triangle vs Cross-Sectional-Area GT

Every method below is pushed through the *same* seeds / stopping /
tracking; only the peaks differ. MSMT-CSD and PRISM-plus-x get the same
3-shell data. PRISM-plus additionally reports the posterior-mean and
CV-pruned connectomes (doc 007 §6.17–6.18) under this tracker.

dipy 1.12.1 note: FORCEModel.generate has no `seed` (added on dipy master
in #4130); the library draw is therefore not bit-reproducible here.
"""
import argparse, time
from dataclasses import replace
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import eudx_tracking
from dipy.tracking.utils import connectivity_matrix, seeds_from_mask

from dmipy_jax.validation.connectivity_metrics import connectivity_dice_f1
from dmipy_jax.validation.force_disco import disco_subject_path

DISCO = disco_subject_path(1)


def load_protocol_data(snr: int):
    bvals, bvecs = read_bvals_bvecs(str(DISCO / "DiSCo_gradients.bvals"),
                                    str(DISCO / "DiSCo_gradients_dipy.bvecs"))
    keep = bvals < 3100
    gtab = gradient_table(bvals[keep], bvecs=bvecs[keep])
    img = nib.load(DISCO / f"highRes_DiSCo1_DWI_RicianNoise-snr{snr}.nii.gz")
    data = np.asarray(img.dataobj)[..., keep].astype(np.float32)
    affine = img.affine
    mask = nib.load(DISCO / "highRes_DiSCo1_mask.nii.gz").get_fdata().astype(bool)
    labels = nib.load(DISCO / "highRes_DiSCo1_ROIs.nii.gz").get_fdata().astype(int)
    rois_mask = nib.load(DISCO / "highRes_DiSCo1_ROIs-mask.nii.gz").get_fdata()
    gt = np.loadtxt(DISCO / "DiSCo1_Connectivity_Matrix_Cross-Sectional_Area.txt")
    seed_mask = binary_erosion(rois_mask * mask, iterations=1)
    seeds = seeds_from_mask(seed_mask, affine, density=2)
    return dict(data=data, gtab=gtab, affine=affine, mask=mask, labels=labels,
                seeds=seeds, gt=gt, bvals=bvals[keep], bvecs=bvecs[keep])


def track_protocol(pam, d):
    sl = Streamlines(eudx_tracking(d["seeds"], BinaryStoppingCriterion(d["mask"]), d["affine"],
                                   pam=pam, step_size=0.5, random_seed=42))
    sl = sl[np.array([len(s) for s in sl]) > 1]
    cm = connectivity_matrix(sl, d["affine"], d["labels"])[1:, 1:].astype(float)
    return cm, len(sl)


def score(cm, gt):
    lt = np.tril(np.ones(gt.shape), -1) > 0
    r, _ = pearsonr(gt[lt], cm[lt])
    dsc = connectivity_dice_f1(cm, gt)
    return r, dsc


def run_force(d, num_sims, num_cpus, diffusivity="range"):
    from dipy.reconst.force import FORCEModel, force_peaks
    cfg = {"wm_d_par_range": (0.4e-3, 0.9e-3), "wm_d_perp_range": (0.1e-3, 0.4e-3),
           "gm_d_iso_range": (0.7e-3, 1.2e-3), "csf_d": 3.0e-3} if diffusivity == "range" else \
          {"wm_d_par_range": 0.6e-3, "wm_d_perp_range": 0.3e-3,
           "gm_d_iso_range": (0.7e-3, 1.2e-3), "csf_d": 3.0e-3}
    model = FORCEModel(d["gtab"], compute_odf=True)
    model.generate(num_simulations=num_sims, num_cpus=num_cpus, wm_threshold=0.0,
                   odi_range=(0.01, 0.15), diffusivity_config=cfg, use_cache=True)
    fit = model.fit(d["data"], mask=d["mask"])
    return force_peaks(fit, mask=d["mask"])


def run_msmt(d):
    from dmipy_jax.validation.msmt_baseline import msmt_csd_pam
    pam, _ = msmt_csd_pam(d["data"], d["gtab"], d["mask"], default_sphere)
    pam.affine = d["affine"]
    return pam


def run_prism_plus(d, n_fibres, n_iter, library_size, n_samples, key=0):
    import jax
    from dmipy_jax.validation import prism_uncertainty as pu
    from dmipy_jax.validation.prism_jax import PrismConfig, fit_prism, prism_fit_to_pam
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "vpdc", Path(__file__).with_name("validate_prism_disco_connectivity.py"))
    drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
    bvals_si = np.asarray(d["bvals"]) * 1e6; bvecs = np.asarray(d["bvecs"])
    out = {"data": d["data"], "mask": d["mask"]}
    init_dirs, init_fracs = drv._warm_start(out, bvals_si, bvecs, n_fibres, library_size)
    cfg = replace(PrismConfig(n_fibres=n_fibres, n_iter=n_iter, loss="nll", learn_diffusivities=True,
                              d_par=0.6e-9, tortuosity=True, use_restricted=False,
                              separate_extra_dpar=True))
    fit = fit_prism(d["data"], d["mask"], bvals_si, bvecs, cfg, init_dirs, init_fracs)
    pam = prism_fit_to_pam(fit, default_sphere, peak_frac_min=0.10, affine=d["affine"])
    post = pu.laplace_fixel_posterior(fit, d["data"], bvals_si, bvecs)
    # posterior direction samples → connectomes under the protocol tracker
    S = post.sample_dirs(jax.random.PRNGKey(key), n_samples)
    from dmipy_jax.validation.disco_tracking import peaks_to_pam
    idx = np.argwhere(d["mask"]); K = post.dirs.shape[1]
    cms = []
    for s in range(n_samples):
        pd = np.zeros(d["mask"].shape + (5, 3)); pv = np.zeros(d["mask"].shape + (5,))
        for n, (i, j, k) in enumerate(idx):
            for f in range(min(K, 5)):
                v = post.wm_fracs[n, f]
                if v >= 0.10:
                    pd[i, j, k, f] = S[s, n, f]; pv[i, j, k, f] = v
        cms.append(track_protocol(peaks_to_pam(pd, pv, d["mask"], default_sphere, d["affine"]), d)[0])
    C = np.stack(cms)
    res = {"mean": C.mean(0), "sd": C.std(0), "lo": np.percentile(C, 5, axis=0)}
    return pam, res, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[50, 10])
    ap.add_argument("--methods", nargs="+", default=["force", "force-fixed", "msmt", "prism-plus"])
    ap.add_argument("--num-sims", type=int, default=1_000_000)
    ap.add_argument("--num-cpus", type=int, default=16)
    ap.add_argument("--n-fibres", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--library-size", type=int, default=500_000)
    ap.add_argument("--n-samples", type=int, default=20)
    args = ap.parse_args()

    from dmipy_jax.validation import prism_uncertainty as pu
    rows = []
    for snr in args.snrs:
        d = load_protocol_data(snr)
        print(f"\n##### SNR {snr}: {d['data'].shape[-1]} volumes, {len(d['seeds'])} seeds", flush=True)
        for m in args.methods:
            t0 = time.time()
            if m == "force":
                pam = run_force(d, args.num_sims, args.num_cpus, "range")
            elif m == "force-fixed":
                pam = run_force(d, args.num_sims, args.num_cpus, "fixed")
            elif m == "msmt":
                pam = run_msmt(d)
            elif m == "prism-plus":
                pam, post, fit = run_prism_plus(d, args.n_fibres, args.n_iter, args.library_size, args.n_samples)
            cm, n = track_protocol(pam, d)
            r, dsc = score(cm, d["gt"])
            print(f"  {m:12s} r={r:.4f}  Dice={dsc['dice']:.3f}  FP={dsc['fp']}  streamlines={n:,}  ({time.time()-t0:.0f}s)", flush=True)
            rows.append((snr, m, r, dsc["dice"]))
            if m == "prism-plus":
                for label, c in (("  + posterior mean", post["mean"]),
                                 ("  + CV<0.3 pruned", pu.threshold_connectome(post, "cv", cv_max=0.3))):
                    r2, d2 = score(c, d["gt"])
                    print(f"  {label:18s} r={r2:.4f}  Dice={d2['dice']:.3f}  FP={d2['fp']}", flush=True)
                    rows.append((snr, m + label.strip(), r2, d2["dice"]))
                print(f"    D∥={fit.d_par*1e9:.2f} D∥ex={fit.d_par_extra*1e9:.2f} f_i={fit.fintra.mean():.3f}", flush=True)
    np.savez("validation/disco_force_protocol_results.npz",
             rows=np.array([(str(a), str(b), c, e) for a, b, c, e in rows]))
    print("\nPaper §3.2 reports: FORCE r = 0.868 (SNR 10), 0.894 (SNR 50); CSD 0.847.")


if __name__ == "__main__":
    main()

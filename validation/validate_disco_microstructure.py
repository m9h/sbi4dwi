#!/usr/bin/env python3
"""
DiSCo intra-axonal volume fraction under the FORCE-paper protocol data
(3 shells ≤ 3100), doc 008 §8 item 1: does the compartment-attribution fix
of §7.2 turn into a headline number against FORCE (tuned r = 0.918,
default r = 0.679, doc 005 §2) *without* per-dataset retuning?

Variants of the PRISM-JAX plus-x fit (warm-started as in doc 007 §8.3):
  plus-x            as run for the connectome (CSF + GM balls, tortuosity, decoupled D∥,ex)
  no-iso            CSF/GM disabled (DiSCo is strands + extra-strand space only)
  no-iso-tscale     + learnable global tortuosity scale s in D⊥ = s·D∥,ex·(1 − f_i)
  no-iso-no-tort    + free D⊥ instead of tortuosity
  fixed-D           PRISM's fixed D∥ = 1.7 / D⊥ = 0.4 (as published, in-vivo values)
  fixed-D-disco     PRISM with the DiSCo-tuned D∥ = 0.6 / D⊥ = 0.3 (the retune FORCE needs)
Reports Pearson r and mean bias of f_i (WM-internal) and of the voxel-level
intra fraction f_i·f_wm + f_res against Strand_Intra_Volume_Fraction, in
the mask, and the fitted global diffusivities.
"""
import argparse, json, time, importlib.util
from dataclasses import replace
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.stats import pearsonr
from dmipy_jax.validation.force_disco import disco_subject_path
from dmipy_jax.validation.prism_jax import PrismConfig, fit_prism

spec = importlib.util.spec_from_file_location("vdfp", Path(__file__).with_name("validate_disco_force_protocol.py"))
vdfp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vdfp)
spec2 = importlib.util.spec_from_file_location("vpdc", Path(__file__).with_name("validate_prism_disco_connectivity.py"))
vpdc = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(vpdc)


def variants(n_iter, learn_dpar=0.6e-9):
    base = replace(PrismConfig(n_fibres=3, n_iter=n_iter, loss="nll"), learn_diffusivities=True, d_par=learn_dpar,
                   tortuosity=True, use_restricted=False, separate_extra_dpar=True)
    return {
        "plus-x": base,
        "no-iso": replace(base, use_isotropic=False),
        "no-iso-tscale": replace(base, use_isotropic=False, tortuosity_scale=True),
        "no-iso-no-tort": replace(base, use_isotropic=False, tortuosity=False),
        "fixed-D": PrismConfig(n_fibres=3, n_iter=n_iter, loss="nll"),
        "fixed-D-disco": PrismConfig(n_fibres=3, n_iter=n_iter, loss="nll", d_par=0.6e-9, d_perp=0.3e-9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[30, 10])
    ap.add_argument("--n-iter", type=int, default=300); ap.add_argument("--library-size", type=int, default=300_000)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--out", default="validation/disco_microstructure_results.json")
    a = ap.parse_args()
    gt_vf = nib.load(disco_subject_path(1) / "highRes_DiSCo1_Strand_Intra_Volume_Fraction.nii.gz").get_fdata()
    results = {}
    for snr in a.snrs:
        d = vdfp.load_protocol_data(snr)
        bvals_si = np.asarray(d["bvals"]) * 1e6; bvecs = np.asarray(d["bvecs"]); mask = d["mask"]
        g = gt_vf[mask]
        print(f"\n=== SNR {snr}: {int(mask.sum()):,} voxels, GT intra-VF mean {g.mean():.3f} sd {g.std():.3f}", flush=True)
        t0 = time.time()
        init_dirs, init_fracs = vpdc._warm_start({"data": d["data"], "mask": mask}, bvals_si, bvecs, 3, a.library_size)
        print(f"  warm start {time.time()-t0:.0f}s", flush=True)
        res = {}
        for name, cfg in variants(a.n_iter).items():
            if a.variants and name not in a.variants: continue
            t0 = time.time()
            fit = fit_prism(d["data"], mask, bvals_si, bvecs, cfg, init_dirs, init_fracs)
            fwm = fit.wm_fracs.sum(1); fiso = fit.fracs[:, 0] + fit.fracs[:, 1]; fres = fit.fracs[:, -1]
            fvox = fit.fintra * fwm + fres
            r_i = pearsonr(fit.fintra, g)[0]; r_v = pearsonr(fvox, g)[0]
            res[name] = {"r_fintra": float(r_i), "bias_fintra": float((fit.fintra - g).mean()),
                         "r_voxel": float(r_v), "bias_voxel": float((fvox - g).mean()),
                         "f_iso_mean": float(fiso.mean()), "f_res_mean": float(fres.mean()),
                         "dpar": float(fit.d_par * 1e9), "dpar_ex": float((fit.d_par_extra or fit.d_par) * 1e9),
                         "dperp": float(np.mean(fit.d_perp) * 1e9), "sigma": float(fit.sigma or 0)}
            o = res[name]
            print(f"  {name:15s} r(f_i)={o['r_fintra']:.3f} bias {o['bias_fintra']:+.3f} | r(voxel)={o['r_voxel']:.3f} bias {o['bias_voxel']:+.3f} | "
                  f"f_iso {o['f_iso_mean']:.2f} f_res {o['f_res_mean']:.2f} | D∥ {o['dpar']:.2f} D∥ex {o['dpar_ex']:.2f} D⊥ {o['dperp']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        results[snr] = res
        json.dump(results, open(a.out, "w"), indent=1)
    print("\nFORCE reference (doc 005 §2, b=1900 single shell, SNR 30): default r = 0.679, DiSCo-tuned r = 0.918")


if __name__ == "__main__":
    main()

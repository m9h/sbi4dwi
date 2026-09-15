#!/usr/bin/env python3
"""FORCE ND vs DiSCo Strand_Intra_Volume_Fraction on the FORCE-paper protocol
data (3 shells ≤ 3100), dipy-master venv (numpy + dipy + nibabel only).
Default in-vivo priors vs the authors' DiSCo ex-vivo ranges vs the paper's
narrow bands — the like-for-like reference for validate_disco_microstructure.py."""
import argparse, json, time
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.stats import pearsonr
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.force import FORCEModel

DISCO = Path.home() / ".dipy" / "disco" / "disco_1"
PRIORS = {
    "default": None,
    "authors-disco": {"wm_d_par_range": (0.4e-3, 0.9e-3), "wm_d_perp_range": (0.1e-3, 0.4e-3),
                      "gm_d_iso_range": (0.7e-3, 1.2e-3), "csf_d": 3.0e-3},
    "paper-narrow": {"wm_d_par_range": (0.54e-3, 0.66e-3), "wm_d_perp_range": (0.32e-3, 0.38e-3),
                     "gm_d_iso_range": (0.7e-3, 1.2e-3), "csf_d": 3.0e-3},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[30, 10]); ap.add_argument("--num-sims", type=int, default=1_000_000)
    ap.add_argument("--num-cpus", type=int, default=10); ap.add_argument("--out", default="validation/external/force_disco_ndi.json")
    a = ap.parse_args()
    bvals, bvecs = read_bvals_bvecs(str(DISCO / "DiSCo_gradients.bvals"), str(DISCO / "DiSCo_gradients_dipy.bvecs"))
    keep = bvals < 3100; gtab = gradient_table(bvals[keep], bvecs=bvecs[keep])
    mask = nib.load(DISCO / "highRes_DiSCo1_mask.nii.gz").get_fdata().astype(bool)
    gt = nib.load(DISCO / "highRes_DiSCo1_Strand_Intra_Volume_Fraction.nii.gz").get_fdata()[mask]
    results = {}
    for snr in a.snrs:
        data = np.asarray(nib.load(DISCO / f"highRes_DiSCo1_DWI_RicianNoise-snr{snr}.nii.gz").dataobj)[..., keep].astype(np.float32)
        results[snr] = {}
        for name, cfg in PRIORS.items():
            t0 = time.time()
            model = FORCEModel(gtab, compute_odf=False)
            model.generate(num_simulations=a.num_sims, num_cpus=a.num_cpus, wm_threshold=0.0, odi_range=(0.01, 0.15),
                           diffusivity_config=cfg, use_cache=True)
            fit = model.fit(data, mask=mask)
            nd = np.asarray(fit.nd)[mask]; wm = np.asarray(fit.wm_fraction)[mask]
            r_nd = pearsonr(nd, gt)[0]; r_ndwm = pearsonr(nd * wm, gt)[0]
            results[snr][name] = {"r_nd": float(r_nd), "bias_nd": float((nd - gt).mean()), "r_nd_wm": float(r_ndwm),
                                  "bias_nd_wm": float((nd * wm - gt).mean()), "wm_mean": float(wm.mean())}
            o = results[snr][name]
            print(f"SNR {snr} {name:14s} r(ND)={o['r_nd']:.3f} bias {o['bias_nd']:+.3f} | r(ND·f_wm)={o['r_nd_wm']:.3f} bias {o['bias_nd_wm']:+.3f} | f_wm {o['wm_mean']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

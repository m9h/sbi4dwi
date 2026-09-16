#!/usr/bin/env python3
"""
DIPY plug-in demo (doc 008 §9 item 3): refine MSMT-CSD and FORCE peaks with
`dmipy_jax.validation.dipy_refine.refine_peaks` on DiSCo under the FORCE
authors' protocol, and track everything with the same protocol tracker.

Rows per SNR: peaks as produced (MSMT-CSD, FORCE with the authors' DiSCo
ranges), the refined MAP fixels, and the CV-pruned posterior connectome
from the Laplace posterior — Pearson r / Dice vs the CSA ground truth,
plus intra-VF r for the refined fit. Runtimes per stage.
"""
import argparse, time, importlib.util, json
from pathlib import Path
import numpy as np, jax, nibabel as nib
from scipy.stats import pearsonr
from dipy.data import default_sphere
from dmipy_jax.validation import dipy_refine as dr
from dmipy_jax.validation.force_disco import disco_subject_path

vdfp = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location("vdfp", Path(__file__).with_name("validate_disco_force_protocol.py"))); s.loader.exec_module(vdfp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[30, 10]); ap.add_argument("--n-post", type=int, default=20)
    ap.add_argument("--num-sims", type=int, default=500_000); ap.add_argument("--num-cpus", type=int, default=6)
    ap.add_argument("--out", default="validation/dipy_refine_results.json")
    a = ap.parse_args()
    gt_vf = nib.load(disco_subject_path(1) / "highRes_DiSCo1_Strand_Intra_Volume_Fraction.nii.gz").get_fdata()
    results = {}
    for snr in a.snrs:
        d = vdfp.load_protocol_data(snr); mask = d["mask"]
        print(f"\n=== SNR {snr}", flush=True)
        sources = {}
        t0 = time.time(); sources["msmt"] = vdfp.run_msmt(d); print(f"  MSMT-CSD peaks {time.time()-t0:.0f}s", flush=True)
        t0 = time.time(); sources["force"] = vdfp.run_force(d, a.num_sims, a.num_cpus); print(f"  FORCE peaks {time.time()-t0:.0f}s", flush=True)
        results[snr] = {}
        for name, pam in sources.items():
            cm, _ = vdfp.track_protocol(pam, d); r0, dsc0 = vdfp.score(cm, d["gt"])
            t0 = time.time(); pam_ref, post, fit = dr.refine_peaks(d["data"], d["gtab"], mask, pam, n_fibres=3, cfg=dr.default_config(3, d_par_init=0.6e-9), affine=d["affine"]); t_ref = time.time() - t0
            cm1, _ = vdfp.track_protocol(pam_ref, d); r1, dsc1 = vdfp.score(cm1, d["gt"])
            t0 = time.time()
            cms = np.array([vdfp.track_protocol(p, d)[0] for p in dr.posterior_pams(post, mask, a.n_post, jax.random.PRNGKey(0), affine=d["affine"])])
            mean_cm = cms.mean(0); cv = cms.std(0) / np.maximum(mean_cm, 1e-9); r2, dsc2 = vdfp.score(np.where(cv < 0.3, mean_cm, 0.0), d["gt"]); t_post = time.time() - t0
            r_vf = pearsonr(fit.fintra, gt_vf[mask])[0]
            results[snr][name] = {"peaks": (r0, dsc0["dice"]), "refined": (r1, dsc1["dice"]), "posterior_cv": (r2, dsc2["dice"]), "intra_vf_r": float(r_vf), "t_refine": t_ref, "t_post": t_post}
            print(f"  {name:5s}: peaks r={r0:.3f} Dice={dsc0['dice']:.2f}  →  refined MAP r={r1:.3f} Dice={dsc1['dice']:.2f}  →  CV-pruned posterior r={r2:.3f} Dice={dsc2['dice']:.2f}   "
                  f"intra-VF r={r_vf:.3f}  D∥={fit.d_par*1e9:.2f}  (refine {t_ref:.0f}s, {a.n_post} posterior connectomes {t_post:.0f}s)", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

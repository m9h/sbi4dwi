#!/usr/bin/env python3
"""
Is the amortised posteriors' crossing error breadth or bias? (doc 008 §7.4)

For SBI_dMRI (validation/external/sbi_dmri_*_samples.npz) and our flow
(validation/flow_sbi_results.npz), compare per crossing angle the mean
best-match angular error with the posterior's own σ_θ (RMS angle of the
direction samples about their mode). err ≈ σ_θ → the posterior is honest
but broad (an information / amortisation limit); err ≫ σ_θ → confidently
wrong (bias, e.g. mode averaging under label switching); err ≪ σ_θ →
over-dispersed. Pure numpy on cached samples.
"""
import numpy as np, sys
from pathlib import Path


def best_match_err(dirs, fr, gt, fmin=0.05):
    out = np.full(len(gt), np.nan)
    for n in range(len(gt)):
        es = []
        for g in gt[n]:
            if np.linalg.norm(g) == 0: continue
            cand = [d for d, f in zip(dirs[n], fr[n]) if f >= fmin and np.linalg.norm(d) > 0]
            if cand: es.append(np.degrees(np.arccos(np.clip(max(abs(float(g @ d)) for d in cand), -1, 1))))
        if es: out[n] = np.mean(es)
    return out


def report(name, dirs, fr, sig, gt, ang):
    err = best_match_err(dirs, fr, gt)
    print(f"\n{name}")
    print("  angle   mean err   median σ_θ(main)  σ_θ(2nd)   err/σ   |err − σ|<5° share")
    rows = []
    for av in sorted(set(ang.tolist())):
        m = ang == av; e = np.nanmean(err[m]); s1 = np.median(sig[m, 0]); s2 = np.median(sig[m, 1]) if sig.shape[1] > 1 else np.nan
        share = np.mean(np.abs(err[m] - sig[m, 0]) < 5)
        rows.append((av, e, s1, s2, e / max(s1, 1e-6), share))
        print(f"  {int(av):>4d}  {e:8.2f}°  {s1:12.2f}°  {s2:8.2f}°  {e/max(s1,1e-6):6.2f}  {100*share:6.0f}%")
    e_all = np.nanmean(err); s_all = np.median(sig[:, 0])
    print(f"  all   {e_all:8.2f}°  {s_all:12.2f}°            {e_all/max(s_all,1e-6):6.2f}")
    return rows


def main():
    z = np.load("validation/external/synthetic_snr30.npz", allow_pickle=True)
    gt = z["gt_dirs"].copy(); ang = z["angle"]; gt[ang == 0, 1] = 0
    for tag in ("synthetic_snr30", "synthetic_snr30_cluster", "synthetic_snr30_odi0.2", "synthetic_snr10", "synthetic_snr10_cluster"):
        p = Path(f"validation/external/sbi_dmri_{tag}.json_samples.npz")
        if not p.exists(): continue
        s = np.load(p)
        zz = np.load(f"validation/external/{tag.replace('_cluster', '')}.npz", allow_pickle=True)
        g2 = zz["gt_dirs"].copy(); a2 = zz["angle"]; g2[a2 == 0, 1] = 0
        report(f"SBI_dMRI {tag}", s["dirs"], s["fracs"], s["sigma_deg"], g2, a2)
    p = Path("validation/flow_sbi_results.npz")
    if p.exists():
        s = np.load(p); report("sbi4dwi flow NPE (synthetic SNR30, clustered)", s["dirs"], s["fracs"], s["sigma_deg"], gt, ang)


if __name__ == "__main__":
    main()

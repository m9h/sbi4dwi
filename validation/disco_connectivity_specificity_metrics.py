#!/usr/bin/env python3
"""
Compute Lin's CCC + Dice/F1 across dmipy-JAX, FORCE, MRtrix DiSCo
connectivity matrices to supplement §21's Pearson-r table.

Reports per-SNR:
  - Pearson r (count alignment)
  - Lin's CCC (penalises scale bias)
  - Dice/F1 at threshold = 0 (any nonzero count → "connected")
  - Precision, recall, TP/FP/FN
"""

from pathlib import Path

import numpy as np

from validation.lib.connectivity_metrics import summarize_method


METHODS = {
    "dmipy-JAX (B2)": "validation/dmipy_disco_connectivity_results.npz",
    # §17.5 r=0.298/0.211/0.322 lives in *_default.npz (the tuned-library
    # run produced all-zero ODFs → zero streamlines → nan r, doc 005 §4d).
    "FORCE (§17.5)":  "validation/force_disco_connectivity_results_default.npz",
    "MRtrix (§18)":   "validation/mrtrix_disco_connectivity_results.npz",
}

SNRS = (10, 30, 50)


def load_cmats(path: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    d = np.load(path)
    gt = d["gt"]
    cmats = {snr: d[f"cmat_snr{snr}"] for snr in SNRS}
    return cmats, gt


def main():
    print("=" * 78)
    print("DiSCo connectivity specificity-aware metrics (upper triangle, k>0)")
    print("=" * 78)

    rows = []
    for name, path in METHODS.items():
        path = Path(path)
        if not path.exists():
            print(f"  skip {name}: {path} not found")
            continue
        cmats, gt = load_cmats(path)
        summary = summarize_method(cmats, gt, threshold=0.0)
        for snr in SNRS:
            s = summary[snr]
            rows.append({"method": name, "snr": snr, **s})

    # Print as table
    print()
    hdr = f"{'method':<18s} {'SNR':>3s} {'pearson':>8s} " \
          f"{'CCC_norm':>9s} {'Dice':>6s} {'prec':>6s} {'rec':>6s} " \
          f"{'TP':>4s}/{'FP':>4s}/{'FN':>3s}/{'TN':>3s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:<18s} {r['snr']:>3d} "
              f"{r['pearson_r']:>8.4f} "
              f"{r['lin_ccc_sumnorm']:>9.4f} "
              f"{r['dice']:>6.3f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['tp']:>4d}/{r['fp']:>4d}/{r['fn']:>3d}/{r['tn']:>3d}")
    print("\n(CCC_norm = Lin's CCC after sum-normalising both matrices to "
          "the probability simplex — required since pred and GT live in "
          "different units. Dice/F1 are equivalent for binary "
          "classification.)")

    # Threshold sweep on dmipy-JAX SNR=30 only (Dice as a function of cutoff)
    print()
    print("=" * 78)
    print("dmipy-JAX (B2) Dice / precision / recall vs threshold @ SNR=30")
    print("=" * 78)
    cmats, gt = load_cmats(Path(METHODS["dmipy-JAX (B2)"]))
    cmat30 = cmats[30]
    iu = np.triu_indices(cmat30.shape[0], k=1)
    upper = cmat30[iu]
    percentiles = [0, 25, 50, 70, 80, 90, 95]
    thresholds = [float(np.percentile(upper[upper > 0], p)) if p > 0 else 0.0
                  for p in percentiles]
    from validation.lib.connectivity_metrics import connectivity_dice_f1
    print(f"  {'pctile':>6s} {'thr':>8s}  {'Dice':>6s} {'prec':>6s} "
          f"{'rec':>6s}  {'TP':>4s}/{'FP':>4s}/{'FN':>3s}")
    for p, t in zip(percentiles, thresholds):
        d = connectivity_dice_f1(cmat30, gt, threshold=t)
        tag = f"p{p:>3d}" if p > 0 else "any>0"
        print(f"  {tag:>6s} {t:>8.1f}  {d['dice']:>6.3f} "
              f"{d['precision']:>6.3f} {d['recall']:>6.3f}  "
              f"{d['tp']:>4d}/{d['fp']:>4d}/{d['fn']:>3d}")

    # Save NPZ for downstream use
    out = Path("validation/disco_specificity_metrics.npz")
    np.savez(
        out,
        rows=np.array(rows, dtype=object),
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

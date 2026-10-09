#!/usr/bin/env python3
"""
DiSCo connectivity-matrix benchmark — FORCE paper §3.2 metric.

Per-SNR pipeline:
  1. Fit FORCE on DiSCo subject 1 single-shell b=1900 with the default
     library (the tuned library has the wm_threshold=1.0 ODF-suppression
     bug documented in doc 005 §4d).
  2. force_peaks(fit) → PeaksAndMetrics with peak directions.
  3. LocalTracking (Euler integration) on those peaks; seeds throughout
     the ROI mask.
  4. dipy.tracking.utils.connectivity_matrix → 16×16 connectivity.
  5. Pearson r on upper-triangle vs DiSCo ground-truth
     Connectivity_Matrix_Cross-Sectional_Area.

Reproduces (approximately) FORCE paper §3.2 reported numbers:
  r = 0.868 at SNR=10, r = 0.894 at SNR=50.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation.lib.force_disco_connectivity import (
    connectivity_pearson,
    lin_ccc,
    load_gt_connectivity,
    run_force_connectivity,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--seed-density", type=int, default=2)
    ap.add_argument("--library", choices=["tuned", "default"], default="tuned",
                    help="Which FORCE library to use. tuned = paper §3.2 "
                         "DiSCo-aligned priors; default = in-vivo priors.")
    ap.add_argument(
        "--rebalance",
        nargs=3, type=float, metavar=("P1", "P2", "P3"),
        default=None,
        help="Rebalance the library to target (P1, P2, P3) fractions of "
             "(1f, 2f, 3f) entries. Example: --rebalance 0.8 0.1 0.1 "
             "applies the Dirichlet(8,1,1) prior the audit recommended.",
    )
    args = ap.parse_args()

    gt = load_gt_connectivity(subject=1)
    print(f"Ground-truth connectivity: 16×16, "
          f"{int((gt > 0).sum())} nonzero off-diagonal entries.")
    print(f"Library: {args.library}")
    rebalance_fibres = None
    if args.rebalance is not None:
        p1, p2, p3 = args.rebalance
        rebalance_fibres = {1: p1, 2: p2, 3: p3}
        print(f"Rebalanced fibre-count fractions: {rebalance_fibres}")

    results = {}
    iu = np.triu_indices(16, k=1)
    for snr in args.snrs:
        print(f"\n=== SNR = {snr} ===")
        res = run_force_connectivity(
            subject=1, snr=snr, tuned=(args.library == "tuned"),
            seed_density=args.seed_density,
            rebalance_fibres=rebalance_fibres,
        )
        cmat = res["connectivity"]
        r = connectivity_pearson(cmat, gt)
        # CCC: penalises systematic bias. Use the upper-triangle entries
        # so it's comparable to Pearson r above.
        mask_ut = np.zeros_like(gt, dtype=bool)
        mask_ut[iu] = True
        ccc = lin_ccc(cmat, gt, mask=mask_ut)
        results[snr] = {
            "cmat": cmat,
            "r": r,
            "ccc": ccc,
            "n_streamlines": res["streamlines_count"],
            "n_seeds": res["n_seeds"],
        }
        print(f"  n_seeds:        {res['n_seeds']:,}")
        print(f"  n_streamlines:  {res['streamlines_count']:,}")
        print(f"  Pearson r vs GT (upper-triangle): {r:.4f}")
        print(f"  Lin CCC vs GT (upper-triangle):   {ccc:.4f}")

    # Save NPZ
    rebal_tag = ""
    if rebalance_fibres is not None:
        rebal_tag = "_reb" + "".join(
            f"{int(p*100):02d}" for p in [
                rebalance_fibres.get(1, 0),
                rebalance_fibres.get(2, 0),
                rebalance_fibres.get(3, 0),
            ]
        )
    suffix = f"_{args.library}{rebal_tag}"
    out_npz = Path(f"validation/force_disco_connectivity_results{suffix}.npz")
    np.savez(
        out_npz,
        snrs=np.array(args.snrs),
        r=np.array([results[s]["r"] for s in args.snrs]),
        ccc=np.array([results[s]["ccc"] for s in args.snrs]),
        n_streamlines=np.array([results[s]["n_streamlines"] for s in args.snrs]),
        gt=gt,
        **{f"cmat_snr{s}": results[s]["cmat"] for s in args.snrs},
    )
    print(f"\nWrote {out_npz}")

    # Figure: GT vs reconstructed matrices + Pearson curve
    n = len(args.snrs)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.5 * (n + 1), 6), dpi=130)
    # GT in first column (both rows the same)
    for row in range(2):
        ax = axes[row, 0]
        ax.imshow(gt, cmap="hot", vmin=0, vmax=gt.max())
        ax.set_title("Ground-truth\n(Cross-Sect Area)" if row == 0 else "")
        ax.set_xticks([]); ax.set_yticks([])

    for col, snr in enumerate(args.snrs):
        cmat = results[snr]["cmat"]
        r = results[snr]["r"]
        # Recon matrix
        ax = axes[0, col + 1]
        ax.imshow(cmat, cmap="hot", vmin=0, vmax=cmat.max() if cmat.max() > 0 else 1)
        ax.set_title(f"FORCE @ SNR={snr}\n(streamline counts)")
        ax.set_xticks([]); ax.set_yticks([])
        # Scatter GT vs recon
        ax = axes[1, col + 1]
        iu = np.triu_indices(16, k=1)
        x = gt[iu]
        y = cmat[iu]
        ax.scatter(x, y, s=8, alpha=0.6)
        ax.set_xlabel("GT cross-sect area")
        ax.set_ylabel("FORCE streamline count")
        ax.set_title(f"r = {r:.3f}")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "DiSCo subject 1 connectivity matrix: FORCE peaks → LocalTracking "
        "→ ROI connectivity\n"
        "(default-library fit, single-shell b=1900, ROI-mask seeding, "
        "paper §3.2 reproduction)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = Path(f"validation/force_disco_connectivity{suffix}.png")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Wrote {out_png}")

    # Print summary table
    print("\n" + "=" * 72)
    print("DiSCo connectivity-matrix (upper-triangle, 120 ROI pairs)")
    print("=" * 72)
    print(f"  {'SNR':>4s}  {'Pearson r':>10s}  {'CCC':>8s}  {'Paper §3.2 r':>14s}")
    paper_vals = {10: 0.868, 50: 0.894}
    for snr in args.snrs:
        paper = f"{paper_vals.get(snr, '—'):.3f}" if snr in paper_vals else "—"
        print(f"  {snr:>4d}  {results[snr]['r']:>10.4f}  "
              f"{results[snr]['ccc']:>8.4f}  {paper:>14s}")


if __name__ == "__main__":
    main()

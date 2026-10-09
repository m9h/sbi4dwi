#!/usr/bin/env python3
"""
dmipy-JAX dictionary matcher on DiSCo (scalar microstructure recovery).

Counterpart to `validation/validate_force_disco_phantom.py` — same
benchmark setup, but uses sbi4dwi's `DictionaryMatcher` instead of
dipy's `FORCEModel`. Direct head-to-head with the §16 results.

Pipeline:
  1. Load DiSCo subject 1 single-shell b=1900 at SNR ∈ {10, 30, 50}.
  2. Generate a 200K-entry dmipy-JAX 2-stick + Bingham + iso library
     with DiSCo-tuned diffusivity priors (per FORCE paper §3.2).
  3. Match every brain-mask voxel via cosine similarity.
  4. Compare matched NDI (= 1 - f_iso) and FA-proxy to DiSCo GT
     Strand_Intra_Volume_Fraction and DTI FA via Pearson r + Lin CCC.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from validation.lib.dmipy_disco_dict import fit_dmipy_dict_on_disco
from validation.lib.force_disco import load_disco_subject
from validation.lib.force_disco_connectivity import lin_ccc
from validation.lib.force_inter_method import (
    fit_dti_baseline, masked_pearson,
)


def render_figure(maps_by_snr: dict, gt_ndi: np.ndarray, mask: np.ndarray, out: Path):
    snrs = sorted(maps_by_snr.keys())
    z = int(np.median(np.argwhere(mask.any(axis=(0, 1)))[:, 0]))

    fig, axes = plt.subplots(2, len(snrs) + 1, figsize=(3.5 * (len(snrs) + 1), 6.5), dpi=130)
    # GT in column 0 (both rows the same — just the NDI panel)
    axes[0, 0].imshow(np.rot90(gt_ndi[:, :, z]), cmap="viridis", vmin=0, vmax=1)
    axes[0, 0].set_title("Ground-truth NDI"); axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for col, snr in enumerate(snrs):
        m = maps_by_snr[snr]
        ax = axes[0, col + 1]
        ax.imshow(np.rot90(m["ndi"][:, :, z]), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"dmipy-JAX NDI\nSNR={snr}  r={m['r_ndi']:.3f}\nCCC={m['ccc_ndi']:.3f}")
        ax.axis("off")
        ax = axes[1, col + 1]
        # scatter
        a = m["ndi"][mask].ravel(); b = gt_ndi[mask].ravel()
        valid = np.isfinite(a) & np.isfinite(b)
        a = a[valid]; b = b[valid]
        if a.size > 8000:
            idx = np.random.default_rng(0).choice(a.size, 8000, replace=False)
            a = a[idx]; b = b[idx]
        ax.scatter(b, a, s=2, alpha=0.25)
        lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("GT NDI"); ax.set_ylabel("dmipy-JAX NDI")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "dmipy-JAX DictionaryMatcher on DiSCo — NDI recovery (counterpart to §16.3)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--library-size", type=int, default=200_000)
    args = ap.parse_args()

    print("=" * 78)
    print("dmipy-JAX DictionaryMatcher on DiSCo — Option A: scalar comparison")
    print("=" * 78)

    # GT + DTI baseline are SNR-independent (only the GT) but we want
    # to compare to DTI FA at each SNR. Load fixtures.
    out0 = load_disco_subject(subject=1, snr=30, single_shell_b=1900)
    mask = out0["mask"]
    rois = out0["rois"]
    gt_ndi = out0["gt"]["ndi"]
    affine = nib.load(
        Path.home() / ".dipy" / "disco" / "disco_1" / "highRes_DiSCo1_DWI.nii.gz"
    ).affine

    results = {}
    for snr in args.snrs:
        print(f"\n--- SNR = {snr} ---")
        fit = fit_dmipy_dict_on_disco(
            subject=1, snr=snr,
            library_size=args.library_size,
        )
        r_ndi = masked_pearson(fit["ndi"], gt_ndi, mask)
        ccc_ndi = lin_ccc(fit["ndi"], gt_ndi, mask=mask)
        print(f"  dmipy NDI mean (in mask): {fit['ndi'][mask].mean():.4f}  "
              f"(GT mean: {gt_ndi[mask].mean():.4f})")
        print(f"  Pearson r vs GT NDI: {r_ndi:.4f}")
        print(f"  Lin CCC vs GT NDI:   {ccc_ndi:.4f}")

        # FA proxy: also do DTI as reference
        snr_out = load_disco_subject(subject=1, snr=snr, single_shell_b=1900)
        dti = fit_dti_baseline(snr_out["data"], snr_out["gtab"], mask)
        r_fa = masked_pearson(fit["fa_proxy"], dti["fa"], mask)
        print(f"  dmipy FA-proxy vs DTI FA Pearson r: {r_fa:.4f}")

        results[snr] = {
            **fit, "r_ndi": float(r_ndi), "ccc_ndi": float(ccc_ndi),
            "r_fa": float(r_fa), "dti_fa": dti["fa"],
        }

    # Save NPZ
    out_npz = Path("validation/dmipy_disco_results.npz")
    np.savez(
        out_npz,
        snrs=np.array(args.snrs),
        r_ndi=np.array([results[s]["r_ndi"] for s in args.snrs]),
        ccc_ndi=np.array([results[s]["ccc_ndi"] for s in args.snrs]),
        r_fa=np.array([results[s]["r_fa"] for s in args.snrs]),
        gt_ndi=gt_ndi, mask=mask,
        **{f"ndi_snr{s}": results[s]["ndi"] for s in args.snrs},
    )
    print(f"\nWrote {out_npz}")

    render_figure(results, gt_ndi, mask, Path("validation/dmipy_disco.png"))

    # Compare to dipy FORCE §16.3 / §16.7
    print("\n" + "=" * 72)
    print("dmipy-JAX vs dipy FORCE on DiSCo NDI recovery")
    print("=" * 72)
    print(f"  {'SNR':>4s}  {'dmipy r':>9s}  {'dmipy CCC':>10s}  "
          f"{'FORCE r §16':>12s}  {'FORCE CCC':>10s}")
    force_force = {
        10: (0.879, "~0.78"),  # §16.7 + CCC from §17.6 framing
        30: (0.918, "~0.85"),
        50: (0.922, "~0.86"),
    }
    for snr in args.snrs:
        f_r, f_ccc = force_force.get(snr, ("—", "—"))
        print(f"  {snr:>4d}  {results[snr]['r_ndi']:>9.4f}  "
              f"{results[snr]['ccc_ndi']:>10.4f}  "
              f"{f_r if isinstance(f_r, str) else f'{f_r:.3f}':>12s}  "
              f"{f_ccc:>10s}")


if __name__ == "__main__":
    main()

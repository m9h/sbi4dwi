#!/usr/bin/env python3
"""
Stanford HARDI inter-method comparison: FORCE vs DTI.

Replicates the FORCE paper's Figure C1 logic on Stanford HARDI:
fit DTI on the same data, compare voxel-wise against FORCE-derived
FA / MD / RD on the brain mask. The paper reports DTI/MSDKI agreement
with FORCE (Figure 4) and FA/MD/RD matches (Figure C1) as a baseline
sanity check.

Output:
  - validation/force_dti_inter_method.png — paired FA/MD/RD maps + scatter
  - Brain-mask Pearson correlations printed to stdout
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation.lib.force_inter_method import (
    fit_dti_baseline,
    load_force_maps,
    masked_pearson,
)
from validation.lib.force_stanford import load_stanford_hardi


def render_inter_method_figure(
    force_maps: dict, dti_maps: dict, mask: np.ndarray, out: Path,
) -> dict:
    """3 metrics × {FORCE, DTI, scatter} = 9-panel figure.

    Returns the per-metric Pearson correlation dict.
    """
    metrics = [
        ("fa", "FA", "gray", 0.0, 1.0),
        ("md", "MD (mm²/s)", "hot", None, None),
        ("rd", "RD (mm²/s)", "hot", None, None),
    ]

    z = int(np.median(np.argwhere(mask.any(axis=(0, 1)))[:, 0]))

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), dpi=130)
    correlations = {}

    for row, (key, label, cmap, vmin, vmax) in enumerate(metrics):
        force_map = force_maps[key]
        dti_map = dti_maps[key]

        # Auto-vmin/vmax from FORCE if not specified, share between panels
        v_lo = vmin if vmin is not None else float(np.percentile(force_map[mask], 1))
        v_hi = vmax if vmax is not None else float(np.percentile(force_map[mask], 99))

        for col, (mp, title) in enumerate([
            (force_map, f"FORCE {label}"),
            (dti_map, f"DTI {label}"),
        ]):
            ax = axes[row, col]
            ax.imshow(np.rot90(mp[:, :, z]), cmap=cmap, vmin=v_lo, vmax=v_hi)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Scatter in the third column
        ax = axes[row, 2]
        f = force_map[mask].ravel()
        d = dti_map[mask].ravel()
        valid = np.isfinite(f) & np.isfinite(d)
        f, d = f[valid], d[valid]
        # Subsample for speed if many voxels
        if f.size > 5000:
            idx = np.random.default_rng(0).choice(f.size, 5000, replace=False)
            f, d = f[idx], d[idx]
        r = masked_pearson(force_map, dti_map, mask)
        correlations[key] = r
        ax.scatter(d, f, s=2, alpha=0.3, color="C0")
        lo = min(f.min(), d.min()); hi = max(f.max(), d.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.7, label="y = x")
        ax.set_xlabel(f"DTI {label}")
        ax.set_ylabel(f"FORCE {label}")
        ax.set_title(f"r = {r:.3f}", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Stanford HARDI: FORCE vs DTI inter-method comparison\n"
        f"(brain-mask voxels, z={z}; mirrors FORCE paper Fig. C1)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    return correlations


def main():
    print("=" * 78)
    print("Stanford HARDI inter-method: FORCE vs DTI")
    print("=" * 78)

    print("\nLoading Stanford HARDI...")
    data, affine, mask, gtab = load_stanford_hardi()
    print(f"  brain voxels: {int(mask.sum()):,}")

    print("\nLoading cached FORCE maps...")
    force_maps = load_force_maps()
    print(f"  FORCE maps loaded: {sorted(force_maps.keys())}")

    print("\nFitting DTI on Stanford HARDI...")
    dti_maps = fit_dti_baseline(data, gtab, mask)
    for k in ("fa", "md", "rd"):
        v = dti_maps[k][mask]
        print(f"  DTI {k:>3s}: min={v.min():.5f}  mean={v.mean():.5f}  max={v.max():.5f}")

    out_png = Path("validation/force_dti_inter_method.png")
    correlations = render_inter_method_figure(force_maps, dti_maps, mask, out_png)

    print("\n" + "=" * 78)
    print("Brain-mask Pearson correlations: FORCE vs DTI")
    print("=" * 78)
    for k, r in correlations.items():
        print(f"  {k:>4s}: r = {r:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Replicate the FORCE paper's Stanford HARDI demonstration (Figure 6) with
dipy 1.12.1's `FORCEModel`.

Goal: confirm the upstream implementation produces sensible NODDI-style
maps on its design data, before any comparison claims about FORCE vs
dmipy-JAX. This is the methodological reset called for in doc 004 §13
after the matched-conditions sweep revealed our synthetic was missing
the extra-axonal zeppelin compartment.

Output:
  - validation/force_stanford_maps.png — 6-panel slice figure mirroring
    the paper's Figure 6 (FA, MD, NDI, ODI, FW, num_fibers).
  - Cached per-voxel maps at FORCE_STANFORD_CACHE
    (default ~/.cache/force_stanford/force_stanford_maps.npz).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation.lib.force_stanford import (
    fit_force_or_load_cached_maps,
    load_stanford_hardi,
)


CACHE = str(Path.home() / ".cache" / "dipy_force" / "force_matched_500k.npz")


def render_six_panel(maps: dict, out: Path):
    """Render FA, MD, NDI (=nd), ODI (= dispersion), FW (= csf_fraction),
    num_fibers on an axial slice — mirrors paper Figure 6."""
    mask = maps["mask"]
    z_choices = np.argwhere(mask.any(axis=(0, 1)))[:, 0]
    z = int(np.median(z_choices)) - 5  # tutorial uses mid_z - 5

    panels = [
        ("FA", maps["fa"], "gray", 0.0, 1.0),
        ("MD", maps["md"], "hot", None, None),
        ("NDI", maps["nd"], "viridis", 0.0, 1.0),
        ("ODI / dispersion", maps["dispersion"], "viridis", 0.0, 1.0),
        ("FW (csf_fraction)", maps["csf_fraction"], "Blues", 0.0, 1.0),
        ("Num fibers", maps["num_fibers"], "magma", 0.0, 3.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=130)
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes.flat, panels):
        kw = {"cmap": cmap}
        if vmin is not None:
            kw["vmin"], kw["vmax"] = vmin, vmax
        slice_ = np.rot90(arr[:, :, z])
        im = ax.imshow(slice_, **kw)
        ax.set_title(f"{title}  (z={z})", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "FORCE on Stanford HARDI (dipy 1.12.1, 500K-entry library, "
        "n_neighbors=50)\nMirrors FORCE paper Fig. 6 + tutorial reconst_force.py",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    if not Path(CACHE).exists():
        raise SystemExit(
            f"FORCE library cache not found at {CACHE}. "
            "Run validation/validate_force_matched.py first to build it."
        )

    print("Loading Stanford HARDI...")
    data, affine, mask, gtab = load_stanford_hardi()
    n_brain = int(mask.sum())
    print(f"  shape: {data.shape}  brain voxels: {n_brain:,}")

    print(f"\nFitting FORCE on {n_brain:,} voxels...")
    maps = fit_force_or_load_cached_maps(library_cache=CACHE, n_jobs=-1)

    out_png = Path("validation/force_stanford_maps.png")
    render_six_panel(maps, out_png)

    # Print summary stats over the brain
    print("\nMap summary (within brain mask):")
    for key in ("fa", "md", "nd", "dispersion", "wm_fraction",
                "gm_fraction", "csf_fraction", "num_fibers"):
        arr = maps[key][mask]
        print(f"  {key:>14s}  min={arr.min():>7.4f}  mean={arr.mean():>7.4f}  "
              f"max={arr.max():>7.4f}")


if __name__ == "__main__":
    main()

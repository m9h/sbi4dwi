#!/usr/bin/env python3
"""
DiSCo phantom benchmark — FORCE paper §3.2 protocol.

Single-shell b≈2000 extraction (DiSCo's b=1900) on subject 1 highRes
(40³ voxels). Three methods fit:
  - FORCE (dipy 1.12.1, 500K-entry library generated for the DiSCo gtab)
  - DTI (dipy TensorModel) — sanity baseline
  - CSD (dipy ConstrainedSphericalDeconvModel) — paper's stated baseline

Ground-truth comparisons (brain-mask, Pearson r against
``highRes_DiSCo1_Strand_Intra_Volume_Fraction``):
  - FORCE NDI (``fit.nd``)
  - FORCE FA (vs DTI FA round-trip — sanity)

Output:
  - validation/force_disco.png — 4-panel: GT NDI, FORCE NDI, FORCE FA,
    scatter GT vs FORCE NDI
  - Pearson correlations printed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation.lib.force_disco import load_disco_subject
from validation.lib.force_inter_method import (
    fit_dti_baseline,
    masked_pearson,
)


DIPY_FORCE_DISCO_CACHE_DEFAULT = Path.home() / ".cache" / "dipy_force" / "force_disco_500k.npz"
DIPY_FORCE_DISCO_CACHE_TUNED = Path.home() / ".cache" / "dipy_force" / "force_disco_500k_tuned.npz"
DISCO_FORCE_FIT_CACHE_DEFAULT = Path.home() / ".cache" / "force_disco" / "force_disco_fit.npz"
DISCO_FORCE_FIT_CACHE_TUNED = Path.home() / ".cache" / "force_disco" / "force_disco_fit_tuned.npz"


def get_or_make_force_library(gtab, n_sims=500_000, tuned: bool = False):
    """Build or load a FORCE library for the DiSCo gtab.

    Parameters
    ----------
    tuned : bool
        If True, use the DiSCo-aligned diffusivity priors from FORCE paper
        §3.2 (D_∥ ∈ [0.54, 0.66]×10⁻³, D_⊥ ∈ [0.32, 0.38]×10⁻³ mm²/s,
        wm_threshold=1.0 to disable GM/CSF compartments). Otherwise use
        the in-vivo defaults.
    """
    from dipy.reconst.force import (
        generate_force_simulations,
        load_force_simulations,
        save_force_simulations,
    )

    cache_path = DIPY_FORCE_DISCO_CACHE_TUNED if tuned else DIPY_FORCE_DISCO_CACHE_DEFAULT
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        tag = "TUNED" if tuned else "DEFAULT"
        print(f"Loading {tag} DiSCo FORCE library from {cache_path}")
        return load_force_simulations(str(cache_path))

    kwargs = dict(gtab=gtab, num_simulations=n_sims, num_cpus=1, verbose=True)
    if tuned:
        # FORCE paper §3.2 DiSCo-aligned priors (units: mm²/s)
        kwargs["diffusivity_config"] = {
            "wm_d_par_range": (0.00054, 0.00066),
            "wm_d_perp_range": (0.00032, 0.00038),
            "gm_d_iso_range": (0.0007, 0.0012),  # paper disables iso via wm_threshold
            "csf_d": 0.003,
        }
        kwargs["wm_threshold"] = 1.0  # purely-WM library, no GM/CSF mixing
        print(f"Generating TUNED DiSCo FORCE library (DiSCo diffusivities, "
              f"WM-only) (n={n_sims:,}, single-CPU, ~15 min)...")
    else:
        print(f"Generating DEFAULT FORCE library (in-vivo priors) "
              f"(n={n_sims:,}, single-CPU, ~15 min)...")

    sims = generate_force_simulations(**kwargs)
    save_force_simulations(sims, str(cache_path))
    print(f"Saved to {cache_path}")
    return sims


def fit_or_load_force_disco(data, gtab, mask, sims, n_neighbors=50, n_jobs=-1, tuned=False):
    import warnings as _w
    _w.filterwarnings("ignore")
    from dipy.reconst.force import FORCEModel

    cache = DISCO_FORCE_FIT_CACHE_TUNED if tuned else DISCO_FORCE_FIT_CACHE_DEFAULT
    if cache.exists():
        print(f"Loading cached FORCE fit from {cache}")
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    model = FORCEModel(gtab, simulations=sims, n_neighbors=n_neighbors)
    print(f"Fitting FORCE on {int(mask.sum()):,} masked voxels (this takes a few minutes)...")
    fit = model.fit(data, mask=mask, n_jobs=n_jobs)

    out = {
        "fa": np.asarray(fit.fa, dtype=np.float32),
        "md": np.asarray(fit.md, dtype=np.float32),
        "rd": np.asarray(fit.rd, dtype=np.float32),
        "nd": np.asarray(fit.nd, dtype=np.float32),
        "dispersion": np.asarray(fit.dispersion, dtype=np.float32),
        "wm_fraction": np.asarray(fit.wm_fraction, dtype=np.float32),
        "num_fibers": np.asarray(fit.num_fibers, dtype=np.float32),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def render_disco_figure(force_maps, dti_maps, gt, mask, out_png):
    z = int(np.median(np.argwhere(mask.any(axis=(0, 1)))[:, 0]))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), dpi=130)

    panels = [
        (axes[0, 0], gt["ndi"], "Ground-truth NDI (Intra VF)", "viridis", 0, 1),
        (axes[0, 1], force_maps["nd"], "FORCE NDI (fit.nd)", "viridis", 0, 1),
        (axes[0, 2], np.abs(force_maps["nd"] - gt["ndi"]) * mask,
         "|FORCE − GT| NDI", "hot", 0, 0.5),
        (axes[1, 0], dti_maps["fa"], "DTI FA", "gray", 0, 1),
        (axes[1, 1], force_maps["fa"], "FORCE FA", "gray", 0, 1),
        (axes[1, 2], force_maps["dispersion"], "FORCE ODI / dispersion",
         "viridis", 0, 1),
    ]
    for ax, arr, title, cmap, vmin, vmax in panels:
        im = ax.imshow(np.rot90(arr[:, :, z]), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "DiSCo phantom benchmark (subject 1 highRes, single-shell b=1900, SNR=30)\n"
        "FORCE vs ground-truth Intra Volume Fraction; DTI baseline",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Wrote {out_png}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", action="store_true",
                    help="Use FORCE paper §3.2 DiSCo-aligned diffusivity priors "
                         "instead of the in-vivo defaults.")
    ap.add_argument("--snr", type=int, default=30,
                    help="DiSCo SNR variant (10/20/30/40/50)")
    args = ap.parse_args()

    print("=" * 78)
    print(f"DiSCo phantom benchmark — {'TUNED' if args.tuned else 'DEFAULT'} library")
    print("=" * 78)

    print(f"\nLoading DiSCo subject 1 (single-shell b=1900, SNR={args.snr})...")
    out = load_disco_subject(subject=1, snr=args.snr, single_shell_b=1900)
    data = out["data"]
    mask = out["mask"]
    gtab = out["gtab"]
    gt = out["gt"]
    print(f"  data: {data.shape}, mask: {int(mask.sum()):,} voxels")

    print("\nGetting FORCE library (regenerates if not cached)...")
    sims = get_or_make_force_library(gtab, n_sims=500_000, tuned=args.tuned)

    print("\nFitting FORCE...")
    force_maps = fit_or_load_force_disco(data, gtab, mask, sims, tuned=args.tuned)
    for k in ("fa", "nd", "dispersion"):
        v = force_maps[k][mask]
        print(f"  FORCE {k:>4s}: min={v.min():.4f}  mean={v.mean():.4f}  max={v.max():.4f}")

    print("\nFitting DTI baseline...")
    dti_maps = fit_dti_baseline(data, gtab, mask)
    for k in ("fa", "md"):
        v = dti_maps[k][mask]
        print(f"  DTI {k:>3s}: min={v.min():.4f}  mean={v.mean():.4f}  max={v.max():.4f}")

    # Ground-truth Pearson correlations
    print("\n" + "=" * 78)
    print("Brain-mask Pearson correlations vs ground truth")
    print("=" * 78)
    r_ndi = masked_pearson(force_maps["nd"], gt["ndi"], mask)
    r_force_dti_fa = masked_pearson(force_maps["fa"], dti_maps["fa"], mask)
    print(f"  FORCE NDI vs GT Intra Volume Fraction: r = {r_ndi:.4f}")
    print(f"  FORCE FA  vs DTI FA (sanity round-trip): r = {r_force_dti_fa:.4f}")

    suffix = "_tuned" if args.tuned else ""
    out_png = Path(f"validation/force_disco{suffix}.png")
    render_disco_figure(force_maps, dti_maps, gt, mask, out_png)

    # Save the result npz for later
    out_npz = Path(f"validation/force_disco_results{suffix}.npz")
    np.savez(
        out_npz,
        force_nd=force_maps["nd"],
        force_fa=force_maps["fa"],
        force_dispersion=force_maps["dispersion"],
        dti_fa=dti_maps["fa"],
        dti_md=dti_maps["md"],
        gt_ndi=gt["ndi"],
        gt_diameter=gt["diameter"],
        mask=mask,
        r_ndi=r_ndi,
        r_force_dti_fa=r_force_dti_fa,
    )
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()

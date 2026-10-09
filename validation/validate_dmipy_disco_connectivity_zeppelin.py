#!/usr/bin/env python3
"""
dmipy-JAX connectivity on DiSCo with stick + zeppelin compartments (§22).

Same eudx_tracking + connectivity_matrix pipeline as §21, swapping the
3D-stick-only simulator for the new stick+zeppelin simulator that adds
an extra-axonal compartment per fibre with NODDI tortuosity coupling.
Tests whether the richer biophysical model closes the remaining 0.07
Pearson r gap to the paper (r=0.82 → 0.89 target).

Pipeline (identical to §21 except simulator + PAM adapter):
  500K library generation → match → build PAM → eudx_tracking →
  connectivity_matrix.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import nibabel as nib
import numpy as np

import jax
import jax.numpy as jnp

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.matcher import DictionaryMatcher
from dmipy_jax.library.storage import SimulationLibrary
from validation.lib.connectivity_metrics import summarize_method
from validation.lib.dmipy_disco_dict import (
    build_disco_tuned_3d_stick_zeppelin_simulator,
    dmipy_zeppelin_params_to_pam_single,
)
from validation.lib.force_disco import (
    disco_subject_path, load_disco_subject,
)
from validation.lib.force_disco_connectivity import (
    connectivity_pearson, load_gt_connectivity,
)


def build_pam_from_zeppelin_fit(params_maps: dict, mask: np.ndarray,
                                  sphere) -> "PeaksAndMetrics":
    """Construct PeaksAndMetrics from per-voxel 9-param zeppelin params."""
    from dipy.direction.peaks import PeaksAndMetrics

    shape = mask.shape
    peak_dirs = np.zeros(shape + (5, 3), dtype=np.float64)
    peak_values = np.zeros(shape + (5,), dtype=np.float64)
    peak_indices = -np.ones(shape + (5,), dtype=np.int32)

    keys = ["d_par", "theta1", "phi1", "theta2", "phi2",
            "odi", "v_ic", "f1", "f_iso"]
    param_stack = np.stack([params_maps[k] for k in keys], axis=-1)

    idx = np.argwhere(mask)
    for i, j, k in idx:
        pd, pv, pi = dmipy_zeppelin_params_to_pam_single(
            param_stack[i, j, k], sphere,
        )
        peak_dirs[i, j, k] = pd
        peak_values[i, j, k] = pv
        peak_indices[i, j, k] = pi

    pam = PeaksAndMetrics()
    pam.peak_dirs = peak_dirs
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices
    pam.sphere = sphere
    pam.affine = np.eye(4)
    pam.shm_coeff = None
    pam.B = None
    return pam


def run_zeppelin_connectivity(
    subject: int = 1,
    snr: int = 30,
    library_size: int = 500_000,
    seed_density: int = 2,
    step_size: float = 0.5,
    max_angle: float = 45.0,
    pmf_threshold: float = 0.1,
    single_shell_b: float | None = 1900,
) -> dict:
    import warnings as _w
    _w.filterwarnings("ignore")
    from dipy.data import default_sphere
    from dipy.tracking.tracker import eudx_tracking
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.utils import connectivity_matrix, seeds_from_mask

    out = load_disco_subject(subject=subject, snr=snr,
                              single_shell_b=single_shell_b)
    data = out["data"]
    mask = out["mask"]
    rois = out["rois"]
    bvals = np.asarray(out["gtab"].bvals) * 1e6
    bvecs = np.asarray(out["gtab"].bvecs)
    acq = JaxAcquisition(bvalues=jnp.asarray(bvals),
                         gradient_directions=jnp.asarray(bvecs))
    affine = nib.load(
        disco_subject_path(subject) / f"highRes_DiSCo{subject}_DWI.nii.gz"
    ).affine

    sim = build_disco_tuned_3d_stick_zeppelin_simulator(acq)
    print(f"  generating stick+zeppelin library: {library_size:,} entries...")
    gen = LibraryGenerator(sim, chunk_size=20_000)
    params, signals = gen.generate(library_size, key=jax.random.PRNGKey(1))
    jax.block_until_ready(signals)
    lib = SimulationLibrary(
        params=params, signals=signals,
        parameter_names=sim.parameter_names,
    )
    matcher = DictionaryMatcher(lib, k_best=10)

    print(f"  matching {int(mask.sum()):,} voxels...")
    maps = matcher.match_volume(data, mask=mask, batch_size=4096)

    print(f"  building PeaksAndMetrics from matched params...")
    sphere = default_sphere
    pam = build_pam_from_zeppelin_fit(maps, mask, sphere)

    sc_mask = ((rois > 0) | mask).astype(np.uint8)
    sc = BinaryStoppingCriterion(sc_mask)
    seed_mask = (rois > 0) & mask
    seeds = seeds_from_mask(seed_mask, affine, density=seed_density)
    print(f"  tracking ({len(seeds):,} seeds)...")
    streamline_gen = eudx_tracking(
        seeds, sc, affine,
        pam=pam,
        max_cross=None,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        step_size=step_size,
        min_len=4, max_len=300,
        return_all=True,
        random_seed=0,
    )
    streamlines = Streamlines(streamline_gen)
    streamlines = Streamlines([s for s in streamlines if len(s) >= 2])
    print(f"  streamlines: {len(streamlines):,}")

    if len(streamlines) == 0:
        cmat16 = np.zeros((16, 16), dtype=np.float64)
    else:
        cmat, _ = connectivity_matrix(
            streamlines, affine, rois.astype(np.int32),
            return_mapping=True, mapping_as_streamlines=False, symmetric=True,
        )
        cmat16 = cmat[1:17, 1:17].astype(np.float64)

    # Report v_ic histogram inside the brain mask
    v_ic_inside = maps["v_ic"][mask]
    print(f"  v_ic recovered (in-mask): mean={v_ic_inside.mean():.3f} "
          f"std={v_ic_inside.std():.3f} "
          f"[{v_ic_inside.min():.3f}, {v_ic_inside.max():.3f}]")

    return {
        "streamlines_count": len(streamlines),
        "n_seeds": int(len(seeds)),
        "connectivity": cmat16,
        "v_ic_mean": float(v_ic_inside.mean()),
        "v_ic_std": float(v_ic_inside.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--library-size", type=int, default=500_000)
    ap.add_argument("--multi-shell", action="store_true",
                    help="Use all 4 DiSCo shells instead of single-shell "
                         "b=1925. §23 follow-up.")
    args = ap.parse_args()

    shell_tag = "multi-shell (all 4)" if args.multi_shell else "single-shell b=1925"
    suffix = "_multishell" if args.multi_shell else ""
    print("=" * 78)
    print(f"dmipy-JAX connectivity on DiSCo (§22 stick+zeppelin, {shell_tag})")
    print("=" * 78)

    gt = load_gt_connectivity(subject=1)
    print(f"GT: 16×16, {int((gt[np.triu_indices(16, k=1)] > 0).sum())} "
          f"nonzero upper-triangle pairs.")

    results = {}
    cmats = {}
    single_shell_b = None if args.multi_shell else 1900
    for snr in args.snrs:
        print(f"\n=== SNR = {snr} ===")
        res = run_zeppelin_connectivity(
            snr=snr, library_size=args.library_size,
            single_shell_b=single_shell_b,
        )
        r = connectivity_pearson(res["connectivity"], gt)
        results[snr] = {**res, "r": float(r)}
        cmats[snr] = res["connectivity"]
        print(f"  Pearson r vs GT (upper-triangle): {r:.4f}")

    # Specificity-aware summary
    summary = summarize_method(cmats, gt, threshold=0.0)

    out_npz = Path(
        f"validation/dmipy_disco_connectivity_zeppelin_results{suffix}.npz"
    )
    np.savez(
        out_npz,
        snrs=np.array(args.snrs),
        r=np.array([results[s]["r"] for s in args.snrs]),
        ccc_norm=np.array([summary[s]["lin_ccc_sumnorm"] for s in args.snrs]),
        dice=np.array([summary[s]["dice"] for s in args.snrs]),
        precision=np.array([summary[s]["precision"] for s in args.snrs]),
        recall=np.array([summary[s]["recall"] for s in args.snrs]),
        v_ic_mean=np.array([results[s]["v_ic_mean"] for s in args.snrs]),
        v_ic_std=np.array([results[s]["v_ic_std"] for s in args.snrs]),
        n_streamlines=np.array([results[s]["streamlines_count"] for s in args.snrs]),
        gt=gt,
        **{f"cmat_snr{s}": results[s]["connectivity"] for s in args.snrs},
    )
    print(f"\nWrote {out_npz}")

    # Head-to-head vs §21 stick-only and FORCE paper
    print("\n" + "=" * 78)
    print("§22 stick+zeppelin vs §21 stick-only vs FORCE paper")
    print("=" * 78)
    stick_r = {10: 0.7611, 30: 0.7905, 50: 0.8228}
    paper_r = {10: 0.868, 50: 0.894}
    hdr = f"  {'SNR':>3s} {'r (§22)':>9s} {'r (§21)':>9s} " \
          f"{'Δ':>6s} {'paper':>6s} {'CCC':>6s} {'Dice':>6s} {'rec':>6s}"
    print(hdr)
    for snr in args.snrs:
        s = summary[snr]
        delta = s["pearson_r"] - stick_r.get(snr, float("nan"))
        paper = paper_r.get(snr, float("nan"))
        paper_str = f"{paper:>6.3f}" if not np.isnan(paper) else "    —"
        print(f"  {snr:>3d} {s['pearson_r']:>9.4f} "
              f"{stick_r.get(snr, float('nan')):>9.4f} "
              f"{delta:>+6.3f} {paper_str} "
              f"{s['lin_ccc_sumnorm']:>6.3f} {s['dice']:>6.3f} "
              f"{s['recall']:>6.3f}")


if __name__ == "__main__":
    main()

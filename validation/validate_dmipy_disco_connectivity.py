#!/usr/bin/env python3
"""
dmipy-JAX connectivity on DiSCo (Option B2 — full tractography).

Uses the 3D-orientation 2-stick simulator (per-fibre θ + φ instead of
§20's θ-only planar parametrisation) to recover proper 3D fibre
directions, then builds a PeaksAndMetrics object and runs eudx_tracking
+ tck2connectome — identical downstream pipeline to §17.5.

Direct head-to-head with §17.5 (FORCE peaks): same DiSCo data, same
tractography, same connectivity metric. Tests the hypothesis from §20
that dmipy-JAX's win on spatial-pattern recovery (r=0.98) translates
into meaningfully different connectivity matrices.
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import jax
import jax.numpy as jnp

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.matcher import DictionaryMatcher
from dmipy_jax.library.storage import SimulationLibrary
from validation.lib.dmipy_disco_dict import (
    build_disco_tuned_3d_two_stick_simulator,
    dmipy_params_to_pam_single,
)
from validation.lib.force_disco import (
    disco_subject_path, load_disco_subject,
)
from validation.lib.force_disco_connectivity import (
    connectivity_pearson, load_gt_connectivity,
)


def build_pam_from_dict_fit(params_maps: dict, mask: np.ndarray,
                             sphere) -> "PeaksAndMetrics":
    """Construct a dipy PeaksAndMetrics from per-voxel dmipy-JAX params.

    params_maps is the dict returned by DictionaryMatcher.match_volume —
    keys are the simulator parameter names mapped to (X, Y, Z) arrays.
    """
    from dipy.direction.peaks import PeaksAndMetrics

    shape = mask.shape
    peak_dirs = np.zeros(shape + (5, 3), dtype=np.float64)
    peak_values = np.zeros(shape + (5,), dtype=np.float64)
    peak_indices = -np.ones(shape + (5,), dtype=np.int32)

    # Stack the 8 params at each voxel
    keys = ["d_par", "theta1", "phi1", "theta2", "phi2", "odi", "f1", "f_iso"]
    param_stack = np.stack(
        [params_maps[k] for k in keys], axis=-1,
    )  # (X, Y, Z, 8)

    idx = np.argwhere(mask)
    for i, j, k in idx:
        pd, pv, pi = dmipy_params_to_pam_single(param_stack[i, j, k], sphere)
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


def run_dmipy_connectivity(
    subject: int = 1,
    snr: int = 30,
    library_size: int = 500_000,
    seed_density: int = 2,
    step_size: float = 0.5,
    max_angle: float = 45.0,
    pmf_threshold: float = 0.1,
    single_shell_b: float | None = 1900,
) -> dict:
    """End-to-end dmipy-JAX dict → tractography → connectivity pipeline."""
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

    # 1. Generate 3D library
    sim = build_disco_tuned_3d_two_stick_simulator(acq)
    print(f"  generating 3D dmipy-JAX library: {library_size:,} entries...")
    gen = LibraryGenerator(sim, chunk_size=20_000)
    params, signals = gen.generate(library_size, key=jax.random.PRNGKey(1))
    jax.block_until_ready(signals)
    lib = SimulationLibrary(
        params=params, signals=signals,
        parameter_names=sim.parameter_names,
    )
    matcher = DictionaryMatcher(lib, k_best=10)

    # 2. Match every brain-mask voxel
    print(f"  matching {int(mask.sum()):,} voxels...")
    maps = matcher.match_volume(data, mask=mask, batch_size=4096)

    # 3. Build PeaksAndMetrics
    print(f"  building PeaksAndMetrics from matched params...")
    sphere = default_sphere
    pam = build_pam_from_dict_fit(maps, mask, sphere)

    # 4. eudx_tracking
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

    # 5. Connectivity matrix
    if len(streamlines) == 0:
        cmat16 = np.zeros((16, 16), dtype=np.float64)
    else:
        cmat, _ = connectivity_matrix(
            streamlines, affine, rois.astype(np.int32),
            return_mapping=True, mapping_as_streamlines=False, symmetric=True,
        )
        cmat16 = cmat[1:17, 1:17].astype(np.float64)

    return {
        "streamlines_count": len(streamlines),
        "n_seeds": int(len(seeds)),
        "connectivity": cmat16,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--library-size", type=int, default=500_000)
    ap.add_argument("--multi-shell", action="store_true",
                    help="Use all 4 DiSCo shells (b=0/1000/1925/3094/13192) "
                         "instead of single-shell b=1925. §23 follow-up.")
    args = ap.parse_args()

    shell_tag = "multi-shell (all 4)" if args.multi_shell else "single-shell b=1925"
    suffix = "_multishell" if args.multi_shell else ""
    print("=" * 78)
    print(f"dmipy-JAX connectivity on DiSCo (Option B2 — 3D 2-stick, {shell_tag})")
    print("=" * 78)

    gt = load_gt_connectivity(subject=1)
    print(f"GT: 16×16, {int((gt > 0).sum())} nonzero off-diagonal entries.")

    results = {}
    single_shell_b = None if args.multi_shell else 1900
    for snr in args.snrs:
        print(f"\n=== SNR = {snr} ===")
        res = run_dmipy_connectivity(
            snr=snr, library_size=args.library_size,
            single_shell_b=single_shell_b,
        )
        r = connectivity_pearson(res["connectivity"], gt)
        results[snr] = {**res, "r": float(r)}
        print(f"  Pearson r vs GT (upper-triangle): {r:.4f}")

    out_npz = Path(
        f"validation/dmipy_disco_connectivity_results{suffix}.npz"
    )
    np.savez(
        out_npz,
        snrs=np.array(args.snrs),
        r=np.array([results[s]["r"] for s in args.snrs]),
        n_streamlines=np.array([results[s]["streamlines_count"] for s in args.snrs]),
        gt=gt,
        **{f"cmat_snr{s}": results[s]["connectivity"] for s in args.snrs},
    )
    print(f"\nWrote {out_npz}")

    # Comparison vs §17 / §18
    print("\n" + "=" * 72)
    print("dmipy-JAX vs dipy FORCE vs MRtrix SD_STREAM (DiSCo connectivity Pearson r)")
    print("=" * 72)
    force_r = {10: 0.298, 30: 0.211, 50: 0.322}  # §17.5
    mrtrix_r = {10: 0.142, 30: 0.081, 50: 0.131}  # §18.2
    print(f"  {'SNR':>4s}  {'dmipy-JAX':>10s}  {'FORCE':>8s}  "
          f"{'MRtrix':>8s}  {'Paper':>8s}")
    paper = {10: 0.868, 50: 0.894}
    for snr in args.snrs:
        p = f"{paper.get(snr, '—'):.3f}" if snr in paper else "—"
        print(f"  {snr:>4d}  {results[snr]['r']:>10.4f}  "
              f"{force_r.get(snr, '—'):>8.4f}  "
              f"{mrtrix_r.get(snr, '—'):>8.4f}  "
              f"{p:>8s}")


if __name__ == "__main__":
    main()

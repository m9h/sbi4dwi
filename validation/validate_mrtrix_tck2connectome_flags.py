#!/usr/bin/env python3
"""
Sweep `tck2connectome` normalisation flags on the DiSCo MRtrix
SD_STREAM streamlines to test whether one of them closes the §18 gap
(r ≈ 0.13 currently vs paper §3.2 r = 0.868).

We re-use the §18 SD_STREAM pipeline up to the tractography step, cache
the resulting .tck file, then re-run `tck2connectome` for each flag
combination.

Flag combos tested:
  - raw            (baseline; current §18 setting)
  - invnodevol     (-scale_invnodevol)
  - length         (-scale_length)
  - invlength      (-scale_invlength)
  - invnodevol+invlength
  - invnodevol+length
  - stat_mean      (-stat_edge mean)
  - sift_proxy     (-scale_invlength -scale_invnodevol)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from validation.lib.force_disco_connectivity import (
    connectivity_pearson,
    load_gt_connectivity,
)

# Reuse the existing pipeline from §18
sys.path.insert(0, str(Path(__file__).parent))
from validate_mrtrix_disco_connectivity import (
    _ensure_imp_shim_dir,  # noqa: F401  (kept for compatibility)
    _run,
    extract_single_shell_nii,
    DISCO_ROOT,
)


def build_streamlines(snr: int, n_streamlines: int, workdir: Path) -> dict:
    """Run the §18 pipeline up to + including SD_STREAM. Return paths to
    `tck`, `rois_label_mif`, and the workdir."""
    dwi_nii, bval, bvec = extract_single_shell_nii(
        snr, target_b=1900.0, tol=50.0,
        out_path=workdir / "dwi_ss.nii.gz",
    )
    dwi_mif = workdir / "dwi.mif"
    _run(["mrconvert", str(dwi_nii), str(dwi_mif),
          "-fslgrad", str(bvec), str(bval)])

    mask_disco = (
        nib.load(DISCO_ROOT / "highRes_DiSCo1_mask.nii.gz").get_fdata() > 0
    )
    rois = nib.load(DISCO_ROOT / "highRes_DiSCo1_ROIs.nii.gz").get_fdata().astype(int)
    sc_mask_arr = ((rois > 0) | mask_disco).astype(np.uint8)
    affine = nib.load(DISCO_ROOT / "highRes_DiSCo1_DWI.nii.gz").affine
    sc_path = workdir / "sc_mask.nii.gz"
    nib.Nifti1Image(sc_mask_arr, affine).to_filename(sc_path)
    sc_mif = workdir / "sc_mask.mif"
    _run(["mrconvert", str(sc_path), str(sc_mif)])

    tensor_mif = workdir / "tensor.mif"
    _run(["dwi2tensor", str(dwi_mif), str(tensor_mif), "-nthreads", "0"])
    fa_mif = workdir / "fa.mif"
    v1_mif = workdir / "v1.mif"
    _run(["tensor2metric", str(tensor_mif),
          "-fa", str(fa_mif), "-vector", str(v1_mif), "-nthreads", "0"])

    fa_nii = workdir / "fa.nii.gz"
    _run(["mrconvert", str(fa_mif), str(fa_nii)])
    fa_arr = nib.load(fa_nii).get_fdata()
    fa_in_mask = fa_arr[sc_mask_arr > 0]
    thr = float(np.percentile(fa_in_mask, 80))
    sf_arr = ((fa_arr >= thr) & (sc_mask_arr > 0)).astype(np.uint8)
    sf_path = workdir / "sf_mask.nii.gz"
    nib.Nifti1Image(sf_arr, affine).to_filename(sf_path)
    sf_mif = workdir / "sf_mask.mif"
    _run(["mrconvert", str(sf_path), str(sf_mif)])

    response = workdir / "response.txt"
    _run(["amp2response", str(dwi_mif), str(sf_mif), str(v1_mif),
          str(response), "-shells", "1900", "-nthreads", "0"])

    fod = workdir / "fod.mif"
    _run(["dwi2fod", "csd", str(dwi_mif), str(response), str(fod),
          "-mask", str(sc_mif), "-nthreads", "0", "-lmax", "8"])

    rois_path = workdir / "rois_seed.nii.gz"
    seed_mask = ((rois > 0) & mask_disco).astype(np.uint8)
    nib.Nifti1Image(seed_mask, affine).to_filename(rois_path)
    rois_mif = workdir / "rois_seed.mif"
    _run(["mrconvert", str(rois_path), str(rois_mif)])

    tck = workdir / "tracks.tck"
    _run(["tckgen", "-algorithm", "SD_STREAM", str(fod), str(tck),
          "-seed_image", str(rois_mif), "-mask", str(sc_mif),
          "-select", str(n_streamlines),
          "-step", "0.5", "-angle", "45",
          "-minlength", "2", "-maxlength", "200",
          "-nthreads", "0"])

    rois_label_path = workdir / "rois_label.nii.gz"
    nib.Nifti1Image(rois.astype(np.int32), affine).to_filename(rois_label_path)
    rois_label_mif = workdir / "rois_label.mif"
    _run(["mrconvert", str(rois_label_path), str(rois_label_mif),
          "-datatype", "uint32"])

    return {"tck": tck, "rois_label_mif": rois_label_mif, "workdir": workdir}


def compute_connectome(tck: Path, rois_label_mif: Path,
                       extra_flags: list[str], workdir: Path) -> np.ndarray:
    """Run tck2connectome with the given extra flags, return 16×16 matrix."""
    name = "_".join(f.replace("-", "") for f in extra_flags) or "raw"
    out = workdir / f"connectome_{name}.csv"
    cmd = [
        "tck2connectome", str(tck), str(rois_label_mif), str(out),
        "-symmetric", "-zero_diagonal", "-force", "-nthreads", "0",
    ] + extra_flags
    _run(cmd)
    cmat = np.loadtxt(out, delimiter=",")
    assert cmat.shape == (16, 16), f"{cmat.shape} from {extra_flags}"
    return cmat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=int, default=30)
    ap.add_argument("--n-streamlines", type=int, default=100_000)
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix=f"tck2conn_flag_sweep_snr{args.snr}_"))
    print(f"workdir: {workdir}")

    try:
        print(f"\n=== Generating streamlines once (SNR={args.snr}, "
              f"{args.n_streamlines} tracks) ===")
        paths = build_streamlines(args.snr, args.n_streamlines, workdir)

        flag_sets = [
            ("raw",                       []),
            ("invnodevol",                ["-scale_invnodevol"]),
            ("length",                    ["-scale_length"]),
            ("invlength",                 ["-scale_invlength"]),
            ("invnodevol+invlength",      ["-scale_invnodevol", "-scale_invlength"]),
            ("invnodevol+length",         ["-scale_invnodevol", "-scale_length"]),
            ("stat_mean",                 ["-stat_edge", "mean"]),
            ("stat_max",                  ["-stat_edge", "max"]),
        ]

        gt = load_gt_connectivity(subject=1)
        results = {}
        print(f"\n=== Flag sweep ===")
        for name, flags in flag_sets:
            print(f"\n--- {name}: {flags} ---")
            cmat = compute_connectome(paths["tck"], paths["rois_label_mif"],
                                       flags, workdir)
            r = connectivity_pearson(cmat, gt)
            results[name] = {"r": float(r), "cmat": cmat}
            print(f"  Pearson r vs GT: {r:.4f}")

        # Summary
        print("\n" + "=" * 60)
        print(f"tck2connectome flag sweep @ SNR={args.snr}, "
              f"{args.n_streamlines} streamlines")
        print("=" * 60)
        print(f"{'flag set':<24s}  {'Pearson r':>10s}")
        for name in [n for n, _ in flag_sets]:
            print(f"  {name:<22s}  {results[name]['r']:>10.4f}")
        print(f"  {'PAPER §3.2 reported':<22s}  {0.868 if args.snr == 10 else (0.894 if args.snr == 50 else '—'):>10}")

        out_npz = Path(f"validation/mrtrix_tck2conn_flags_snr{args.snr}.npz")
        np.savez(out_npz,
                 names=np.array([n for n, _ in flag_sets]),
                 r=np.array([results[n]["r"] for n, _ in flag_sets]),
                 gt=gt,
                 **{f"cmat_{n}": results[n]["cmat"] for n, _ in flag_sets})
        print(f"\nWrote {out_npz}")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

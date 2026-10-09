#!/usr/bin/env python3
"""
MRtrix3 SD_STREAM connectivity reproduction on DiSCo as an independent
reference against the FORCE §17 pipeline.

The system MRtrix 3.0.4 ``dwi2response`` Python wrapper is broken under
Python 3.12 (imp removal + downstream NameErrors). We replace that step
with a manual response-function estimate using only C++ binaries:

  1. dwi2tensor → tensor.
  2. tensor2metric → FA + V1 (principal eigenvector).
  3. Threshold high-FA voxels = single-fibre mask.
  4. amp2response with that mask + V1 → SH response coefficients.
  5. dwi2fod csd → FOD.
  6. tckgen -algorithm SD_STREAM → deterministic streamlines.
  7. tck2connectome → 16×16 connectivity.
  8. Pearson r vs DiSCo GT cross-sectional-area (upper triangle).

The control question: if SD_STREAM also gets r ≈ 0.3, the paper's r=0.87
is unreachable from public dipy/MRtrix on DiSCo regardless of tool — the
gap is somewhere in the connectivity-metric definition or seeding
strategy. If SD_STREAM reaches r ≈ 0.87, the gap is in dipy's FORCE peak
extraction pipeline specifically.
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


DISCO_ROOT = Path.home() / ".dipy" / "disco" / "disco_1"


IMP_SHIM = '''
"""Minimal shim for Python 3.12, where the stdlib `imp` module was removed.
MRtrix3 3.0.4's dwi2response wrapper still imports `imp.find_module` /
`imp.load_module`. Backed by `importlib.util.spec_from_file_location` /
`module_from_spec` / `Loader.exec_module`."""
import importlib.util
import os


def find_module(name, path=None):
    spec = None
    if path is None:
        path = [os.getcwd()]
    for p in path:
        f = os.path.join(p, name, "__init__.py")
        if os.path.exists(f):
            spec = importlib.util.spec_from_file_location(name, f, submodule_search_locations=[os.path.join(p, name)])
            return None, os.path.join(p, name), ("", "", 5)  # PKG_DIRECTORY = 5
        f = os.path.join(p, name + ".py")
        if os.path.exists(f):
            spec = importlib.util.spec_from_file_location(name, f)
            return None, f, (".py", "r", 1)  # PY_SOURCE = 1
    raise ImportError(name)


def load_module(name, file, pathname, description):
    suffix, mode, type_ = description
    if type_ == 5:  # package
        init_file = os.path.join(pathname, "__init__.py")
        spec = importlib.util.spec_from_file_location(
            name, init_file, submodule_search_locations=[pathname]
        )
    else:
        spec = importlib.util.spec_from_file_location(name, pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sys
    sys.modules[name] = module
    return module
'''


def _ensure_imp_shim_dir() -> Path:
    """Write an `imp` shim into a stable location and return that dir
    so we can prepend it to PYTHONPATH for MRtrix Python helpers."""
    shim_dir = Path.home() / ".cache" / "mrtrix_py312_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_path = shim_dir / "imp.py"
    if not shim_path.exists() or shim_path.read_text() != IMP_SHIM:
        shim_path.write_text(IMP_SHIM)
    return shim_dir


def _run(cmd: list[str], extra_env: dict | None = None, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    import os as _os
    env = _os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        return subprocess.run(
            cmd, check=True, capture_output=True, text=True,
            env=env, **kwargs,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ! command failed (exit {e.returncode})", file=sys.stderr)
        print(f"  ! stderr:\n{e.stderr}", file=sys.stderr)
        raise


def extract_single_shell_nii(snr: int, target_b: float, tol: float, out_path: Path):
    """Save a single-shell DWI volume + matching bvals/bvecs for MRtrix."""
    src = DISCO_ROOT / f"highRes_DiSCo1_DWI_RicianNoise-snr{snr}.nii.gz"
    img = nib.load(src)
    data = img.get_fdata()
    bvals = np.loadtxt(DISCO_ROOT / "DiSCo_gradients.bvals")
    bvecs = np.loadtxt(DISCO_ROOT / "DiSCo_gradients_dipy.bvecs")
    # dipy convention: bvecs as (N, 3) if loaded by read_bvals_bvecs,
    # but np.loadtxt of the file as on disk gives (3, N). Detect:
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T

    keep = (bvals < tol) | (np.abs(bvals - target_b) <= tol)
    sub_data = data[..., keep]
    sub_bvals = bvals[keep]
    sub_bvecs = bvecs[keep]

    nib.Nifti1Image(sub_data.astype(np.float32), img.affine, img.header).to_filename(out_path)
    np.savetxt(out_path.with_suffix(".bval"), sub_bvals[None, :], fmt="%.1f")
    np.savetxt(out_path.with_suffix(".bvec"), sub_bvecs.T, fmt="%.10f")
    return out_path, out_path.with_suffix(".bval"), out_path.with_suffix(".bvec")


def run_mrtrix_pipeline(snr: int, n_streamlines: int = 100_000) -> np.ndarray:
    """Run the full MRtrix SD_STREAM pipeline on DiSCo and return the
    16×16 connectivity matrix."""
    workdir = Path(tempfile.mkdtemp(prefix=f"mrtrix_disco_snr{snr}_"))
    try:
        print(f"  workdir: {workdir}")
        # Extract single-shell b=1900 + b=0
        dwi_nii, bval, bvec = extract_single_shell_nii(
            snr, target_b=1900.0, tol=50.0,
            out_path=workdir / "dwi_ss.nii.gz",
        )

        # Convert to MRtrix .mif
        dwi_mif = workdir / "dwi.mif"
        _run(["mrconvert", str(dwi_nii), str(dwi_mif),
              "-fslgrad", str(bvec), str(bval)])

        # Mask: WM + ROI cylinders so streamlines can enter ROI labels
        mask_disco = (
            nib.load(DISCO_ROOT / "highRes_DiSCo1_mask.nii.gz").get_fdata() > 0
        )
        rois = nib.load(DISCO_ROOT / "highRes_DiSCo1_ROIs.nii.gz").get_fdata().astype(int)
        sc_mask = ((rois > 0) | mask_disco).astype(np.uint8)
        sc_path = workdir / "sc_mask.nii.gz"
        nib.Nifti1Image(
            sc_mask, nib.load(DISCO_ROOT / "highRes_DiSCo1_DWI.nii.gz").affine,
        ).to_filename(sc_path)
        sc_mif = workdir / "sc_mask.mif"
        _run(["mrconvert", str(sc_path), str(sc_mif)])

        # Step 1: tensor → FA + principal eigenvector (all C++)
        tensor_mif = workdir / "tensor.mif"
        _run(["dwi2tensor", str(dwi_mif), str(tensor_mif), "-nthreads", "0"])
        fa_mif = workdir / "fa.mif"
        v1_mif = workdir / "v1.mif"
        _run(["tensor2metric", str(tensor_mif),
              "-fa", str(fa_mif), "-vector", str(v1_mif),
              "-nthreads", "0"])

        # Step 2: single-fibre mask = top-quantile FA voxels in WM mask
        # (replaces the broken `dwi2response tournier`)
        # mrconvert FA to NIfTI so nibabel can read it
        fa_nii = workdir / "fa.nii.gz"
        _run(["mrconvert", str(fa_mif), str(fa_nii)])
        fa_arr = nib.load(fa_nii).get_fdata()
        # Keep voxels with FA above the 80th percentile within the SC mask
        sc_arr = sc_mask  # already computed above
        fa_in_mask = fa_arr[sc_arr > 0]
        thr = float(np.percentile(fa_in_mask, 80))
        sf_mask_arr = ((fa_arr >= thr) & (sc_arr > 0)).astype(np.uint8)
        sf_path = workdir / "sf_mask.nii.gz"
        nib.Nifti1Image(
            sf_mask_arr,
            nib.load(DISCO_ROOT / "highRes_DiSCo1_DWI.nii.gz").affine,
        ).to_filename(sf_path)
        sf_mif = workdir / "sf_mask.mif"
        _run(["mrconvert", str(sf_path), str(sf_mif)])
        print(f"  single-fibre voxels (FA >= {thr:.3f}): {int(sf_mask_arr.sum())}")

        # Step 3: response function via amp2response (all C++).
        # Restrict to the diffusion-weighted shell so we get a single-row
        # response vector that `dwi2fod csd` can consume (otherwise it
        # produces one row per shell and dwi2fod rejects the matrix).
        response = workdir / "response.txt"
        _run(["amp2response", str(dwi_mif), str(sf_mif), str(v1_mif),
              str(response), "-shells", "1900", "-nthreads", "0"])

        # Step 4: CSD FOD
        fod = workdir / "fod.mif"
        _run(["dwi2fod", "csd", str(dwi_mif), str(response), str(fod),
              "-mask", str(sc_mif), "-nthreads", "0", "-lmax", "8"])

        # Step 5: SD_STREAM tractography
        rois_path = workdir / "rois_seed.nii.gz"
        seed_mask = ((rois > 0) & mask_disco).astype(np.uint8)
        nib.Nifti1Image(
            seed_mask, nib.load(DISCO_ROOT / "highRes_DiSCo1_DWI.nii.gz").affine,
        ).to_filename(rois_path)
        rois_mif = workdir / "rois_seed.mif"
        _run(["mrconvert", str(rois_path), str(rois_mif)])

        tck = workdir / "tracks.tck"
        _run(["tckgen", "-algorithm", "SD_STREAM", str(fod), str(tck),
              "-seed_image", str(rois_mif),
              "-mask", str(sc_mif),
              "-select", str(n_streamlines),
              "-step", "0.5",
              "-angle", "45",
              "-minlength", "2",
              "-maxlength", "200",
              "-nthreads", "0"])

        # Connectome
        rois_label_path = workdir / "rois_label.nii.gz"
        nib.Nifti1Image(
            rois.astype(np.int32),
            nib.load(DISCO_ROOT / "highRes_DiSCo1_DWI.nii.gz").affine,
        ).to_filename(rois_label_path)
        rois_label_mif = workdir / "rois_label.mif"
        _run(["mrconvert", str(rois_label_path), str(rois_label_mif),
              "-datatype", "uint32"])

        connectome = workdir / "connectome.csv"
        _run(["tck2connectome", str(tck), str(rois_label_mif), str(connectome),
              "-symmetric", "-zero_diagonal", "-nthreads", "0"])

        cmat = np.loadtxt(connectome, delimiter=",")
        # tck2connectome with 16-label parcellation produces 16×16
        assert cmat.shape == (16, 16), f"Unexpected connectome shape: {cmat.shape}"
        return cmat
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--n-streamlines", type=int, default=100_000)
    args = ap.parse_args()

    gt = load_gt_connectivity(subject=1)
    print(f"Ground-truth connectivity: 16×16, "
          f"{int((gt > 0).sum())} nonzero off-diagonal entries.")

    results = {}
    for snr in args.snrs:
        print(f"\n=== MRtrix SD_STREAM @ SNR = {snr} ===")
        cmat = run_mrtrix_pipeline(snr, n_streamlines=args.n_streamlines)
        r = connectivity_pearson(cmat, gt)
        results[snr] = {"cmat": cmat, "r": r}
        print(f"  Pearson r vs GT (upper-triangle): {r:.4f}")

    # Save NPZ
    out_npz = Path("validation/mrtrix_disco_connectivity_results.npz")
    np.savez(
        out_npz,
        snrs=np.array(args.snrs),
        r=np.array([results[s]["r"] for s in args.snrs]),
        gt=gt,
        **{f"cmat_snr{s}": results[s]["cmat"] for s in args.snrs},
    )
    print(f"\nWrote {out_npz}")

    print("\n" + "=" * 72)
    print("MRtrix3 SD_STREAM connectivity-matrix Pearson r (upper-triangle)")
    print("=" * 72)
    print(f"  {'SNR':>4s}  {'r':>8s}   Paper §3.2 r")
    paper = {10: 0.868, 50: 0.894}
    for snr in args.snrs:
        p = f"{paper.get(snr, '—'):.3f}" if snr in paper else "—"
        print(f"  {snr:>4d}  {results[snr]['r']:>8.4f}   {p:>12s}")


if __name__ == "__main__":
    main()

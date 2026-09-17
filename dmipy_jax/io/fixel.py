"""
Minimal MRtrix3 fixel-directory writer (doc 008 §9.4, plug-in packaging).

A fixel directory holds
  index.mif       (X,Y,Z,2) uint32  — per voxel: number of fixels, offset into the fixel list
  directions.mif  (N,3)     float32 — unit directions per fixel
  <name>.mif      (N,1)     float32 — one value per fixel (fraction, f_i, σ_θ, …)
written here as uncompressed .mif (text header + raw little-endian data,
"float32le"/"uint32le", C-ordered arrays with MRtrix's default layout).
Readable by `mrview`, `fixel2voxel`, `fixelconnectivity`, `tcksift2 -fixel`.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def _write_mif(path: Path, arr: np.ndarray, vox=(1.0, 1.0, 1.0), transform=None, dtype="float32le"):
    arr = np.ascontiguousarray(arr)
    nd = arr.ndim
    # MRtrix default layout "+0,+1,+2,…" = axis 0 fastest; write the buffer in Fortran order to match.
    layout = ",".join(f"+{i}" for i in range(nd))
    tf = np.eye(4) if transform is None else np.asarray(transform, float).copy()
    # MRtrix's 'transform' is rigid: voxel size lives in 'vox', so strip the scaling from an affine
    col = np.linalg.norm(tf[:3, :3], axis=0); col[col == 0] = 1.0; tf[:3, :3] = tf[:3, :3] / col
    hdr = ["mrtrix image", f"dim: {','.join(str(s) for s in arr.shape)}",
           f"vox: {','.join(str(v) for v in (list(vox) + [1.0] * (nd - 3))[:nd])}",
           f"layout: {layout}", f"datatype: {dtype}"]
    for r in range(3):
        hdr.append("transform: " + ",".join(f"{x:g}" for x in tf[r, :4]))
    header = "\n".join(hdr) + "\nfile: . "
    # offset = header length including the trailing 'END\n'; compute iteratively
    off = len(header.encode()) + 12
    for _ in range(3):
        cand = f"{header}{off}\nEND\n"; n = len(cand.encode())
        if n == off: break
        off = n
    data = arr.astype(np.uint32 if dtype.startswith("uint32") else np.float32).astype("<u4" if dtype.startswith("uint32") else "<f4")
    with open(path, "wb") as f:
        f.write(f"{header}{off}\nEND\n".encode()); f.write(data.tobytes(order="F"))


def write_fixel_directory(out_dir, mask: np.ndarray, dirs: np.ndarray, values: dict, vox=(1.0, 1.0, 1.0), transform=None,
                          frac_min: float = 0.0):
    """dirs (N,K,3) and values {name: (N,K)} for the N mask voxels (row order = np.argwhere(mask));
    fixels with ``values['fraction'] < frac_min`` (or zero direction) are dropped."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    idx = np.argwhere(mask); N, K = dirs.shape[:2]
    keep = np.linalg.norm(dirs, axis=2) > 0
    if "fraction" in values and frac_min > 0:
        keep &= np.asarray(values["fraction"]) >= frac_min
    index = np.zeros(mask.shape + (2,), np.uint32); fdirs = []; fvals = {k: [] for k in values}
    off = 0
    for n, (i, j, k) in enumerate(idx):
        m = keep[n]; c = int(m.sum())
        index[i, j, k] = (c, off)
        if c:
            fdirs.append(dirs[n][m])
            for name in values: fvals[name].append(np.asarray(values[name])[n][m])
            off += c
    fdirs = np.concatenate(fdirs) if fdirs else np.zeros((0, 3))
    _write_mif(out / "index.mif", index, vox, transform, "uint32le")
    _write_mif(out / "directions.mif", fdirs.astype(np.float32)[:, :, None], vox, transform)          # (N,3,1)
    for name, v in fvals.items():
        arr = (np.concatenate(v) if v else np.zeros(0)).astype(np.float32)[:, None, None]             # (N,1,1)
        _write_mif(out / f"{name}.mif", arr, vox, transform)
    return int(off)


def fixels_from_posterior(post, fit):
    """Convenience: (dirs, values) for write_fixel_directory from a FixelPosterior + PrismFit."""
    K = post.dirs.shape[1]
    return post.dirs, {"fraction": post.wm_fracs, "sigma_theta_deg": post.sigma_deg,
                       "f_intra": np.repeat(fit.fintra[:, None], K, axis=1)}

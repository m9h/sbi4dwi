#!/usr/bin/env python3
"""
Whole-brain runtime of the refinement stage (doc 008 §8 item 7) on Stanford
HARDI (in vivo, 160 directions, b = 2000): CSD peaks → refine_peaks (K = 3
plus-x MAP, 300 iterations) → Laplace posterior, timed per stage, against
FORCE (dipy 1.12.1, 500K library, default in-vivo prior) on the same mask.
No ground truth here — this is throughput and sanity (D∥ learned, f_i range,
agreement of refined peaks with CSD peaks).
"""
import argparse, time, json
import numpy as np, jax
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, default_sphere
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.direction import peaks_from_model
from dmipy_jax.validation import dipy_refine as dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=300); ap.add_argument("--force-sims", type=int, default=500_000)
    ap.add_argument("--num-cpus", type=int, default=6); ap.add_argument("--slices", type=int, default=0, help="0 = whole brain; else central N axial slices")
    ap.add_argument("--out", default="validation/runtime_hardi.json")
    a = ap.parse_args()
    fdwi, fbval, fbvec = get_fnames(name="stanford_hardi")
    data, affine = load_nifti(fdwi); bvals, bvecs = read_bvals_bvecs(fbval, fbvec); gtab = gradient_table(bvals, bvecs=bvecs)
    _, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3, numpass=1)
    if a.slices:
        z0 = data.shape[2] // 2; keep = np.zeros_like(mask); keep[:, :, z0 - a.slices // 2:z0 + (a.slices + 1) // 2] = True; mask &= keep
    N = int(mask.sum()); print(f"Stanford HARDI {data.shape}, {N:,} mask voxels, {len(bvals)} measurements", flush=True)
    res = {"n_voxels": N, "n_meas": int(len(bvals))}
    t0 = time.time()
    response, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    csd = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=8)
    pam = peaks_from_model(csd, data, default_sphere, relative_peak_threshold=0.5, min_separation_angle=25, mask=mask, npeaks=5, parallel=True, num_processes=a.num_cpus)
    res["t_csd"] = time.time() - t0; print(f"CSD peaks: {res['t_csd']:.0f}s", flush=True)
    t0 = time.time(); pam_ref, post, fit = dr.refine_peaks(data.astype(np.float32), gtab, mask, pam, n_fibres=3, cfg=dr.default_config(3, n_iter=a.n_iter), affine=affine, laplace=False)
    jax.block_until_ready(fit.fintra); res["t_refine"] = time.time() - t0
    print(f"refine (K=3, {a.n_iter} it): {res['t_refine']:.0f}s = {1e3*res['t_refine']/N:.2f} ms/voxel; D∥={fit.d_par*1e9:.2f} D∥ex={(fit.d_par_extra or fit.d_par)*1e9:.2f} "
          f"f_i mean {fit.fintra.mean():.3f} [{np.percentile(fit.fintra, 5):.2f}, {np.percentile(fit.fintra, 95):.2f}]", flush=True)
    from dmipy_jax.validation import prism_uncertainty as pu
    t0 = time.time(); post = pu.laplace_fixel_posterior(fit, data.astype(np.float32), np.asarray(gtab.bvals) * 1e6, np.asarray(gtab.bvecs)); res["t_laplace"] = time.time() - t0
    print(f"Laplace posterior: {res['t_laplace']:.0f}s; median σ_θ (main fixel) {np.median(post.sigma_deg[:, 0]):.2f}°", flush=True)
    # agreement of refined main peak with CSD main peak
    pd0 = np.asarray(pam.peak_dirs)[mask][:, 0]; pd1 = np.asarray(pam_ref.peak_dirs)[mask][:, 0]
    ok = (np.linalg.norm(pd0, axis=1) > 0) & (np.linalg.norm(pd1, axis=1) > 0)
    ang = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", pd0[ok], pd1[ok])), 0, 1)))
    res["main_peak_agreement_deg_median"] = float(np.median(ang)); print(f"main-peak agreement CSD vs refined: median {np.median(ang):.1f}°, 90th pct {np.percentile(ang, 90):.1f}°", flush=True)
    res["dpar"] = float(fit.d_par * 1e9); res["fintra_mean"] = float(fit.fintra.mean())
    try:
        from dipy.reconst.force import FORCEModel, force_peaks
        t0 = time.time(); model = FORCEModel(gtab, compute_odf=False); model.generate(num_simulations=a.force_sims, num_cpus=a.num_cpus, use_cache=True); res["t_force_lib"] = time.time() - t0
        t0 = time.time(); ffit = model.fit(data, mask=mask); _ = force_peaks(ffit, mask=mask); res["t_force_fit"] = time.time() - t0
        print(f"FORCE: library {res['t_force_lib']:.0f}s (cached after first), fit + peaks {res['t_force_fit']:.0f}s = {1e3*res['t_force_fit']/N:.2f} ms/voxel", flush=True)
    except Exception as e:
        print("FORCE skipped:", e)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

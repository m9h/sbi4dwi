#!/usr/bin/env python3
"""
B1 (doc 008 §10): in-vivo scan–rescan reproducibility on HCP-YA retest
subject 105923 (3 shells × 90 directions, 1.25 mm; first visit from
HCP_1200, second from HCP_Retest, both in their own T1w-ACPC space).

Per session (``--stage session --session test|retest``):
  MSMT-CSD peaks (dipy)  →  PRISM-JAX plus-x refinement (K = 3, NLL,
  learned D∥, tortuosity, CSF/GM on) + Laplace fixel posterior  →  FORCE
  (dipy 1.12.1, 1M library)  →  DTI FA (b ≤ 1000)  →  deterministic
  tractography per peak set (MSMT, refine MAP, FORCE, refine posterior
  samples) → Desikan 68-ROI connectomes.

Cross-session (``--stage compare``): rigid T1w registration retest → test,
peak/fixel and scalar maps resampled into the test grid, directions
rotated; then in the WM intersection
  * f_i test–retest agreement (refine f_i, FORCE ND, ND·f_wm, FA reference):
    Pearson r, CCC, mean |Δ|, within-subject CoV;
  * main-fixel angular disagreement and fixel-count agreement per method;
  * calibration: Laplace σ (test ⊕ retest) vs observed Δθ, by decile;
  * connectome reproducibility per method (Pearson on log(1+w), Dice of
    the support), the CV-pruned posterior connectome, and whether the
    posterior edge CV predicts the scan–rescan edge disagreement.

    setsid nohup uv run python validation/validate_hcp_retest.py --stage all > log &
"""
import argparse, json, time, os
from pathlib import Path
import numpy as np, nibabel as nib

ROOT = Path("/data/datasets/hcp"); OUT = ROOT / "b1"
SUBJ = "105923"
WM_LABELS = [2, 41, 251, 252, 253, 254, 255, 77, 7, 46, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58, 28, 60]   # WM + deep GM (tracking mask)
WM_ONLY = [2, 41, 251, 252, 253, 254, 255, 77]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_session(session, slices=0):
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    d = ROOT / session / SUBJ / "T1w"
    img = nib.load(d / "Diffusion" / "data.nii.gz"); affine = img.affine
    bvals, bvecs = read_bvals_bvecs(str(d / "Diffusion" / "bvals"), str(d / "Diffusion" / "bvecs"))
    bvals = np.round(bvals / 100) * 100
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    mask = nib.load(d / "Diffusion" / "nodif_brain_mask.nii.gz").get_fdata() > 0
    aparc = resample_labels(nib.load(d / "aparc+aseg.nii.gz"), img.shape[:3], affine)
    if slices:
        z0 = mask.shape[2] // 2; keep = np.zeros_like(mask); keep[:, :, z0 - slices // 2:z0 + (slices + 1) // 2] = True; mask &= keep
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, affine, gtab, mask, aparc


def resample_labels(lab_img, shape, affine):
    """Nearest-neighbour resampling of a label volume onto (shape, affine)."""
    from scipy.ndimage import map_coordinates
    ijk = np.indices(shape).reshape(3, -1).astype(float)
    world = affine[:3, :3] @ ijk + affine[:3, 3:4]
    src = np.linalg.inv(lab_img.affine); s = src[:3, :3] @ world + src[:3, 3:4]
    lab = np.asarray(lab_img.dataobj)
    out = map_coordinates(lab, s, order=0, mode="constant", cval=0)
    return out.reshape(shape).astype(np.int32)


def desikan_rois(aparc):
    """aparc+aseg cortical labels 1001–1035 / 2001–2035 → 1..68 (0 = none)."""
    labs = [l for l in range(1001, 1036) if l != 1004] + [l for l in range(2001, 2036) if l != 2004]
    rois = np.zeros(aparc.shape, np.int32)
    for n, l in enumerate(labs):
        rois[aparc == l] = n + 1
    return rois, labs


# --------------------------------------------------------------------------- #
# per-method peaks
# --------------------------------------------------------------------------- #
def pam_arrays(pam, mask, k=5):
    pd = np.asarray(pam.peak_dirs)[mask][:, :k]; pv = np.asarray(pam.peak_values)[mask][:, :k]
    nrm = np.linalg.norm(pd, axis=-1, keepdims=True); pd = np.where(nrm > 0, pd / np.maximum(nrm, 1e-12), 0.0)
    return pd.astype(np.float32), pv.astype(np.float32)


def build_pam(dirs, vals, mask, affine, sphere, chunk=200_000):
    """Vectorised PeaksAndMetrics from in-mask (N,K,3) dirs / (N,K) values."""
    from dipy.direction.peaks import PeaksAndMetrics
    N, K = vals.shape; shape = mask.shape
    pdirs = np.zeros(shape + (5, 3)); pvals = np.zeros(shape + (5,)); pidx = -np.ones(shape + (5,), np.int32)
    idx = np.where(mask)
    order = np.argsort(-vals, axis=1); d = np.take_along_axis(dirs, order[:, :, None], 1); v = np.take_along_axis(vals, order, 1)
    d = np.where(v[..., None] > 0, d, 0.0); v = np.where(v > 0, v, 0.0)
    v = v / np.maximum(v.max(1, keepdims=True), 1e-12)               # per-voxel relative peak values: pmf_threshold is relative for every method
    pind = -np.ones((N, K), np.int32); V = sphere.vertices.astype(np.float32)
    for s in range(0, N, chunk):
        dd = d[s:s + chunk].reshape(-1, 3); pi = np.argmax(np.abs(dd @ V.T), axis=1).reshape(-1, K)
        pind[s:s + chunk] = np.where(v[s:s + chunk] > 0, pi, -1)
    tmp = np.zeros((N, 5, 3)); tmp[:, :K] = d; pdirs[idx] = tmp
    tmp = np.zeros((N, 5)); tmp[:, :K] = v; pvals[idx] = tmp
    tmp = -np.ones((N, 5), np.int32); tmp[:, :K] = pind; pidx[idx] = tmp
    pam = PeaksAndMetrics(); pam.peak_dirs = pdirs; pam.peak_values = pvals; pam.peak_indices = pidx
    pam.sphere = sphere; pam.affine = affine; pam.shm_coeff = None; pam.B = None
    return pam


def msmt_peaks(data, gtab, mask, affine, num_cpus):
    from dipy.reconst.mcsd import auto_response_msmt, MultiShellDeconvModel, multi_shell_fiber_response
    from dipy.direction import peaks_from_model
    from dipy.data import default_sphere
    from dipy.core.gradients import unique_bvals_tolerance
    wm, gm, csf = auto_response_msmt(gtab, data)
    ubvals = unique_bvals_tolerance(gtab.bvals)
    response = multi_shell_fiber_response(sh_order_max=8, bvals=ubvals, wm_rf=wm, gm_rf=gm, csf_rf=csf)
    model = MultiShellDeconvModel(gtab, response, sh_order_max=8)
    pam = peaks_from_model(model, data, default_sphere, relative_peak_threshold=0.3, min_separation_angle=25, mask=mask, npeaks=5,
                           parallel=num_cpus > 1, num_processes=num_cpus)
    return pam, {"wm": np.asarray(wm).tolist(), "gm": np.asarray(gm).tolist(), "csf": np.asarray(csf).tolist()}


def fa_map(data, gtab, mask):
    from dipy.reconst.dti import TensorModel
    from dipy.core.gradients import gradient_table
    keep = gtab.bvals <= 1050
    g = gradient_table(gtab.bvals[keep], bvecs=gtab.bvecs[keep], b0_threshold=50)
    fit = TensorModel(g).fit(data[..., keep], mask=mask)
    return np.nan_to_num(fit.fa)[mask].astype(np.float32)


# --------------------------------------------------------------------------- #
# tracking / connectomes
# --------------------------------------------------------------------------- #
def track_connectome(pam, track_mask, seed_mask, rois, affine, n_rois, step=0.625, max_angle=45.0, seed_density=1, random_seed=0):
    from dipy.tracking.tracker import eudx_tracking
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.utils import connectivity_matrix, seeds_from_mask
    sc = BinaryStoppingCriterion(track_mask.astype(np.uint8))
    seeds = seeds_from_mask(seed_mask, affine, density=seed_density)
    gen = eudx_tracking(seeds, sc, affine, pam=pam, max_cross=None, max_angle=max_angle, pmf_threshold=0.1, step_size=step,
                        min_len=10, max_len=400, return_all=True, random_seed=random_seed, nbr_threads=0)
    sl = Streamlines(gen)
    if len(sl) == 0:
        return np.zeros((n_rois, n_rois)), 0
    cm = connectivity_matrix(sl, affine, rois.astype(np.int32), symmetric=True)
    return cm[1:n_rois + 1, 1:n_rois + 1].astype(np.float64), int(len(sl))


# --------------------------------------------------------------------------- #
# stage: one session
# --------------------------------------------------------------------------- #
def run_session(session, a):
    from dipy.data import default_sphere
    from dmipy_jax.validation import dipy_refine as dr
    out = OUT / session; out.mkdir(parents=True, exist_ok=True)
    data, affine, gtab, mask, aparc = load_session(session, a.slices)
    N = int(mask.sum()); log(f"{session}: {data.shape}, {N:,} mask voxels")
    rois, labs = desikan_rois(aparc); n_rois = len(labs)
    wm = np.isin(aparc, WM_ONLY); track_mask = (np.isin(aparc, WM_LABELS) | (rois > 0)) & mask; seed_mask = wm & mask
    np.savez_compressed(out / "grid.npz", affine=affine, mask=mask, aparc=aparc, rois=rois, wm=wm, track_mask=track_mask, labels=np.array(labs))
    res = {"n_mask": N, "n_wm": int((wm & mask).sum()), "n_rois": n_rois}
    # 1. DTI FA
    t0 = time.time(); fa = fa_map(data, gtab, mask); np.save(out / "fa.npy", fa); res["t_fa"] = time.time() - t0; log(f"FA done {res['t_fa']:.0f}s")
    # 2. MSMT-CSD peaks
    if not (out / "msmt.npz").exists() or a.redo:
        t0 = time.time(); pam, resp = msmt_peaks(data, gtab, mask, affine, a.num_cpus); pd, pv = pam_arrays(pam, mask)
        np.savez_compressed(out / "msmt.npz", dirs=pd, vals=pv); json.dump(resp, open(out / "msmt_response.json", "w"))
        res["t_msmt"] = time.time() - t0; log(f"MSMT peaks {res['t_msmt']:.0f}s; mean #peaks {np.mean((pv > 0).sum(1)):.2f}")
    z = np.load(out / "msmt.npz"); msmt_d, msmt_v = z["dirs"], z["vals"]
    # 3. refine + Laplace
    if not (out / "refine.npz").exists() or a.redo:
        import jax
        pam_in = build_pam(msmt_d, msmt_v, mask, affine, default_sphere)
        t0 = time.time(); pam_ref, post, fit = dr.refine_peaks(data, gtab, mask, pam_in, n_fibres=3, cfg=dr.default_config(3, n_iter=a.n_iter), affine=affine, laplace=False)
        jax.block_until_ready(fit.fintra); res["t_refine"] = time.time() - t0
        from dmipy_jax.validation import prism_uncertainty as pu
        t0 = time.time(); post = pu.laplace_fixel_posterior(fit, data, np.asarray(gtab.bvals) * 1e6, np.asarray(gtab.bvecs)); res["t_laplace"] = time.time() - t0
        np.savez_compressed(out / "refine.npz", dirs=fit.dirs.astype(np.float32), wm_fracs=fit.wm_fracs.astype(np.float32), fracs=fit.fracs.astype(np.float32),
                            fintra=fit.fintra.astype(np.float32), d_par=fit.d_par, d_par_extra=fit.d_par_extra or fit.d_par, d_perp=fit.d_perp,
                            sigma_deg=post.sigma_deg.astype(np.float32), cov=post.cov.astype(np.float32), e1=post.e1.astype(np.float32), e2=post.e2.astype(np.float32))
        res["d_par"] = float(fit.d_par * 1e9); res["d_par_extra"] = float((fit.d_par_extra or fit.d_par) * 1e9); res["fintra_wm_mean"] = float(fit.fintra[wm[mask]].mean())
        log(f"refine {res['t_refine']:.0f}s, Laplace {res['t_laplace']:.0f}s; D∥={res['d_par']:.2f} D∥ex={res['d_par_extra']:.2f} f_i(WM)={res['fintra_wm_mean']:.3f} median σ={np.median(post.sigma_deg[:, 0]):.2f}°")
    # 4. FORCE
    if not (out / "force.npz").exists() or a.redo:
        from dipy.reconst.force import FORCEModel, force_peaks
        t0 = time.time(); model = FORCEModel(gtab, compute_odf=False)
        model.generate(num_simulations=a.force_sims, num_cpus=a.num_cpus, use_cache=True); res["t_force_lib"] = time.time() - t0
        t0 = time.time(); ffit = model.fit(data, mask=mask); fpam = force_peaks(ffit, mask=mask); res["t_force"] = time.time() - t0
        fd, fv = pam_arrays(fpam, mask)
        np.savez_compressed(out / "force.npz", dirs=fd, vals=fv, nd=np.asarray(ffit.nd)[mask].astype(np.float32), wm=np.asarray(ffit.wm_fraction)[mask].astype(np.float32),
                            nfib=np.asarray(ffit.num_fibers)[mask].astype(np.float32), unc_nd=np.asarray(ffit.uncertainty_nd)[mask].astype(np.float32))
        log(f"FORCE library {res['t_force_lib']:.0f}s, fit+peaks {res['t_force']:.0f}s")
    del data
    # 5. connectomes
    if not (out / "connectomes.npz").exists() or a.redo or a.redo_connectomes:
        from dmipy_jax.validation import prism_uncertainty as pu
        import jax
        R = np.load(out / "refine.npz"); F = np.load(out / "force.npz"); cms = {}; counts = {}
        keep = R["wm_fracs"] >= 0.10
        sets = {"msmt": (msmt_d, msmt_v), "refine": (R["dirs"], np.where(keep, R["wm_fracs"], 0.0)), "force": (F["dirs"], F["vals"])}
        for name, (d, v) in sets.items():
            t0 = time.time(); cm, n = track_connectome(build_pam(d, v, mask, affine, default_sphere), track_mask, seed_mask, rois, affine, n_rois)
            cms[name] = cm; counts[name] = n; log(f"connectome {name}: {n:,} streamlines, {int((cm > 0).sum() // 2)} edges, {time.time()-t0:.0f}s")
        post = pu.FixelPosterior(dirs=R["dirs"], cov=R["cov"], e1=R["e1"], e2=R["e2"], sigma_deg=R["sigma_deg"], wm_fracs=R["wm_fracs"], mask=mask)
        S = post.sample_dirs(jax.random.key(0), a.n_post); samples = []
        for s in range(a.n_post):
            t0 = time.time(); cm, n = track_connectome(build_pam(S[s], np.where(keep, R["wm_fracs"], 0.0), mask, affine, default_sphere), track_mask, seed_mask, rois, affine, n_rois, random_seed=s + 1)
            samples.append(cm); log(f"posterior sample {s}: {n:,} streamlines, {time.time()-t0:.0f}s")
        samples = np.stack(samples)
        np.savez_compressed(out / "connectomes.npz", msmt=cms["msmt"], refine=cms["refine"], force=cms["force"], posterior=samples, counts=json.dumps(counts))
    json.dump(res, open(out / f"session_results.json", "w"), indent=1)
    log(f"{session} done")


# --------------------------------------------------------------------------- #
# stage: compare
# --------------------------------------------------------------------------- #
def rigid_t1(session_moving="retest", session_static="test"):
    from dipy.align import affine_registration
    mv = nib.load(ROOT / session_moving / SUBJ / "T1w" / "T1w_acpc_dc_restore_brain.nii.gz"); st = nib.load(ROOT / session_static / SUBJ / "T1w" / "T1w_acpc_dc_restore_brain.nii.gz")
    m = np.asarray(mv.dataobj, np.float32); s = np.asarray(st.dataobj, np.float32)
    xformed, reg = affine_registration(m, s, moving_affine=mv.affine, static_affine=st.affine, pipeline=["rigid"], level_iters=[1000, 200, 50], sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1])
    ok = (s > 0) & (xformed > 0)
    r_before = np.corrcoef(m[ok], s[ok])[0, 1] if m.shape == s.shape else np.nan; r_after = np.corrcoef(xformed[ok], s[ok])[0, 1]
    return reg, float(r_before), float(r_after)


def resample_to_test(reg, grid_t, grid_r):
    """For each test in-mask voxel: retest voxel index (nearest) under the rigid map; -1 if outside retest mask."""
    At, Ar = grid_t["affine"], grid_r["affine"]; mt, mr = grid_t["mask"], grid_r["mask"]
    ijk = np.argwhere(mt).T.astype(float); world = At[:3, :3] @ ijk + At[:3, 3:4]
    wm_ = reg[:3, :3] @ world + reg[:3, 3:4]                          # static → moving world
    r_ijk = np.rint(np.linalg.inv(Ar)[:3, :3] @ wm_ + np.linalg.inv(Ar)[:3, 3:4]).astype(int)
    inside = np.all((r_ijk >= 0) & (r_ijk < np.array(mr.shape)[:, None]), axis=0)
    lin = -np.ones(ijk.shape[1], np.int64)
    flat_index = -np.ones(mr.shape, np.int64); flat_index[mr] = np.arange(mr.sum())
    lin[inside] = flat_index[r_ijk[0, inside], r_ijk[1, inside], r_ijk[2, inside]]
    return lin                                                          # (N_test,) index into retest in-mask arrays or -1


def ccc(x, y):
    mx, my = x.mean(), y.mean(); vx, vy = x.var(), y.var(); return float(2 * np.mean((x - mx) * (y - my)) / (vx + vy + (mx - my) ** 2))


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", a, b)), 0, 1)))


def conn_metrics(A, B):
    iu = np.triu_indices_from(A, 1); a, b = A[iu], B[iu]
    r = float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]); sa, sb = a > 0, b > 0
    dice = float(2 * (sa & sb).sum() / max(sa.sum() + sb.sum(), 1))
    return {"r_log": r, "dice": dice, "n_edges": [int(sa.sum()), int(sb.sum())]}


def run_compare(a):
    T = {k: np.load(OUT / "test" / k) for k in ["grid.npz", "msmt.npz", "refine.npz", "force.npz", "connectomes.npz"]}
    Rr = {k: np.load(OUT / "retest" / k) for k in ["grid.npz", "msmt.npz", "refine.npz", "force.npz", "connectomes.npz"]}
    fa_t, fa_r = np.load(OUT / "test" / "fa.npy"), np.load(OUT / "retest" / "fa.npy")
    res = {}
    t0 = time.time(); reg, r0, r1 = rigid_t1(); res["registration"] = {"affine": reg.tolist(), "t1_corr_before": r0, "t1_corr_after": r1,
                                                                     "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(reg[:3, :3]) - 1) / 2, -1, 1)))),
                                                                     "translation_mm": np.linalg.norm(reg[:3, 3]).tolist()}
    log(f"rigid registration {time.time()-t0:.0f}s: T1 corr {r0:.3f} → {r1:.3f}, rotation {res['registration']['rotation_deg']:.2f}°, |t| {res['registration']['translation_mm']:.2f} mm")
    Rm = reg[:3, :3]; Rrot = Rm / np.cbrt(np.linalg.det(Rm))            # static→moving rotation; directions: d_static = Rrot.T d_moving
    lin = resample_to_test(reg, T["grid.npz"], Rr["grid.npz"])
    gt, gr = T["grid.npz"], Rr["grid.npz"]
    wm_t = gt["wm"][gt["mask"]]; wm_r_full = gr["wm"][gr["mask"]]
    ok = lin >= 0; wm_r = np.zeros_like(wm_t); wm_r[ok] = wm_r_full[lin[ok]]
    sel = ok & wm_t & wm_r; res["n_wm_both"] = int(sel.sum()); res["wm_dice"] = float(2 * sel.sum() / (wm_t.sum() + wm_r_full.sum()))
    log(f"WM intersection {sel.sum():,} voxels (Dice of WM masks {res['wm_dice']:.3f})")
    def pull(x):   # retest in-mask array → test in-mask order
        y = np.zeros((len(lin),) + x.shape[1:], x.dtype); y[ok] = x[lin[ok]]; return y
    # ---- scalars
    sc = {"fa": (fa_t, pull(fa_r)), "refine_fi": (T["refine.npz"]["fintra"], pull(Rr["refine.npz"]["fintra"])),
          "force_nd": (T["force.npz"]["nd"], pull(Rr["force.npz"]["nd"])),
          "force_nd_wm": (T["force.npz"]["nd"] * T["force.npz"]["wm"], pull(Rr["force.npz"]["nd"] * Rr["force.npz"]["wm"]))}
    res["scalars"] = {}
    for k, (x, y) in sc.items():
        x, y = x[sel].astype(float), y[sel].astype(float); d = x - y
        res["scalars"][k] = {"r": float(np.corrcoef(x, y)[0, 1]), "ccc": ccc(x, y), "mean_abs_diff": float(np.abs(d).mean()), "bias": float(d.mean()),
                             "wscv": float(np.sqrt(np.mean(d ** 2) / 2) / np.mean((x + y) / 2)), "mean": float(x.mean())}
        o = res["scalars"][k]; log(f"  {k:12s} r={o['r']:.3f} CCC={o['ccc']:.3f} |Δ|={o['mean_abs_diff']:.3f} bias={o['bias']:+.3f} wsCV={100*o['wscv']:.1f}% (mean {o['mean']:.3f})")
    # ---- fixels
    res["fixels"] = {}
    sets = {"msmt": (T["msmt.npz"]["dirs"], T["msmt.npz"]["vals"] > 0, Rr["msmt.npz"]["dirs"], Rr["msmt.npz"]["vals"] > 0),
            "refine": (T["refine.npz"]["dirs"], T["refine.npz"]["wm_fracs"] >= 0.1, Rr["refine.npz"]["dirs"], Rr["refine.npz"]["wm_fracs"] >= 0.1),
            "force": (T["force.npz"]["dirs"], T["force.npz"]["vals"] > 0, Rr["force.npz"]["dirs"], Rr["force.npz"]["vals"] > 0)}
    for k, (dt, kt, dr_, kr) in sets.items():
        dr_ = pull(dr_) @ Rrot; kr = pull(kr)                              # rotate retest dirs into test frame: (d_m^T R) = (R^T d_m)^T
        nt, nr = kt.sum(1), kr.sum(1); both = sel & (nt > 0) & (nr > 0)
        ang = angle_deg(dt[both, 0], dr_[both, 0])                        # main fixel
        # nearest retest fixel to each test fixel (all fixels)
        allang = []
        for f in range(dt.shape[1]):
            m = both & kt[:, f]
            dots = np.abs(np.einsum("ij,ikj->ik", dt[m, f], dr_[m])); dots[~kr[m]] = -1
            allang.append(np.degrees(np.arccos(np.clip(dots.max(1), 0, 1))))
        allang = np.concatenate(allang)
        res["fixels"][k] = {"main_median_deg": float(np.median(ang)), "main_p90_deg": float(np.percentile(ang, 90)), "main_within10": float((ang < 10).mean()),
                            "all_median_deg": float(np.median(allang)), "all_within10": float((allang < 10).mean()),
                            "count_agree": float((nt[sel] == nr[sel]).mean()), "mean_count": [float(nt[sel].mean()), float(nr[sel].mean())], "n": int(both.sum())}
        o = res["fixels"][k]; log(f"  {k:7s} main fixel Δθ median {o['main_median_deg']:.2f}° p90 {o['main_p90_deg']:.1f}° (<10°: {100*o['main_within10']:.1f}%) | all fixels median {o['all_median_deg']:.2f}° | count agree {100*o['count_agree']:.1f}% (mean {o['mean_count'][0]:.2f}/{o['mean_count'][1]:.2f})")
        if k == "refine":
            st, sr = T["refine.npz"]["sigma_deg"][:, 0], pull(Rr["refine.npz"]["sigma_deg"])[:, 0]
            covt, covr = T["refine.npz"]["cov"][:, 0], pull(Rr["refine.npz"]["cov"])[:, 0]
            tr = (np.trace(covt, axis1=1, axis2=2) + np.trace(covr, axis1=1, axis2=2)) * (180 / np.pi) ** 2
            pred_median = np.sqrt(np.maximum(tr, 0) * np.log(2))       # 2-D isotropic Gaussian with total variance tr
            scomb = np.sqrt(st ** 2 + sr ** 2)[both]; pm = pred_median[both]
            from scipy.stats import spearmanr
            q = np.quantile(scomb, np.linspace(0, 1, 11)); bins = []
            for i in range(10):
                m = (scomb >= q[i]) & (scomb <= q[i + 1])
                bins.append({"sigma_comb_median": float(np.median(scomb[m])), "pred_median_deg": float(np.median(pm[m])), "obs_median_deg": float(np.median(ang[m])), "n": int(m.sum())})
            res["calibration"] = {"spearman_sigma_vs_dtheta": float(spearmanr(scomb, ang)[0]), "bins": bins,
                                  "obs_over_pred_median": float(np.median(ang) / np.median(pm)), "frac_within_pred_p90": float(np.mean(ang < np.sqrt(np.maximum(tr[both], 0) * np.log(10))))}
            log(f"  calibration: Spearman(σ, Δθ)={res['calibration']['spearman_sigma_vs_dtheta']:.3f}; obs/pred median {res['calibration']['obs_over_pred_median']:.2f}; within predicted 90% {100*res['calibration']['frac_within_pred_p90']:.1f}%")
            for b in bins: log(f"    σ_comb {b['sigma_comb_median']:5.2f}°  pred median {b['pred_median_deg']:5.2f}°  observed {b['obs_median_deg']:5.2f}°  (n={b['n']:,})")
    # ---- connectomes
    Ct, Cr = T["connectomes.npz"], Rr["connectomes.npz"]; res["connectomes"] = {}
    for k in ["msmt", "refine", "force"]:
        res["connectomes"][k] = conn_metrics(Ct[k], Cr[k]); o = res["connectomes"][k]; log(f"  connectome {k:7s}: r(log)={o['r_log']:.3f} Dice={o['dice']:.3f} edges {o['n_edges']}")
    Pt, Pr = Ct["posterior"], Cr["posterior"]; mt_, mr_ = Pt.mean(0), Pr.mean(0); sdt, sdr = Pt.std(0), Pr.std(0)
    res["connectomes"]["posterior_mean"] = conn_metrics(mt_, mr_)
    for cv in [0.2, 0.3, 0.5]:
        pt = np.where(sdt / np.maximum(mt_, 1e-9) < cv, mt_, 0); pr = np.where(sdr / np.maximum(mr_, 1e-9) < cv, mr_, 0)
        res["connectomes"][f"posterior_cv{cv}"] = conn_metrics(pt, pr); o = res["connectomes"][f"posterior_cv{cv}"]
        log(f"  connectome posterior CV<{cv}: r(log)={o['r_log']:.3f} Dice={o['dice']:.3f} edges {o['n_edges']}")
    o = res["connectomes"]["posterior_mean"]; log(f"  connectome posterior mean: r(log)={o['r_log']:.3f} Dice={o['dice']:.3f}")
    # does the test-session edge CV predict scan–rescan edge disagreement?
    iu = np.triu_indices_from(mt_, 1); m = (mt_[iu] > 0) & (mr_[iu] > 0)
    cv_t = (sdt / np.maximum(mt_, 1e-9))[iu][m]; dis = (np.abs(mt_ - mr_) / np.maximum((mt_ + mr_) / 2, 1e-9))[iu][m]
    from scipy.stats import spearmanr
    res["connectomes"]["edge_cv_vs_disagreement_spearman"] = float(spearmanr(cv_t, dis)[0]); res["connectomes"]["n_shared_edges"] = int(m.sum())
    log(f"  edge CV (test) vs scan–rescan disagreement: Spearman {res['connectomes']['edge_cv_vs_disagreement_spearman']:.3f} over {m.sum()} shared edges")
    json.dump(res, open(OUT / "compare_results.json", "w"), indent=1); log("compare done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["session", "compare", "all"], default="all"); ap.add_argument("--session", choices=["test", "retest"])
    ap.add_argument("--slices", type=int, default=0); ap.add_argument("--n-iter", type=int, default=300); ap.add_argument("--n-post", type=int, default=10)
    ap.add_argument("--force-sims", type=int, default=1_000_000); ap.add_argument("--num-cpus", type=int, default=16); ap.add_argument("--redo", action="store_true"); ap.add_argument("--redo-connectomes", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global OUT
    if a.out: OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("session", "all"):
        for s in ([a.session] if a.session else ["test", "retest"]):
            run_session(s, a)
    if a.stage in ("compare", "all"):
        run_compare(a)


if __name__ == "__main__":
    main()

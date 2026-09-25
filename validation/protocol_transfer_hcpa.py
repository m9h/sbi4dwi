#!/usr/bin/env python3
"""
Protocol transfer HCP-YA → HCP-A by digital twin (doc 008 §12, UCSF GO/TED planning).

For each HCP retest subject the fitted plus-x tissue (first visit, WM) is
taken as ground truth and forward-simulated on
  * the subject's own HCP-YA scheme  (b 1000/2000/3000 × 90 + 18 b0, 1.25 mm; measured WM b0 SNR),
  * the HCP-A / Lifespan scheme      (b 1500 × 93 + b 3000 × 92 + 28 b0, each in AP and PA → 398 volumes,
                                       1.5 mm: SNR × (1.5/1.25)^3 = 1.73 relative to HCP-YA),
  * HCP-A directions at HCP-YA SNR    (isolates the shell/direction effect from the voxel-volume effect),
with Rician noise, then refitted with the default (CSF+GM balls) and the
WM-only (no-iso) plus-x models from a common perturbed initialisation.
Reports bias / sd of f_i and the main-fixel angular error against the
generating truth per protocol, and Fisher/CRLB relative precisions for
canonical voxels per protocol (as acquired and per minute of scan time).

    uv run python validation/protocol_transfer_hcpa.py --subjects 105923 103818 111312 114823 115320
"""
import argparse, json, time, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, nibabel as nib
from dipy.core.sphere import disperse_charges, HemiSphere
sys.path.insert(0, "validation"); import validate_hcp_retest as V
from dmipy_jax.validation import dipy_refine as dr, prism_jax as pj

SCAN_MIN = {"hcp_ya": 59.0, "hcp_a": 21.4}          # dMRI scan time per protocol (YA: 6 × 9:50; A: 2 runs × 2 PE ≈ 21 min)


def electrostatic(n, seed):
    rng = np.random.default_rng(seed); th = np.arccos(rng.uniform(0, 1, n)); ph = rng.uniform(0, 2 * np.pi, n)
    hs, _ = disperse_charges(HemiSphere(theta=th, phi=ph), 5000); return hs.vertices


def hcpa_scheme():
    """Lifespan dMRI: dir98 (b1500 46 + b3000 46 + 6 b0) and dir99 (47 + 46 + 6 b0), AP and PA each."""
    d15 = electrostatic(93, 0); d30 = electrostatic(92, 1)
    bv = np.concatenate([np.zeros(14), np.full(93, 1500.0), np.full(92, 3000.0)])
    gv = np.concatenate([np.tile([[1.0, 0, 0]], (14, 1)), d15, d30])
    order = np.random.default_rng(2).permutation(len(bv))
    bv, gv = bv[order], gv[order]
    return np.concatenate([bv, bv]), np.concatenate([gv, gv])          # AP + PA


def measured_b0_snr(subject, wm):
    d = V.ROOT / "test" / subject / "T1w" / "Diffusion"; b = np.loadtxt(d / "bvals"); img = nib.load(d / "data.nii.gz")
    idx = np.where(b < 50)[0]; X = np.stack([np.asarray(img.dataobj[..., i])[wm] for i in idx], 1)
    return float(np.median(X.mean(1) / X.std(1, ddof=1)))


def truth_phys(R, cfg):
    N = len(R["fintra"]); fi = jnp.asarray(R["fintra"])
    keys = R.files if hasattr(R, "files") else R.keys()
    dpex = float(R["d_par_extra"]) if "d_par_extra" in keys else float(R["d_par"])      # no-iso variant fits saved D∥ only
    return {"s0": jnp.ones(N), "fracs": jnp.asarray(R["fracs"]), "dirs": jnp.asarray(R["dirs"]), "fintra": fi,
            "d_par": jnp.asarray(float(R["d_par"])), "d_par_extra": jnp.asarray(dpex), "d_perp": dpex * (1.0 - fi)}


def rician(key, s, sigma):
    k1, k2 = jax.random.split(key); return jnp.sqrt((s + sigma * jax.random.normal(k1, s.shape)) ** 2 + (sigma * jax.random.normal(k2, s.shape)) ** 2)


def metrics(fit, R, wm_thr=0.1):
    t_fi, e_fi = R["fintra"], fit.fintra; d = e_fi - t_fi
    tmain = R["dirs"][np.arange(len(t_fi)), np.argmax(R["wm_fracs"], 1)]; emain = fit.dirs[np.arange(len(t_fi)), np.argmax(fit.wm_fracs, 1)]
    ang = V.angle_deg(tmain, emain); nt = (R["wm_fracs"] >= wm_thr).sum(1); ne = (fit.wm_fracs >= wm_thr).sum(1)
    return {"fi_bias": float(d.mean()), "fi_sd": float(d.std()), "fi_mad": float(np.abs(d).mean()), "fi_r": float(np.corrcoef(t_fi, e_fi)[0, 1]),
            "main_angle_median": float(np.median(ang)), "main_angle_p90": float(np.percentile(ang, 90)), "main_within10": float((ang < 10).mean()),
            "count_agree": float((nt == ne).mean()), "d_par": float(fit.d_par * 1e9), "d_par_extra": float((fit.d_par_extra or fit.d_par) * 1e9),
            "f_iso_mean": float(fit.fracs[:, :2].sum(1).mean())}


def crlb_table(schemes, snr, cfg):
    """CRLB relative sd of (f_i, D∥, D∥ex, f_csf, f_gm, fibre-1 polar angle) for canonical voxels, Gaussian noise σ = 1/SNR."""
    cases = {"single fibre": ([[0, 0, 1.0]], [0.9]), "2 fibres 60°": ([[0, 0, 1.0], [0, np.sin(np.pi / 3), np.cos(np.pi / 3)]], [0.5, 0.4]),
             "3 fibres": ([[0, 0, 1.0], [0, 1.0, 0], [1.0, 0, 0]], [0.4, 0.3, 0.2])}
    out = {}
    for name, (dirs, wf) in cases.items():
        K = 3; D = np.zeros((K, 3)); D[:len(dirs)] = np.array(dirs); F = np.zeros(K + 3); F[2:2 + len(wf)] = wf; F[0] = 0.05; F[1] = 0.05
        F[2:2 + len(wf)] *= (0.9 / F[2:].sum()) if F[2:].sum() > 0 else 1; F[-1] = 0
        for k in range(len(wf), K): D[k] = [1, 0, 0]
        def theta_to_sig(th, bvals, bvecs):
            fi, dpar, dpex, fcsf, fgm, pol = th
            d0 = jnp.array([jnp.sin(pol), 0.0, jnp.cos(pol)]); dd = jnp.asarray(D).at[0].set(d0)
            fr = jnp.asarray(F).at[0].set(fcsf).at[1].set(fgm); fr = fr.at[2:2 + len(wf)].set(fr[2:2 + len(wf)] * (1 - fcsf - fgm) / fr[2:2 + len(wf)].sum())
            phys = {"s0": jnp.ones(1), "fracs": fr[None], "dirs": dd[None], "fintra": jnp.array([fi]), "d_par": dpar, "d_par_extra": dpex, "d_perp": dpex * (1 - fi)}
            return pj.forward(phys, bvals, bvecs, cfg)[0]
        th0 = jnp.array([0.5, 1.7e-9, 1.2e-9, 0.05, 0.05, 1e-3])
        scale = jnp.array([1.0, 1e-9, 1e-9, 1.0, 1.0, 1.0])
        out[name] = {}
        for sname, (bv, gv, s) in schemes.items():
            J = jax.jacfwd(lambda t: theta_to_sig(t * scale, jnp.asarray(bv * 1e6, jnp.float32), jnp.asarray(gv, jnp.float32)))(th0 / scale)
            Fi = (J.T @ J) * s ** 2
            sd = np.sqrt(np.diag(np.linalg.pinv(np.asarray(Fi, np.float64))))
            Ff = np.asarray(Fi, np.float64)[np.ix_([0, 3, 4, 5], [0, 3, 4, 5])]; sdf = np.sqrt(np.diag(np.linalg.pinv(Ff)))   # diffusivities fixed (global in the fit)
            out[name][sname] = {"sd_fi": float(sd[0]), "sd_dpar_rel": float(sd[1] / 1.7), "sd_dpex_rel": float(sd[2] / 1.2), "sd_fcsf": float(sd[3]), "sd_fgm": float(sd[4]), "sd_angle_deg": float(np.degrees(sd[5])),
                                "fixedD_sd_fi": float(sdf[0]), "fixedD_sd_fcsf": float(sdf[1]), "fixedD_sd_angle_deg": float(np.degrees(sdf[3]))}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", nargs="+", default=["105923"]); ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--out", default="validation/hcp_retest/protocol_transfer_hcpa.json"); ap.add_argument("--slab", type=int, default=0)
    a = ap.parse_args()
    bA, gA = hcpa_scheme(); res = {"hcpa_scheme": {"n_vol": int(len(bA)), "shells": {str(int(b)): int((bA == b).sum()) for b in np.unique(bA)}}, "subjects": {}}
    cfg_iso = dr.default_config(3, n_iter=a.n_iter); cfg_wm = dr.default_config(3, n_iter=a.n_iter, wm_only=True)
    for subj in a.subjects:
        t_sub = time.time(); base = V.ROOT / "b1" / subj / "test"; G = np.load(base / "grid.npz"); R = np.load(base / "refine.npz")
        mask = G["mask"]; wm = G["wm"] & mask
        if a.slab:
            z0 = mask.shape[2] // 2; keep = np.zeros_like(wm); keep[:, :, z0 - a.slab // 2:z0 + (a.slab + 1) // 2] = True; wm &= keep
        sel = wm[mask]                                        # WM subset of the in-mask arrays
        Rw = {k: (R[k][sel] if R[k].ndim >= 1 and R[k].shape[0] == sel.shape[0] else R[k]) for k in R.files}
        d = V.ROOT / "test" / subj / "T1w" / "Diffusion"; bY = np.round(np.loadtxt(d / "bvals") / 100) * 100; gY = np.loadtxt(d / "bvecs").T
        snr_ya = measured_b0_snr(subj, wm); snr_a = snr_ya * (1.5 / 1.25) ** 3
        schemes = {"hcp_ya": (bY, gY, snr_ya), "hcp_a": (bA, gA, snr_a), "hcp_a_at_ya_snr": (bA, gA, snr_ya)}
        V.log(f"{subj}: {sel.sum():,} WM voxels, HCP-YA WM b0 SNR {snr_ya:.1f} → HCP-A {snr_a:.1f}")
        # each model is tested against its own fitted tissue (a self-consistent digital twin per model)
        Rn = np.load(base / "refine_noiso.npz"); Rnw = {k: (Rn[k][sel] if Rn[k].ndim >= 1 and Rn[k].shape[0] == sel.shape[0] else Rn[k]) for k in Rn.files}
        res["subjects"][subj] = {"n_wm": int(sel.sum()), "snr_ya": snr_ya, "snr_a": snr_a, "arms": {}}
        for cname, cfg, Rt in [("default", cfg_iso, Rw), ("noiso", cfg_wm, Rnw)]:
            phys = truth_phys(Rt, cfg); rng = np.random.default_rng(0)
            init_dirs = Rt["dirs"] + rng.normal(0, 0.1, Rt["dirs"].shape); init_dirs /= np.linalg.norm(init_dirs, axis=-1, keepdims=True)
            init_fracs = np.clip(Rt["fracs"] + rng.normal(0, 0.02, Rt["fracs"].shape), 0.01, None); init_fracs /= init_fracs.sum(1, keepdims=True)
            for sname, (bv, gv, snr) in schemes.items():
                bvals = jnp.asarray(bv * 1e6, jnp.float32); bvecs = jnp.asarray(gv, jnp.float32)
                y = np.asarray(rician(jax.random.key({"hcp_ya": 1, "hcp_a": 2, "hcp_a_at_ya_snr": 3}[sname] + (0 if cname == "default" else 10)), pj.forward(phys, bvals, bvecs, cfg), 1.0 / snr))
                vol = np.zeros(mask.shape + (len(bv),), np.float32); vol[wm] = y
                t0 = time.time(); fit = pj.fit_prism(vol, wm, np.asarray(bvals), np.asarray(bvecs), cfg, init_dirs, init_fracs); jax.block_until_ready(fit.fintra)
                m = metrics(fit, Rt); m["t_fit"] = time.time() - t0; res["subjects"][subj]["arms"][f"{sname}/{cname}"] = m
                V.log(f"  {sname:16s} {cname:7s}: f_i bias {m['fi_bias']:+.3f} sd {m['fi_sd']:.3f} r {m['fi_r']:.3f} | main fixel {m['main_angle_median']:.2f}° (<10° {100*m['main_within10']:.0f}%) | count agree {100*m['count_agree']:.0f}% | D∥ {m['d_par']:.2f} f_iso {m['f_iso_mean']:.3f} ({m['t_fit']:.0f}s)")
                del vol
        if subj == a.subjects[0]:
            res["crlb"] = {"default": crlb_table(schemes, None, cfg_iso), "scan_min": SCAN_MIN}
            for case, per in res["crlb"]["default"].items():
                for sname, o in per.items():
                    tmin = SCAN_MIN.get(sname, SCAN_MIN["hcp_a"])
                    V.log(f"  CRLB {case:13s} {sname:16s}: sd f_i {o['sd_fi']:.3f}  D∥ {100*o['sd_dpar_rel']:.1f}%  D∥ex {100*o['sd_dpex_rel']:.1f}%  f_csf {o['sd_fcsf']:.3f}  angle {o['sd_angle_deg']:.2f}°  | D fixed: f_i {o['fixedD_sd_fi']:.3f} f_csf {o['fixedD_sd_fcsf']:.3f} angle {o['fixedD_sd_angle_deg']:.2f}°  | f_i per √min {o['fixedD_sd_fi']*np.sqrt(tmin):.3f}")
        V.log(f"{subj} done in {time.time()-t_sub:.0f}s")
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

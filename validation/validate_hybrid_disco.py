#!/usr/bin/env python3
"""
Hybrid posterior on DiSCo under the FORCE authors' protocol (doc 008 §9.2):
flow proposal (validation/flow_disco_k3.eqx, trained by train_flow_k.py on the
3-shell DiSCo scheme, K = 3, D∥ as a flow parameter) → PRISM-JAX plus-x MAP
→ Laplace posterior → protocol tractography → connectome vs CSA ground truth.

Compared initialisations of the identical MAP: flow, dictionary (doc 007
§8.3), none. Reports intra-VF r, MAP connectome Pearson r / Dice, the
CV-pruned posterior connectome (doc 007 §8.3's best row), and wall time.
"""
import argparse, time, sys, importlib.util
from dataclasses import replace
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, nibabel as nib
from scipy.stats import pearsonr
from dipy.data import default_sphere
sys.path.insert(0, str(Path(__file__).parent))
from diagnose_flow_variants import build_flow
from train_flow_k import dirs_fracs_from_samples
from dmipy_jax.validation import prism_jax as pj, prism_uncertainty as pu
from dmipy_jax.validation.force_disco import disco_subject_path
from dmipy_jax.validation.disco_tracking import peaks_to_pam

vdfp = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location("vdfp", Path(__file__).with_name("validate_disco_force_protocol.py"))); s.loader.exec_module(vdfp)
vpdc = importlib.util.module_from_spec(s2 := importlib.util.spec_from_file_location("vpdc", Path(__file__).with_name("validate_prism_disco_connectivity.py"))); s2.loader.exec_module(vpdc)


def cluster_k(V, F, K, n_iter=10):
    """K-means on the sphere (antipodal) over pooled direction samples → dirs (K,3), fracs (K,)."""
    D = (V[:, :, None] * V[:, None, :]).mean(0); w, E = np.linalg.eigh(D)
    c = np.stack([E[:, -1 - k] for k in range(K)])
    for _ in range(n_iter):
        lab = np.argmax(np.abs(V @ c.T), axis=1)
        for k in range(K):
            m = V[lab == k]
            if len(m) >= 3:
                c[k] = np.linalg.eigh((m[:, :, None] * m[:, None, :]).mean(0))[1][:, -1]
    fr = np.zeros(K)
    for k in range(K):
        m = lab == k
        if m.sum() >= 3: fr[k] = F[m].mean() * m.mean() * K
    o = np.argsort(-fr); return c[o], fr[o]


def flow_proposal(flow, meta, y, n_samples, key, chunk=512):
    K = int(meta["K"]); lo, hi = meta["lo"], meta["hi"]; N = y.shape[0]; param = str(meta["param"]) if "param" in meta.files else "hemi"; n = {"hemi": 2, "dyad": 5}[param]
    lo_j, span_j = jnp.asarray(lo), jnp.asarray(hi - lo)
    sample = jax.jit(lambda k, x: jax.vmap(lambda kk, xx: flow.sample(kk, (n_samples,), condition=xx))(jax.random.split(k, x.shape[0]), x) * span_j + lo_j)
    dirs = np.zeros((N, K, 3)); fr = np.zeros((N, K + 3)); fi = np.zeros(N); dpar = np.zeros(N)
    for i0 in range(0, N, chunk):
        key, sk = jax.random.split(key); S = np.asarray(sample(sk, jnp.asarray(y[i0:i0 + chunk])))       # (B,S,P)
        for j in range(S.shape[0]):
            d, f = dirs_fracs_from_samples(S[j], K, param)
            c, w = cluster_k(d.reshape(-1, 3), f.T.reshape(-1), K)
            w = np.clip(w, 0.02, 1.0); dirs[i0 + j] = c
            fr[i0 + j] = np.concatenate([[0.02, max(1 - w.sum(), 0.02)], w, [0.02]]); fr[i0 + j] /= fr[i0 + j].sum()
            fi[i0 + j] = S[j][:, n * K + K].mean(); dpar[i0 + j] = S[j][:, n * K + K + 1].mean()
    return dirs, fr, fi, dpar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[30, 10]); ap.add_argument("--flow", default="validation/flow_disco_k3.eqx")
    ap.add_argument("--n-samples", type=int, default=200); ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--n-post", type=int, default=20, help="posterior connectome samples (CV pruning)")
    ap.add_argument("--inits", nargs="+", default=["flow", "dict", "none"])
    a = ap.parse_args()
    meta = np.load(a.flow + ".meta.npz", allow_pickle=True); K = int(meta["K"])
    gt_vf = nib.load(disco_subject_path(1) / "highRes_DiSCo1_Strand_Intra_Volume_Fraction.nii.gz").get_fdata()
    for snr in a.snrs:
        d = vdfp.load_protocol_data(snr); mask = d["mask"]; bvals = np.asarray(d["bvals"]) * 1e6; bvecs = np.asarray(d["bvecs"])
        flow = eqx.tree_deserialise_leaves(a.flow, build_flow(jax.random.key(0), len(meta["names"]), len(bvals), int(meta["hidden"]), int(meta["depth"]), embed=False))
        y = d["data"][mask]; b0 = bvals < 50e6; y = y / np.maximum(y[:, b0].mean(1, keepdims=True), 1e-6)
        print(f"\n=== SNR {snr}: {len(y):,} voxels", flush=True)
        inits = {}
        if "flow" in a.inits:
            t0 = time.time(); fd, ff, fi, dp = flow_proposal(flow, meta, y, a.n_samples, jax.random.key(3))
            print(f"  flow proposal {time.time()-t0:.0f}s: f_i r={pearsonr(fi, gt_vf[mask])[0]:.3f} mean {fi.mean():.3f}; D∥ mean {dp.mean():.2f}", flush=True)
            inits["flow"] = (fd, ff, float(np.median(dp)) * 1e-9)
        if "dict" in a.inits:
            t0 = time.time(); dd, df = vpdc._warm_start({"data": d["data"], "mask": mask}, bvals, bvecs, K, 300_000)
            print(f"  dictionary warm start {time.time()-t0:.0f}s", flush=True); inits["dict"] = (dd, df, 0.6e-9)
        if "none" in a.inits:
            inits["none"] = (None, None, 0.6e-9)
        for name, (idirs, ifr, dpar0) in inits.items():
            t0 = time.time()
            cfg = replace(pj.PrismConfig(n_fibres=K, n_iter=a.n_iter, loss="nll"), learn_diffusivities=True, d_par=dpar0,
                          tortuosity=True, use_restricted=False, separate_extra_dpar=True)
            fit = pj.fit_prism(d["data"], mask, bvals, bvecs, cfg, idirs, ifr); t_map = time.time() - t0
            r_vf = pearsonr(fit.fintra, gt_vf[mask])[0]
            pam = pj.prism_fit_to_pam(fit, default_sphere, peak_frac_min=0.10, affine=d["affine"])
            cm, _ = vdfp.track_protocol(pam, d); r_map, dsc = vdfp.score(cm, d["gt"])
            t0 = time.time(); post = pu.laplace_fixel_posterior(fit, d["data"], bvals, bvecs); t_lap = time.time() - t0
            S = post.sample_dirs(jax.random.PRNGKey(0), a.n_post); idx = np.argwhere(mask); cms = []
            for s_ in range(a.n_post):
                pd = np.zeros(mask.shape + (5, 3)); pv = np.zeros(mask.shape + (5,))
                for n, (i, j, k) in enumerate(idx):
                    for f in range(min(K, 5)):
                        v = post.wm_fracs[n, f]
                        if v >= 0.10: pd[i, j, k, f] = S[s_, n, f]; pv[i, j, k, f] = v
                cms.append(vdfp.track_protocol(peaks_to_pam(pd, pv, mask, default_sphere, d["affine"]), d)[0])
            cms = np.array(cms); mean_cm = cms.mean(0); cv = cms.std(0) / np.maximum(mean_cm, 1e-9)
            pruned = np.where(cv < 0.3, mean_cm, 0.0); r_cv, dsc_cv = vdfp.score(pruned, d["gt"])
            print(f"  {name:5s} init: intra-VF r={r_vf:.3f}  D∥={fit.d_par*1e9:.2f}  MAP connectome r={r_map:.3f} Dice={dsc['dice']:.2f}  "
                  f"CV<0.3 posterior r={r_cv:.3f} Dice={dsc_cv['dice']:.2f}  (MAP {t_map:.0f}s, Laplace {t_lap:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

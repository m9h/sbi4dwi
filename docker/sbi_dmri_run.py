#!/usr/bin/env python3
"""
Run Nottingham SBI_dMRI (Ball-and-Sticks NPE) on an exported benchmark.

Uses their own forward model, priors, noise policy and post-processing
modules directly (their config-driven pipeline is CPU-only), with the
current `sbi` API, on the GPU. One NSF posterior is trained per
acquisition (the flow conditions on the raw attenuation signal, so it is
acquisition-specific — as in their paper's acquisition-specific variant).

  sbi_dmri_run.py --npz /data/synthetic_snr30.npz --kind synthetic --nfib 2 --n-train 1000000
  sbi_dmri_run.py --npz /data/substrate_signals.npz --kind substrate --nfib 2
  sbi_dmri_run.py --nii dwi.nii.gz --bvals b --bvecs v --mask m.nii.gz --nfib 3 --out-samples s.npz
"""
import argparse, json, time, sys
import numpy as np, torch

sys.path.insert(0, "/workspace/sbi_dmri")
from models.ball_and_sticks.simulator import GradientTable, BallAndSticksAttenuation
from priors.ball_and_sticks import BallAndSticksPriorConfig, build_ball_and_sticks_priors
from models.ball_and_sticks.postprocess import BallAndSticksLayout, sph_to_cart
from tools.noise import NoiseConfig, apply_noise_policy

try:
    from sbi.inference import NPE as _NPE
except ImportError:                      # sbi < 0.23
    from sbi.inference import SNPE as _NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform


def gtab_from_arrays(bvals_mm2, bvecs, device):
    """Their GradientTable drops b0s and expects s/mm² bvals; build via temp files."""
    import tempfile, os
    keep = bvals_mm2 > 50
    d = tempfile.mkdtemp()
    np.savetxt(os.path.join(d, "bvals"), bvals_mm2[keep][None], fmt="%.1f")
    np.savetxt(os.path.join(d, "bvecs"), bvecs[keep].T, fmt="%.6f")
    return GradientTable.read_bvals_bvecs(os.path.join(d, "bvals"), os.path.join(d, "bvecs"), device=device), keep


def train_posterior(gtab, nfib, modelnum, n_train, device, snr_min, snr_max, epochs, seed):
    torch.manual_seed(seed)
    sim = BallAndSticksAttenuation(gtab=gtab, device=device)
    cfg = BallAndSticksPriorConfig(nfib=nfib, modelnum=modelnum, hemisphere=True, include_snr=False)
    prior = build_ball_and_sticks_priors(cfg)
    theta = prior.sample((n_train,)).to(device)
    with torch.no_grad():
        x_clean = torch.cat([sim(theta[i:i + 65536], nfib=nfib, modelnum=modelnum)
                             for i in range(0, n_train, 65536)])
    ncfg = NoiseConfig(noise_type="rician", strategy="random", snr_min=snr_min, snr_max=snr_max,
                       n_levels=8, snr_fixed=30.0, snr_fixed_jitter=0.0, device=device)
    x_noisy, snr = apply_noise_policy(x_clean, ncfg)
    # the flow needs a box prior object for sbi; wrap their MultipleIndependent sample bounds
    lo = theta.min(0).values.cpu(); hi = theta.max(0).values.cpu()
    box = BoxUniform(low=lo - 1e-6, high=hi + 1e-6, device=device)
    net = posterior_nn(model="nsf", num_transforms=8, hidden_features=64)
    inf = _NPE(prior=box, density_estimator=net, device=device)
    inf.append_simulations(theta.to(device), x_noisy.to(device))
    t0 = time.time()
    de = inf.train(training_batch_size=4096, max_num_epochs=epochs, learning_rate=5e-4,
                   show_train_summary=False)
    print(f"trained NSF on {n_train:,} sims in {time.time()-t0:.0f}s", flush=True)
    # Sample from the density estimator directly: the DirectPosterior wrapper
    # rejection-samples against the box prior and, for observations where the
    # flow leaks mass outside the box, loops for hours (observed on the GB10).
    # Samples are clipped to the box instead.
    return (de, lo.to(device), hi.to(device)), BallAndSticksLayout(nfib=nfib, modelnum=modelnum)


def cluster_modes(v_all, f_all, K=2, n_iter=10):
    """Label-switching-robust fixels: pool all fibres' direction samples of one
    voxel, 2-means on the sphere (antipodal), read each mode as a fixel."""
    D = (v_all[:, :, None] * v_all[:, None, :]).mean(0); w, E = np.linalg.eigh(D)
    c = np.stack([E[:, -1 - k] for k in range(K)])
    for _ in range(n_iter):
        lab = np.argmax(np.abs(v_all @ c.T), axis=1)
        for k in range(K):
            m = v_all[lab == k]
            if len(m) >= 3:
                Dk = (m[:, :, None] * m[:, None, :]).mean(0); c[k] = np.linalg.eigh(Dk)[1][:, -1]
    dirs, fr, sig = np.zeros((K, 3)), np.zeros(K), np.full(K, 90.0)
    for k in range(K):
        m = v_all[lab == k]; f = f_all[lab == k]
        dirs[k] = c[k]
        if len(m) >= 3:
            fr[k] = f.mean() * len(m) / len(v_all) * K
            sig[k] = np.degrees(np.sqrt(np.mean(np.arccos(np.clip(np.abs(m @ c[k]), 0, 1)) ** 2)))
    o = np.argsort(-fr)
    return dirs[o], fr[o], sig[o]


def infer(posterior, layout, x, n_samples, device, chunk=256, cluster=False):
    """x: (N,G) attenuation (b0-normalised, b0s removed). Returns dirs (N,nfib,3),
    fracs (N,nfib), sigma_deg (N,nfib) from posterior samples, fibres sorted by
    fraction, direction = principal eigenvector of the sample dyadic. Uses
    sbi's batched sampling (one call per chunk of voxels) — the per-voxel
    loop their pipeline uses took ~1 s/voxel on the GB10."""
    N = x.shape[0]; K = layout.nfib
    dirs = np.zeros((N, K, 3)); fr = np.zeros((N, K)); sig = np.zeros((N, K))
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    for i0 in range(0, N, chunk):
        xb = xt[i0:i0 + chunk]
        de, lo, hi = posterior
        with torch.no_grad():
            sb = de.sample((n_samples,), condition=xb)                                   # (S,B,P)
        sb = torch.clamp(sb, lo, hi).permute(1, 0, 2).cpu().numpy()                      # (B,S,P)
        for j in range(sb.shape[0]):
            s = sb[j]
            f = np.stack([s[:, layout.idx_f(k + 1)] for k in range(K)], 1)
            if cluster:
                vs = [np.stack([np.sin(s[:, layout.idx_theta(k + 1)]) * np.cos(s[:, layout.idx_phi(k + 1)]),
                                np.sin(s[:, layout.idx_theta(k + 1)]) * np.sin(s[:, layout.idx_phi(k + 1)]),
                                np.cos(s[:, layout.idx_theta(k + 1)])], 1) for k in range(K)]
                dirs[i0 + j], fr[i0 + j], sig[i0 + j] = cluster_modes(np.concatenate(vs), f.T.reshape(-1), K)
                continue
            order = np.argsort(-f.mean(0))
            for jj, k in enumerate(order):
                th, ph = s[:, layout.idx_theta(k + 1)], s[:, layout.idx_phi(k + 1)]
                v = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
                D = (v[:, :, None] * v[:, None, :]).mean(0)
                w, V = np.linalg.eigh(D); mu = V[:, -1]
                dirs[i0 + j, jj] = mu; fr[i0 + j, jj] = f[:, k].mean()
                cos = np.clip(np.abs(v @ mu), 0, 1)
                sig[i0 + j, jj] = np.degrees(np.sqrt(np.mean(np.arccos(cos) ** 2)))
        if i0 == 0:
            print(f"  first chunk of {xb.shape[0]} voxels sampled", flush=True)
    return dirs, fr, sig


def best_match(pred_dirs, pred_fr, gt, fmin=0.05):
    errs, hits, tot = [], 0, 0
    for n in range(gt.shape[0]):
        for g in gt[n]:
            if np.linalg.norm(g) == 0: continue
            tot += 1
            cand = [d for d, f in zip(pred_dirs[n], pred_fr[n]) if f >= fmin]
            if not cand: continue
            e = np.degrees(np.arccos(np.clip(max(abs(float(g @ d)) for d in cand), -1, 1)))
            errs.append(e); hits += int(e < 20)
    return (float(np.mean(errs)) if errs else float("nan")), hits / tot if tot else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz"); ap.add_argument("--kind", choices=["synthetic", "substrate"])
    ap.add_argument("--nfib", type=int, default=2); ap.add_argument("--modelnum", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=1_000_000); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=500); ap.add_argument("--snr-min", type=float, default=5.0)
    ap.add_argument("--snr-max", type=float, default=80.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cluster", action="store_true",
                    help="label-switching-robust fixel extraction (pool + 2-means on the sphere) "
                         "instead of the published sort-by-fraction")
    ap.add_argument("--de-cache", default=None, help="torch.save/load the trained density estimator here")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
    z = np.load(a.npz, allow_pickle=True)
    bvals = np.asarray(z["bvals"], float); bvals = bvals if bvals.max() < 1e5 else bvals / 1e6
    gtab, keep = gtab_from_arrays(bvals, np.asarray(z["bvecs"], float), device)
    import os
    if a.de_cache and os.path.exists(a.de_cache):
        de, lo, hi = torch.load(a.de_cache, map_location=device, weights_only=False)
        post, layout = (de.to(device), lo.to(device), hi.to(device)), BallAndSticksLayout(nfib=a.nfib, modelnum=a.modelnum)
        print("loaded density estimator from", a.de_cache, flush=True)
    else:
        post, layout = train_posterior(gtab, a.nfib, a.modelnum, a.n_train, device, a.snr_min, a.snr_max, a.epochs, a.seed)
        if a.de_cache:
            torch.save((post[0].cpu(), post[1].cpu(), post[2].cpu()), a.de_cache); post = (post[0].to(device), post[1], post[2])
    res = {}

    def prep(data, mask):
        y = data[mask]; b0 = y[:, ~keep].mean(1, keepdims=True); return y[:, keep] / np.maximum(b0, 1e-6)

    if a.kind == "synthetic":
        data, mask = z["data"], z["mask"]; ang = z["angle"]; gt = z["gt_dirs"].copy(); gt[ang == 0, 1] = 0
        t0 = time.time(); dirs, fr, sig = infer(post, layout, prep(data, mask), a.n_samples, device, cluster=a.cluster)
        e, r = best_match(dirs, fr, gt); res["overall"] = (e, r)
        print(f"SBI_dMRI synthetic: overall err={e:.2f}° recall={100*r:.1f}%  median σ_θ={np.median(sig[:,0]):.2f}°  ({time.time()-t0:.0f}s)", flush=True)
        for av in sorted(set(ang.tolist())):
            m = ang == av; e, r = best_match(dirs[m], fr[m], gt[m]); res[str(int(av))] = (e, r)
            print(f"  {int(av):>3d}  {e:6.2f}° / {100*r:5.1f}%   σ_θ={np.median(sig[m,0]):.2f}°")
        np.savez((a.out or "sbi_dmri_out") + "_samples.npz", dirs=dirs, fracs=fr, sigma_deg=sig)
    else:
        keys = sorted({k.rsplit("_", 1)[0] for k in z.files if k.endswith("_data")})
        for k in keys:
            data = z[f"{k}_data"]; mask = np.ones(data.shape[:3], bool); axes = z[f"{k}_axes"]; fi = float(z[f"{k}_f_intra"])
            N = int(mask.sum()); gt = np.zeros((N, 2, 3)); gt[:, 0] = axes[0]
            if axes.shape[0] > 1: gt[:, 1] = axes[1]
            dirs, fr, sig = infer(post, layout, prep(data, mask), a.n_samples, device, cluster=a.cluster)
            e, r = best_match(dirs, fr, gt)
            print(f"SBI_dMRI {k:14s} err={e:6.2f}° recall={100*r:5.1f}%  f_sticks={fr.sum(1).mean():.3f} (geom f_i {fi:.3f})", flush=True)
            res[k] = (e, r, float(fr.sum(1).mean()), fi)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

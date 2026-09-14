#!/usr/bin/env python3
"""
sbi4dwi amortised flow posterior (doc 008 §5 run 3): train a conditional
normalising flow (NPE) on the PRISM K=2 forward model for the PRISM
3-shell scheme, then on the synthetic benchmark report angular error /
recall from posterior means, 90 % cone coverage from posterior samples,
and SBC rank uniformity on held-out simulations — next to the Laplace
numbers of doc 007 §6.12.

Parameters: [θ1, φ1, θ2, φ2, f_wm1, f_wm2, f_i]; fibre 1 is the larger
fraction by construction of the prior (f_wm1 ∈ (0.3,1), f_wm2 ∈ (0,0.6),
f_wm2 ≤ 1−f_wm1 via clipping in the forward) — the same layout as the
dictionary warm start. Directions are antipodally symmetric, so posterior
samples are folded to the hemisphere of the posterior mean before
computing dyadics.
"""
import argparse, time
import numpy as np, jax, jax.numpy as jnp
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.validation.prism_jax import PrismConfig, forward, angular_error_best_match
from dmipy_jax.validation import prism_synthetic as ps

NAMES = ["theta1", "phi1", "theta2", "phi2", "f_wm1", "f_wm2", "f_i"]
RANGES = {"theta1": (0, np.pi), "phi1": (0, 2 * np.pi), "theta2": (0, np.pi), "phi2": (0, 2 * np.pi),
          "f_wm1": (0.3, 1.0), "f_wm2": (0.0, 0.6), "f_i": (0.2, 0.8)}


def make_simulator(bvals_si, bvecs, snr_range):
    cfg = PrismConfig(n_fibres=2)
    acq = JaxAcquisition(bvalues=jnp.asarray(bvals_si), gradient_directions=jnp.asarray(bvecs))

    def fwd(p, acq):
        t1, p1, t2, p2, f1, f2, fi = p
        d = jnp.stack([jnp.array([jnp.sin(t1) * jnp.cos(p1), jnp.sin(t1) * jnp.sin(p1), jnp.cos(t1)]),
                       jnp.array([jnp.sin(t2) * jnp.cos(p2), jnp.sin(t2) * jnp.sin(p2), jnp.cos(t2)])])
        f2c = jnp.minimum(f2, 1.0 - f1)
        fr = jnp.array([0.0, 0.0, f1, f2c, jnp.maximum(1.0 - f1 - f2c, 0.0)])
        phys = {"s0": jnp.ones(1), "fracs": fr[None], "dirs": d[None], "fintra": jnp.array([fi]),
                "d_par": jnp.asarray(cfg.d_par), "d_perp": jnp.asarray(cfg.d_perp), "sigma": None}
        return forward(phys, acq.bvalues, acq.gradient_directions, cfg)[0]

    return ModelSimulator(forward_fn=fwd, parameter_names=NAMES, parameter_ranges=RANGES,
                          acquisition=acq, noise_type="rician", snr=30.0, snr_range=snr_range)


def sph(t, p):
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)


def summarise(samples):
    """samples (S,7) → dirs (2,3), fracs (2,), sigma_deg (2,), tangent cov (2,2,2)."""
    dirs, fr, sig, covs = [], [], [], []
    for k, (it, ip, ifr) in enumerate(((0, 1, 4), (2, 3, 5))):
        v = sph(samples[:, it], samples[:, ip])
        D = (v[:, :, None] * v[:, None, :]).mean(0); w, V = np.linalg.eigh(D); mu = V[:, -1]
        v = np.where((v @ mu)[:, None] < 0, -v, v)                      # fold antipodes
        e1, e2 = V[:, 0], V[:, 1]
        t = np.stack([v @ e1, v @ e2], 1)                               # tangent offsets
        dirs.append(mu); fr.append(samples[:, ifr].mean()); covs.append(np.cov(t.T) + 1e-12 * np.eye(2))
        sig.append(np.degrees(np.sqrt(np.mean(np.arccos(np.clip(v @ mu, -1, 1)) ** 2))))
    return np.array(dirs), np.array(fr), np.array(sig), np.array(covs), (V[:, 0], V[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=30000); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--snr-eval", type=float, default=30); ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--n-sbc", type=int, default=300); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6); ap.add_argument("--flow", default="spline")
    a = ap.parse_args()

    b = ps.make_benchmark(snr=a.snr_eval)
    sim = make_simulator(b["bvals"], b["bvecs"], snr_range=(8.0, 60.0))
    cfg = SBIPipelineConfig(model_name="prism_k2", parameter_names=NAMES, parameter_ranges=RANGES,
                            inference_mode="flow", flow_type=a.flow, knots=8, hidden_dim=a.hidden, depth=a.depth,
                            noise_type="rician", snr_range=(8.0, 60.0), learning_rate=5e-4,
                            batch_size=a.batch, n_steps=a.n_steps, lr_schedule="warmup_cosine", warmup_steps=1000)
    t0 = time.time()
    flow, losses = train_sbi(cfg, sim, key=jax.random.PRNGKey(0), print_every=2000)
    print(f"flow trained: {a.n_steps} steps × {a.batch} = {a.n_steps*a.batch/1e6:.1f}M sims, "
          f"loss {losses[0]:.2f} → {np.mean(losses[-200:]):.2f}, {time.time()-t0:.0f}s", flush=True)

    # --- benchmark: posterior per voxel
    y = b["data"][b["mask"]]; b0 = b["bvals"] < 50e6; y = y / y[:, b0].mean(1, keepdims=True)
    N = y.shape[0]; K = 2
    dirs = np.zeros((N, K, 3)); fr = np.zeros((N, K)); sig = np.zeros((N, K)); covs = np.zeros((N, K, 2, 2))
    key = jax.random.PRNGKey(1)
    sample = jax.jit(lambda k, x: flow.sample(k, (a.n_samples,), condition=x))
    t0 = time.time()
    for i in range(N):
        key, sk = jax.random.split(key)
        s = np.asarray(sample(sk, jnp.asarray(y[i])))
        d, f, sg, cv, _ = summarise(s)
        order = np.argsort(-f); dirs[i] = d[order]; fr[i] = f[order]; sig[i] = sg[order]; covs[i] = cv[order]
    print(f"posterior sampling {N} voxels × {a.n_samples}: {time.time()-t0:.0f}s", flush=True)
    gt = b["gt_dirs"].copy(); gt[b["angle"] == 0, 1] = 0
    err, rec = angular_error_best_match(dirs, fr, gt)
    # 90 % cone coverage: Mahalanobis of truth in the sample tangent covariance vs χ²₂(0.9)
    from scipy.stats import chi2
    q = chi2.ppf(0.9, 2); hits = tot = 0; per = {}
    for i in range(N):
        for g in gt[i]:
            if np.linalg.norm(g) == 0: continue
            k = int(np.argmax(np.abs(dirs[i] @ g))); mu = dirs[i, k]
            gs = g if g @ mu >= 0 else -g
            # tangent basis consistent with summarise(): eigenvectors of the dyadic
            e1 = np.cross(mu, [1, 0, 0] if abs(mu[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(mu, e1)
            # covariance in this basis from σ (isotropic approx) — use sample cov rotated: approximate with sigma
            s2 = np.radians(sig[i, k]) ** 2 / 2
            t = np.array([gs @ e1, gs @ e2]); m = (t @ t) / max(s2, 1e-10)
            hit = m <= q; hits += hit; tot += 1
            per.setdefault(int(b["angle"][i]), []).append(hit)
    print(f"\nFLOW posterior @ SNR {a.snr_eval:.0f}: err={err:.2f}° recall={100*rec:.1f}%  "
          f"90% cone coverage={100*hits/tot:.1f}%  median σ_θ(main)={np.median(sig[:,0]):.2f}°")
    print("angle  coverage  median σ_θ  mean err")
    for av in list(ps.ANGLES) + [0]:
        m = b["angle"] == av
        e, _ = angular_error_best_match(dirs[m], fr[m], gt[m])
        print(f"  {av:>3d}  {100*np.mean(per[av]):6.1f}%  {np.median(sig[m,0]):8.2f}°  {e:6.2f}°")

    # --- SBC on held-out simulations (rank of truth among posterior samples, per parameter)
    kk = jax.random.PRNGKey(7); th, xs = sim.sample_and_simulate(kk, a.n_sbc)
    ranks = np.zeros((a.n_sbc, len(NAMES)))
    for i in range(a.n_sbc):
        kk, sk = jax.random.split(kk); s = np.asarray(sample(sk, xs[i]))
        ranks[i] = (s < np.asarray(th[i])[None]).sum(0)
    # uniformity: KS distance of rank/L to U(0,1) per parameter, and 90% coverage
    from scipy.stats import kstest
    print("\nSBC on %d held-out sims (%d samples): param  KS-p  90%%-coverage" % (a.n_sbc, a.n_samples))
    for j, nme in enumerate(NAMES):
        u = ranks[:, j] / a.n_samples; p = kstest(u, "uniform").pvalue
        cov = np.mean((u > 0.05) & (u < 0.95))
        print(f"  {nme:7s}  {p:6.3f}  {100*cov:5.1f}%")
    np.savez("validation/flow_sbi_results.npz", dirs=dirs, fracs=fr, sigma_deg=sig, ranks=ranks, losses=np.array(losses))


if __name__ == "__main__":
    main()

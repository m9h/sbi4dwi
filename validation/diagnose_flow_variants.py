#!/usr/bin/env python3
"""
Why is our amortised flow posterior unresolved at crossings? (doc 008 §7.1)

doc 008 §6.3: spline NPE on the PRISM K=2 model is SBC-calibrated but
gives 18.5° / 58 % recall on the synthetic crossings while Nottingham's
NSF on a comparable model gives 5–10°. Candidate causes, one knob each,
same architecture, budget and evaluation for every variant:

  base       θ∈(0,π), φ∈(0,2π)                      — as run in §6.3
  hemi       θ∈(0,π/2)                               — removes the antipodal doubling of every mode
  disk       (x,y) in the unit disk → upper hemisphere — removes the φ wrap discontinuity as well
  hemi-emb   hemi + MLP embedding 193→64 of the signal, trained jointly
  hemi-2x    hemi with twice the training budget
  hemi-fixed hemi trained SBI_dMRI-style: fixed 1M set, epochs, batch 4096

Reported per variant: angular error / recall by crossing angle from the
mode-clustered posterior, median σ_θ, the *antipodal split* (fraction of
fibre-1 samples in the hemisphere opposite its mode; ≈ 0.5 means the flow
represents each fibre as two antipodal modes) and the *label-switch*
fraction (fibre-1 samples closer to the fibre-2 mode), plus SBC 90 %
coverage on held-out simulations.
"""
import argparse, time, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, optax
import flowjax.bijections as bij, flowjax.distributions as dist, flowjax.flows as flows
from dmipy_jax.validation.prism_jax import PrismConfig, forward, angular_error_best_match
from dmipy_jax.validation import prism_synthetic as ps
sys.path.insert(0, str(Path(__file__).parent))
from validate_flow_sbi import summarise, sph                      # mode clustering of doc 008 §6.3

TWO_PI = 2 * np.pi
VARIANTS = ["base", "hemi", "disk", "hemi-emb", "hemi-2x", "hemi-fixed"]


def param_spec(variant):
    """names, lows, highs, and a function samples(S,7)->dirs(S,2,3)."""
    if variant.startswith("disk"):
        names = ["x1", "y1", "x2", "y2", "f_wm1", "f_wm2", "f_i"]
        lo = np.array([-1, -1, -1, -1, 0.3, 0.0, 0.2]); hi = np.array([1, 1, 1, 1, 1.0, 0.6, 0.8])
    else:
        tmax = np.pi if variant == "base" else np.pi / 2
        names = ["theta1", "phi1", "theta2", "phi2", "f_wm1", "f_wm2", "f_i"]
        lo = np.array([0, 0, 0, 0, 0.3, 0.0, 0.2]); hi = np.array([tmax, TWO_PI, tmax, TWO_PI, 1.0, 0.6, 0.8])
    return names, lo, hi


def dirs_from_params_np(s, variant):
    if variant.startswith("disk"):
        def d(x, y):
            r2 = np.clip(x ** 2 + y ** 2, 0, 0.999); return np.stack([x, y, np.sqrt(1 - r2)], -1)
        return np.stack([d(s[:, 0], s[:, 1]), d(s[:, 2], s[:, 3])], 1)
    return np.stack([sph(s[:, 0], s[:, 1]), sph(s[:, 2], s[:, 3])], 1)


def make_forward(variant, bvals_si, bvecs):
    cfg = PrismConfig(n_fibres=2); bv = jnp.asarray(bvals_si); gv = jnp.asarray(bvecs)

    def one(p):
        if variant.startswith("disk"):
            def d(x, y):
                r2 = jnp.clip(x ** 2 + y ** 2, 0, 0.999); return jnp.array([x, y, jnp.sqrt(1 - r2)])
            dd = jnp.stack([d(p[0], p[1]), d(p[2], p[3])])
        else:
            t1, p1, t2, p2 = p[0], p[1], p[2], p[3]
            dd = jnp.stack([jnp.array([jnp.sin(t1) * jnp.cos(p1), jnp.sin(t1) * jnp.sin(p1), jnp.cos(t1)]),
                            jnp.array([jnp.sin(t2) * jnp.cos(p2), jnp.sin(t2) * jnp.sin(p2), jnp.cos(t2)])])
        f1, f2, fi = p[4], p[5], p[6]
        f2c = jnp.minimum(f2, 1.0 - f1)
        fr = jnp.array([0.0, 0.0, f1, f2c, jnp.maximum(1.0 - f1 - f2c, 0.0)])
        phys = {"s0": jnp.ones(1), "fracs": fr[None], "dirs": dd[None], "fintra": jnp.array([fi]),
                "d_par": jnp.asarray(cfg.d_par), "d_perp": jnp.asarray(cfg.d_perp), "sigma": None}
        return forward(phys, bv, gv, cfg)[0]
    return jax.vmap(one)


class EmbFlow(eqx.Module):
    flow: eqx.Module
    mlp: eqx.Module

    def _emb(self, c):
        return self.mlp(c) if c.ndim == 1 else jax.vmap(self.mlp)(c)

    def log_prob(self, x, condition=None):
        return self.flow.log_prob(x, condition=self._emb(condition))

    def sample(self, key, shape=(), condition=None):
        return self.flow.sample(key, shape, condition=self._emb(condition))


def build_flow(key, theta_dim, cond_dim, hidden, depth, embed):
    k1, k2 = jax.random.split(key)
    flow = flows.masked_autoregressive_flow(
        key=k1, base_dist=dist.StandardNormal((theta_dim,)), cond_dim=64 if embed else cond_dim,
        transformer=bij.RationalQuadraticSpline(knots=8, interval=3.0), flow_layers=depth, nn_width=hidden, nn_depth=2)
    if embed:
        return EmbFlow(flow, eqx.nn.MLP(cond_dim, 64, width_size=256, depth=2, activation=jax.nn.gelu, key=k2))
    return flow


def train(flow, fwd, lo, hi, key, n_steps, batch, snr_range, lr=5e-4, fixed_set=None, print_every=2000):
    """Streamed training (fresh sims every step) or fixed-set epochs.
    Loss = −mean log q(θ_norm | x); θ normalised to [0,1]; x is the
    b0-normalised Rician-noisy signal (b0 = index 0, always 1 after normalisation)."""
    lo_j, span_j = jnp.asarray(lo), jnp.asarray(hi - lo)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(1000, n_steps // 4), n_steps, lr * 0.01)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = opt.init(eqx.filter(flow, eqx.is_inexact_array))

    def simulate(k, th_norm):
        k1, k2, k3 = jax.random.split(k, 3)
        s = fwd(th_norm * span_j + lo_j)
        snr = jnp.exp(jax.random.uniform(k1, (s.shape[0], 1), minval=jnp.log(snr_range[0]), maxval=jnp.log(snr_range[1])))
        sig = 1.0 / snr
        n = jnp.sqrt((s + sig * jax.random.normal(k2, s.shape)) ** 2 + (sig * jax.random.normal(k3, s.shape)) ** 2)
        return n / n[:, :1]

    @eqx.filter_jit
    def step(flow, opt_state, k):
        k1, k2 = jax.random.split(k)
        th = jax.random.uniform(k1, (batch, len(lo)))
        x = simulate(k2, th)
        loss, g = eqx.filter_value_and_grad(lambda f: -jnp.mean(f.log_prob(th, condition=x)))(flow)
        upd, opt_state = opt.update(g, opt_state, eqx.filter(flow, eqx.is_inexact_array))
        return eqx.apply_updates(flow, upd), opt_state, loss

    @eqx.filter_jit
    def step_fixed(flow, opt_state, th, x):
        loss, g = eqx.filter_value_and_grad(lambda f: -jnp.mean(f.log_prob(th, condition=x)))(flow)
        upd, opt_state = opt.update(g, opt_state, eqx.filter(flow, eqx.is_inexact_array))
        return eqx.apply_updates(flow, upd), opt_state, loss

    losses = []; t0 = time.time()
    if fixed_set is None:
        for i in range(n_steps):
            key, sk = jax.random.split(key)
            flow, opt_state, l = step(flow, opt_state, sk); losses.append(float(l))
            if i % print_every == 0:
                print(f"    step {i}: loss {np.mean(losses[-200:]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        n_set, = fixed_set
        key, k1, k2 = jax.random.split(key, 3)
        TH = jax.random.uniform(k1, (n_set, len(lo)))
        X = jnp.concatenate([simulate(jax.random.fold_in(k2, j), TH[j:j + 65536]) for j in range(0, n_set, 65536)])
        per_epoch = n_set // batch; i = 0
        while i < n_steps:
            key, sk = jax.random.split(key); perm = jax.random.permutation(sk, n_set)
            for e in range(per_epoch):
                idx = perm[e * batch:(e + 1) * batch]
                flow, opt_state, l = step_fixed(flow, opt_state, TH[idx], X[idx]); losses.append(float(l)); i += 1
                if i % print_every == 0:
                    print(f"    step {i} (epoch {i // per_epoch}): loss {np.mean(losses[-200:]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
                if i >= n_steps: break
    return flow, np.array(losses)


def evaluate(flow, variant, lo, hi, b, n_samples, key):
    y = b["data"][b["mask"]]; y = y / y[:, :1]
    N = y.shape[0]; lo_j, span_j = jnp.asarray(lo), jnp.asarray(hi - lo)
    sample = jax.jit(lambda k, x: flow.sample(k, (n_samples,), condition=x) * span_j + lo_j)
    dirs = np.zeros((N, 2, 3)); fr = np.zeros((N, 2)); sig = np.zeros((N, 2)); anti = np.zeros(N); switch = np.zeros(N)
    for i in range(N):
        key, sk = jax.random.split(key); s = np.asarray(sample(sk, jnp.asarray(y[i])))
        V = dirs_from_params_np(s, variant)                         # (S,2,3)
        # the clustering summariser expects θ,φ columns: convert back
        th = np.arccos(np.clip(V[..., 2], -1, 1)); ph = np.arctan2(V[..., 1], V[..., 0]) % TWO_PI
        s2 = np.column_stack([th[:, 0], ph[:, 0], th[:, 1], ph[:, 1], s[:, 4], s[:, 5], s[:, 6]])
        d, f, sg, _, _ = summarise(s2); o = np.argsort(-f); dirs[i], fr[i], sig[i] = d[o], f[o], sg[o]
        # antipodal split and label switching, from the un-clustered fibre-1 samples
        v1 = V[:, 0]; D = (v1[:, :, None] * v1[:, None, :]).mean(0); mu = np.linalg.eigh(D)[1][:, -1]
        anti[i] = min(np.mean(v1 @ mu < 0), np.mean(v1 @ mu > 0))
        if np.linalg.norm(dirs[i, 1]) > 0 and fr[i, 1] > 0.05:
            switch[i] = np.mean(np.abs(v1 @ dirs[i, 1]) > np.abs(v1 @ dirs[i, 0]))
    gt = b["gt_dirs"].copy(); gt[b["angle"] == 0, 1] = 0
    err, rec = angular_error_best_match(dirs, fr, gt)
    rows = {}
    for av in list(ps.ANGLES) + [0]:
        m = b["angle"] == av; e, r = angular_error_best_match(dirs[m], fr[m], gt[m])
        rows[int(av)] = (e, r, float(np.median(sig[m, 0])), float(np.mean(anti[m])), float(np.mean(switch[m])))
    return {"err": err, "recall": rec, "sigma": float(np.median(sig[:, 0])), "anti": float(anti.mean()),
            "switch": float(switch.mean()), "rows": rows}


def sbc(flow, fwd, lo, hi, key, n, n_samples, snr_range=(8.0, 60.0)):
    lo_j, span_j = jnp.asarray(lo), jnp.asarray(hi - lo)
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    th = jax.random.uniform(k1, (n, len(lo))); s = fwd(th * span_j + lo_j)
    snr = jnp.exp(jax.random.uniform(k2, (n, 1), minval=jnp.log(snr_range[0]), maxval=jnp.log(snr_range[1])))
    x = jnp.sqrt((s + jax.random.normal(k3, s.shape) / snr) ** 2 + (jax.random.normal(k4, s.shape) / snr) ** 2); x = x / x[:, :1]
    sample = jax.jit(lambda k, xx: flow.sample(k, (n_samples,), condition=xx))
    ranks = np.zeros((n, len(lo)))
    for i in range(n):
        key, sk = jax.random.split(key); ss = np.asarray(sample(sk, x[i])); ranks[i] = (ss < np.asarray(th[i])[None]).sum(0)
    u = ranks / n_samples
    return np.mean((u > 0.05) & (u < 0.95), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--n-steps", type=int, default=20000); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=400); ap.add_argument("--n-sbc", type=int, default=200)
    ap.add_argument("--snr-eval", type=float, default=30); ap.add_argument("--fixed-n", type=int, default=1_000_000)
    ap.add_argument("--out", default="validation/flow_variants_results.npz")
    a = ap.parse_args()
    b = ps.make_benchmark(snr=a.snr_eval)
    results = {}
    for variant in a.variants:
        names, lo, hi = param_spec(variant)
        fwd = make_forward(variant, b["bvals"], b["bvecs"])
        flow = build_flow(jax.random.key(0), 7, len(b["bvals"]), a.hidden, a.depth, embed=variant.endswith("emb"))
        n_steps = a.n_steps * (2 if variant.endswith("2x") else 1); batch = a.batch; fixed = None
        if variant.endswith("fixed"):
            batch = 4096; fixed = (a.fixed_n,); n_steps = 60 * (a.fixed_n // 4096)     # 60 epochs, as SBI_dMRI
        ck = Path(f"validation/flow_variant_{variant}_{a.hidden}x{a.depth}.eqx")
        print(f"\n=== {variant}: {n_steps} steps × {batch}  ({'fixed 1M set' if fixed else 'streamed'})", flush=True)
        t0 = time.time()
        if ck.exists():
            flow = eqx.tree_deserialise_leaves(ck, flow); losses = np.zeros(2); print("  loaded", ck)
        else:
            flow, losses = train(flow, fwd, lo, hi, jax.random.key(1), n_steps, batch, (8.0, 60.0), fixed_set=fixed)
            eqx.tree_serialise_leaves(ck, flow)
        print(f"  trained in {time.time()-t0:.0f}s, loss {losses[:200].mean():.3f} → {losses[-200:].mean():.3f}", flush=True)
        t0 = time.time(); ev = evaluate(flow, variant, lo, hi, b, a.n_samples, jax.random.key(2))
        cov = sbc(flow, fwd, lo, hi, jax.random.key(3), a.n_sbc, a.n_samples)
        print(f"  {variant}: err={ev['err']:.2f}°  recall={100*ev['recall']:.1f}%  median σ_θ={ev['sigma']:.1f}°  "
              f"antipodal split={ev['anti']:.2f}  label-switch={ev['switch']:.2f}  SBC 90% cov={np.round(100*cov).astype(int).tolist()}  ({time.time()-t0:.0f}s)")
        print("   angle    err   recall  σ_θ   anti  switch")
        for av, (e, r, sg, an, sw) in ev["rows"].items():
            print(f"   {av:>4d}  {e:6.2f}° {100*r:6.1f}% {sg:5.1f}° {an:5.2f} {sw:5.2f}")
        results[variant] = {**ev, "sbc_cov": cov.tolist(), "final_loss": float(losses[-200:].mean())}
        np.savez(a.out, **{f"{v}_rows": np.array([[k, *r] for k, r in res["rows"].items()]) for v, res in results.items()},
                 summary=np.array([[v, res["err"], res["recall"], res["sigma"], res["anti"], res["switch"], res["final_loss"]] for v, res in results.items()], dtype=object))
    print("\nvariant     err    recall   σ_θ   anti  switch  loss")
    for v, r in results.items():
        print(f"{v:11s} {r['err']:6.2f}° {100*r['recall']:6.1f}% {r['sigma']:5.1f}° {r['anti']:5.2f} {r['switch']:5.2f}  {r['final_loss']:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Tier C3 (doc 008 §10): emulator-in-the-loop inversion against Monte Carlo.

1. Train an Equinox MLP ensemble  f(θ, b, ĝ) → log S  on the MC substrate
   library (`generate_mc_library.py`): θ = (f_intra, angle, c2_1, c2_2, tortuous),
   ĝ in the substrate frame (bundle 1 along z, bundle 2 = Rx(angle)·z). Per-
   measurement inputs make the emulator usable at any voxel orientation:
   at inversion the gradient directions are rotated into the substrate
   frame by a learnable rotation (bundle-1 direction + roll).
2. Held-out accuracy of the emulator (RMSE of S, per shell).
3. Inversion of held-out MC signals (Rician SNR 30): Optimistix
   Levenberg–Marquardt through the emulator with random restarts, over
   (θ, rotation); ensemble spread as a first uncertainty; recovered
   f_intra / angle / c2 vs truth. Same signals fitted with the
   stick+zeppelin plus-x (dispersed, no-iso, free D⊥) for comparison.
"""
import argparse, json, time
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, optax, h5py
from dmipy_jax.validation import prism_jax as pj

THETA_NAMES = ["f_intra", "angle_deg", "c2_1", "c2_2", "tortuous"]
LO = np.array([0.05, 0.0, 0.5, 0.5, 0.0]); HI = np.array([0.85, 90.0, 1.0, 1.0, 1.0])


def load_library(path):
    with h5py.File(path) as f:
        P, D, S = f["params"][:], f["descriptors"][:], f["signals"][:]; bvals, bvecs = f["bvals"][:], f["bvecs"][:]
    ang = P[:, 1]; tort = P[:, 2]; c2_2 = np.where(np.isnan(D[:, 2]), D[:, 1], D[:, 2])
    theta = np.column_stack([D[:, 0], ang, D[:, 1], c2_2, tort])
    ok = np.isfinite(theta).all(1) & (S > 0).all(1)
    return theta[ok], S[ok], bvals, bvecs


class Emulator(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, width=256, depth=4):
        self.mlp = eqx.nn.MLP(5 + 1 + 6, 1, width, depth, activation=jax.nn.gelu, key=key)

    def __call__(self, theta_n, b_n, g):
        # symmetric features of ĝ (antipodal invariance): squares and products
        feat = jnp.array([g[0] ** 2, g[1] ** 2, g[2] ** 2, g[0] * g[1], g[0] * g[2], g[1] * g[2]])
        x = jnp.concatenate([theta_n, jnp.array([b_n]), feat])
        return self.mlp(x)[0]                          # log S

    def signal(self, theta_n, b_n, G):
        return jnp.exp(jax.vmap(lambda g, b: self(theta_n, b, g))(G, b_n))


def train_ensemble(theta, S, bvals, bvecs, n_models=5, steps=6000, key=0):
    tn = (theta - LO) / (HI - LO); bn = bvals / 3e9
    X_theta = jnp.asarray(np.repeat(tn, len(bvals), 0)); X_b = jnp.asarray(np.tile(bn, len(theta)))
    X_g = jnp.asarray(np.tile(bvecs, (len(theta), 1))); Y = jnp.asarray(np.log(np.clip(S, 1e-4, 1)).reshape(-1))
    models = []
    for m in range(n_models):
        k = jax.random.key(key + m); em = Emulator(k)
        opt = optax.adam(optax.warmup_cosine_decay_schedule(0, 2e-3, 300, steps, 1e-5)); st = opt.init(eqx.filter(em, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(em, st, k):
            idx = jax.random.randint(k, (4096,), 0, Y.shape[0])
            def loss(e): return jnp.mean((jax.vmap(e)(X_theta[idx], X_b[idx], X_g[idx]) - Y[idx]) ** 2)
            l, g = eqx.filter_value_and_grad(loss)(em); u, st = opt.update(g, st, eqx.filter(em, eqx.is_inexact_array)); return eqx.apply_updates(em, u), st, l
        kk = jax.random.key(100 + m)
        for i in range(steps):
            kk, sk = jax.random.split(kk); em, st, l = step(em, st, sk)
        models.append(em); print(f"  emulator {m}: final batch MSE(log S) {float(l):.5f}", flush=True)
    return models


def ens_signal(models, theta_n, b_n, G):
    return jnp.stack([m.signal(theta_n, b_n, G) for m in models])       # (E, M)


def rotation(v_axis_angle):
    """Rodrigues rotation from a 3-vector (axis × angle)."""
    th = jnp.linalg.norm(v_axis_angle) + 1e-12; k = v_axis_angle / th
    K = jnp.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return jnp.eye(3) + jnp.sin(th) * K + (1 - jnp.cos(th)) * (K @ K)


def invert(models, y, bvals, bvecs, n_restarts=8, key=0):
    """LM through the ensemble mean; params = (θ_n (5), rotvec (3)). Returns best θ, rotation, ensemble spread."""
    import optimistix as optx
    b_n = jnp.asarray(bvals / 3e9); G0 = jnp.asarray(bvecs); yj = jnp.asarray(y)

    def resid(p, _):
        tn = jax.nn.sigmoid(p[:5]); R = rotation(p[5:]); G = G0 @ R          # rotate gradients into the substrate frame
        pred = jnp.mean(ens_signal(models, tn, b_n, G), 0)
        return pred - yj
    solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-8)
    best = None; rng = np.random.default_rng(key)
    for r in range(n_restarts):
        p0 = jnp.concatenate([jnp.asarray(rng.normal(0, 1.0, 5)), jnp.asarray(rng.normal(0, 1.5, 3))])
        sol = optx.least_squares(resid, solver, p0, max_steps=200, throw=False)
        c = float(jnp.sum(resid(sol.value, None) ** 2))
        if best is None or c < best[0]: best = (c, sol.value)
    p = best[1]; tn = jax.nn.sigmoid(p[:5]); R = rotation(p[5:])
    theta = np.asarray(tn) * (HI - LO) + LO
    spread = np.asarray(jnp.std(ens_signal(models, tn, b_n, G0 @ R), 0)).mean()
    axis1 = np.asarray(R @ jnp.array([0.0, 0.0, 1.0]))                            # bundle-1 direction in the lab frame
    return theta, axis1, best[0], spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="/data/datasets/sbi4dwi_mc_library/mc_library.h5")
    ap.add_argument("--n-models", type=int, default=5); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=40); ap.add_argument("--snr", type=float, default=30)
    ap.add_argument("--out", default="validation/emulator_inversion_results.json")
    a = ap.parse_args()
    theta, S, bvals, bvecs = load_library(a.library)
    rng = np.random.default_rng(0); idx = rng.permutation(len(theta)); test = idx[:a.n_test]; train = idx[a.n_test:]
    print(f"library: {len(theta)} substrates ({len(train)} train / {len(test)} test), f_intra {theta[:,0].min():.2f}–{theta[:,0].max():.2f}", flush=True)
    t0 = time.time(); models = train_ensemble(theta[train], S[train], bvals, bvecs, a.n_models, a.steps)
    print(f"trained {a.n_models} emulators in {time.time()-t0:.0f}s", flush=True)
    # held-out emulator accuracy
    b_n = jnp.asarray(bvals / 3e9); G = jnp.asarray(bvecs); err = []
    for i in test:
        tn = jnp.asarray((theta[i] - LO) / (HI - LO)); pred = np.asarray(jnp.mean(ens_signal(models, tn, b_n, G), 0)); err.append(pred - S[i])
    err = np.array(err); shells = np.round(bvals / 1e9).astype(int)
    print("emulator held-out RMSE by shell: " + "  ".join(f"b={s}: {np.sqrt(np.mean(err[:, shells == s] ** 2)):.4f}" for s in np.unique(shells)), flush=True)
    # inversion of held-out signals with noise, in a random lab orientation
    res = []; t0 = time.time()
    for n, i in enumerate(test):
        Rl = rotation(jnp.asarray(rng.normal(0, 1.5, 3))); G_lab = np.asarray(G @ Rl.T)     # observe with rotated gradients ⇔ rotated substrate
        s = 1 / a.snr; y = np.sqrt((S[i] + rng.normal(0, s, S[i].shape)) ** 2 + rng.normal(0, s, S[i].shape) ** 2)
        th, axis1, cost, spread = invert(models, y, bvals, G_lab, key=n)
        true_axis1 = np.asarray(Rl @ jnp.array([0.0, 0.0, 1.0]))
        ang_err = np.degrees(np.arccos(min(1.0, abs(float(axis1 @ true_axis1)))))
        res.append({"true": theta[i].tolist(), "est": th.tolist(), "axis_err_deg": float(ang_err), "cost": cost, "spread": float(spread)})
        if n < 8 or n % 10 == 0:
            print(f"  [{n}] f_i {theta[i,0]:.2f}→{th[0]:.2f}  angle {theta[i,1]:.0f}→{th[1]:.0f}  c2 {theta[i,2]:.2f}→{th[2]:.2f}  tort {theta[i,4]:.0f}→{th[4]:.2f}  axis err {ang_err:.1f}°  ({time.time()-t0:.0f}s)", flush=True)
    T = np.array([r["true"] for r in res]); E = np.array([r["est"] for r in res])
    print("\nheld-out inversion (SNR 30): " + "  ".join(f"{n}: r={np.corrcoef(T[:, k], E[:, k])[0,1]:.2f} bias={np.mean(E[:, k]-T[:, k]):+.3f} sd={np.std(E[:, k]-T[:, k]):.3f}" for k, n in enumerate(THETA_NAMES)))
    print(f"bundle-1 axis error: median {np.median([r['axis_err_deg'] for r in res]):.1f}°")
    json.dump({"results": res, "names": THETA_NAMES}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

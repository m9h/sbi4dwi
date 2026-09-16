#!/usr/bin/env python3
"""
Train an amortised flow proposal for the PRISM K-fibre model on an arbitrary
scheme (hybrid posterior, doc 008 §9). Parameters (hemisphere θ ≤ π/2):
  [θ_k, φ_k]×K, f_wm_k×K (cumulatively clipped to sum ≤ 1), f_i, D∥ (µm²/ms)
D∥ is a *flow parameter* so one proposal covers in-vivo (1.7–2.5) and
ex-vivo / DiSCo (0.5–0.9) regimes; D⊥ = D∥(1 − f_i) (tortuosity), the
remainder 1 − Σf_wm goes to the GM ball. Same trainer as
diagnose_flow_variants.py (streamed simulations, Rician SNR range).

  train_flow_k.py --scheme disco --K 3 --out validation/flow_disco_k3.eqx
"""
import argparse, time, sys, importlib.util
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
sys.path.insert(0, str(Path(__file__).parent))
from diagnose_flow_variants import build_flow, train
from dmipy_jax.validation.prism_jax import PrismConfig, forward

TWO_PI = 2 * np.pi


def load_scheme(name):
    if name == "disco":
        spec = importlib.util.spec_from_file_location("vdfp", Path(__file__).with_name("validate_disco_force_protocol.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        d = m.load_protocol_data(30)
        return np.asarray(d["bvals"]) * 1e6, np.asarray(d["bvecs"])
    if name == "prism":
        from dmipy_jax.validation.prism_synthetic import prism_scheme
        return prism_scheme()
    z = np.load(name); b = z["bvals"]; return (b if b.max() > 1e5 else b * 1e6), z["bvecs"]


PER_FIBRE = {"hemi": 2, "dyad": 5}


def spec_k(K, dpar_range=(0.3, 2.6), param="hemi"):
    names, lo, hi = [], [], []
    for k in range(K):
        if param == "dyad":
            names += [f"{c}{k+1}" for c in ("xx", "yy", "xy", "xz", "yz")]; lo += [0, 0, -0.5, -0.5, -0.5]; hi += [1, 1, 0.5, 0.5, 0.5]
        else:
            names += [f"theta{k+1}", f"phi{k+1}"]; lo += [0, 0]; hi += [np.pi / 2, TWO_PI]
    fmax = [1.0, 0.6, 0.4, 0.3, 0.3][:K]; fmin = [0.2] + [0.0] * (K - 1)
    for k in range(K):
        names.append(f"f_wm{k+1}"); lo.append(fmin[k]); hi.append(fmax[k])
    names += ["f_i", "d_par"]; lo += [0.2, dpar_range[0]]; hi += [0.85, dpar_range[1]]
    return names, np.array(lo, float), np.array(hi, float)


def make_forward_k(K, bvals_si, bvecs, param="hemi"):
    cfg = PrismConfig(n_fibres=K); bv = jnp.asarray(bvals_si); gv = jnp.asarray(bvecs); n = PER_FIBRE[param]

    def one(p):
        dirs = []
        for k in range(K):
            q = p[n * k:n * (k + 1)]
            if param == "dyad":
                xx, yy, xy, xz, yz = q; M = jnp.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, 1 - xx - yy]])
                dirs.append(jnp.linalg.eigh(M)[1][:, -1])
            else:
                t, ph = q
                dirs.append(jnp.array([jnp.sin(t) * jnp.cos(ph), jnp.sin(t) * jnp.sin(ph), jnp.cos(t)]))
        dd = jnp.stack(dirs)
        f = p[n * K:n * K + K]; fi = p[n * K + K]; dpar = p[n * K + K + 1] * 1e-9
        # cumulative clip so Σ f_wm ≤ 1; remainder → GM ball
        fc = []; used = 0.0
        for k in range(K):
            fk = jnp.minimum(f[k], 1.0 - used); fc.append(fk); used = used + fk
        fc = jnp.stack(fc)
        fr = jnp.concatenate([jnp.array([0.0]), jnp.array([jnp.maximum(1.0 - used, 0.0)]), fc, jnp.array([0.0])])
        phys = {"s0": jnp.ones(1), "fracs": fr[None], "dirs": dd[None], "fintra": jnp.array([fi]),
                "d_par": dpar, "d_perp": dpar * (1.0 - fi), "sigma": None}
        return forward(phys, bv, gv, cfg)[0]
    return jax.vmap(one)


def dirs_fracs_from_samples(s, K, param="hemi"):
    """(S,P) samples → per-sample dirs (S,K,3) and wm fracs (S,K)."""
    n = PER_FIBRE[param]
    if param == "dyad":
        ds = []
        for k in range(K):
            xx, yy, xy, xz, yz = s[:, n * k:n * (k + 1)].T
            M = np.stack([np.stack([xx, xy, xz], -1), np.stack([xy, yy, yz], -1), np.stack([xz, yz, 1 - xx - yy], -1)], -2)
            ds.append(np.linalg.eigh(M)[1][..., -1])
        d = np.stack(ds, 1)
    else:
        d = np.stack([np.stack([np.sin(s[:, 2 * k]) * np.cos(s[:, 2 * k + 1]), np.sin(s[:, 2 * k]) * np.sin(s[:, 2 * k + 1]),
                                np.cos(s[:, 2 * k])], -1) for k in range(K)], 1)
    return d, s[:, n * K:n * K + K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="disco"); ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=30000); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--snr-range", type=float, nargs=2, default=[8.0, 60.0])
    ap.add_argument("--out", required=True); ap.add_argument("--param", choices=["hemi", "dyad"], default="dyad")
    a = ap.parse_args()
    bvals, bvecs = load_scheme(a.scheme); names, lo, hi = spec_k(a.K, param=a.param)
    fwd = make_forward_k(a.K, bvals, bvecs, param=a.param)
    flow = build_flow(jax.random.key(0), len(names), len(bvals), a.hidden, a.depth, embed=False)
    print(f"scheme {a.scheme}: {len(bvals)} measurements; K={a.K}, {len(names)} params: {names}", flush=True)
    t0 = time.time()
    flow, losses = train(flow, fwd, lo, hi, jax.random.key(1), a.n_steps, a.batch, tuple(a.snr_range))
    eqx.tree_serialise_leaves(a.out, flow)
    np.savez(a.out + ".meta.npz", names=np.array(names), lo=lo, hi=hi, K=a.K, scheme=a.scheme, hidden=a.hidden, depth=a.depth, param=a.param,
             losses=losses)
    print(f"trained {a.n_steps}×{a.batch} in {time.time()-t0:.0f}s, loss {losses[:200].mean():.3f} → {losses[-200:].mean():.3f}; saved {a.out}")


if __name__ == "__main__":
    main()

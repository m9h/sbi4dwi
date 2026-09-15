#!/usr/bin/env python3
"""
Exchange model against MCMRSimulator permeable cylinders (doc 008 §9.3):
fit the closed-form Kärger stick+zeppelin (f_i, τ_ex, D∥, D⊥; fibre along z)
to MCMR signals for the DiSCo single-Δ protocol and the Fisher-optimised
multi-Δ protocol, at several permeabilities, with Rician noise repeats →
empirical scatter of τ_ex vs the CRLB of design_exchange_protocol.py.
"""
import argparse, glob, json, re
import numpy as np, pandas as pd, jax, jax.numpy as jnp
from scipy.optimize import least_squares
from dmipy_jax.validation import prism_exchange as px

jax.config.update("jax_enable_x64", True)


def load(perm_tag, name):
    df = pd.read_csv(f"validation/mcmr/perm_{perm_tag}_{name}.csv")
    return df


def fit(df, y, x0=(0.5, 30.0, 1.7, 0.5)):
    b = jnp.asarray(df.b.values * 1e9); g = jnp.asarray(df[["gx", "gy", "gz"]].values)
    delta = jnp.asarray(df.delta.values * 1e-3); Delta = jnp.asarray(df.Delta.values * 1e-3); u = jnp.array([0.0, 0.0, 1.0])
    def model(p):
        f, tau, dpar, dperp = p
        return px.karger_pgse(b, g, u, f, dpar * 1e-9, dpar * 0.85e-9, dperp * 1e-9, tau * 1e-3, delta, Delta)
    fm = jax.jit(model); jac = jax.jit(jax.jacfwd(model))
    r = least_squares(lambda p: np.asarray(fm(jnp.asarray(p))) - y, x0, jac=lambda p: np.asarray(jac(jnp.asarray(p))),
                      bounds=([0.05, 1.0, 0.3, 0.02], [0.95, 2000.0, 3.0, 2.0]), max_nfev=400)
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=float, nargs="+", default=[30, 50]); ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", default="validation/exchange_mcmr_results.json")
    a = ap.parse_args()
    tags = sorted({re.search(r"perm_(.+?)_(disco|optimised)\.csv", f).group(1) for f in glob.glob("validation/mcmr/perm_*_disco.csv")})
    results = {}
    for tag in tags:
        meta = dict(kv.split("=") for kv in open(f"validation/mcmr/perm_{tag}_meta.txt").read().split())
        print(f"\n### permeability {tag}: f_intra {float(meta['f_intra']):.3f}, {meta['n_cyl']} cylinders", flush=True)
        results[tag] = {"f_intra": float(meta["f_intra"])}
        for name in ("disco", "optimised"):
            df = load(tag, name); S = df.S.values
            x, rmse = fit(df, S)
            print(f"  {name:9s} clean fit: f_i={x[0]:.3f} τ_ex={x[1]:.1f} ms D∥={x[2]:.2f} D⊥={x[3]:.2f}  rmse {rmse:.4f}", flush=True)
            results[tag][name] = {"clean": x.tolist(), "rmse": rmse}
            rng = np.random.default_rng(0)
            for snr in a.snrs:
                est = []
                for _ in range(a.repeats):
                    s = 1.0 / snr; y = np.sqrt((S + rng.normal(0, s, S.shape)) ** 2 + rng.normal(0, s, S.shape) ** 2)
                    est.append(fit(df, y, x0=tuple(x))[0])
                est = np.array(est); med = np.median(est, 0); sd = np.std(est, 0)
                print(f"    SNR {snr:.0f} × {a.repeats}: f_i {med[0]:.3f} ± {sd[0]:.3f}   τ_ex {med[1]:.1f} ± {sd[1]:.1f} ms (rel {sd[1]/max(med[1],1e-6):.2f})   "
                      f"D∥ {med[2]:.2f} ± {sd[2]:.2f}   D⊥ {med[3]:.2f} ± {sd[3]:.2f}", flush=True)
                results[tag][name][f"snr{snr:.0f}"] = {"median": med.tolist(), "sd": sd.tolist()}
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

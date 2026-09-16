#!/usr/bin/env python3
"""
Acquisition design for the exchange model (doc 008 §9 item 2): which PGSE
protocol makes the exchange time τ_ex identifiable?

Fisher information of θ = (f_i, τ_ex, D∥, D⊥) under the closed-form
Kärger stick+zeppelin (`prism_exchange.karger_pgse`), single fibre along
z, Gaussian noise σ = 1/SNR *at TE = 0* scaled by the T2 decay
exp(−(Δ + δ + t_ro)/T2) of each measurement — so a longer Δ buys
time-dependence but costs signal, and the optimiser has to trade them.
Averaged over 5 orientations of the fibre relative to the scheme.

Protocols compared at equal measurement count (M = 3 shells × 32 dirs):
  disco-single   δ = 17.7, Δ = 35.8 ms, b = 1000/2000/3000 (the DiSCo / PRISM timing)
  fixed-3Δ       the same shells at Δ = 20 / 45 / 80 ms (δ = 10)
  optimised      Δ_k and b_k per shell, continuous, gradient ascent on
                 log det F (D-optimal) with a τ_ex-weighted objective,
                 δ = 10 ms, Δ ∈ [15, 120] ms, b ∈ [0.3, 5] ms/µm²
Reports CRLB relative SD of every parameter for each protocol at SNR 30
and 50, over a grid of true τ_ex (10 / 25 / 50 / 100 ms).
"""
import argparse, json
import numpy as np, jax, jax.numpy as jnp, optax
from dmipy_jax.validation import prism_exchange as px

jax.config.update("jax_enable_x64", True)
T2 = 70e-3; T_RO = 10e-3


def directions(n, seed=0):
    rng = np.random.default_rng(seed); g = rng.normal(size=(n, 3)); return g / np.linalg.norm(g, axis=1, keepdims=True)


def fibre_dirs(n=5, seed=1):
    return directions(n, seed)


def protocol_signal(theta, log_b, log_Delta, delta, dirs_meas, fibre):
    """theta = (f_i, τ_ex, D∥, D⊥) in (–, s, m²/s, m²/s); shells k with b_k, Δ_k."""
    f_i, tau, dpar, dperp = theta
    S = []
    for k in range(len(log_b)):
        b = jnp.exp(log_b[k]) * 1e9; D = jnp.exp(log_Delta[k])
        S.append(px.karger_pgse(jnp.full(dirs_meas.shape[0], b), dirs_meas, fibre, f_i, dpar, dpar * 0.85, dperp, tau, delta, D)
                 * jnp.exp(-(D + delta + T_RO) / T2))
    return jnp.concatenate(S)


def fisher(theta, log_b, log_Delta, delta, dirs_meas, fibres, snr):
    """Averaged over fibre orientations; parameters in log space for scale-free CRLB (relative SD)."""
    def sig(log_theta, fib):
        return protocol_signal(jnp.exp(log_theta), log_b, log_Delta, delta, dirs_meas, fib)
    F = 0.0
    for fib in fibres:
        J = jax.jacfwd(sig)(jnp.log(theta), fib)
        F = F + J.T @ J * snr ** 2
    return F / len(fibres)


def crlb_rel_sd(F):
    return jnp.sqrt(jnp.diag(jnp.linalg.inv(F + 1e-12 * jnp.eye(F.shape[0]))))


def optimise(theta_grid, delta, dirs_meas, fibres, snr, n_shells=3, steps=400, lr=0.05, w_tau=3.0, gmax=None):
    """Maximise Σ_θgrid [log det F + w_tau · log F_tau,tau] over (log b_k, log Δ_k)."""
    lb = jnp.log(jnp.array([1.0, 2.0, 3.0][:n_shells])); lD = jnp.log(jnp.array([0.02, 0.045, 0.08][:n_shells]))
    params = {"lb": lb, "lD": lD}
    def obj(p):
        lb_ = jnp.clip(p["lb"], jnp.log(0.3), jnp.log(5.0)); lD_ = jnp.clip(p["lD"], jnp.log(max(0.015, delta + 0.002)), jnp.log(0.12))
        if gmax is not None:      # b ≤ γ² G² δ² (Δ − δ/3): cap b at what the gradient can deliver
            b_cap = (px.GAMMA * gmax * delta) ** 2 * (jnp.exp(lD_) - delta / 3.0) * 1e-9
            lb_ = jnp.minimum(lb_, jnp.log(b_cap))
        tot = 0.0
        for th in theta_grid:
            F = fisher(th, lb_, lD_, delta, dirs_meas, fibres, snr)
            tot = tot + jnp.linalg.slogdet(F + 1e-9 * jnp.eye(4))[1] + w_tau * jnp.log(1.0 / jnp.linalg.inv(F + 1e-9 * jnp.eye(4))[1, 1])
        return -tot / len(theta_grid)
    opt = optax.adam(lr); st = opt.init(params); g = jax.jit(jax.value_and_grad(obj))
    for i in range(steps):
        v, gr = g(params); upd, st = opt.update(gr, st); params = optax.apply_updates(params, upd)
        params = {"lb": jnp.clip(params["lb"], jnp.log(0.3), jnp.log(5.0)), "lD": jnp.clip(params["lD"], jnp.log(max(0.015, delta + 0.002)), jnp.log(0.12))}
    if gmax is not None:
        b_cap = (px.GAMMA * gmax * delta) ** 2 * (jnp.exp(params["lD"]) - delta / 3.0) * 1e-9
        params["lb"] = jnp.minimum(params["lb"], jnp.log(b_cap))
    return params, float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=float, nargs="+", default=[30, 50]); ap.add_argument("--n-dirs", type=int, default=32)
    ap.add_argument("--taus-ms", type=float, nargs="+", default=[10, 25, 50, 100]); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="validation/exchange_protocol_design.json")
    ap.add_argument("--gmax", type=float, default=None, help="gradient limit in T/m (e.g. 0.08 clinical, 0.3 Connectom); caps b per shell")
    ap.add_argument("--delta", type=float, default=10e-3, help="pulse duration for the fixed-3Δ and optimised protocols (s)")
    a = ap.parse_args()
    dirs_meas = jnp.asarray(directions(a.n_dirs)); fibres = jnp.asarray(fibre_dirs())
    base = {"f_i": 0.55, "dpar": 1.7e-9, "dperp": 0.5e-9}
    grid = [jnp.array([base["f_i"], t * 1e-3, base["dpar"], base["dperp"]]) for t in a.taus_ms]
    protocols = {
        "disco-single": (jnp.log(jnp.array([1.0, 2.0, 3.0])), jnp.log(jnp.array([0.0358] * 3)), 17.7e-3),
        "fixed-3Δ":     (jnp.log(jnp.array([1.0, 2.0, 3.0])), jnp.log(jnp.array([0.02, 0.045, 0.08])), 10e-3),
    }
    results = {}
    names = ["f_i", "tau_ex", "D_par", "D_perp"]
    for snr in a.snrs:
        print(f"\n=== SNR {snr:.0f} (at TE=0; T2 = {T2*1e3:.0f} ms penalises long Δ)", flush=True)
        p_opt, v = optimise(grid, a.delta, dirs_meas, fibres, snr, steps=a.steps, gmax=a.gmax)
        protocols["optimised"] = (p_opt["lb"], p_opt["lD"], a.delta)
        protocols["fixed-3Δ"] = (protocols["fixed-3Δ"][0], protocols["fixed-3Δ"][1], a.delta)
        print(f"  optimised (Gmax {a.gmax}, δ = {a.delta*1e3:.0f} ms): b = {np.round(np.exp(np.asarray(p_opt['lb'])), 2).tolist()} ms/µm²,  Δ = {np.round(np.exp(np.asarray(p_opt['lD']))*1e3, 1).tolist()} ms")
        results[snr] = {}
        print("  protocol       τ_ex   " + "  ".join(f"{n:>8s}" for n in names) + "   (CRLB relative SD)")
        for pname, (lb, lD, delta) in protocols.items():
            results[snr][pname] = {"b": np.exp(np.asarray(lb)).tolist(), "Delta_ms": (np.exp(np.asarray(lD)) * 1e3).tolist(), "crlb": {}}
            for th in grid:
                F = fisher(th, lb, lD, delta, dirs_meas, fibres, snr); sd = np.asarray(crlb_rel_sd(F))
                results[snr][pname]["crlb"][f"{float(th[1])*1e3:.0f}"] = sd.tolist()
                print(f"  {pname:13s} {float(th[1])*1e3:5.0f}ms " + "  ".join(f"{s:8.3f}" for s in sd))
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

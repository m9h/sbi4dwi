#!/usr/bin/env python3
"""Fisher-information design of a coronal RESOLVE (readout-segmented EPI) diffusion block for the orbit
(Graves orbitopathy / TED): extraocular muscle (EOM) oedema fraction and optic-nerve intra-axonal fraction.

Model per tissue (Gaussian-noise Fisher information, noise σ at the scheme's TE):
  EOM        : zeppelin muscle (D∥, D⊥) + free water f_w (3.0e-9), T2 35 ms  → target f_w (oedema), MD
  optic nerve: stick f_i + tortuous zeppelin + CSF f_csf (sheath), T2 70 ms  → target f_i, f_csf
Orientation enters as two nuisance angles per tissue under a weak Gaussian prior (sd ~30°).
RESOLVE timing: single TE set by b_max, TE = TE0 + k·sqrt(b_max/1000); time = n_vol · n_seg · TR; a budget B min allows
n_avg = floor(B / time) repeats (information × n_avg). Everything is a design calculation, not a measurement.
    uv run python validation/design_orbit_resolve.py [--budget 8] [--snr0 25]
"""
import argparse, functools, itertools, json
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

@functools.lru_cache(maxsize=None)
def dirs_n(n, seed=0):
    """n roughly uniform unit vectors (electrostatic repulsion, antipodal)."""
    rng = np.random.default_rng(seed); v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(2000):
        d = v[:, None] - v[None]; d2 = (d ** 2).sum(-1) + np.eye(n); f = (d / d2[..., None] ** 1.5).sum(1)
        da = v[:, None] + v[None]; da2 = (da ** 2).sum(-1) + np.eye(n); f += (da / da2[..., None] ** 1.5).sum(1)
        v += 0.01 * (f - (f * v).sum(1, keepdims=True) * v); v /= np.linalg.norm(v, axis=1, keepdims=True)
    v.setflags(write=False); return v

def scheme(shells):
    """shells: [(b s/mm², n_dirs)] → bvals (SI), bvecs; one b0 per 8 DW volumes (min 1)."""
    b, g = [], []
    for bv, n in shells:
        g.append(dirs_n(n, seed=int(bv))); b.append(np.full(n, bv, float))
    n_dw = sum(n for _, n in shells); n0 = max(1, n_dw // 8)
    b.append(np.zeros(n0)); g.append(np.tile([0, 0, 1.0], (n0, 1)))
    return np.concatenate(b) * 1e6, np.concatenate(g)

# --- tissue forward models (unit S0 at TE=0; T2 weighting applied per compartment) ---
def _unit(pol, az):
    return jnp.array([jnp.sin(pol) * jnp.cos(az), jnp.sin(pol) * jnp.sin(az), jnp.cos(pol)])

def eom_signal(th, b, g, te):
    f_w, d_par, d_perp, pol, az = th; n = _unit(pol, az)             # muscle long axis is a nuisance (gaze, anatomy)
    gn2 = (g @ n) ** 2; zep = jnp.exp(-b * (d_perp + (d_par - d_perp) * gn2))
    return (1 - f_w) * jnp.exp(-te / 0.035) * zep + f_w * jnp.exp(-te / 2.0) * jnp.exp(-b * 3.0e-9)

def nerve_signal(th, b, g, te):
    f_i, f_csf, d_par, pol, az = th; n = _unit(pol, az); gn2 = (g @ n) ** 2
    stick = jnp.exp(-b * d_par * gn2); d_perp = d_par * (1 - f_i); zep = jnp.exp(-b * (d_perp + (d_par - d_perp) * gn2))
    return (1 - f_csf) * jnp.exp(-te / 0.070) * (f_i * stick + (1 - f_i) * zep) + f_csf * jnp.exp(-te / 2.0) * jnp.exp(-b * 3.0e-9)

# orientation (polar, azimuth) is estimated jointly in both tissues; set off-axis so no gradient set is accidentally aligned
TISSUES = {"eom": (eom_signal, jnp.array([0.15, 1.9e-9, 1.4e-9, 1.2, 0.4]), jnp.array([1.0, 1e-9, 1e-9, 1.0, 1.0]), ["f_w", "D∥", "D⊥", "pol", "az"]),
           "nerve": (nerve_signal, jnp.array([0.55, 0.15, 1.7e-9, 0.5, 0.7]), jnp.array([1.0, 1.0, 1e-9, 1.0, 1.0]), ["f_i", "f_csf", "D∥", "pol", "az"])}

def te_ms(b_max, te0=42.0, k=22.0):           # RESOLVE 2 mm, 5 segments at 3 T: ~64 ms at b=1000, ~75 ms at b=2000
    return te0 + k * np.sqrt(max(b_max, 1e-9) / 1000.0)

def crlb(shells, a):
    b, g = scheme(shells); b_max = max(bv for bv, _ in shells); te = te_ms(b_max) * 1e-3
    n_vol = len(b); t_min = n_vol * a.n_seg * a.tr / 60.0; n_avg = a.budget / t_min if t_min <= a.budget else 0.0
    out = {"n_vol": n_vol, "te_ms": te_ms(b_max), "t_min": t_min, "n_avg": n_avg}
    if n_avg == 0: return out
    sigma = 1.0 / a.snr0                                                 # σ relative to S0 at TE=0 (b0 SNR at TE=0 = snr0)
    for name, (fn, th0, scale, labels) in TISSUES.items():
        J = jax.jacfwd(lambda t: fn(t * scale, jnp.asarray(b), jnp.asarray(g), te))(th0 / scale)
        Fi = np.asarray(J.T @ J) / sigma ** 2 * n_avg
        Fi[3, 3] += 1.0 / a.angle_prior_rad ** 2; Fi[4, 4] += 1.0 / a.angle_prior_rad ** 2      # weak orientation prior (Bayesian CRLB) keeps 3-direction schemes finite
        sd = np.sqrt(np.diag(np.linalg.inv(Fi)))
        out[name] = {lab: float(s * (1.0 if sc == 1.0 else sc / th)) for lab, s, sc, th in zip(labels[:3], sd[:3], np.asarray(scale)[:3], np.asarray(th0)[:3])}  # fractions absolute, D relative
        out[name]["angle_deg"] = float(np.degrees(np.sqrt(sd[3] ** 2 + (sd[4] * np.sin(float(th0[3]))) ** 2)))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=8.0, help="minutes"); ap.add_argument("--snr0", type=float, default=25.0, help="b0 SNR at TE=0, 2×2×3 mm")
    ap.add_argument("--tr", type=float, default=3.6, help="s, ~20 coronal 3 mm slices"); ap.add_argument("--n-seg", type=int, default=5); ap.add_argument("--out", default="validation/orbit_resolve_design.json"); ap.add_argument("--angle-prior-rad", type=float, default=0.5, help="sd of the Gaussian prior on each fibre-orientation angle (~30°)")
    a = ap.parse_args()
    named = {"clinical b1000×3": [(1000, 3)], "b1000×6": [(1000, 6)], "b500/1000×6": [(500, 6), (1000, 6)], "b800/1500×12": [(800, 12), (1500, 12)],
             "b600/1200/2000×10": [(600, 10), (1200, 10), (2000, 10)], "b400/1000×12": [(400, 12), (1000, 12)], "b300/800/1500×8": [(300, 8), (800, 8), (1500, 8)]}
    res = {"settings": vars(a), "named": {}, "search": []}
    print(f"budget {a.budget} min, TR {a.tr} s, {a.n_seg} segments, b0 SNR(TE=0) {a.snr0}\n{'scheme':28s} {'vol':>4s} {'TE':>5s} {'min':>5s} {'avg':>4s} | EOM f_w  D⊥%   angle | nerve f_i  f_csf  D∥%   angle   (CRLB sd)")
    def row(n, o):
        if o["n_avg"] == 0: return f"{n:28s} {o['n_vol']:4d} {o['te_ms']:5.1f} {o['t_min']:5.1f}  over budget"
        return f"{n:28s} {o['n_vol']:4d} {o['te_ms']:5.1f} {o['t_min']:5.1f} {o['n_avg']:4.1f} | {o['eom']['f_w']:.3f}  {min(100*o['eom']['D⊥'],999):5.1f}  {min(o['eom']['angle_deg'],99):4.1f}° | {o['nerve']['f_i']:.3f}  {o['nerve']['f_csf']:.3f}  {min(100*o['nerve']['D∥'],999):5.1f}  {min(o['nerve']['angle_deg'],99):4.1f}°"
    for n, sh in named.items():
        o = crlb(sh, a); res["named"][n] = {"shells": sh, **o}; print(row(n, o))
    # exhaustive search: 1–3 shells, b ∈ 250..2500, dirs ∈ {3,6,12}; objective = sd(f_w) + sd(f_i) (equal weight, both fractions)
    bs = [250, 500, 750, 1000, 1250, 1500, 2000, 2500]; ns = [6, 12, 20]; cand = [(b, n) for b in bs for n in ns]
    for k in (1, 2, 3):
        for combo in itertools.combinations(cand, k):
            if len({b for b, _ in combo}) < k: continue
            o = crlb(list(combo), a)
            if o["n_avg"] == 0: continue
            res["search"].append({"shells": list(combo), "obj": o["eom"]["f_w"] + o["nerve"]["f_i"], **o})
    res["search"].sort(key=lambda r: r["obj"]); print("\nbest by sd(f_w)+sd(f_i):")
    for r in res["search"][:8]: print(row(" + ".join(f"b{b}×{n}" for b, n in r["shells"]), r))
    best_eom = min(res["search"], key=lambda r: r["eom"]["f_w"]); best_nerve = min(res["search"], key=lambda r: r["nerve"]["f_i"])
    print("\nbest for EOM f_w alone:  " + row(" + ".join(f"b{b}×{n}" for b, n in best_eom["shells"]), best_eom)); print("best for nerve f_i alone: " + row(" + ".join(f"b{b}×{n}" for b, n in best_nerve["shells"]), best_nerve))
    res["search"] = res["search"][:50]; json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()

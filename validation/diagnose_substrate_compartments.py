#!/usr/bin/env python3
"""
Why is f_i biased on CATERPillar substrates? (doc 008 §7.2)

Every Gaussian-compartment model over-estimates the intra-axonal fraction
on the MC substrates (doc 008 §6.4), while SBI_dMRI's ball+sticks sum
lands near geometry. This driver resolves the bias into its sources by
fitting the *compartment-resolved* MC signals (S_intra, S_extra are
simulated separately) with the directions fixed at ground truth, so
orientation estimation is out of the picture:

  F1  oracle:   S = f S_in + (1-f) S_ex, f by linear LSQ         → sanity (f == geometry)
  F2  intra:    S_in vs stick(GT axis) / stick over the *true* fODF (axon
                segment directions) / zeppelin(GT axis)           → is the intra signal stick-like?
  F3  extra:    S_ex vs zeppelin(GT axes) and per-shell radial D  → is the extra signal Gaussian?
  F4  total, directions fixed, six microstructure models:
        sz-free      stick+zeppelin, f, D∥ (shared), D⊥ free
        sz-tort      stick+zeppelin, D⊥ = (1-f) D∥ (tortuosity), f, D∥ free
        sz-tort-x    same with decoupled D∥,ex (plus-x)
        sz-fixed     PRISM's fixed D∥ = 1.7, D⊥ = 0.4, f free
        ball-stick   f, D free (SBI_dMRI's model)
        sz-fodf      stick over the true fODF + zeppelin, f, D∥, D⊥ free
      and two half-oracles that attribute the bias:
        in-oracle    measured S_in + zeppelin (extra model error only)
        ex-oracle    stick(GT) + measured S_ex (intra model error only)

Clean signals isolate model error; the same fits on 64 Rician SNR-30
voxels show how much noise adds. Scheme: PRISM 3 shells × 64 (b ≤ 3000).
"""
import argparse, json, time
import numpy as np, jax
from scipy.optimize import least_squares
from dmipy_jax.validation import substrate_benchmark as sb
from dmipy_jax.validation.prism_synthetic import prism_scheme


def segment_dirs(sub):
    """True fODF: unit directions and lengths of consecutive sphere-centre
    segments within each axon (CATERPillar writes spheres in growth order)."""
    C, I = sub.centers_m, sub.axon_ids
    dirs, w = [], []
    for a in np.unique(I):
        c = C[I == a]
        if len(c) < 2:
            continue
        d = np.diff(c, axis=0); n = np.linalg.norm(d, axis=1); ok = (n > 0) & (n < 2e-6)
        dirs.append(d[ok] / n[ok, None]); w.append(n[ok])
    return np.concatenate(dirs), np.concatenate(w)


# ---- numpy forward models; b in SI, g (M,3), u (3,) or (K,3) with weights
def stick(b, g, u, dpar):
    return np.exp(-b * dpar * (g @ u) ** 2)


def stick_fodf(b, g, U, w, dpar):
    c2 = (g @ U.T) ** 2                                   # (M,K)
    return (np.exp(-b[:, None] * dpar * c2) * w[None]).sum(1) / w.sum()


def zeppelin(b, g, u, dpar, dperp):
    return np.exp(-b * (dperp + (dpar - dperp) * (g @ u) ** 2))


def zep_multi(b, g, axes, dpar, dperp):
    return np.mean([zeppelin(b, g, u, dpar, dperp) for u in axes], axis=0)


def ball(b, d):
    return np.exp(-b * d)


def fit(resid, x0, lo, hi):
    r = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=2000)
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def all_fits(S, S_in, S_ex, b, g, axes, U, w, f_geom):
    """Returns dict name -> (f, params dict, rmse)."""
    out = {}
    u0 = axes[0]
    # F1 oracle
    A = np.stack([S_in - S_ex], 1); f = float(np.linalg.lstsq(A, S - S_ex, rcond=None)[0][0])
    out["F1-oracle"] = (f, {}, float(np.sqrt(np.mean((f * S_in + (1 - f) * S_ex - S) ** 2))))
    # F2 intra-only (single bundle uses axis; crossings use both axes with equal weight)
    def intra_model(dpar, kind, dperp=0.0):
        if kind == "stick":
            return np.mean([stick(b, g, u, dpar) for u in axes], 0)
        if kind == "fodf":
            return stick_fodf(b, g, U, w, dpar)
        return zep_multi(b, g, axes, dpar, dperp)
    x, r = fit(lambda p: intra_model(p[0] * 1e-9, "stick") - S_in, [2.0], [0.1], [3.5])
    out["F2-in-stick"] = (np.nan, {"dpar": x[0]}, r)
    x, r = fit(lambda p: intra_model(p[0] * 1e-9, "fodf") - S_in, [2.0], [0.1], [3.5])
    out["F2-in-fodf"] = (np.nan, {"dpar": x[0]}, r)
    x, r = fit(lambda p: intra_model(p[0] * 1e-9, "zep", p[1] * 1e-9) - S_in, [2.0, 0.1], [0.1, 0.0], [3.5, 3.0])
    out["F2-in-zep"] = (np.nan, {"dpar": x[0], "dperp": x[1]}, r)
    # F3 extra-only
    x, r = fit(lambda p: zep_multi(b, g, axes, p[0] * 1e-9, p[1] * 1e-9) - S_ex, [1.5, 0.5], [0.05, 0.0], [3.5, 3.0])
    ex = {"dpar": x[0], "dperp": x[1]}
    # per-shell apparent radial D of S_ex (directions within 25° of perpendicular to every axis)
    perp = np.all(np.abs(g @ axes.T) < np.cos(np.radians(65)), axis=1) & (b > 0)
    for bv in np.unique(b[b > 0]):
        m = perp & (np.isclose(b, bv))
        ex[f"Dperp_b{int(bv/1e6)}"] = float(-np.log(np.clip(S_ex[m], 1e-6, 1)).mean() / bv * 1e9) if m.any() else np.nan
    out["F3-ex-zep"] = (np.nan, ex, r)
    # F4 total fits, directions fixed
    def sz(f, dpar_in, dpar_ex, dperp, fodf=False):
        s_in = stick_fodf(b, g, U, w, dpar_in) if fodf else np.mean([stick(b, g, u, dpar_in) for u in axes], 0)
        return f * s_in + (1 - f) * zep_multi(b, g, axes, dpar_ex, dperp)
    specs = {
        "sz-free":    (lambda p: sz(p[0], p[1]*1e-9, p[1]*1e-9, p[2]*1e-9), [0.5, 2.0, 0.5], [0, 0.1, 0], [1, 3.5, 3.0], ["f", "dpar", "dperp"]),
        "sz-tort":    (lambda p: sz(p[0], p[1]*1e-9, p[1]*1e-9, (1-p[0])*p[1]*1e-9), [0.5, 2.0], [0, 0.1], [1, 3.5], ["f", "dpar"]),
        "sz-tort-x":  (lambda p: sz(p[0], p[1]*1e-9, p[2]*1e-9, (1-p[0])*p[2]*1e-9), [0.5, 2.0, 2.0], [0, 0.1, 0.1], [1, 3.5, 3.5], ["f", "dpar", "dpar_ex"]),
        "sz-fixed":   (lambda p: sz(p[0], 1.7e-9, 1.7e-9, 0.4e-9), [0.5], [0], [1], ["f"]),
        "ball-stick": (lambda p: p[0]*np.mean([stick(b, g, u, p[1]*1e-9) for u in axes], 0) + (1-p[0])*ball(b, p[1]*1e-9), [0.5, 2.0], [0, 0.1], [1, 3.5], ["f", "d"]),
        "ball-stick-2d": (lambda p: p[0]*np.mean([stick(b, g, u, p[1]*1e-9) for u in axes], 0) + (1-p[0])*ball(b, p[2]*1e-9), [0.5, 2.0, 1.0], [0, 0.1, 0.0], [1, 3.5, 3.5], ["f", "d_stick", "d_ball"]),
        "sz-fodf":    (lambda p: sz(p[0], p[1]*1e-9, p[1]*1e-9, p[2]*1e-9, fodf=True), [0.5, 2.0, 0.5], [0, 0.1, 0], [1, 3.5, 3.0], ["f", "dpar", "dperp"]),
        "in-oracle":  (lambda p: p[0]*S_in + (1-p[0])*zep_multi(b, g, axes, p[1]*1e-9, p[2]*1e-9), [0.5, 2.0, 0.5], [0, 0.05, 0], [1, 3.5, 3.0], ["f", "dpar_ex", "dperp"]),
        "ex-oracle":  (lambda p: p[0]*np.mean([stick(b, g, u, p[1]*1e-9) for u in axes], 0) + (1-p[0])*S_ex, [0.5, 2.0], [0, 0.1], [1, 3.5], ["f", "dpar"]),
        "ex-oracle-fodf": (lambda p: p[0]*stick_fodf(b, g, U, w, p[1]*1e-9) + (1-p[0])*S_ex, [0.5, 2.0], [0, 0.1], [1, 3.5], ["f", "dpar"]),
    }
    for name, (model, x0, lo, hi, names) in specs.items():
        x, r = fit(lambda p: model(p) - S, x0, lo, hi)
        out[name] = (float(x[0]), dict(zip(names[1:], map(float, x[1:]))), r)
    return out


def simulate(sub, b, g, n, estimator="abs", seed=0, delta=6e-3, Delta=12e-3):
    """Like sb.simulate_substrate_signal but with a choice of signal estimator:
    'abs' = |mean e^{iφ}| (the benchmark's; positively biased by ~√(π/4n) where
    the true signal is ~0) or 'real' = Re mean e^{iφ} (unbiased for a
    symmetric phase distribution)."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    R_in = np.asarray(sb.pgse_lobe_sums(sub, True, 2.0e-9, 1e-5, n, delta, Delta, k1))
    R_ex = np.asarray(sb.pgse_lobe_sums(sub, False, 2.0e-9, 1e-5, n, delta, Delta, k2))
    G = np.sqrt(b / (sb.GAMMA ** 2 * delta ** 2 * (Delta - delta / 3.0)))
    def sig(R):
        z = np.exp(1j * sb.GAMMA * G[:, None] * (g @ R.T)).mean(1)
        return np.abs(z) if estimator == "abs" else np.real(z)
    S_in, S_ex = sig(R_in), sig(R_ex); f = sub.f_intra
    return f * S_in + (1 - f) * S_ex, S_in, S_ex


def full_fit_ablation(data, b, g, n_iter):
    """The actual PRISM-JAX plus-x pipeline on the noisy volume, with one
    knob turned at a time. Reports the WM-internal f_i (what doc 008 tabulates),
    the WM / isotropic fractions and the voxel-level intra fraction f_i·f_wm."""
    from dataclasses import replace
    from dmipy_jax.validation import prism_jax as pj
    base = replace(pj.PrismConfig(n_fibres=2, n_iter=n_iter, loss="nll"), learn_diffusivities=True,
                   tortuosity=True, use_restricted=False, separate_extra_dpar=True,
                   lam_diffusivity_prior=1.0, diffusivity_prior_centre=2.0e-9, diffusivity_prior_sd=0.2, d_par=2.0e-9)
    variants = {
        "plus-x":          base,
        "no-tortuosity":   replace(base, tortuosity=False),
        "no-D-prior":      replace(base, lam_diffusivity_prior=0.0),
        "no-spatial":      replace(base, lam_spatial=0.0, lam_continuity=0.0),
        "no-sparsity":     replace(base, lam_sparse=0.0, lam_repulsion=0.0),
        "mse-loss":        replace(base, loss="mse"),
        "fixed-D-prism":   pj.PrismConfig(n_fibres=2, n_iter=n_iter, loss="nll"),
        "fixed-D-nores":   replace(pj.PrismConfig(n_fibres=2, n_iter=n_iter, loss="nll"), use_restricted=False),
    }
    mask = np.ones(data.shape[:3], bool); out = {}
    for name, cfg in variants.items():
        t0 = time.time(); fit = pj.fit_prism(data, mask, b, g, cfg)
        fwm = fit.wm_fracs.sum(1); fiso = fit.fracs[:, 0] + fit.fracs[:, 1]; fres = fit.fracs[:, -1]
        out[name] = {"fintra": float(fit.fintra.mean()), "f_wm": float(fwm.mean()), "f_iso": float(fiso.mean()),
                     "f_csf": float(fit.fracs[:, 0].mean()), "f_gm": float(fit.fracs[:, 1].mean()), "f_res": float(fres.mean()),
                     "f_voxel": float((fit.fintra * fwm + fres).mean()),
                     "dpar": float(fit.d_par * 1e9), "dpar_ex": float((fit.d_par_extra or fit.d_par) * 1e9),
                     "dperp": float(np.mean(fit.d_perp) * 1e9)}
        o = out[name]
        print(f"    {name:15s} f_i={o['fintra']:.3f}  f_wm={o['f_wm']:.2f} f_iso={o['f_iso']:.2f} (csf {o['f_csf']:.2f} gm {o['f_gm']:.2f}) "
              f"f_res={o['f_res']:.2f}  f_i·f_wm+res={o['f_voxel']:.3f}  D∥={o['dpar']:.2f} D∥ex={o['dpar_ex']:.2f} D⊥={o['dperp']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 45, 90])
    ap.add_argument("--geoms", nargs="+", default=["straight", "tortuous"])
    ap.add_argument("--n-particles", type=int, default=4000, help="benchmark value")
    ap.add_argument("--particle-sweep", type=int, nargs="*", default=[4000, 32000],
                    help="MC estimator check: |z| vs Re z at these particle counts (angle 0 only)")
    ap.add_argument("--estimator", choices=["abs", "real"], default="abs")
    ap.add_argument("--snr", type=float, default=30); ap.add_argument("--n-noisy", type=int, default=64)
    ap.add_argument("--full-fit", action="store_true", help="run the PRISM-JAX plus-x ablation on the noisy volume")
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--out", default="validation/substrate_compartment_diagnostic.json")
    args = ap.parse_args()
    bvals, bvecs = prism_scheme(); b = np.asarray(bvals); g = np.asarray(bvecs)
    results = {}
    for geom in args.geoms:
        df = sb.caterpillar_bundle(icvf=0.5, box_um=10.0, tortuous=int(geom == "tortuous"),
                                   beading=0.3 if geom == "tortuous" else 0.0)
        for ang in args.angles:
            sub = sb.make_crossing_substrate(df, ang, 10.0)
            t0 = time.time()
            S, S_in, S_ex = simulate(sub, b, g, args.n_particles, args.estimator)
            U, w = segment_dirs(sub)
            if ang == 0 and args.particle_sweep:
                # MC estimator floor: S_ex at b=3000 and the sz-free / ball-stick f under |z| vs Re z
                print(f"\n### {geom}: MC estimator check (angle 0)")
                sweep = {}
                for n in args.particle_sweep:
                    for est in ("abs", "real"):
                        S_, Si_, Se_ = simulate(sub, b, g, n, est)
                        fits_ = all_fits(S_, Si_, Se_, b, g, sub.axes, U, w, sub.f_intra)
                        row = {"S_ex_b3000": float(Se_[-64:].mean()), "S_in_b3000": float(Si_[-64:].mean()),
                               "f_sz_free": fits_["sz-free"][0], "f_sz_tort": fits_["sz-tort"][0],
                               "f_ball_stick": fits_["ball-stick"][0], "f_sz_fixed": fits_["sz-fixed"][0],
                               "dperp_ex": fits_["F3-ex-zep"][1]["dperp"], "Dperp_b3000": fits_["F3-ex-zep"][1]["Dperp_b3000"]}
                        sweep[f"{n}_{est}"] = row
                        print(f"  n={n:>6d} {est:4s}: S_ex(b3000)={row['S_ex_b3000']:.4f}  S_in(b3000)={row['S_in_b3000']:.4f}  "
                              f"f: sz-free {row['f_sz_free']:.3f} sz-tort {row['f_sz_tort']:.3f} ball-stick {row['f_ball_stick']:.3f} "
                              f"sz-fixed {row['f_sz_fixed']:.3f}  D⊥ex={row['dperp_ex']:.2f} D⊥(b3000)={row['Dperp_b3000']:.2f}", flush=True)
                results[f"{geom}_mc_estimator"] = sweep
            # fODF concentration: <cos²> about each axis, weighted
            c2 = [float(((U @ a) ** 2 * w).sum() / w.sum()) for a in sub.axes]
            print(f"\n### {geom} {ang:.0f}°: f_geom={sub.f_intra:.3f}  MC {time.time()-t0:.0f}s  "
                  f"fODF <cos²> about axes {np.round(c2, 3)}  S_in(b3000)={S_in[-64:].mean():.3f} S_ex(b3000)={S_ex[-64:].mean():.3f}", flush=True)
            key = f"{geom}_{int(ang)}"
            res = {"f_geom": sub.f_intra, "c2": c2}
            clean = all_fits(S, S_in, S_ex, b, g, sub.axes, U, w, sub.f_intra)
            res["clean"] = {k: {"f": v[0], **v[1], "rmse": v[2]} for k, v in clean.items()}
            print(f"  {'model':16s} {'f':>6s} {'Δf':>6s} {'rmse':>7s}  params")
            for k, v in clean.items():
                df_ = v[0] - sub.f_intra if not np.isnan(v[0]) else np.nan
                pr = "  ".join(f"{a}={c:.2f}" for a, c in v[1].items())
                print(f"  {k:16s} {v[0]:6.3f} {df_:+6.3f} {v[2]:7.4f}  {pr}")
            # noisy: mean f over independent Rician voxels for the main models
            data, _ = sb.build_volume(S, (args.n_noisy, 1, 1), args.snr, seed=0)
            noisy = {}
            for i in range(args.n_noisy):
                y = data[i, 0, 0] / data[i, 0, 0, b < 50e6].mean()
                fi = all_fits(y, S_in, S_ex, b, g, sub.axes, U, w, sub.f_intra)
                for k in ("sz-free", "sz-tort", "sz-tort-x", "sz-fixed", "ball-stick", "sz-fodf"):
                    noisy.setdefault(k, []).append(fi[k][0])
            res["noisy"] = {k: {"f_mean": float(np.mean(v)), "f_sd": float(np.std(v))} for k, v in noisy.items()}
            print(f"  SNR {args.snr:.0f} × {args.n_noisy}: " + "  ".join(f"{k} {np.mean(v):.3f}±{np.std(v):.3f}" for k, v in noisy.items()))
            if args.full_fit:
                print(f"  full PRISM-JAX fits on the {args.n_noisy}-voxel SNR-{args.snr:.0f} volume (geometry f={sub.f_intra:.3f}):")
                vol, _ = sb.build_volume(S, (8, 8, 1), args.snr, seed=0)
                res["full_fit"] = full_fit_ablation(vol, b, g, args.n_iter)
            results[key] = res
            json.dump(results, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

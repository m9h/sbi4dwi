#!/usr/bin/env python3
"""
Axonal permeability on CATERPillar substrates (doc 008 §14.4), after the
permeability rung of Nguyen-Duc et al. 2026: undulating + beaded axons at
packing 0.5, membrane permeability κ = 0 … 4·10⁻⁵ m/s, where the Standard
Model's f fell by 75 %. Two questions:

1. What does PRISM-JAX (no-iso, with the §14.3 glial sphere) read for f_i
   on the paper's single-Δ protocol as κ rises? (Expect the same collapse —
   no single-Δ model can tell exchange from a smaller intra pool.)
2. Does the Kärger kernel of §9.3 recover f_i and τ_ex on the *realistic*
   permeable substrate, on the paper's protocol (single Δ: should not) and on
   the Fisher-designed multi-Δ protocol (should), the way it did on
   MCMRSimulator's straight cylinders?

Ground-truth τ_ex comes from the walkers themselves: the fraction of
initially intra-axonal walkers found extra-axonal after T = Δ + δ, read
through the two-site Kärger solution.
"""
import argparse, json, time
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import least_squares
from validation.lib import substrate_benchmark as sb
from validation.lib import prism_exchange as px
from validation.validate_substrate_ladder import chuv_scheme, RUNGS
from dmipy_jax.prism import prism_jax as pj
from dmipy_jax.prism import prism_uncertainty as pu


def optimised_scheme(n_dirs=32, seed=0):
    """§9.3 Fisher-designed protocol: b 1.4 / 5 / 5 ms/µm², δ 10 ms, Δ 15 / 15 / 51.5 ms, 32 dirs each (+1 b0)."""
    from dipy.core.sphere import disperse_charges, HemiSphere
    rng = np.random.default_rng(seed)
    hs, _ = disperse_charges(HemiSphere(theta=np.pi * rng.random(n_dirs), phi=2 * np.pi * rng.random(n_dirs)), 5000)
    d = hs.vertices
    shells = [(1.4e9, 10e-3, 15e-3), (5.0e9, 10e-3, 15e-3), (5.0e9, 10e-3, 51.5e-3)]
    bv = np.concatenate([[0.0]] + [np.full(n_dirs, b) for b, _, _ in shells])
    gv = np.concatenate([np.zeros((1, 3))] + [d] * len(shells))
    de = np.concatenate([[10e-3]] + [np.full(n_dirs, dl) for _, dl, _ in shells])
    De = np.concatenate([[15e-3]] + [np.full(n_dirs, Dl) for _, _, Dl in shells])
    return bv, gv, de, De


def karger_fit(bvals, bvecs, delta, Delta, u, y, x0=(0.5, 30.0, 2.2, 0.6, 0.5)):
    """(f_i, τ_ex [ms], D∥, D∥ex/D∥, D⊥ [µm²/ms]) least squares with the closed-form
    Kärger kernel, fibre u known. The extra-cellular axial D is tied to the
    stick's by a ratio ≤ 1: with f free, the stick and the zeppelin are
    label-exchangeable otherwise and the fit can land on the mirror solution
    (a small slow "intra" pool). Returned x[3] is the ratio."""
    b = jnp.asarray(bvals); g = jnp.asarray(bvecs); de = jnp.asarray(delta); De = jnp.asarray(Delta); u = jnp.asarray(u)

    def model(p):
        f, tau, dpar, ratio, dperp = p
        return px.karger_pgse(b, g, u, f, dpar * 1e-9, ratio * dpar * 1e-9, dperp * 1e-9, tau * 1e-3, de, De)
    fm = jax.jit(model); jac = jax.jit(jax.jacfwd(model))
    r = least_squares(lambda p: np.asarray(fm(jnp.asarray(p))) - y, x0, jac=lambda p: np.asarray(jac(jnp.asarray(p))),
                      bounds=([0.05, 1.0, 0.3, 0.3, 0.02], [0.95, 2000.0, 3.0, 1.0, 2.0]), max_nfev=400)
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.0, 1e-5, 2e-5, 4e-5])
    ap.add_argument("--icvf", type=float, default=0.5)
    ap.add_argument("--box-um", type=float, default=20.0)
    ap.add_argument("--snr", type=float, default=50)
    ap.add_argument("--n-particles", type=int, default=10000)
    ap.add_argument("--dt", type=float, default=4e-6)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="validation/substrate_permeability")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from dmipy_jax.simulation.substrates.caterpillar import CATERPillarCLI
    kw = dict(RUNGS["beading"]); kw["icvf"] = args.icvf
    cli = CATERPillarCLI()
    t0 = time.time()
    df = cli.generate(cli.default_config(box_um=args.box_um, threads=args.threads, **kw), cells="all")
    sub = sb.make_crossing_substrate(df, 0.0, args.box_um, seed=args.seed)
    fr = sub.fractions
    f_in = fr["intra"] / (fr["intra"] + fr["extra"])
    u = sub.axes[0]
    print(f"substrate: {len(df)} spheres / {df.id.nunique()} axons, intra {fr['intra']:.3f} extra {fr['extra']:.3f} → f {f_in:.3f}; "
          f"mean inner radius {1e6*np.mean(sub.radii_m):.2f} µm, gen {time.time()-t0:.0f}s", flush=True)

    D_in, D_ex = 2.5e-9, 1.5e-9
    bv_c, gv_c = chuv_scheme()
    de_c, De_c = 16.5e-3, 55.5e-3
    bv_o, gv_o, de_o, De_o = optimised_scheme()
    timings = sorted({(float(a), float(b)) for a, b in zip(de_o, De_o)} | {(de_c, De_c)})
    results = {"f_gt": f_in, "fractions": fr, "mean_radius_m": float(np.mean(sub.radii_m))}
    saved = {"bvals_chuv": bv_c, "bvecs_chuv": gv_c, "bvals_opt": bv_o, "bvecs_opt": gv_o, "delta_opt": de_o, "Delta_opt": De_o,
             "axis": u, "f_gt": f_in}
    key = jax.random.PRNGKey(args.seed)
    for kappa in args.kappas:
        tag = f"{kappa:.0e}"
        print(f"\n### κ = {kappa:.1e} m/s", flush=True)
        res = {"kappa": kappa}
        R = {}; tau_meas = {}
        for (de, De) in timings:
            t0 = time.time()
            key, k = jax.random.split(key)
            Rk, c0, c1 = sb.pgse_lobe_sums_permeable(sub, kappa, D_in, D_ex, args.dt, args.n_particles, de, De, k)
            R[(de, De)] = np.asarray(Rk)
            tau = sb.exchange_time_from_transfer(c0, c1, de + De, f_in)
            tau_meas[(de, De)] = tau
            p_out = float(np.mean(np.asarray(c1)[np.asarray(c0) == 0] == 1))
            print(f"  walk δ {1e3*de:.1f} Δ {1e3*De:.1f} ms: {time.time()-t0:.0f}s, intra→extra fraction {p_out:.3f} "
                  f"→ τ_ex {1e3*tau:.1f} ms; intra fraction at end {float(np.mean(np.asarray(c1)==0)):.3f}", flush=True)
        res["tau_ex_measured_ms"] = {f"{1e3*de:.1f}/{1e3*De:.1f}": 1e3 * t for (de, De), t in tau_meas.items()}
        tau_gt = 1e3 * tau_meas[(de_c, De_c)]       # longest walk: best-determined

        # --- signals ---
        S_c = np.asarray(sb.signals_from_lobe_sums(R[(de_c, De_c)], bv_c, gv_c, de_c, De_c))
        S_o = np.zeros(len(bv_o))
        for (de, De) in timings:
            m = np.isclose(de_o, de) & np.isclose(De_o, De)
            if m.any():
                S_o[m] = np.asarray(sb.signals_from_lobe_sums(R[(de, De)], bv_o[m], gv_o[m], de, De))
        sh = bv_c > 4000e6
        print(f"  S(b5000, CHUV) {S_c[sh].mean():.3f}", flush=True)
        saved.update({f"k{tag}_S_chuv": S_c, f"k{tag}_S_opt": S_o})

        # --- 1. PRISM on the paper's protocol ---
        data, mask = sb.build_volume(S_c, (8, 8, 1), args.snr, seed=args.seed)
        saved[f"k{tag}_data_chuv"] = data
        N = int(mask.sum()); gt = np.zeros((N, 2, 3)); gt[:, 0] = u
        base = pj.PrismConfig(n_fibres=2, n_iter=args.n_iter, loss="nll", learn_diffusivities=True, tortuosity=True,
                              use_isotropic=False, use_restricted=False, disperse=True, separate_extra_dpar=True,
                              lam_diffusivity_prior=1.0, diffusivity_prior_centre=2.5e-9, diffusivity_prior_sd=0.2,
                              d_par=2.5e-9, d_par_range=(0.3e-9, 3.5e-9), small_delta=de_c, big_delta=De_c)
        for meth, cfg in {"plus-x-disp-noiso": base,
                          "plus-x-disp-noiso-sphere": replace(base, use_restricted=True, learn_restricted=True,
                                                              restricted_kernel="sphere", sphere_radius=4e-6,
                                                              sphere_radius_range=(1e-6, 10e-6))}.items():
            t0 = time.time()
            fit = pj.fit_prism(data, mask, bv_c, gv_c, cfg)
            post = pu.laplace_fixel_posterior(fit, data, bv_c, gv_c)
            err, rec = pj.angular_error_best_match(fit.dirs, fit.wm_fracs, gt)
            fi = float(np.nanmean(fit.fintra)); fres = float(np.mean(fit.f_res)) if cfg.use_restricted else 0.0
            print(f"  {meth:24s} err={err:4.2f}° f_i={fi:.3f}±{float(np.nanmean(post.fintra_sd)):.3f} ({100*(fi-f_in)/f_in:+.0f} %) "
                  f"f_res={fres:.3f} D∥={fit.d_par*1e9:.2f} D∥ex={fit.d_par_extra*1e9:.2f} D⊥={fit.d_perp*1e9:.2f} ({time.time()-t0:.0f}s)", flush=True)
            res[meth] = dict(err=err, recall=rec, fi=fi, fi_sd=float(np.nanmean(post.fintra_sd)), f_res=fres,
                             d_par=fit.d_par, d_par_extra=fit.d_par_extra, d_perp=fit.d_perp)

        # --- 2. Kärger kernel on both protocols ---
        for name, (bv, gv, de, De, S) in {"karger-chuv": (bv_c, gv_c, np.full(len(bv_c), de_c), np.full(len(bv_c), De_c), S_c),
                                          "karger-optimised": (bv_o, gv_o, de_o, De_o, S_o)}.items():
            t0 = time.time()
            x, rmse = karger_fit(bv, gv, de, De, u, S)
            print(f"  {name:18s} clean: f_i={x[0]:.3f} ({100*(x[0]-f_in)/f_in:+.0f} %) τ_ex={x[1]:.1f} ms (GT {tau_gt:.1f}) "
                  f"D∥={x[2]:.2f} D∥ex={x[3]*x[2]:.2f} D⊥={x[4]:.2f} rmse {rmse:.4f} ({time.time()-t0:.0f}s)", flush=True)
            res[name] = {"clean": x.tolist(), "rmse": rmse, "tau_gt_ms": tau_gt}
            rng = np.random.default_rng(args.seed)
            est = []
            for _ in range(args.repeats):
                s = 1.0 / args.snr
                y = np.sqrt((S + rng.normal(0, s, S.shape)) ** 2 + rng.normal(0, s, S.shape) ** 2)
                est.append(karger_fit(bv, gv, de, De, u, y, x0=tuple(x))[0])
            est = np.array(est); med = np.median(est, 0); sd = np.std(est, 0)
            print(f"  {'':18s} SNR {args.snr:.0f} × {args.repeats}: f_i {med[0]:.3f} ± {sd[0]:.3f}  τ_ex {med[1]:.1f} ± {sd[1]:.1f} ms  "
                  f"D∥ {med[2]:.2f} ± {sd[2]:.2f}  D∥ex/D∥ {med[3]:.2f} ± {sd[3]:.2f}  D⊥ {med[4]:.2f} ± {sd[4]:.2f}", flush=True)
            res[name][f"snr{args.snr:.0f}"] = {"median": med.tolist(), "sd": sd.tolist()}
        results[tag] = res
        np.savez(args.out + "_signals.npz", **saved)
        json.dump(results, open(args.out + "_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()

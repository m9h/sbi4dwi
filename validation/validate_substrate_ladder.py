#!/usr/bin/env python3
"""
Realism ladder on CATERPillar substrates, after Nguyen-Duc et al. 2026
(Imaging Neuroscience; bioRxiv 10.64898/2026.01.28.702302): the Standard
Model's f was accurate within ~1 % on clean substrates but +20 % with
beading, +35 % with impermeable astrocytes, +39 % with myelin. Here the same
ladder is run through PRISM-JAX with its Laplace posterior, so the question
is not only "how biased is f_i" but "does the posterior width say so".

Each rung is one CATERPillar generator setting (20 µm box, their diffusion
protocol: b = 0/500/1000/2000/3000/5000 s/mm², 60 directions per shell,
δ = 16.5 ms, Δ = 55.5 ms, D_in = 2.5, D_ex = 1.5 µm²/ms, SNR 50 on b0).
Signals from the JAX walker (validation/lib/substrate_benchmark.py): intra,
extra and — when present — glial compartments, myelin without water.

Rungs: baseline (straight), undulation, beading, dispersion, myelin,
astrocytes. Permeability is not on this ladder: the walker has no membrane
transit (doc 008 §9.3 covers exchange with MCMRSimulator cylinders).

Ground truth f is reported two ways: f_water = intra / all water (what a
fitted intra fraction should match when glia are modelled as "other") and
f_axon = intra / (intra + extra), the paper's definition.
"""
import argparse, json, time
from dataclasses import replace
import numpy as np
from validation.lib import substrate_benchmark as sb
from dmipy_jax.prism import prism_jax as pj
from dmipy_jax.prism import prism_uncertainty as pu


def chuv_scheme(n_dirs=60, seed=0):
    """Nguyen-Duc et al. protocol: six shells × 60 directions (+1 b0)."""
    from dipy.core.sphere import disperse_charges, HemiSphere
    rng = np.random.default_rng(seed)
    hs, _ = disperse_charges(HemiSphere(theta=np.pi * rng.random(n_dirs), phi=2 * np.pi * rng.random(n_dirs)), 5000)
    d = hs.vertices
    shells = [500.0, 1000.0, 2000.0, 3000.0, 5000.0]
    bv = np.concatenate([[0.0], np.repeat(shells, n_dirs)])
    gv = np.concatenate([np.zeros((1, 3))] + [d] * len(shells))
    return bv * 1e6, gv


# generator settings per rung (fractions of the voxel; paper values where they
# transfer to a 20 µm box — glial process length is shortened from 30 to 10 µm
# so processes fit the box, somata keep r = 3 µm)
RUNGS = {
    "baseline":   dict(icvf=0.6, tortuous=False),
    "undulation": dict(icvf=0.6, tortuous=True, epsilon=0.2),
    "beading":    dict(icvf=0.6, tortuous=True, epsilon=0.2, beading=0.3, beading_std=0.3),
    "dispersion": dict(icvf=0.6, tortuous=True, epsilon=0.2, c2=0.90),
    "myelin":     dict(icvf=0.0, myelin_icvf=0.6, tortuous=False),
    "astrocytes": dict(icvf=0.6, tortuous=False, glia_soma=0.01, glia_processes=0.06, glia_process_length=10.0),
}


def generate(rung, box_um, threads):
    from dmipy_jax.simulation.substrates.caterpillar import CATERPillarCLI
    cli = CATERPillarCLI()
    kw = dict(RUNGS[rung]); kw.setdefault("c2", 0.98)
    cfg = cli.default_config(box_um=box_um, threads=threads, **kw)
    return cli.generate(cfg, cells="all")


METHOD_SETS = {
    # doc 008 §14.2: plus-x + dispersion with and without the isotropic balls
    "iso": ["plus-x-disp-noiso", "plus-x-disp-iso", "plus-x-disp-iso-res"],
    # doc 008 §14.3: the restricted pool as a glial compartment (no CSF/GM balls)
    "glia": ["plus-x-disp-noiso", "plus-x-disp-noiso-res", "plus-x-disp-noiso-resD", "plus-x-disp-noiso-sphere"],
    # D_res free up to 1.0 turns the pool into a hindered-isotropic absorber (takes 11–15 % on clean rungs);
    # cap it at the glial range (MC glia: 0.04) and let the sphere's D be the intrinsic 2.5
    "glia-bounded": ["plus-x-disp-noiso", "plus-x-disp-noiso-resDslow", "plus-x-disp-noiso-sphere25"],
}


def fit_methods(n_iter, delta, Delta, which="iso"):
    """plus-x + dispersion (doc 008 §10.3 winner), D∥ prior at the intrinsic 2.5.
    -noiso: no isotropic balls (WM-only);  -iso: CSF+GM balls;  -res: PRISM's
    restricted ball D_res 0.2 fixed;  -resD: restricted ball with learnable
    D_res ∈ [0.01, 1.0];  -sphere: GPD soma with learnable radius ∈ [1, 10] µm
    and the stick's D∥ inside."""
    base = pj.PrismConfig(n_fibres=2, n_iter=n_iter, loss="nll", learn_diffusivities=True, tortuosity=True,
                          use_restricted=False, disperse=True, separate_extra_dpar=True, lam_diffusivity_prior=1.0,
                          diffusivity_prior_centre=2.5e-9, diffusivity_prior_sd=0.2, d_par=2.5e-9,
                          d_par_range=(0.3e-9, 3.5e-9), small_delta=delta, big_delta=Delta)
    noiso = replace(base, use_isotropic=False)
    all_methods = {
        "plus-x-disp-noiso": noiso,
        "plus-x-disp-iso": base,
        "plus-x-disp-iso-res": replace(base, use_restricted=True),
        "plus-x-disp-noiso-res": replace(noiso, use_restricted=True),
        "plus-x-disp-noiso-resD": replace(noiso, use_restricted=True, learn_restricted=True, d_res=0.1e-9,
                                          d_res_range=(0.01e-9, 1.0e-9)),
        "plus-x-disp-noiso-sphere": replace(noiso, use_restricted=True, learn_restricted=True,
                                            restricted_kernel="sphere", sphere_radius=4e-6,
                                            sphere_radius_range=(1e-6, 10e-6)),
        "plus-x-disp-noiso-resDslow": replace(noiso, use_restricted=True, learn_restricted=True, d_res=0.05e-9,
                                              d_res_range=(0.005e-9, 0.3e-9)),
        "plus-x-disp-noiso-sphere25": replace(noiso, use_restricted=True, learn_restricted=True,
                                              restricted_kernel="sphere", sphere_radius=4e-6, d_sphere=2.5e-9,
                                              sphere_radius_range=(1e-6, 8e-6)),
    }
    return {m: all_methods[m] for m in METHOD_SETS[which]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=list(RUNGS))
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 60])
    ap.add_argument("--box-um", type=float, default=20.0)
    ap.add_argument("--snr", type=float, default=50)
    ap.add_argument("--n-particles", type=int, default=4000)
    ap.add_argument("--dt", type=float, default=4e-6, help="axon compartments (step 0.14 µm at D 2.5)")
    ap.add_argument("--dt-glia", type=float, default=2e-6, help="glial walkers (process radius ≥ 0.15 µm)")
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="validation/substrate_ladder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", choices=list(METHOD_SETS), default="iso")
    ap.add_argument("--from-npz", default=None,
                    help="re-fit the noisy volumes saved by an earlier run (CATERPillar is not seed-stable, "
                         "so this is the only way to compare models on the same substrates)")
    ap.add_argument("--evidence", action="store_true", help="per-voxel Laplace log-evidence per method")
    args = ap.parse_args()

    bvals, bvecs = chuv_scheme()
    delta, Delta = 16.5e-3, 55.5e-3
    D_in, D_ex = 2.5e-9, 1.5e-9
    results = {}
    saved = {"bvals": bvals, "bvecs": bvecs, "snr": args.snr, "delta": delta, "Delta": Delta}
    prev = np.load(args.from_npz) if args.from_npz else None
    if prev is not None:
        bvals, bvecs = prev["bvals"], prev["bvecs"]
    for rung in args.rungs:
        t0 = time.time()
        if prev is None:
            df = generate(rung, args.box_um, args.threads)
            ax = df[df.type == 0]
            print(f"\n### {rung}: {len(ax)} axon spheres / {ax.id.nunique()} axons, {int((df.type == 1).sum())} glial spheres, "
                  f"gen {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"\n### {rung} (volumes from {args.from_npz})", flush=True)
        for ang in args.angles:
            key = f"{rung}_{int(ang)}"
            if prev is None:
                t0 = time.time()
                sub = sb.make_crossing_substrate(df, ang, args.box_um, seed=args.seed)
                fr = sub.fractions
                print(f"  angle {ang:>3.0f}: box fractions intra {fr['intra']:.3f} myelin {fr['myelin']:.3f} glia {fr['glia']:.3f} "
                      f"extra {fr['extra']:.3f} → f_water {sub.f_intra:.3f} f_axon {sub.f_axon_water:.3f}  ({time.time()-t0:.0f}s)", flush=True)
                t0 = time.time()
                import jax
                k_in, k_ex, k_gl = jax.random.split(jax.random.PRNGKey(args.seed), 3)
                sig = lambda R: np.asarray(sb.signals_from_lobe_sums(R, bvals, bvecs, delta, Delta))
                S_in = sig(sb.pgse_lobe_sums(sub, "intra", D_in, args.dt, args.n_particles, delta, Delta, k_in))
                S_ex = sig(sb.pgse_lobe_sums(sub, "extra", D_ex, args.dt, args.n_particles, delta, Delta, k_ex))
                S_gl = np.full(len(bvals), np.nan)
                water = fr["intra"] + fr["extra"] + fr["glia"]
                S = (fr["intra"] * S_in + fr["extra"] * S_ex) / water
                if sub.has_glia and fr["glia"] > 0:
                    # glial processes (r ≥ 0.15 µm) need the finer step
                    S_gl = sig(sb.pgse_lobe_sums(sub, "glia", D_in, args.dt_glia, args.n_particles, delta, Delta, k_gl))
                    S = S + fr["glia"] / water * S_gl
                sh = bvals > 4000e6
                print(f"    MC {time.time()-t0:.0f}s  S(b5000) intra {S_in[sh].mean():.3f} extra {S_ex[sh].mean():.3f} "
                      f"glia {np.nanmean(S_gl[sh]) if np.isfinite(S_gl).any() else float('nan'):.3f} total {S[sh].mean():.3f}", flush=True)
                data, mask = sb.build_volume(S, (8, 8, 1), args.snr, seed=args.seed)
                axes = sub.axes
                saved.update({f"{key}_signal": S, f"{key}_intra": S_in, f"{key}_extra": S_ex, f"{key}_glia": S_gl,
                              f"{key}_data": data, f"{key}_axes": axes,
                              f"{key}_fractions": np.array([fr["intra"], fr["myelin"], fr["glia"], fr["extra"]])})
                np.savez(args.out + "_signals.npz", **saved)
            else:
                if f"{key}_data" not in prev.files:
                    print(f"  angle {ang:>3.0f}: not in npz, skipped"); continue
                data, axes = prev[f"{key}_data"], prev[f"{key}_axes"]
                fi_, fm_, fg_, fe_ = prev[f"{key}_fractions"]
                fr = {"intra": float(fi_), "myelin": float(fm_), "glia": float(fg_), "extra": float(fe_)}
                mask = np.ones(data.shape[:3], bool)
            f_water = fr["intra"] / (fr["intra"] + fr["glia"] + fr["extra"])
            f_axon = fr["intra"] / (fr["intra"] + fr["extra"])
            f_glia_water = fr["glia"] / (fr["intra"] + fr["glia"] + fr["extra"])
            print(f"  angle {ang:>3.0f}: f_water {f_water:.3f} f_axon {f_axon:.3f} glia/water {f_glia_water:.3f}", flush=True)
            N = int(mask.sum())
            gt = np.zeros((N, 2, 3)); gt[:, 0] = axes[0]
            if ang > 0:
                gt[:, 1] = axes[1]
            for meth, cfg in fit_methods(args.n_iter, delta, Delta, args.methods).items():
                t0 = time.time()
                fit = pj.fit_prism(data, mask, bvals, bvecs, cfg)
                post = pu.laplace_fixel_posterior(fit, data, bvals, bvecs)
                ev = float(np.mean(pu.voxel_log_evidence(fit, data, bvals, bvecs))) if args.evidence else float("nan")
                err, rec = pj.angular_error_best_match(fit.dirs, fit.wm_fracs, gt)
                fi = float(np.nanmean(fit.fintra)); fi_sd = float(np.nanmean(post.fintra_sd))
                fiso = float(np.mean(fit.fracs[:, 0] + fit.fracs[:, 1])) if cfg.use_isotropic else 0.0
                fres = float(np.mean(fit.f_res)) if cfg.use_restricted else 0.0
                z = (fi - f_water) / max(fi_sd, 1e-6)
                rest = ""
                if cfg.use_restricted:
                    rest = (f" r_s={fit.sphere_radius*1e6:.1f}µm" if cfg.restricted_kernel == "sphere"
                            else f" D_res={fit.d_res*1e9:.3f}")
                print(f"    {meth:24s} err={err:5.2f}° recall={100*rec:5.1f}%  f_i={fi:.3f}±{fi_sd:.3f} "
                      f"(Δ vs f_water {fi-f_water:+.3f}, z={z:+.1f}; vs f_axon {fi-f_axon:+.3f})  "
                      f"f_iso={fiso:.3f} f_res={fres:.3f}{rest}  D∥={fit.d_par*1e9:.2f} D∥ex={fit.d_par_extra*1e9:.2f}  "
                      f"σθ={np.nanmean(post.sigma_deg[:, 0]):.1f}°  logZ={ev:.1f}  ({time.time()-t0:.0f}s)", flush=True)
                results[f"{key}|{meth}"] = dict(err=err, recall=rec, fi=fi, fi_sd=fi_sd, f_water=f_water,
                                                f_axon=f_axon, f_glia_water=f_glia_water, f_iso=fiso, f_res=fres,
                                                d_par=fit.d_par, d_par_extra=fit.d_par_extra, d_res=fit.d_res,
                                                sphere_radius=fit.sphere_radius, log_evidence=ev,
                                                sigma_deg=float(np.nanmean(post.sigma_deg[:, 0])), fractions=fr)
            json.dump(results, open(args.out + "_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()

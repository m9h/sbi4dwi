#!/usr/bin/env python3
"""
Tier B5 (doc 008 §10): choose the PRISM-JAX regularisation weights from data
instead of by hand. Measurement-split cross-validation: fit on a random 80 %
of the gradient directions (b0s always kept), score the Rician NLL of the
held-out 20 % under the fitted model, per hyper-parameter setting, averaged
over splits. The fit itself is the differentiable plus-x MAP; the outer
loop is a small grid over (λ_spatial, λ_sparse, λ_repulsion, D-prior λ)
because the inner Rprop loop is a fori_loop rather than an Optimistix
solve, so implicit differentiation is not available yet — the CV number
is the honest answer to "how were the weights chosen" either way.

Runs on DiSCo (authors' protocol, SNR 30) and on one CATERPillar substrate
volume, and reports held-out NLL, plus the downstream metric where GT
exists (intra-VF r on DiSCo; angular error on the substrate) for the CV
optimum vs PRISM's defaults.
"""
import argparse, json, time, importlib.util, itertools
from dataclasses import replace
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, nibabel as nib
from scipy.stats import pearsonr
from dmipy_jax.validation import prism_jax as pj
from dmipy_jax.validation.force_disco import disco_subject_path

vdfp = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location("vdfp", Path(__file__).with_name("validate_disco_force_protocol.py"))); s.loader.exec_module(vdfp)

# NLL data term is ~1e2 per voxel; PRISM's MSE-scale defaults (0.01–0.02) are inert under it, so the grid spans the NLL scale.
GRID = {"lam_spatial": [0.0, 1.0, 10.0, 100.0], "lam_sparse": [0.0, 1.0, 10.0], "lam_repulsion": [0.0, 1.0, 10.0], "lam_diffusivity_prior": [0.0, 1.0]}


def heldout_nll(fit, data, mask, bvals, bvecs, idx_test):
    """Rician NLL of held-out measurements under the fitted parameters (σ from the fit)."""
    from dmipy_jax.validation.prism_jax import forward, rician_nll
    y = jnp.asarray(data[mask][:, idx_test]); bv = jnp.asarray(bvals[idx_test]); gv = jnp.asarray(bvecs[idx_test])
    phys = {"s0": jnp.asarray(fit.s0), "fracs": jnp.asarray(fit.fracs), "dirs": jnp.asarray(fit.dirs), "fintra": jnp.asarray(fit.fintra),
            "d_par": jnp.asarray(fit.d_par), "d_perp": jnp.asarray(fit.d_perp), "sigma": jnp.asarray(fit.sigma if fit.sigma else 0.05)}
    if fit.d_par_extra is not None: phys["d_par_extra"] = jnp.asarray(fit.d_par_extra)
    if fit.odi is not None:
        phys["odi"] = jnp.asarray(fit.odi); phys["kappa"] = 1.0 / jnp.tan(jnp.pi * phys["odi"] / 2.0)
    pred = forward(phys, bv, gv, fit.cfg)
    return float(jnp.mean(rician_nll(pred, y, phys["sigma"])))


def cv_score(data, mask, bvals, bvecs, cfg, n_splits, seed=0):
    rng = np.random.default_rng(seed); M = len(bvals); b0 = bvals < 50e6; dwi = np.where(~b0)[0]
    scores = []
    for s in range(n_splits):
        test = rng.choice(dwi, size=len(dwi) // 5, replace=False); train = np.setdiff1d(np.arange(M), test)
        fit = pj.fit_prism(data[..., train], mask, bvals[train], bvecs[train], cfg)
        scores.append(heldout_nll(fit, data, mask, bvals, bvecs, test))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=3); ap.add_argument("--n-iter", type=int, default=200)
    ap.add_argument("--dataset", choices=["disco", "substrate"], default="disco"); ap.add_argument("--snr", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.dataset == "disco":
        d = vdfp.load_protocol_data(a.snr); data, mask = d["data"], d["mask"]; bvals = np.asarray(d["bvals"]) * 1e6; bvecs = np.asarray(d["bvecs"])
        gt = nib.load(disco_subject_path(1) / "highRes_DiSCo1_Strand_Intra_Volume_Fraction.nii.gz").get_fdata()[mask]; K = 3; dpar0 = 0.6e-9
        metric = lambda fit: ("intra-VF r", float(pearsonr(fit.fintra, gt)[0]))
    else:
        z = np.load("validation/external/substrate_signals.npz"); key = "straight_45"
        data = z[f"{key}_data"]; mask = np.ones(data.shape[:3], bool); bvals = z["bvals"]; bvecs = z["bvecs"]; K = 2; dpar0 = 2.0e-9
        axes = z[f"{key}_axes"]; gtd = np.zeros((int(mask.sum()), 2, 3)); gtd[:, 0] = axes[0]; gtd[:, 1] = axes[1]
        metric = lambda fit: ("angular err", float(pj.angular_error_best_match(fit.dirs, fit.wm_fracs, gtd)[0]))
    base = replace(pj.PrismConfig(n_fibres=K, n_iter=a.n_iter, loss="nll"), learn_diffusivities=True, d_par=dpar0, tortuosity=True,
                   use_restricted=False, separate_extra_dpar=True, use_isotropic=(a.dataset == "disco"))
    rows = []; t0 = time.time()
    names = list(GRID); combos = list(itertools.product(*GRID.values()))
    print(f"{a.dataset}: {len(combos)} settings × {a.n_splits} splits, {int(mask.sum()):,} voxels", flush=True)
    for vals in combos:
        cfg = replace(base, **dict(zip(names, vals)), lam_continuity=vals[0] / 2)
        m, sd = cv_score(data, mask, bvals, bvecs, cfg, a.n_splits)
        fit = pj.fit_prism(data, mask, bvals, bvecs, cfg); mname, mval = metric(fit)
        rows.append({**dict(zip(names, vals)), "cv_nll": m, "cv_sd": sd, mname: mval})
        print(f"  {dict(zip(names, vals))}  held-out NLL {m:.4f} ± {sd:.4f}   {mname} {mval:.3f}   ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["cv_nll"]); best = rows[0]
    default = next((r for r in rows if r["lam_spatial"] == 0.0 and r["lam_sparse"] == 0.0 and r["lam_repulsion"] == 0.0 and r["lam_diffusivity_prior"] == (1.0 if a.dataset == "substrate" else 0.0)), None)   # ≈ PRISM defaults under NLL (inert priors)
    print(f"\nCV optimum: {best}")
    if default: print(f"PRISM default: {default}")
    json.dump({"rows": rows, "best": best, "default": default}, open(a.out or f"validation/hyperparam_cv_{a.dataset}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

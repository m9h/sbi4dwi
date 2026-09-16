"""
DIPY-facing refinement stage (doc 008 §8 item 6 / §9 item 3): take any
DIPY ``PeaksAndMetrics`` (CSD, MSMT-CSD, FORCE, …) as the initialisation,
run the differentiable, spatially regularised PRISM-JAX fit from it, and
return (a) a refined ``PeaksAndMetrics`` that drops straight into DIPY's
trackers and (b) the Laplace fixel posterior (tangent-plane covariance
per fixel) for uncertainty-aware tractography (CV pruning, doc 007 §8.3).

    pam_ref, post = refine_peaks(data, gtab, mask, pam_in)

The gradient table is DIPY's (b in s/mm²); everything else is SI inside.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Optional, Tuple
import numpy as np
from dmipy_jax.validation import prism_jax as pj, prism_uncertainty as pu


def default_config(n_fibres: int, n_iter: int = 300, d_par_init: float = 1.7e-9, wm_only: bool = False) -> pj.PrismConfig:
    """The plus-x model: learned diffusivities, decoupled extra-cellular D∥,
    tortuosity, no restricted pool; isotropic compartments off for WM-only tissue."""
    return replace(pj.PrismConfig(n_fibres=n_fibres, n_iter=n_iter, loss="nll"), learn_diffusivities=True, d_par=d_par_init,
                   tortuosity=True, use_restricted=False, separate_extra_dpar=True, use_isotropic=not wm_only)


def init_from_pam(pam, mask: np.ndarray, n_fibres: int, frac_min: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """(init_dirs (N,K,3), init_fracs (N,K+3)) from peak_dirs / peak_values in the mask."""
    pd = np.asarray(pam.peak_dirs)[mask][:, :n_fibres]; pv = np.asarray(pam.peak_values)[mask][:, :n_fibres].astype(float)
    N = pd.shape[0]
    dirs = np.zeros((N, n_fibres, 3)); rng = np.random.default_rng(0)
    for k in range(n_fibres):
        v = pd[:, k]; nrm = np.linalg.norm(v, axis=1); ok = nrm > 0
        dirs[ok, k] = v[ok] / nrm[ok, None]
        r = rng.normal(size=(int((~ok).sum()), 3)); dirs[~ok, k] = r / np.linalg.norm(r, axis=1, keepdims=True)
    w = pv / np.maximum(pv.sum(1, keepdims=True), 1e-12); w = np.where(w > frac_min, w, frac_min)
    fr = np.full((N, n_fibres + 3), 0.02); fr[:, 2:2 + n_fibres] = w * 0.9
    fr /= fr.sum(1, keepdims=True)
    return dirs, fr


def refine_peaks(data: np.ndarray, gtab, mask: np.ndarray, pam, n_fibres: int = 3, cfg: Optional[pj.PrismConfig] = None,
                 sphere=None, peak_frac_min: float = 0.10, affine=None, laplace: bool = True):
    """Refine DIPY peaks with the PRISM-JAX MAP; return (PeaksAndMetrics, FixelPosterior | None, PrismFit)."""
    from dipy.data import default_sphere
    sphere = sphere or default_sphere
    bvals_si = np.asarray(gtab.bvals, float) * 1e6; bvecs = np.asarray(gtab.bvecs, float)
    cfg = cfg or default_config(n_fibres)
    init_dirs, init_fracs = init_from_pam(pam, mask, n_fibres)
    fit = pj.fit_prism(data, mask, bvals_si, bvecs, cfg, init_dirs, init_fracs)
    pam_ref = pj.prism_fit_to_pam(fit, sphere, peak_frac_min=peak_frac_min, affine=affine if affine is not None else getattr(pam, "affine", np.eye(4)))
    post = pu.laplace_fixel_posterior(fit, data, bvals_si, bvecs) if laplace else None
    return pam_ref, post, fit


def posterior_pams(post, mask: np.ndarray, n_samples: int, key, sphere=None, affine=None, frac_min: float = 0.10):
    """Yield one PeaksAndMetrics per posterior direction sample (for CV-pruned connectomes)."""
    import jax
    from dipy.data import default_sphere
    from dmipy_jax.validation.disco_tracking import peaks_to_pam
    sphere = sphere or default_sphere; affine = np.eye(4) if affine is None else affine
    S = post.sample_dirs(key, n_samples); idx = np.argwhere(mask); K = post.dirs.shape[1]
    for s in range(n_samples):
        pd = np.zeros(mask.shape + (5, 3)); pv = np.zeros(mask.shape + (5,))
        for n, (i, j, k) in enumerate(idx):
            for f in range(min(K, 5)):
                v = post.wm_fracs[n, f]
                if v >= frac_min:
                    pd[i, j, k, f] = S[s, n, f]; pv[i, j, k, f] = v
        yield peaks_to_pam(pd, pv, mask, sphere, affine)

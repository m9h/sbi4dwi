"""
Per-fixel uncertainty for PRISM-JAX fits (doc 007 §3.4).

PRISM is a point estimate. Here the MAP from ``fit_prism`` is turned into
a per-voxel **Laplace posterior** over the fibre directions (and the
fractions / f_i), by taking the Hessian of the per-voxel data term at the
MAP with the global parameters (σ, D∥, D⊥) held fixed. Directions are
parameterised in the tangent plane of each MAP direction so the
posterior over a unit vector is a 2-D Gaussian there.

What this buys:
  - an angular uncertainty (σ_θ, in degrees, per fibre) and a full 2×2
    tangent covariance per fixel;
  - direction *samples* → many PeaksAndMetrics → posterior-sampled
    tractography → per-edge credible intervals on the connectome;
  - a calibration check on the synthetic benchmark (coverage of the true
    direction by the 90 % cone), the fixel analogue of SBC.

Approximation stated up front: the spatial priors couple voxels, and
the Laplace step here ignores that coupling (block-diagonal Hessian).
That makes the intervals conservative where the prior is informative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from dmipy_jax.validation import prism_jax as pj


def _tangent_basis(n):
    """Two unit vectors orthogonal to n (n: (...,3))."""
    a = jnp.where(jnp.abs(n[..., 0:1]) < 0.9,
                  jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), n.shape),
                  jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), n.shape))
    e1 = jnp.cross(n, a)
    e1 = e1 / jnp.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = jnp.cross(n, e1)
    return e1, e2


@dataclass
class FixelPosterior:
    dirs: np.ndarray          # (N,K,3) MAP directions
    cov: np.ndarray           # (N,K,2,2) tangent-plane covariance
    e1: np.ndarray            # (N,K,3)
    e2: np.ndarray            # (N,K,3)
    sigma_deg: np.ndarray     # (N,K) sqrt(max eigenvalue of cov), degrees
    wm_fracs: np.ndarray      # (N,K)
    mask: np.ndarray

    def sample_dirs(self, key, n_samples: int) -> np.ndarray:
        """(S,N,K,3) unit-vector samples from the tangent Gaussians."""
        # symmetric sqrt via eigh (cholesky fails on near-singular float32 blocks)
        w, V = np.linalg.eigh(0.5 * (self.cov + np.swapaxes(self.cov, -1, -2)))
        L = V * np.sqrt(np.maximum(w, 0.0))[..., None, :]           # (N,K,2,2)
        z = np.asarray(jax.random.normal(key, (n_samples,) + self.cov.shape[:2] + (2,)))
        t = np.einsum("nkij,snkj->snki", L, z)                     # (S,N,K,2)
        v = (self.dirs[None] + t[..., 0:1] * self.e1[None] + t[..., 1:2] * self.e2[None])
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def laplace_fixel_posterior(fit: pj.PrismFit, data: np.ndarray, bvals_si, bvecs,
                            jitter: float = 1e-6, dir_prior_sd_rad: float = 1.0) -> FixelPosterior:
    """Laplace approximation around the MAP, per voxel, over
    [tangent coords (K×2), f_i logit, fraction logits (K+3)]. Only the
    direction block is returned as a covariance; the rest is marginalised
    by inverting the full per-voxel Hessian.

    ``dir_prior_sd_rad``: a weak Gaussian prior (default 1 rad ≈ 57°) on
    every tangent coordinate. A fibre with ~zero fraction has no curvature
    in the data term; without the prior its clipped-eigenvalue inverse
    leaks through the (small) cross-fibre coupling into the *main*
    fibre's covariance. The prior bounds any fixel's σ_θ at ~57° and
    leaves well-determined fixels untouched (their precision is 10³–10⁵×
    larger)."""
    cfg = fit.cfg
    K = cfg.n_fibres
    mask = fit.mask
    y = np.asarray(data, np.float32)[mask]
    b0 = np.asarray(bvals_si) < 50e6
    scale = float(y[:, b0].mean())
    y = jnp.asarray(y / scale)
    bvals = jnp.asarray(np.asarray(bvals_si, np.float32))
    bv = jnp.asarray(np.asarray(bvecs, np.float32))

    dirs0 = jnp.asarray(fit.dirs, jnp.float32)
    e1, e2 = _tangent_basis(dirs0)
    s0 = jnp.asarray(fit.s0 / scale, jnp.float32)
    fi_logit0 = jnp.log(fit.fintra / (1 - fit.fintra)).astype(jnp.float32)
    frac_logit0 = jnp.log(jnp.clip(jnp.asarray(fit.fracs, jnp.float32), 1e-6, 1.0))
    sigma = None if fit.sigma is None else jnp.asarray(fit.sigma / scale)
    d_par = jnp.asarray(fit.d_par); d_perp_g = jnp.asarray(fit.d_perp)
    d_par_ex = jnp.asarray(fit.d_par if fit.d_par_extra is None else fit.d_par_extra)
    odi0 = None if fit.odi is None else jnp.asarray(fit.odi, jnp.float32)

    n_t = 2 * K
    def unpack_local(th, n, e1n, e2n, s0n, odin):
        t = th[:n_t].reshape(K, 2)
        d = n + t[:, 0:1] * e1n + t[:, 1:2] * e2n
        d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
        fi = jax.nn.sigmoid(th[n_t])
        logits = th[n_t + 1:]
        if not cfg.use_restricted:
            logits = logits.at[-1].set(-1e9)
        fr = jax.nn.softmax(logits)
        phys = {"s0": s0n[None], "fracs": fr[None], "dirs": d[None], "fintra": fi[None],
                "d_par": d_par, "d_par_extra": d_par_ex,
                "d_perp": (d_par_ex * (1 - fi))[None] if cfg.tortuosity else d_perp_g,
                "sigma": sigma}
        if cfg.disperse:
            phys["odi"] = odin[None]
            phys["kappa"] = 1.0 / jnp.tan(jnp.pi * phys["odi"] / 2.0)
        return phys

    def pred_voxel(th, n, e1n, e2n, s0n, odin):
        phys = unpack_local(th, n, e1n, e2n, s0n, odin)
        return pj.forward(phys, bvals, bv, cfg)[0]

    def voxel_cov(yv, n, e1n, e2n, s0n, fil, frl, odin):
        th0 = jnp.concatenate([jnp.zeros(n_t), fil[None], frl])
        # Gauss–Newton / Fisher form: H ≈ Jᵀ J / σ², positive semi-definite by
        # construction. The exact Hessian is indefinite at a not-fully-
        # converged MAP and its clipped inverse produces σ_θ of 10³–10⁴°.
        J = jax.jacfwd(pred_voxel)(th0, n, e1n, e2n, s0n, odin)     # (M,P)
        if sigma is not None:
            s2 = sigma ** 2
        else:
            r = yv - pred_voxel(th0, n, e1n, e2n, s0n, odin)
            s2 = jnp.mean(r ** 2)
        H = J.T @ J / s2
        # weak priors: 1 rad on directions, 3 logit-units on f_i / fractions —
        # keeps the marginal direction variance from leaking through
        # coupling to unconstrained fraction logits
        prior_prec = jnp.concatenate([jnp.full((n_t,), 1.0 / dir_prior_sd_rad ** 2),
                                      jnp.full((H.shape[0] - n_t,), 1.0 / 9.0)])
        H = H + jnp.diag(prior_prec) + jitter * jnp.eye(H.shape[0])
        cov_full = jnp.linalg.inv(0.5 * (H + H.T))
        return cov_full[:n_t, :n_t].reshape(K, 2, K, 2)

    odi_arg = odi0 if odi0 is not None else jnp.zeros((dirs0.shape[0], K), jnp.float32)
    cov = jax.vmap(voxel_cov)(y, dirs0, e1, e2, s0, fi_logit0, frac_logit0, odi_arg)
    cov = np.asarray(cov)
    cov_kk = np.stack([cov[:, k, :, k, :] for k in range(K)], axis=1)   # (N,K,2,2)
    w = np.linalg.eigvalsh(cov_kk)
    sigma_deg = np.degrees(np.sqrt(np.maximum(w[..., -1], 0.0)))
    return FixelPosterior(dirs=np.asarray(fit.dirs), cov=cov_kk, e1=np.asarray(e1),
                          e2=np.asarray(e2), sigma_deg=sigma_deg,
                          wm_fracs=np.asarray(fit.wm_fracs), mask=mask)


def cone_coverage(post: FixelPosterior, gt_dirs: np.ndarray, level: float = 0.9,
                  frac_min: float = 0.05) -> dict:
    """Fraction of (voxel, GT fibre) pairs whose true direction lies inside
    the `level` credible cone of the best-matching predicted fibre.
    Mahalanobis distance in the tangent plane vs χ²₂ quantile."""
    from scipy.stats import chi2
    q = chi2.ppf(level, df=2)
    hits, total, mahal = 0, 0, []
    for n in range(gt_dirs.shape[0]):
        for g in gt_dirs[n]:
            if np.linalg.norm(g) == 0:
                continue
            cand = [k for k in range(post.dirs.shape[1]) if post.wm_fracs[n, k] >= frac_min]
            if not cand:
                continue
            k = max(cand, key=lambda k: abs(g @ post.dirs[n, k]))
            gs = g if g @ post.dirs[n, k] >= 0 else -g
            t = np.array([gs @ post.e1[n, k], gs @ post.e2[n, k]])   # tangent offset (small-angle)
            m = t @ np.linalg.solve(post.cov[n, k] + 1e-12 * np.eye(2), t)
            mahal.append(m)
            hits += int(m <= q)
            total += 1
    return {"coverage": hits / total if total else float("nan"), "n": total,
            "level": level, "mahal": np.array(mahal)}


def posterior_connectivity(post: FixelPosterior, mask, rois, affine, sphere, key,
                           n_samples: int = 20, peak_frac_min: float = 0.10,
                           max_angle: float = 45.0, include_map: bool = True) -> dict:
    """Track once per posterior direction sample → distribution over the
    16×16 connectome. Returns mean, sd, 5/95 % per edge and all samples."""
    from dmipy_jax.validation.disco_tracking import peaks_to_pam, track_connectivity
    S = post.sample_dirs(key, n_samples)
    if include_map:
        S = np.concatenate([post.dirs[None], S], axis=0)
    K = post.dirs.shape[1]
    idx = np.argwhere(mask)
    cmats = []
    for s in range(S.shape[0]):
        pdirs = np.zeros(mask.shape + (5, 3)); pvals = np.zeros(mask.shape + (5,))
        for n, (i, j, k) in enumerate(idx):
            for f in range(min(K, 5)):
                v = post.wm_fracs[n, f]
                if v >= peak_frac_min:
                    pdirs[i, j, k, f] = S[s, n, f]; pvals[i, j, k, f] = v
        pam = peaks_to_pam(pdirs, pvals, mask, sphere, affine)
        cmats.append(track_connectivity(pam, mask, rois, affine, max_angle=max_angle)["connectivity"])
    C = np.stack(cmats)
    return {"samples": C, "map": C[0] if include_map else None,
            "mean": C[1:].mean(0) if include_map else C.mean(0),
            "sd": C[1:].std(0) if include_map else C.std(0),
            "lo": np.percentile(C[1:] if include_map else C, 5, axis=0),
            "hi": np.percentile(C[1:] if include_map else C, 95, axis=0)}

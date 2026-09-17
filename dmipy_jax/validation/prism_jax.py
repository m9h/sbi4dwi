"""
PRISM-JAX: differentiable patch-wise analysis-by-synthesis (doc 006 §4).

A faithful JAX re-implementation of PRISM (Abouagour, Shah, Garyfallidis,
arXiv:2604.00250) so it can be benchmarked on the *same* DiSCo tracking
pipeline as doc 004 §21–§24, plus the levers we use to try to exceed it:

  PRISM (faithful)                     PRISM-plus (ours)
  ---------------------------------    ----------------------------------
  fixed D∥=1.7, D⊥=0.4 ×1e-3 mm²/s     learnable global D∥, D⊥ (banded)
  random direction init                warm start from DictionaryMatcher
  point-estimate directions            (Bingham dispersion: TODO)

Forward model (PRISM eq. 1), per voxel:

    S = S0 · ( f_csf·e^{-b D_csf} + f_gm·e^{-b D_gm}
             + Σ_k f_wm,k · [ f_i·stick_k + (1-f_i)·zeppelin_k ]
             + f_res·e^{-b D_res} )

    D_csf = 3.0, D_gm = 0.9, D_res = 0.2 (×1e-3 mm²/s); f_i shared over k.

Per-voxel learnables: S0 (softplus), K+3 fractions (softmax), K unit
directions (ℓ2-normalised), f_i (sigmoid). Global: log σ (Rician NLL
mode); optionally D∥, D⊥ (PRISM-plus).

Priors (PRISM §2.3), all over the masked voxel set:
  - Huber–Laplacian on fractions vs neighbour mean   (λ_sp=0.01, δ=0.05)
  - direction repulsion   Σ_{k<l} f_k f_l (n_k·n_l)²  (λ_rep=0.01)
  - L1 sparsity on minor fibres  Σ_k min(f_k, τ)     (λ=0.02, τ=0.15)
  - directional continuity  f_k(1 − max_nb (n_k·n')²) (λ=0.005)

Optimiser: Rprop (optax.rprop), whole masked volume jointly. DiSCo is
40³ so no slab stitching is needed.

Units: b-values in SI (s/m²) as everywhere in dmipy-JAX; diffusivities
in m²/s (1.7e-3 mm²/s = 1.7e-9 m²/s).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import i0e


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PrismConfig:
    n_fibres: int = 2
    d_csf: float = 3.0e-9
    d_gm: float = 0.9e-9
    d_res: float = 0.2e-9
    d_par: float = 1.7e-9
    d_perp: float = 0.4e-9
    # PRISM-plus: learn global D∥/D⊥ inside a band (sigmoid-parameterised)
    learn_diffusivities: bool = False
    # PRISM-plus: Szafer–Stanisz tortuosity, D⊥ = D∥·(1 − f_i) per voxel
    # (as in the §22 simulator). Overrides d_perp / d_perp_range.
    tortuosity: bool = False
    # PRISM-plus: weak Gaussian prior on learned log D∥ around cfg.d_par,
    # λ·(log D∥ − log d_par)²/sd². Stops D∥ absorbing noise at low SNR
    # (doc 007 §6.6). 0 disables.
    lam_diffusivity_prior: float = 0.0
    diffusivity_prior_sd: float = 0.3      # in log units (~±30 %)
    diffusivity_prior_centre: Optional[float] = None   # None → cfg.d_par
    # PRISM-plus (doc 007 §6.14 lever): give the extra-cellular zeppelin its
    # own axial diffusivity D∥,ex ≤ D∥ (dense packing hinders the axial
    # direction too); with `tortuosity`, D⊥,ex = D∥,ex·(1 − f_i). Keeps the
    # intra stick's D∥ free to stay near the intrinsic value instead of
    # collapsing to absorb extra-cellular hindrance.
    separate_extra_dpar: bool = False
    d_par_extra_init_ratio: float = 0.7
    # PRISM-plus: Watson (axially-symmetric Bingham) dispersion per fibre,
    # learnable ODI ∈ odi_range. Implemented as a Legendre (Funk–Hecke)
    # convolution of the Watson FOD with the stick / zeppelin kernels, so
    # cost is O(n_legendre) forward evaluations instead of a sphere grid.
    disperse: bool = False
    # Ablation (doc 007 §3.1): drop PRISM's restricted isotropic pool.
    use_restricted: bool = True
    use_isotropic: bool = True       # False → CSF and GM balls disabled (WM-only voxels, e.g. substrates)
    tortuosity_scale: bool = False   # learn a global factor s ∈ (0.5, 2) in D⊥,ex = s·D∥,ex·(1 − f_i)
    odi_init: float = 0.1
    odi_range: tuple[float, float] = (0.01, 0.5)
    n_legendre: int = 13                   # even orders 0..24
    n_quad: int = 128
    d_par_range: tuple[float, float] = (0.3e-9, 3.0e-9)
    d_perp_range: tuple[float, float] = (0.02e-9, 1.5e-9)
    loss: str = "mse"                # "mse" | "nll"
    sigma_init: float = 0.05
    lam_spatial: float = 0.01        # PRISM's MSE-scale value; inert under loss='nll' (doc 008 §10.1) — see nll_default_priors()
    huber_delta: float = 0.05
    lam_repulsion: float = 0.01
    lam_sparse: float = 0.02
    tau_sparse: float = 0.15
    lam_continuity: float = 0.005
    connectivity: int = 6            # 6 | 26
    n_iter: int = 300
    learning_rate: float = 1e-2      # Rprop initial step
    fintra_init: float = 0.5
    seed: int = 0

    def __post_init__(self):
        if self.loss not in ("mse", "nll"):
            raise ValueError(f"loss must be 'mse' or 'nll', got {self.loss!r}")
        if self.connectivity not in (6, 26):
            raise ValueError("connectivity must be 6 or 26")


# --------------------------------------------------------------------------- #
# Neighbour table
# --------------------------------------------------------------------------- #

def nll_default_priors(cfg: "PrismConfig") -> "PrismConfig":
    """Prior weights on the Rician-NLL scale (doc 008 §10.1): λ_spatial 300, λ_sparse 10, λ_repulsion 10."""
    from dataclasses import replace
    return replace(cfg, lam_spatial=300.0, lam_continuity=150.0, lam_sparse=10.0, lam_repulsion=10.0)


def build_neighbour_table(mask: np.ndarray, connectivity: int = 6) -> np.ndarray:
    """(N, n_nb) int32 table of masked-voxel indices; -1 where the
    neighbour is outside the mask/volume. Row order = ``np.argwhere(mask)``."""
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    index = -np.ones(mask.shape, dtype=np.int32)
    index[mask] = np.arange(len(coords), dtype=np.int32)

    if connectivity == 6:
        offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    else:
        offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                   for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    nb = -np.ones((len(coords), len(offsets)), dtype=np.int32)
    shape = np.array(mask.shape)
    for c, off in enumerate(offsets):
        q = coords + np.array(off)
        ok = np.all((q >= 0) & (q < shape), axis=1)
        nb[ok, c] = index[q[ok, 0], q[ok, 1], q[ok, 2]]
    return nb


# --------------------------------------------------------------------------- #
# Parameterisation
# --------------------------------------------------------------------------- #

def _inv_softplus(x: float) -> float:
    return float(np.log(np.expm1(x)))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def init_params(n_vox: int, cfg: PrismConfig, key,
                init_dirs: Optional[np.ndarray] = None,
                init_fracs: Optional[np.ndarray] = None) -> dict:
    """PRISM init: uniform fractions, random directions, f_i=0.5, S0=1.
    ``init_dirs`` (N,K,3) / ``init_fracs`` (N,K+3) give PRISM-plus warm
    start (fractions must sum to 1 per voxel)."""
    K = cfg.n_fibres
    if init_dirs is None:
        dirs_raw = jax.random.normal(key, (n_vox, K, 3))
    else:
        dirs_raw = jnp.asarray(init_dirs, dtype=jnp.float32)
    if init_fracs is None:
        frac_logits = jnp.zeros((n_vox, K + 3))
    else:
        frac_logits = jnp.log(jnp.clip(jnp.asarray(init_fracs), 1e-4, 1.0))
    params = {
        "s0_raw": jnp.full((n_vox,), _inv_softplus(1.0)),
        "frac_logits": frac_logits,
        "dirs_raw": dirs_raw,
        "fintra_logit": jnp.full((n_vox,), _logit(cfg.fintra_init)),
    }
    if cfg.loss == "nll":
        params["log_sigma"] = jnp.asarray(np.log(cfg.sigma_init))
    if cfg.disperse:
        lo, hi = cfg.odi_range
        params["odi_logit"] = jnp.full((n_vox, K), _logit((cfg.odi_init - lo) / (hi - lo)))
    if cfg.learn_diffusivities:
        lo, hi = cfg.d_par_range
        params["dpar_logit"] = jnp.asarray(_logit((cfg.d_par - lo) / (hi - lo)))
        lo, hi = cfg.d_perp_range
        params["dperp_logit"] = jnp.asarray(_logit((cfg.d_perp - lo) / (hi - lo)))
    if cfg.separate_extra_dpar:
        params["dpar_extra_frac_logit"] = jnp.asarray(_logit(cfg.d_par_extra_init_ratio))
    if cfg.tortuosity_scale:
        params["tort_scale_logit"] = jnp.asarray(0.0)     # s = 0.5 + 1.5·sigmoid → 1.25 at init
    return params


def unpack(params: dict, cfg: PrismConfig) -> dict:
    """Raw → physical. Fractions layout: [csf, gm, wm_1..wm_K, res]."""
    dirs = params["dirs_raw"]
    dirs = dirs / jnp.maximum(jnp.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)
    logits = params["frac_logits"]
    if not cfg.use_restricted:
        logits = logits.at[:, -1].set(-1e9)
    if not cfg.use_isotropic:
        logits = logits.at[:, 0].set(-1e9).at[:, 1].set(-1e9)
    out = {
        "s0": jax.nn.softplus(params["s0_raw"]),
        "fracs": jax.nn.softmax(logits, axis=-1),
        "dirs": dirs,
        "fintra": jax.nn.sigmoid(params["fintra_logit"]),
    }
    if cfg.learn_diffusivities:
        lo, hi = cfg.d_par_range
        out["d_par"] = lo + (hi - lo) * jax.nn.sigmoid(params["dpar_logit"])
        lo, hi = cfg.d_perp_range
        out["d_perp"] = lo + (hi - lo) * jax.nn.sigmoid(params["dperp_logit"])
    else:
        out["d_par"] = jnp.asarray(cfg.d_par)
        out["d_perp"] = jnp.asarray(cfg.d_perp)
    # extra-cellular axial diffusivity: shared with the stick unless decoupled
    out["d_par_extra"] = out["d_par"]
    if cfg.separate_extra_dpar:
        out["d_par_extra"] = out["d_par"] * jax.nn.sigmoid(params["dpar_extra_frac_logit"])
    if cfg.tortuosity:
        scale = 0.5 + 1.5 * jax.nn.sigmoid(params["tort_scale_logit"]) if cfg.tortuosity_scale else 1.0
        out["tort_scale"] = scale
        out["d_perp"] = scale * out["d_par_extra"] * (1.0 - out["fintra"])       # (N,)
    if cfg.disperse:
        lo, hi = cfg.odi_range
        out["odi"] = lo + (hi - lo) * jax.nn.sigmoid(params["odi_logit"])   # (N,K)
        out["kappa"] = 1.0 / jnp.tan(jnp.pi * out["odi"] / 2.0)
    out["sigma"] = jnp.exp(params["log_sigma"]) if "log_sigma" in params else None
    return out


# --------------------------------------------------------------------------- #
# Watson dispersion via Funk–Hecke / Legendre convolution
# --------------------------------------------------------------------------- #

def _legendre_even(t, n_terms):
    """Even Legendre polynomials P_0, P_2, …, P_{2(n_terms−1)} at t. Returns
    (n_terms, *t.shape). Bonnet recurrence."""
    p_prev = jnp.ones_like(t)          # P_0
    p_curr = t                         # P_1
    out = [p_prev]
    l = 1
    while len(out) < n_terms:
        p_next = ((2 * l + 1) * t * p_curr - l * p_prev) / (l + 1)   # P_{l+1}
        p_prev, p_curr = p_curr, p_next
        l += 1
        if l % 2 == 0:
            out.append(p_curr)
    return jnp.stack(out)


def _gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(x, jnp.float32), jnp.asarray(w, jnp.float32)


def watson_legendre_coeffs(kappa, n_terms, n_quad):
    """a_l for the Watson FOD f(t) = exp(κ t²)/Z on S² (∫_{S²} f = 1),
    f(v·μ) = Σ_l a_l P_l(v·μ), a_l = (2l+1)/2 ∫ f(t) P_l(t) dt.
    kappa: (...); returns (..., n_terms)."""
    x, w = _gauss_legendre(n_quad)
    k = kappa[..., None]
    g = jnp.exp(k * x ** 2 - k)                      # shift for stability
    z = 2.0 * jnp.pi * jnp.sum(w * g, axis=-1, keepdims=True)
    f = g / z
    P = _legendre_even(x, n_terms)                   # (L,Q)
    l = 2.0 * jnp.arange(n_terms)
    return (2 * l + 1) / 2.0 * jnp.einsum("...q,lq->...l", f * w, P)


def _kernel_quad(bvals, d_par, d_perp, x):
    """K(b,t) = exp(−b(D∥ t² + D⊥(1−t²))) on quadrature nodes.
    d_perp scalar → (M,Q); (N,) → (N,M,Q)."""
    b = bvals[:, None]
    if jnp.ndim(d_perp) == 0:
        return jnp.exp(-b * (d_par * x ** 2 + d_perp * (1 - x ** 2)))
    return jnp.exp(-b[None] * (d_par * x ** 2 + d_perp[:, None, None] * (1 - x ** 2)))


@jax.custom_vjp
def _kernel_legendre_core(bvals, d_par, d_perp, P, x, w):
    K = _kernel_quad(bvals, d_par, d_perp, x)
    return 2.0 * jnp.pi * jnp.einsum("...q,lq->...l", K * w, P)


def _klc_fwd(bvals, d_par, d_perp, P, x, w):
    K = _kernel_quad(bvals, d_par, d_perp, x)
    lam = 2.0 * jnp.pi * jnp.einsum("...q,lq->...l", K * w, P)
    return lam, (bvals, d_perp, K, P, x, w)


def _klc_bwd(res, ct):
    # Analytic adjoint: ∂λ_l/∂D∥ = 2π∫(−b t²) K P_l, ∂λ_l/∂D⊥ = 2π∫(−b(1−t²)) K P_l.
    # Written out so the GPU compiler never sees reverse-mode through the
    # quadrature einsum (which hangs XLA on GB10 in jax 0.8 when fused
    # with the rest of the forward model).
    bvals, d_perp, K, P, x, w = res
    A = 2.0 * jnp.pi * jnp.einsum("...l,lq->...q", ct, P) * w * K     # (...,M,Q)
    b = bvals[:, None]
    g_par = -jnp.sum(A * b * x ** 2)
    g_perp_full = -(A * b * (1 - x ** 2))
    if jnp.ndim(d_perp) == 0:
        g_perp = jnp.sum(g_perp_full)
    else:
        g_perp = jnp.sum(g_perp_full, axis=(-2, -1))
    return (jnp.zeros_like(bvals), g_par, g_perp,
            jnp.zeros_like(P), jnp.zeros_like(x), jnp.zeros_like(w))


_kernel_legendre_core.defvjp(_klc_fwd, _klc_bwd)


def kernel_legendre_coeffs(bvals, d_par, d_perp, n_terms, n_quad):
    """λ_l(b) = 2π ∫ K(b,t) P_l(t) dt for the axially symmetric kernel
    K = exp(−b(D∥ t² + D⊥(1−t²))). d_perp scalar → (M, L); (N,) → (N, M, L).
    Gradients w.r.t. D∥ / D⊥ use an analytic adjoint (see _klc_bwd)."""
    x, w = _gauss_legendre(n_quad)
    P = _legendre_even(x, n_terms)                   # (L,Q)
    return _kernel_legendre_core(bvals, jnp.asarray(d_par), jnp.asarray(d_perp), P, x, w)


def dispersed_signal(a, lam, c):
    """Σ_l a_l λ_l P_l(c). a: (N,K,L); lam: (M,L) or (N,M,L); c: (N,K,M).

    Runs the Bonnet recurrence as a ``lax.scan`` over l = 0 … 2(L−1),
    accumulating the even terms. An unrolled recurrence made the GPU
    backward pass compile exponentially in L on GB10 / jax 0.8 (XLA
    re-materialises the chain inside every fusion); a scan keeps it a
    loop in both directions."""
    L = a.shape[-1]
    n_steps = 2 * L - 1
    # even-l coefficients spread onto every l (odd slots zero)
    a2 = jnp.zeros(a.shape[:-1] + (n_steps,), a.dtype).at[..., ::2].set(a)
    lam2 = jnp.zeros(lam.shape[:-1] + (n_steps,), lam.dtype).at[..., ::2].set(lam)
    if lam2.ndim == 2:
        lam2 = lam2[None, None]                       # (1,1,M,S)
    else:
        lam2 = lam2[:, None]                          # (N,1,M,S)

    def step(carry, j):
        p_prev, p_curr, acc = carry                   # P_{j-1}, P_j, Σ_{<j}
        coef = a2[..., j][..., None] * lam2[..., j]   # (N,K,M)
        acc = acc + coef * p_curr
        p_next = ((2 * j + 1) * c * p_curr - j * p_prev) / (j + 1)
        return (p_curr, p_next, acc), None

    init = (jnp.zeros_like(c), jnp.ones_like(c), jnp.zeros_like(c))   # P_{-1}:=0, P_0=1
    (_, _, out), _ = jax.lax.scan(step, init, jnp.arange(n_steps))
    return out


# --------------------------------------------------------------------------- #
# Forward model
# --------------------------------------------------------------------------- #

def forward(phys: dict, bvals, bvecs, cfg: PrismConfig):
    """(N, M) predicted signal for all masked voxels."""
    b = bvals[None, None, :]                                   # (1,1,M)
    gdotn = jnp.einsum("nkj,mj->nkm", phys["dirs"], bvecs)     # (N,K,M)
    c2 = gdotn ** 2
    d_par = phys["d_par"]
    d_perp = phys["d_perp"]
    if jnp.ndim(d_perp) == 1:                                  # per-voxel (tortuosity)
        d_perp = d_perp[:, None, None]
    d_par_ex = phys.get("d_par_extra", d_par)
    e_stick = jnp.exp(-b * d_par * c2)
    e_zep = jnp.exp(-b * (d_par_ex * c2 + d_perp * (1.0 - c2)))
    fi = phys["fintra"][:, None, None]
    if cfg.disperse:
        a = watson_legendre_coeffs(phys["kappa"], cfg.n_legendre, cfg.n_quad)   # (N,K,L)
        lam_stick = kernel_legendre_coeffs(bvals, d_par, jnp.asarray(0.0),
                                           cfg.n_legendre, cfg.n_quad)
        lam_zep = kernel_legendre_coeffs(bvals, d_par_ex, phys["d_perp"],
                                         cfg.n_legendre, cfg.n_quad)
        e_stick = dispersed_signal(a, lam_stick, gdotn)
        e_zep = dispersed_signal(a, lam_zep, gdotn)
    e_wm = fi * e_stick + (1.0 - fi) * e_zep                   # (N,K,M)

    f = phys["fracs"]                                          # (N,K+3)
    K = cfg.n_fibres
    e_csf = jnp.exp(-bvals * cfg.d_csf)[None, :]
    e_gm = jnp.exp(-bvals * cfg.d_gm)[None, :]
    e_res = jnp.exp(-bvals * cfg.d_res)[None, :]
    s = (f[:, 0:1] * e_csf + f[:, 1:2] * e_gm
         + jnp.einsum("nk,nkm->nm", f[:, 2:2 + K], e_wm))
    if cfg.use_restricted:
        s = s + f[:, -1:] * e_res
    return phys["s0"][:, None] * s


# --------------------------------------------------------------------------- #
# Losses and priors
# --------------------------------------------------------------------------- #

def mse_loss(pred, y):
    """PRISM ℒ_MSE = (1/|ℳ|) Σ_x Σ_m (ŷ−y)²: sum over measurements, mean
    over voxels. The prior weights λ are calibrated to this scale — a
    mean over measurements too would shrink the data term ~M× and let
    the sparsity prior delete real fibres."""
    return jnp.mean(jnp.sum((pred - y) ** 2, axis=-1))


def rician_nll(pred, y, sigma):
    """Mean Rician NLL with learnable σ (PRISM eq. for ℒ_NLL, dropping log y)."""
    s2 = sigma ** 2
    z = y * pred / s2
    log_i0 = jnp.log(jnp.maximum(i0e(z), 1e-30)) + jnp.abs(z)
    nll = jnp.log(s2) + (y ** 2 + pred ** 2) / (2.0 * s2) - log_i0
    return jnp.mean(jnp.sum(nll, axis=-1))          # same scale as mse_loss


def _huber(r, delta):
    a = jnp.abs(r)
    return jnp.where(a < delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))


def huber_laplacian(fracs, nb, delta):
    """Mean over voxels of Σ_channels huber(f − mean_nb f)."""
    valid = (nb >= 0)                                          # (N,n_nb)
    nb_safe = jnp.where(valid, nb, 0)
    f_nb = fracs[nb_safe]                                      # (N,n_nb,C)
    w = valid[..., None].astype(fracs.dtype)
    cnt = jnp.maximum(w.sum(axis=1), 1.0)                      # (N,1)
    mean_nb = (f_nb * w).sum(axis=1) / cnt
    has_nb = (valid.sum(axis=1) > 0)[:, None]
    r = jnp.where(has_nb, fracs - mean_nb, 0.0)
    return jnp.mean(jnp.sum(_huber(r, delta), axis=-1))


def direction_repulsion(fracs_wm, dirs):
    """Σ_{k<l} f_k f_l (n_k·n_l)², mean over voxels."""
    K = dirs.shape[1]
    if K < 2:
        return jnp.asarray(0.0)
    tot = 0.0
    for k in range(K):
        for l in range(k + 1, K):
            cos2 = jnp.sum(dirs[:, k] * dirs[:, l], axis=-1) ** 2
            tot = tot + fracs_wm[:, k] * fracs_wm[:, l] * cos2
    return jnp.mean(tot)


def minor_fibre_sparsity(fracs_wm, tau):
    """L1 below threshold τ: Σ_k min(f_k, τ). Flat above τ so genuine
    fibres are untouched; fractions below τ are pushed to zero."""
    return jnp.mean(jnp.sum(jnp.minimum(fracs_wm, tau), axis=-1))


def directional_continuity(fracs_wm, dirs, nb):
    """f_k · (1 − max over neighbours' fibres of (n_k·n')²)."""
    valid = (nb >= 0)
    nb_safe = jnp.where(valid, nb, 0)
    d_nb = dirs[nb_safe]                                       # (N,n_nb,K,3)
    cos2 = jnp.einsum("nkj,nblj->nkbl", dirs, d_nb) ** 2       # (N,K,n_nb,K)
    cos2 = jnp.where(valid[:, None, :, None], cos2, 1.0)       # no penalty w/o nb
    align = cos2.max(axis=(2, 3))                              # (N,K)
    return jnp.mean(jnp.sum(fracs_wm * (1.0 - align), axis=-1))


def total_loss(params, y, bvals, bvecs, nb, cfg: PrismConfig):
    phys = unpack(params, cfg)
    pred = forward(phys, bvals, bvecs, cfg)
    if cfg.loss == "nll":
        data_term = rician_nll(pred, y, phys["sigma"])
    else:
        data_term = mse_loss(pred, y)
    K = cfg.n_fibres
    f_wm = phys["fracs"][:, 2:2 + K]
    d_prior = 0.0
    if cfg.learn_diffusivities and cfg.lam_diffusivity_prior > 0:
        centre = cfg.d_par if cfg.diffusivity_prior_centre is None else cfg.diffusivity_prior_centre
        z = (jnp.log(phys["d_par"]) - jnp.log(centre)) / cfg.diffusivity_prior_sd
        d_prior = cfg.lam_diffusivity_prior * z ** 2
    reg = (d_prior
           + cfg.lam_spatial * huber_laplacian(phys["fracs"], nb, cfg.huber_delta)
           + cfg.lam_repulsion * direction_repulsion(f_wm, phys["dirs"])
           + cfg.lam_sparse * minor_fibre_sparsity(f_wm, cfg.tau_sparse)
           + cfg.lam_continuity * directional_continuity(f_wm, phys["dirs"], nb))
    return data_term + reg


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #

@dataclass
class PrismFit:
    dirs: np.ndarray        # (N,K,3)
    fracs: np.ndarray       # (N,K+3)  [csf, gm, wm_1..K, res]
    fintra: np.ndarray      # (N,)
    s0: np.ndarray          # (N,)
    d_par: float
    d_perp: float
    sigma: Optional[float]
    loss_history: np.ndarray
    mask: np.ndarray
    cfg: PrismConfig
    odi: Optional[np.ndarray] = None      # (N,K), fibre-sorted, if disperse
    d_par_extra: Optional[float] = None   # extra-cellular axial D (== d_par unless decoupled)

    @property
    def wm_fracs(self) -> np.ndarray:
        return self.fracs[:, 2:2 + self.cfg.n_fibres]


def fit_prism(
    data: np.ndarray,
    mask: np.ndarray,
    bvals_si: np.ndarray,
    bvecs: np.ndarray,
    cfg: PrismConfig = PrismConfig(),
    init_dirs: Optional[np.ndarray] = None,
    init_fracs: Optional[np.ndarray] = None,
) -> PrismFit:
    """Joint whole-mask fit. ``data`` is (X,Y,Z,M) raw magnitude; it is
    scaled by the in-mask mean b0 so S0≈1 (PRISM learns S0 per voxel)."""
    mask = np.asarray(mask, dtype=bool)
    y = np.asarray(data, dtype=np.float32)[mask]               # (N,M)
    bvals_si = np.asarray(bvals_si, dtype=np.float32)
    b0 = bvals_si < 1e6 * 50                                   # b < 50 s/mm²
    scale = float(y[:, b0].mean()) if b0.any() else float(y.mean())
    y = jnp.asarray(y / scale)
    bvals = jnp.asarray(bvals_si)
    bvecs = jnp.asarray(np.asarray(bvecs, dtype=np.float32))
    nb = jnp.asarray(build_neighbour_table(mask, cfg.connectivity))

    key = jax.random.PRNGKey(cfg.seed)
    params = init_params(y.shape[0], cfg, key, init_dirs, init_fracs)
    opt = optax.rprop(cfg.learning_rate)
    opt_state = opt.init(params)

    loss_fn = lambda p: total_loss(p, y, bvals, bvecs, nb, cfg)
    grad_fn = jax.value_and_grad(loss_fn)

    def body(_, carry):
        p, s, hist, i = carry
        val, g = grad_fn(p)
        upd, s = opt.update(g, s, p)
        p = optax.apply_updates(p, upd)
        hist = hist.at[i].set(val)
        return p, s, hist, i + 1

    hist0 = jnp.zeros((cfg.n_iter,), dtype=jnp.float32)
    run = jax.jit(lambda p, s: jax.lax.fori_loop(0, cfg.n_iter, body, (p, s, hist0, 0)))
    params, opt_state, hist, _ = run(params, opt_state)
    phys = unpack(params, cfg)

    # sort fibres by descending fraction per voxel
    f_wm = np.asarray(phys["fracs"][:, 2:2 + cfg.n_fibres])
    order = np.argsort(-f_wm, axis=1)
    dirs = np.take_along_axis(np.asarray(phys["dirs"]), order[..., None], axis=1)
    fracs = np.asarray(phys["fracs"]).copy()
    fracs[:, 2:2 + cfg.n_fibres] = np.take_along_axis(f_wm, order, axis=1)
    odi = (np.take_along_axis(np.asarray(phys["odi"]), order, axis=1)
           if cfg.disperse else None)

    return PrismFit(
        dirs=dirs, fracs=fracs,
        fintra=np.asarray(phys["fintra"]), s0=np.asarray(phys["s0"]) * scale,
        d_par=float(phys["d_par"]), d_perp=float(jnp.mean(phys["d_perp"])),
        sigma=None if phys["sigma"] is None else float(phys["sigma"]) * scale,
        loss_history=np.asarray(hist), mask=mask, cfg=cfg, odi=odi,
        d_par_extra=float(phys["d_par_extra"]),
    )


# --------------------------------------------------------------------------- #
# Adapters + metrics
# --------------------------------------------------------------------------- #

def prism_fit_to_pam(fit: PrismFit, sphere, peak_frac_min: float = 0.05,
                     affine: np.ndarray | None = None):
    """PRISM-JAX fit → dipy PeaksAndMetrics for the shared tracker.
    Fibres with fraction < ``peak_frac_min`` are dropped (soft model
    selection in PRISM is via the sparsity prior; this is the hard cut
    the tracker needs)."""
    from dmipy_jax.validation.disco_tracking import peaks_to_pam

    shape = fit.mask.shape
    K = fit.cfg.n_fibres
    peak_dirs = np.zeros(shape + (5, 3), dtype=np.float64)
    peak_values = np.zeros(shape + (5,), dtype=np.float64)
    idx = np.argwhere(fit.mask)
    kk = min(K, 5)
    for n, (i, j, k) in enumerate(idx):
        for s in range(kk):
            v = fit.wm_fracs[n, s]
            if v >= peak_frac_min:
                peak_dirs[i, j, k, s] = fit.dirs[n, s]
                peak_values[i, j, k, s] = v
    return peaks_to_pam(peak_dirs, peak_values, fit.mask, sphere, affine)


def angular_error_best_match(pred_dirs, pred_fracs, gt_dirs, frac_min=0.05):
    """PRISM metric: each GT fibre paired with its closest predicted fibre
    (fraction ≥ frac_min). Returns (mean error deg, recall)."""
    errs, hits, total = [], 0, 0
    for n in range(gt_dirs.shape[0]):
        for g in gt_dirs[n]:
            if np.linalg.norm(g) == 0:
                continue
            total += 1
            cand = [d for d, f in zip(pred_dirs[n], pred_fracs[n]) if f >= frac_min]
            if not cand:
                continue
            cos = max(abs(float(np.dot(g, d))) for d in cand)
            e = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            errs.append(e)
            hits += int(e < 20.0)
    return (float(np.mean(errs)) if errs else float("nan"),
            hits / total if total else float("nan"))

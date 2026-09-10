"""
Model-misspecification benchmark on CATERPillar substrates (doc 007 §3.5,
doc 006 Phase 2).

Every benchmark so far generated signals from the same stick+zeppelin
model that PRISM fits (in-model), or from DiSCo's straight cylinders.
Here the signal comes from Monte Carlo diffusion in *realistic* axon
geometry — overlapping-sphere axons from CATERPillar with tortuosity and
beading — so the forward model is wrong in ways the estimators are not
told about. Question: which of PRISM / PRISM-plus / MSMT-CSD degrades
gracefully?

Pipeline
  CATERPillar (one population, growth ≈ z) → sphere list A (µm)
  → B = A rotated by the crossing angle about x, both wrapped into the
    periodic voxel box → union SDF (periodic via jnp.mod)
  → intra walkers (inside spheres) and extra walkers (outside spheres),
    Brownian + elastic reflection, positions accumulated over the two
    PGSE lobes → phase for any (G, u) is γ G u·(R₁ − R₂) dt
  → S = f_i S_intra + (1 − f_i) S_extra, f_i measured from the geometry
  → replicated into a small volume with independent Rician noise.

Units: CATERPillar is in µm; everything here is converted to SI.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

GAMMA = 2.6751525e8  # rad/s/T


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _rotation_about_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def bundle_axis(df) -> np.ndarray:
    """Mean principal axis of the axons (unit vector, +z hemisphere)."""
    axes = []
    for _, g in df.groupby("id"):
        c = g[["x", "y", "z"]].values
        if len(c) < 5:
            continue
        c = c - c.mean(0)
        w, v = np.linalg.eigh(c.T @ c)
        a = v[:, -1]
        axes.append(a if a[2] >= 0 else -a)
    a = np.mean(axes, axis=0)
    return a / np.linalg.norm(a)


@dataclass
class Substrate:
    centers_m: np.ndarray      # (S,3) in metres, inside [0, L)^3
    radii_m: np.ndarray        # (S,)
    box_m: float
    axes: np.ndarray           # (n_bundles,3) GT fibre directions
    f_intra: float             # measured intra-cellular volume fraction
    meta: dict


def make_crossing_substrate(df, angle_deg: float, box_um: float, seed: int = 0) -> Substrate:
    """Two copies of one CATERPillar bundle crossing at ``angle_deg``
    (second copy rotated about x, offset by a random shift, wrapped into
    the periodic box). angle 0 → single bundle."""
    rng = np.random.default_rng(seed)
    a = bundle_axis(df)
    c = df[["x", "y", "z"]].values.astype(np.float64)
    r = df["radius"].values.astype(np.float64)
    L = float(box_um)
    cs, rs, axes = [c], [r], [a]
    if angle_deg > 0:
        R = _rotation_about_x(np.radians(angle_deg))
        cb = (c - L / 2) @ R.T + L / 2 + rng.uniform(0, L, 3)
        cs.append(cb); rs.append(r); axes.append(R @ a)
    C = np.mod(np.concatenate(cs), L)
    Rr = np.concatenate(rs)
    sub = Substrate(C * 1e-6, Rr * 1e-6, L * 1e-6, np.array(axes), float("nan"),
                    {"angle_deg": angle_deg, "n_spheres": len(C)})
    sub.f_intra = measure_intra_fraction(sub, seed=seed)
    return sub


def periodic_sdf(sub: Substrate):
    """Union-of-spheres SDF with the query point wrapped into the box, so
    walkers see a periodic tiling. Uses a 3×3×3 neighbour-image check on
    the sphere set nearest the boundaries via wrapping the *difference*."""
    C = jnp.asarray(sub.centers_m); Rr = jnp.asarray(sub.radii_m); L = sub.box_m

    def sdf(p):
        d = p - C
        d = d - L * jnp.round(d / L)          # minimum-image difference
        return jnp.min(jnp.linalg.norm(d, axis=1) - Rr)
    return sdf


def measure_intra_fraction(sub: Substrate, n: int = 200_000, seed: int = 0) -> float:
    sdf = periodic_sdf(sub)
    p = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3)) * sub.box_m
    inside = jax.vmap(sdf)(p) <= 0
    return float(jnp.mean(inside))


def _init_fn(sub: Substrate, intra: bool):
    sdf = periodic_sdf(sub)
    L = sub.box_m
    n_try = 8

    def init(key, n):
        # rejection-sample uniform points in the box, keep those in the
        # requested compartment; oversample and take the first n (loops
        # are avoided so this stays jit-able; with f≈0.3–0.7 a factor 8
        # oversample is always enough)
        p = jax.random.uniform(key, (n_try * n, 3)) * L
        ok = jax.vmap(sdf)(p) <= 0
        ok = ok if intra else ~ok
        idx = jnp.argsort(~ok)[:n]           # ok=True first
        return p[idx]
    return init


# --------------------------------------------------------------------------- #
# Monte Carlo PGSE
# --------------------------------------------------------------------------- #

def _reflect(pos, old, sdf):
    """Elastic reflection about the surface between old (valid) and pos."""
    g = jax.grad(sdf)(pos)
    g = g / (jnp.linalg.norm(g) + 1e-12)
    d = sdf(pos)
    return pos - 2.0 * d * g


def pgse_lobe_sums(sub: Substrate, intra: bool, D: float, dt: float, n_particles: int,
                   delta: float, Delta: float, key):
    """Run walkers for T = Δ + δ and return R1, R2: the position sums
    (× dt) over the first and second gradient lobes, (n,3) each, in metres·s.
    Phase for gradient amplitude G along unit u is then γ G u·(R1 − R2)."""
    sdf = periodic_sdf(sub)
    init = _init_fn(sub, intra)
    n_steps = int(round((Delta + delta) / dt))
    n1 = int(round(delta / dt)); s2 = int(round(Delta / dt))
    lobe = jnp.zeros(n_steps).at[:n1].set(1.0).at[s2:s2 + n1].set(-1.0)   # +1, 0, −1

    def valid(p):
        return (sdf(p) <= 0) if intra else (sdf(p) > 0)

    def step(carry, w):
        pos, acc, k = carry
        k, sk = jax.random.split(k)
        prop = pos + jnp.sqrt(2 * D * dt) * jax.random.normal(sk, pos.shape)

        def fix(p, o):
            bad = ~valid(p)
            p2 = jax.lax.cond(bad, lambda: _reflect(p, o, sdf), lambda: p)
            # if the reflection still lands in the wrong compartment (corner
            # cases), reject the move
            return jax.lax.cond(valid(p2), lambda: p2, lambda: o)
        new = jax.vmap(fix)(prop, pos)
        acc = acc + w * new * dt
        return (new, acc, k), None

    k0, k1 = jax.random.split(key)
    pos0 = init(k0, n_particles)
    (pos, acc, _), _ = jax.lax.scan(step, (pos0, jnp.zeros_like(pos0), k1), lobe)
    return acc          # (n,3) = R1 − R2 already (lobe weights ±1)


def signals_from_lobe_sums(R, bvals_si, bvecs, delta, Delta):
    """|mean exp(iφ)| per measurement. G from b = γ²G²δ²(Δ − δ/3)."""
    G = jnp.sqrt(jnp.asarray(bvals_si) / (GAMMA ** 2 * delta ** 2 * (Delta - delta / 3.0)))
    phase = GAMMA * G[:, None] * (jnp.asarray(bvecs) @ R.T)          # (M,n)
    return jnp.abs(jnp.mean(jnp.exp(1j * phase), axis=1))


def simulate_substrate_signal(sub: Substrate, bvals_si, bvecs, *, D_intra=2.0e-9, D_extra=2.0e-9,
                              delta=17.74e-3, Delta=35.78e-3, dt=2e-5, n_particles=4000, seed=0):
    """Returns (S_total, S_intra, S_extra) on the given scheme."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    R_in = pgse_lobe_sums(sub, True, D_intra, dt, n_particles, delta, Delta, k1)
    R_ex = pgse_lobe_sums(sub, False, D_extra, dt, n_particles, delta, Delta, k2)
    S_in = signals_from_lobe_sums(R_in, bvals_si, bvecs, delta, Delta)
    S_ex = signals_from_lobe_sums(R_ex, bvals_si, bvecs, delta, Delta)
    f = sub.f_intra
    return np.asarray(f * S_in + (1 - f) * S_ex), np.asarray(S_in), np.asarray(S_ex)


# --------------------------------------------------------------------------- #
# Benchmark assembly
# --------------------------------------------------------------------------- #

def caterpillar_bundle(icvf=0.5, box_um=10.0, tortuous=0, beading=0.0, seed=0, threads=8):
    from dmipy_jax.validation.caterpillar import CATERPillarOracle
    o = CATERPillarOracle()
    cfg = o.get_default_config()
    cfg.update({"vox_sizes": [box_um], "axons_without_myelin_icvf": icvf,
                "glial_pop1_icvf_soma": 0.0, "glial_pop1_icvf_branches": 0.0,
                "tortuous": tortuous, "beading_variation": beading,
                "beading_variation_std": beading / 2 if beading > 0 else 0.0,
                "nbr_threads": threads})
    return o.generate(cfg)


def build_volume(signal: np.ndarray, shape=(8, 8, 1), snr: float = 30.0, seed: int = 0):
    """Replicate one substrate signal into a small volume with independent
    Rician noise per voxel (S0 = 1)."""
    rng = np.random.default_rng(seed)
    N = int(np.prod(shape)); s = 1.0 / snr
    clean = np.tile(signal[None], (N, 1))
    noisy = np.sqrt((clean + rng.normal(0, s, clean.shape)) ** 2 + rng.normal(0, s, clean.shape) ** 2)
    data = np.zeros(shape + (len(signal),), np.float32); data[np.ones(shape, bool)] = noisy
    return data, np.ones(shape, bool)

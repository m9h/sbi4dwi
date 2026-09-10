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
  CATERPillar (one population, c₂=⟨cos²ψ⟩≈0.98 so growth ≈ z) → sphere
    list A (µm; the duplicated type-2 rows are dropped)
  → sheet crossing: A fills x < L/2, B = A rotated by the crossing angle
    about x fills x ≥ L/2 (CATERPillar's two-population "sheet" option;
    dense bundles cannot interpenetrate) → union SDF
  → box [0, L]³: intra walkers periodic along z only (axons span the
    box; a wrapped x/y image would land in another axon — the substrate
    is not periodic-consistent, wrapping all axes made the intra space
    percolate, measured 2026-09-09), extra walkers periodic in all axes
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
    centers_m: np.ndarray      # (S,3) in metres
    radii_m: np.ndarray        # (S,)
    box_m: float
    axes: np.ndarray           # (n_bundles,3) GT fibre directions
    f_intra: float             # measured intra-cellular volume fraction
    meta: dict
    axon_ids: np.ndarray = None   # (S,) which axon each sphere belongs to


def make_crossing_substrate(df, angle_deg: float, box_um: float, seed: int = 0) -> Substrate:
    """Two copies of one CATERPillar bundle crossing at ``angle_deg`` in a
    *sheet* configuration (CATERPillar's own two-population option): the
    first copy fills x < L/2, the second — rotated about x through the box
    centre — fills x ≥ L/2. Dense bundles cannot interpenetrate (dropping
    overlapping spheres removed almost all of the second bundle), so the
    voxel is split instead; both bundles keep the single-bundle packing.
    angle 0 → single bundle in the whole box."""
    a = bundle_axis(df)
    c = df[["x", "y", "z"]].values.astype(np.float64)
    r = df["radius"].values.astype(np.float64)
    ids = df["id"].values.astype(np.int32)
    L = float(box_um)
    if angle_deg <= 0:
        C, Rr, I, axes, n_b = c, r, ids, [a], 0
    else:
        R = _rotation_about_x(np.radians(angle_deg))
        cb = (c - L / 2) @ R.T + L / 2
        ka = c[:, 0] < L / 2
        kb = cb[:, 0] >= L / 2
        C = np.concatenate([c[ka], cb[kb]]); Rr = np.concatenate([r[ka], r[kb]])
        I = np.concatenate([ids[ka], ids[kb] + ids.max() + 1])
        axes = [a, R @ a]; n_b = int(kb.sum())
    sub = Substrate(C * 1e-6, Rr * 1e-6, L * 1e-6, np.array(axes), float("nan"),
                    {"angle_deg": angle_deg, "n_spheres": len(C), "n_spheres_b": n_b},
                    axon_ids=I)
    sub.f_intra = measure_intra_fraction(sub, seed=seed)
    return sub


def union_sdf(sub: Substrate, periodic=(False, False, False)):
    """Union-of-spheres SDF. Axes flagged periodic use the minimum-image
    difference (period = box length) so walkers wrapping across that face
    see the substrate's image."""
    C = jnp.asarray(sub.centers_m); Rr = jnp.asarray(sub.radii_m); L = sub.box_m
    per = jnp.asarray(periodic, dtype=jnp.float32)

    def sdf(p):
        d = p - C
        d = d - per * L * jnp.round(d / L)
        return jnp.min(jnp.linalg.norm(d, axis=1) - Rr)
    return sdf


def nearest_axon_fn(sub: Substrate, periodic=(False, False, False)):
    """Axon id of the sphere with the smallest signed distance to p."""
    C = jnp.asarray(sub.centers_m); Rr = jnp.asarray(sub.radii_m); L = sub.box_m
    ids = jnp.asarray(sub.axon_ids); per = jnp.asarray(periodic, dtype=jnp.float32)

    def axon_of(p):
        d = p - C
        d = d - per * L * jnp.round(d / L)
        return ids[jnp.argmin(jnp.linalg.norm(d, axis=1) - Rr)]
    return axon_of


def confine_box(p, L, periodic):
    """Wrap periodic axes with mod, mirror-reflect the others."""
    per = jnp.asarray(periodic, dtype=bool)
    wrapped = jnp.mod(p, L)
    refl = jnp.where(p < 0, -p, p)
    refl = jnp.where(refl > L, 2 * L - refl, refl)
    return jnp.where(per, wrapped, refl)


# Intra walkers: periodic in y and z (axons span the box along z; the
# rotated sheet's axons lie in the y–z plane), reflecting in x. A wrapped
# image that lands in a different axon is rejected by the axon-id check,
# so periodicity can only help a walker continue along its *own* axon.
# Extra walkers: periodic in all three (an image landing inside a sphere
# is rejected).
PERIODIC_INTRA = (False, True, True)
PERIODIC_EXTRA = (True, True, True)


def measure_intra_fraction(sub: Substrate, n: int = 200_000, seed: int = 0) -> float:
    sdf = union_sdf(sub, PERIODIC_INTRA)
    p = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3)) * sub.box_m
    # chunked: vmap over 200k × n_spheres would materialise tens of GB
    inside = jax.lax.map(lambda q: jax.vmap(sdf)(q) <= 0, p.reshape(-1, 2000, 3))
    return float(jnp.mean(inside))


def _init_fn(sub: Substrate, intra: bool):
    sdf = union_sdf(sub, PERIODIC_INTRA if intra else PERIODIC_EXTRA)
    L = sub.box_m
    f = sub.f_intra if intra else 1.0 - sub.f_intra
    n_try = int(np.ceil(3.0 / max(f, 1e-3)))     # oversample so ≥ n valid w.h.p.

    def init(key, n):
        # rejection-sample uniform points in the box, keep those in the
        # requested compartment (oversampled; loops avoided so this stays
        # jit-able)
        p = jax.random.uniform(key, (n_try * n, 3)) * L
        ok = jax.lax.map(lambda q: jax.vmap(sdf)(q) <= 0,
                         p.reshape(-1, 1000, 3)).reshape(-1) if (n_try * n) % 1000 == 0 \
            else jax.vmap(sdf)(p) <= 0
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
    periodic = PERIODIC_INTRA if intra else PERIODIC_EXTRA
    sdf = union_sdf(sub, periodic)
    init = _init_fn(sub, intra)
    L = sub.box_m
    n_steps = int(round((Delta + delta) / dt))
    n1 = int(round(delta / dt)); s2 = int(round(Delta / dt))
    lobe = jnp.zeros(n_steps).at[:n1].set(1.0).at[s2:s2 + n1].set(-1.0)   # +1, 0, −1

    axon_of = nearest_axon_fn(sub, periodic)
    step_sd = jnp.sqrt(2 * D * dt)

    def valid(p):
        return (sdf(p) <= 0) if intra else (sdf(p) > 0)

    def step(carry, w):
        pos, upos, acc, k = carry
        k, sk = jax.random.split(k)
        prop = pos + step_sd * jax.random.normal(sk, pos.shape)
        prop = jax.vmap(lambda p: confine_box(p, L, periodic))(prop)

        def fix(p, o):
            bad = ~valid(p)
            p2 = jax.lax.cond(bad, lambda: _reflect(p, o, sdf), lambda: p)
            # Reject the move if the reflection lands in the wrong compartment,
            # in a *different axon* (the union SDF's nearest surface may belong
            # to a neighbouring axon → tunnelling), or implausibly far away
            # (reflection through a thin wall into the far side).
            dm = p2 - o
            dm = dm - jnp.asarray(periodic, jnp.float32) * L * jnp.round(dm / L)
            ok = valid(p2) & (jnp.linalg.norm(dm) < 4.0 * step_sd)
            if intra:
                ok = ok & (axon_of(p2) == axon_of(o))
            return jax.lax.cond(ok, lambda: p2, lambda: o)
        new = jax.vmap(fix)(prop, pos)
        # unwrapped displacement for the phase (wrapping is a bookkeeping
        # device; the spin physically moved by the minimum-image step)
        step_vec = new - pos
        step_vec = step_vec - jnp.asarray(periodic, jnp.float32) * L * jnp.round(step_vec / L)
        upos = upos + step_vec
        acc = acc + w * upos * dt
        return (new, upos, acc, k), None

    k0, k1 = jax.random.split(key)
    pos0 = init(k0, n_particles)
    (pos, _, acc, _), _ = jax.lax.scan(step, (pos0, pos0, jnp.zeros_like(pos0), k1), lobe)
    return acc          # (n,3) = R1 − R2 already (lobe weights ±1)


def signals_from_lobe_sums(R, bvals_si, bvecs, delta, Delta):
    """|mean exp(iφ)| per measurement. G from b = γ²G²δ²(Δ − δ/3)."""
    G = jnp.sqrt(jnp.asarray(bvals_si) / (GAMMA ** 2 * delta ** 2 * (Delta - delta / 3.0)))
    phase = GAMMA * G[:, None] * (jnp.asarray(bvecs) @ R.T)          # (M,n)
    return jnp.abs(jnp.mean(jnp.exp(1j * phase), axis=1))


def simulate_substrate_signal(sub: Substrate, bvals_si, bvecs, *, D_intra=2.0e-9, D_extra=2.0e-9,
                              delta=6e-3, Delta=12e-3, dt=1e-5, n_particles=4000, seed=0):
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

def caterpillar_bundle(icvf=0.5, box_um=20.0, tortuous=0, beading=0.0, c2=0.98, seed=0, threads=8):
    """One CATERPillar axon population. ``c2`` = ⟨cos²ψ⟩ of growth direction
    vs z (CATERPillar's default 0.5 is an almost isotropic fODF). The
    duplicated type-2 sphere rows are dropped."""
    from dmipy_jax.validation.caterpillar import CATERPillarOracle
    o = CATERPillarOracle()
    cfg = o.get_default_config()
    cfg.update({"vox_sizes": [box_um], "axons_without_myelin_icvf": icvf,
                "glial_pop1_icvf_soma": 0.0, "glial_pop1_icvf_branches": 0.0,
                "tortuous": tortuous, "beading_variation": beading,
                "beading_variation_std": beading / 2 if beading > 0 else 0.0,
                "c2": c2, "nbr_threads": threads})
    df = o.generate(cfg)
    return df[df["type"] == 0].reset_index(drop=True)


def build_volume(signal: np.ndarray, shape=(8, 8, 1), snr: float = 30.0, seed: int = 0):
    """Replicate one substrate signal into a small volume with independent
    Rician noise per voxel (S0 = 1)."""
    rng = np.random.default_rng(seed)
    N = int(np.prod(shape)); s = 1.0 / snr
    clean = np.tile(signal[None], (N, 1))
    noisy = np.sqrt((clean + rng.normal(0, s, clean.shape)) ** 2 + rng.normal(0, s, clean.shape) ** 2)
    data = np.zeros(shape + (len(signal),), np.float32); data[np.ones(shape, bool)] = noisy
    return data, np.ones(shape, bool)

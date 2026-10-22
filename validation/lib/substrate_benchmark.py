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
    """Axons as overlapping spheres (inner radius = axoplasm; outer radius =
    myelin sheath, equal to the inner one when unmyelinated) plus optional
    glial cells as their own sphere sets. Water compartments:

      intra   inside an inner axon sphere
      myelin  between inner and outer axon spheres — no water (no signal)
      glia    inside a glial sphere
      extra   outside every outer axon sphere and every glial sphere

    ``fractions`` are volume fractions of the *box*; ``f_intra`` keeps its
    old meaning (intra / all water) so earlier callers are unchanged."""
    centers_m: np.ndarray      # (S,3) in metres
    radii_m: np.ndarray        # (S,) inner radii
    box_m: float
    axes: np.ndarray           # (n_bundles,3) GT fibre directions
    f_intra: float             # intra / (intra + glia + extra)
    meta: dict
    axon_ids: np.ndarray = None       # (S,) which axon each sphere belongs to
    outer_radii_m: np.ndarray = None  # (S,) == radii_m when unmyelinated
    glia_centers_m: np.ndarray = None # (G,3)
    glia_radii_m: np.ndarray = None   # (G,)
    glia_ids: np.ndarray = None       # (G,)
    fractions: dict = None            # {"intra","myelin","glia","extra"} of the box

    axon_index: np.ndarray = None     # (S,) compact 0..A-1 index per sphere
    axon_start_m: np.ndarray = None   # (A,3) first sphere centre of each axon
    axon_chord_m: np.ndarray = None   # (A,3) last − first sphere centre: the axon's own period

    @property
    def has_glia(self):
        return len(self.glia_radii_m) > 0

    @property
    def f_axon_water(self):
        """The CHUV ground-truth definition: intra / (intra + extra), myelin
        excluded and glia *not* counted as extra-axonal water."""
        fr = self.fractions
        return fr["intra"] / (fr["intra"] + fr["extra"])

    def __post_init__(self):
        if self.outer_radii_m is None:
            self.outer_radii_m = self.radii_m
        if self.glia_centers_m is None:
            self.glia_centers_m = np.zeros((0, 3)); self.glia_radii_m = np.zeros((0,))
            self.glia_ids = np.zeros((0,), np.int32)
        if self.axon_ids is not None and self.axon_index is None:
            uniq, self.axon_index = np.unique(self.axon_ids, return_inverse=True)
            self.axon_index = self.axon_index.astype(np.int32)
            A = len(uniq)
            self.axon_start_m = np.zeros((A, 3)); self.axon_chord_m = np.zeros((A, 3))
            for a in range(A):
                w = np.flatnonzero(self.axon_index == a)      # CSV order runs along the axon
                self.axon_start_m[a] = self.centers_m[w[0]]
                self.axon_chord_m[a] = self.centers_m[w[-1]] - self.centers_m[w[0]]


# Intra-axonal walkers are periodic along *their own axon*: an axon that
# leaves the box at one face continues at its own start, translated by the
# axon's chord (last − first sphere centre). Box-periodic wrapping is wrong
# for anything but a perfectly z-aligned axon — a tilted (c₂ 0.98 ≈ 8°) or
# tortuous axon's image sits at another (x, y), the wrapped move is rejected,
# and the box faces become hard walls: at Δ = 55 ms the diffusion length
# (≈ 19 µm) equals the box, intra axial D collapsed to ≈ 1.1 µm²/ms for an
# intrinsic 2.5 (measured 2026-10-22; the §10.2 tables of doc 008 carry a
# milder version of the same effect at 10 µm / Δ 36 ms).
AXON_IMAGE_PAD_M = 1.5e-6     # spheres this close to an axon end get an image copy at the other end


def intra_augmented(sub: Substrate):
    """Inner axon spheres plus image copies of each axon's first/last
    ``AXON_IMAGE_PAD_M`` of spheres translated by ∓ the chord, so a walker
    near an end sees the continuation. Returns (C, R, axon_index)."""
    C, R, ai = sub.centers_m, sub.radii_m, sub.axon_index
    T = sub.axon_chord_m[ai]; S0 = sub.axon_start_m[ai]
    Tn = np.linalg.norm(T, axis=1)
    ok = Tn > 1e-9
    s = np.einsum("ij,ij->i", C - S0, T) / np.where(ok, Tn, 1.0) ** 2 * np.where(ok, Tn, 0.0)   # metres along chord
    near_start = ok & (s < AXON_IMAGE_PAD_M)
    near_end = ok & (Tn - s < AXON_IMAGE_PAD_M)
    Ca = np.concatenate([C, C[near_start] + T[near_start], C[near_end] - T[near_end]])
    Ra = np.concatenate([R, R[near_start], R[near_end]])
    ia = np.concatenate([ai, ai[near_start], ai[near_end]])
    return Ca, Ra, ia


def _split_sheet(c, L, R, angle_deg):
    """Keep x < L/2 of the original and x ≥ L/2 of the rotated copy."""
    if angle_deg <= 0:
        return np.ones(len(c), bool), np.zeros(len(c), bool), c
    cb = (c - L / 2) @ R.T + L / 2
    return c[:, 0] < L / 2, cb[:, 0] >= L / 2, cb


def make_crossing_substrate(df, angle_deg: float, box_um: float, seed: int = 0) -> Substrate:
    """Two copies of one CATERPillar bundle crossing at ``angle_deg`` in a
    *sheet* configuration (CATERPillar's own two-population option): the
    first copy fills x < L/2, the second — rotated about x through the box
    centre — fills x ≥ L/2. Dense bundles cannot interpenetrate (dropping
    overlapping spheres removed almost all of the second bundle), so the
    voxel is split instead; both bundles keep the single-bundle packing.
    angle 0 → single bundle in the whole box. Glial cells (``type == 1``
    rows, if any) are split the same way."""
    typ = df["type"].values if "type" in df else np.zeros(len(df), int)
    ax_df = df[typ == 0]; gl_df = df[typ == 1]
    a = bundle_axis(ax_df)
    c = ax_df[["x", "y", "z"]].values.astype(np.float64)
    r = ax_df["radius"].values.astype(np.float64)
    ro = ax_df["outer_radius"].values.astype(np.float64) if "outer_radius" in ax_df else r
    ids = ax_df["id"].values.astype(np.int32)
    L = float(box_um)
    R = _rotation_about_x(np.radians(angle_deg)) if angle_deg > 0 else np.eye(3)
    ka, kb, cb = _split_sheet(c, L, R, angle_deg)
    C = np.concatenate([c[ka], cb[kb]]); Rr = np.concatenate([r[ka], r[kb]]); Ro = np.concatenate([ro[ka], ro[kb]])
    I = np.concatenate([ids[ka], ids[kb] + ids.max() + 1])
    axes = [a] if angle_deg <= 0 else [a, R @ a]
    gc = gl_df[["x", "y", "z"]].values.astype(np.float64); gr = gl_df["radius"].values.astype(np.float64)
    gi = gl_df["id"].values.astype(np.int32)
    # CATERPillar grows glial processes past the voxel faces without clipping;
    # under the box-periodic extra-cellular SDF those spheres would wrap back
    # inside as spurious obstacles (measured 2026-10-22: extra space shrank
    # from 0.58 to 0.18 of the box and 45 % of extra walkers froze). Keep the
    # spheres whose centre is inside the box.
    inside = np.all((gc >= 0) & (gc <= L), axis=1)
    gc, gr, gi = gc[inside], gr[inside], gi[inside]
    gka, gkb, gcb = _split_sheet(gc, L, R, angle_deg)
    GC = np.concatenate([gc[gka], gcb[gkb]]); GR = np.concatenate([gr[gka], gr[gkb]])
    GI = np.concatenate([gi[gka], gi[gkb] + (gi.max() + 1 if len(gi) else 0)])
    sub = Substrate(C * 1e-6, Rr * 1e-6, L * 1e-6, np.array(axes), float("nan"),
                    {"angle_deg": angle_deg, "n_spheres": len(C), "n_spheres_b": int(kb.sum()),
                     "n_glia_spheres": len(GC)},
                    axon_ids=I, outer_radii_m=Ro * 1e-6, glia_centers_m=GC * 1e-6, glia_radii_m=GR * 1e-6, glia_ids=GI)
    sub.fractions = measure_fractions(sub, seed=seed)
    water = sub.fractions["intra"] + sub.fractions["glia"] + sub.fractions["extra"]
    sub.f_intra = sub.fractions["intra"] / water
    return sub


GRID_THRESHOLD = 3000     # brute force below this many spheres


def _build_grid(C, Rr, L, n_cells, wrap: bool):
    """Uniform grid of n_cells³ over [0,L]³. Cell (i,j,k) lists every sphere
    whose bounding box, grown by one cell, touches the cell. ``wrap``: cells
    wrap on all axes (box-periodic sets; an image sphere of a non-periodic
    axis is harmless — the min-image distance is only applied on periodic
    axes, so it sits ~L away and never wins the min). Otherwise indices are
    clipped, so spheres (or query points) just outside the box fall into the
    edge cells. The one-cell growth makes the list valid for a query point
    anywhere in the cell *and* for the walker's reflection point, which is
    within a step of it. Lists are padded with a sentinel index that points
    at a far-away dummy sphere."""
    h = L / n_cells
    lo = np.floor((C - Rr[:, None] - h) / h).astype(int)
    hi = np.floor((C + Rr[:, None] + h) / h).astype(int)
    if not wrap:
        lo = np.clip(lo, 0, n_cells - 1); hi = np.clip(hi, 0, n_cells - 1)
    span = hi - lo + 1
    cells, members = [], []
    for s in np.argsort(span.prod(1)):                 # python loop over spheres only
        ii, jj, kk = (np.arange(lo[s, a], hi[s, a] + 1) % n_cells for a in range(3))
        g = np.stack(np.meshgrid(ii, jj, kk, indexing="ij"), -1).reshape(-1, 3)
        q = np.unique((g[:, 0] * n_cells + g[:, 1]) * n_cells + g[:, 2])
        cells.append(q); members.append(np.full(len(q), s, np.int32))
    cells = np.concatenate(cells); members = np.concatenate(members)
    order = np.argsort(cells, kind="stable")
    cells, members = cells[order], members[order]
    counts = np.bincount(cells, minlength=n_cells ** 3)
    m = int(counts.max())
    idx = np.full((n_cells ** 3, m), len(Rr), np.int32)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(len(cells)) - starts[cells]
    idx[cells, pos] = members
    return h, idx


def _spheres_sdf(C, Rr, L, periodic, n_cells: int | None = None):
    """Union-of-spheres SDF (negative inside). Axes flagged periodic use the
    minimum-image difference (period = box length) so walkers wrapping
    across that face see the substrate's image. Empty set → +inf. Above
    ``GRID_THRESHOLD`` spheres a cell list restricts the min to the
    neighbourhood (glial processes are ~10⁵ tiny spheres per voxel); the few
    large spheres (somata) stay brute force so they do not bloat the lists."""
    C = np.asarray(C, np.float64); Rr = np.asarray(Rr, np.float64)
    per = jnp.asarray(periodic, dtype=jnp.float32)
    wrap = bool(np.any(periodic))
    if len(Rr) == 0:
        return lambda p: jnp.inf

    def brute(Cj, Rj):
        def sdf(p):
            d = p - Cj
            d = d - per * L * jnp.round(d / L)
            return jnp.min(jnp.linalg.norm(d, axis=1) - Rj)
        return sdf

    if n_cells is None:
        n_cells = 0 if len(Rr) < GRID_THRESHOLD else int(np.clip(round(L / max(4 * float(np.median(Rr)), L / 32)), 2, 32))
    if n_cells == 0:
        return brute(jnp.asarray(C, jnp.float32), jnp.asarray(Rr, jnp.float32))
    h = L / n_cells
    big = Rr > 2 * h
    parts = []
    if big.any():
        parts.append(brute(jnp.asarray(C[big], jnp.float32), jnp.asarray(Rr[big], jnp.float32)))
    Cs, Rs = C[~big], Rr[~big]
    if len(Rs):
        _, idx = _build_grid(Cs, Rs, L, n_cells, wrap)
        Cpad = jnp.concatenate([jnp.asarray(Cs, jnp.float32), jnp.full((1, 3), 1e3 * L, jnp.float32)])  # sentinel
        Rpad = jnp.concatenate([jnp.asarray(Rs, jnp.float32), jnp.zeros((1,), jnp.float32)])
        idx = jnp.asarray(idx)

        def grid_sdf(p):
            if wrap:
                ijk = jnp.floor(jnp.mod(p, L) / h).astype(jnp.int32) % n_cells
            else:
                ijk = jnp.clip(jnp.floor(p / h).astype(jnp.int32), 0, n_cells - 1)
            nb = idx[(ijk[0] * n_cells + ijk[1]) * n_cells + ijk[2]]
            d = p - Cpad[nb]
            d = d - per * L * jnp.round(d / L)
            return jnp.min(jnp.linalg.norm(d, axis=1) - Rpad[nb])
        parts.append(grid_sdf)
    if len(parts) == 1:
        return parts[0]
    return lambda p: jnp.minimum(parts[0](p), parts[1](p))


def union_sdf(sub: Substrate, periodic=(False, False, False)):
    """SDF of the intra-axonal space (inner spheres)."""
    return _spheres_sdf(sub.centers_m, sub.radii_m, sub.box_m, periodic)


def compartment_sdf(sub: Substrate, comp: str, periodic=(False, False, False)):
    """Signed distance that is ≤ 0 exactly where a walker of compartment
    ``comp`` ("intra" | "glia" | "extra") is allowed to be. For "extra" that
    is the negated SDF of (outer axon spheres ∪ glial spheres); negating
    flips both the gradient and the distance, so the reflection formula is
    unchanged."""
    L = sub.box_m
    if comp == "intra":
        Ca, Ra, _ = intra_augmented(sub)
        return _spheres_sdf(Ca, Ra, L, periodic)
    if comp == "glia":
        return _spheres_sdf(sub.glia_centers_m, sub.glia_radii_m, L, periodic)
    if comp == "extra":
        # obstacles are the spheres whose centre is inside the box: an axon's
        # end spheres poking out of a face are *not* the image of what enters at
        # the opposite face (different (x, y)), so wrapping them in would add
        # spurious obstacles
        ina = np.all((sub.centers_m >= 0) & (sub.centers_m <= L), axis=1)
        s_ax = _spheres_sdf(sub.centers_m[ina], sub.outer_radii_m[ina], L, periodic)
        s_gl = _spheres_sdf(sub.glia_centers_m, sub.glia_radii_m, L, periodic)
        return lambda p: -jnp.minimum(s_ax(p), s_gl(p))
    raise ValueError(comp)


def _nearest_id_fn(C, Rr, ids, L, periodic):
    C = jnp.asarray(C, jnp.float32); Rr = jnp.asarray(Rr, jnp.float32)
    ids = jnp.asarray(ids); per = jnp.asarray(periodic, dtype=jnp.float32)

    def id_of(p):
        d = p - C
        d = d - per * L * jnp.round(d / L)
        return ids[jnp.argmin(jnp.linalg.norm(d, axis=1) - Rr)]
    return id_of


def nearest_axon_fn(sub: Substrate, periodic=(False, False, False)):
    """Compact axon index of the sphere (incl. end images) with the smallest
    signed distance to p."""
    Ca, Ra, ia = intra_augmented(sub)
    return _nearest_id_fn(Ca, Ra, ia, sub.box_m, periodic)


def nearest_glia_fn(sub: Substrate, periodic=(False, False, False)):
    return _nearest_id_fn(sub.glia_centers_m, sub.glia_radii_m, sub.glia_ids, sub.box_m, periodic)


def confine_box(p, L, periodic):
    """Wrap periodic axes with mod, mirror-reflect the others."""
    per = jnp.asarray(periodic, dtype=bool)
    wrapped = jnp.mod(p, L)
    refl = jnp.where(p < 0, -p, p)
    refl = jnp.where(refl > L, 2 * L - refl, refl)
    return jnp.where(per, wrapped, refl)


# Intra walkers: no box periodicity — each axon is periodic along its own
# chord (see ``intra_augmented``), so the SDF is plain Euclidean.
# Extra walkers: periodic in all three (an image landing inside a sphere
# is rejected).
PERIODIC_INTRA = (False, False, False)
PERIODIC_EXTRA = (True, True, True)
# Glial cells are compact (soma + processes) and never span the box, so a
# wrapped image is always a different cell → reflecting walls only.
PERIODIC_GLIA = (False, False, False)
PERIODIC = {"intra": PERIODIC_INTRA, "extra": PERIODIC_EXTRA, "glia": PERIODIC_GLIA}


def _comp_name(comp):
    if isinstance(comp, (bool, np.bool_)):
        return "intra" if comp else "extra"
    return comp


def measure_fractions(sub: Substrate, n: int = 200_000, seed: int = 0) -> dict:
    """Volume fractions of the box: intra, myelin, glia, extra."""
    p = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3)) * sub.box_m
    s_in = union_sdf(sub, PERIODIC_INTRA)
    s_out = _spheres_sdf(sub.centers_m, sub.outer_radii_m, sub.box_m, PERIODIC_INTRA)
    s_gl = compartment_sdf(sub, "glia", PERIODIC_GLIA)

    def chunk(q):
        a = jax.vmap(s_in)(q) <= 0; o = jax.vmap(s_out)(q) <= 0; g = jax.vmap(s_gl)(q) <= 0
        return jnp.stack([a, o & ~a, g & ~o, ~o & ~g], -1)
    # chunked: vmap over 200k × n_spheres would materialise tens of GB
    m = jax.lax.map(chunk, p.reshape(-1, 2000, 3)).reshape(-1, 4).mean(0)
    return dict(zip(["intra", "myelin", "glia", "extra"], map(float, np.asarray(m))))


def measure_intra_fraction(sub: Substrate, n: int = 200_000, seed: int = 0) -> float:
    """intra / all water (glia counted as water, myelin not)."""
    fr = measure_fractions(sub, n, seed)
    return fr["intra"] / (fr["intra"] + fr["glia"] + fr["extra"])


def _init_fn(sub: Substrate, comp):
    comp = _comp_name(comp)
    sdf = compartment_sdf(sub, comp, PERIODIC[comp])
    L = sub.box_m
    fr = sub.fractions or {"intra": sub.f_intra, "extra": 1 - sub.f_intra, "glia": 0.0}
    f = fr[comp]
    n_try = int(np.ceil(3.0 / max(f, 1e-3)))     # oversample so ≥ n valid w.h.p.

    def init(key, n):
        # rejection-sample uniform points in the box, keep those in the
        # requested compartment (oversampled; loops avoided so this stays
        # jit-able)
        p = jax.random.uniform(key, (n_try * n, 3)) * L
        ok = jax.lax.map(lambda q: jax.vmap(sdf)(q) <= 0,
                         p.reshape(-1, 1000, 3)).reshape(-1) if (n_try * n) % 1000 == 0 \
            else jax.vmap(sdf)(p) <= 0
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


def pgse_lobe_sums(sub: Substrate, comp, D: float, dt: float, n_particles: int,
                   delta: float, Delta: float, key):
    """Run walkers of compartment ``comp`` ("intra" | "extra" | "glia"; a bool
    is read as intra/extra) for T = Δ + δ and return R1 − R2: the position
    sums (× dt) over the first minus the second gradient lobe, (n,3), in
    metres·s. Phase for gradient amplitude G along unit u is then
    γ G u·(R1 − R2)."""
    comp = _comp_name(comp)
    periodic = PERIODIC[comp]
    sdf = compartment_sdf(sub, comp, periodic)
    init = _init_fn(sub, comp)
    L = sub.box_m
    n_steps = int(round((Delta + delta) / dt))
    n1 = int(round(delta / dt)); s2 = int(round(Delta / dt))
    lobe = jnp.zeros(n_steps).at[:n1].set(1.0).at[s2:s2 + n1].set(-1.0)   # +1, 0, −1

    # confined walkers must stay in the cell they started in (the union SDF's
    # nearest surface may belong to a neighbour → tunnelling otherwise)
    cell_of = {"intra": nearest_axon_fn(sub, periodic), "glia": nearest_glia_fn(sub, periodic),
               "extra": None}[comp]
    step_sd = jnp.sqrt(2 * D * dt)
    if comp == "intra":
        T = jnp.asarray(sub.axon_chord_m, jnp.float32); S0 = jnp.asarray(sub.axon_start_m, jnp.float32)
        T2 = jnp.maximum(jnp.sum(T ** 2, axis=1), 1e-30)

        def chord_wrap(p):
            """Translate a walker that passed either end of its axon by the
            axon's chord, back into the real chain (a no-op for a zero chord)."""
            a = cell_of(p)
            s = jnp.dot(p - S0[a], T[a]) / T2[a]
            return p - jnp.where(s > 1.0, 1.0, jnp.where(s < 0.0, -1.0, 0.0)) * T[a]

    def valid(p):
        return sdf(p) <= 0

    def step(carry, w):
        pos, upos, acc, k = carry
        k, sk = jax.random.split(k)
        prop = pos + step_sd * jax.random.normal(sk, pos.shape)
        if comp != "intra":          # intra walkers are confined by their axon, not the box
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
            if cell_of is not None:
                ok = ok & (cell_of(p2) == cell_of(o))
            return jax.lax.cond(ok, lambda: p2, lambda: o)
        new = jax.vmap(fix)(prop, pos)
        # unwrapped displacement for the phase (wrapping is a bookkeeping
        # device; the spin physically moved by the minimum-image step)
        step_vec = new - pos
        step_vec = step_vec - jnp.asarray(periodic, jnp.float32) * L * jnp.round(step_vec / L)
        upos = upos + step_vec
        acc = acc + w * upos * dt
        if comp == "intra":
            new = jax.vmap(chord_wrap)(new)        # physical step already counted above
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
                              D_glia=None, delta=6e-3, Delta=12e-3, dt=1e-5, n_particles=4000, seed=0,
                              return_glia=False):
    """Returns (S_total, S_intra, S_extra[, S_glia]) on the given scheme.
    Compartments are weighted by their water fractions (myelin carries no
    water); glial walkers use ``D_glia`` (default = ``D_intra``)."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    R_in = pgse_lobe_sums(sub, "intra", D_intra, dt, n_particles, delta, Delta, k1)
    R_ex = pgse_lobe_sums(sub, "extra", D_extra, dt, n_particles, delta, Delta, k2)
    S_in = signals_from_lobe_sums(R_in, bvals_si, bvecs, delta, Delta)
    S_ex = signals_from_lobe_sums(R_ex, bvals_si, bvecs, delta, Delta)
    fr = sub.fractions or {"intra": sub.f_intra, "extra": 1 - sub.f_intra, "glia": 0.0}
    water = fr["intra"] + fr["extra"] + fr["glia"]
    S = fr["intra"] / water * S_in + fr["extra"] / water * S_ex
    S_gl = np.full(len(bvals_si), np.nan)
    if fr["glia"] > 0 and sub.has_glia:
        R_gl = pgse_lobe_sums(sub, "glia", D_intra if D_glia is None else D_glia, dt, n_particles, delta, Delta, k3)
        S_gl = signals_from_lobe_sums(R_gl, bvals_si, bvecs, delta, Delta)
        S = S + fr["glia"] / water * S_gl
    out = (np.asarray(S), np.asarray(S_in), np.asarray(S_ex))
    return out + (np.asarray(S_gl),) if return_glia else out


# --------------------------------------------------------------------------- #
# Benchmark assembly
# --------------------------------------------------------------------------- #

def caterpillar_bundle(icvf=0.5, box_um=20.0, tortuous=0, beading=0.0, c2=0.98, seed=0, threads=8, backend="legacy"):
    """One CATERPillar axon population. ``c2`` = ⟨cos²ψ⟩ of growth direction
    vs z (CATERPillar's default 0.5 is an almost isotropic fODF). The
    duplicated type-2 sphere rows are dropped."""
    if backend == "cli":
        # 2026-09 CMake/JSON build: un-severed axons, PascalCase config, space-delimited CSV
        from dmipy_jax.simulation.substrates.caterpillar import CATERPillarCLI
        cli = CATERPillarCLI()
        cfg = cli.default_config(box_um=box_um, icvf=icvf, c2=c2, tortuous=bool(tortuous), beading=beading,
                                 beading_std=beading / 2 if beading > 0 else 0.0, threads=threads)
        return cli.generate(cfg, cells="axon")
    from dmipy_jax.simulation.substrates.caterpillar import CATERPillarOracle
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

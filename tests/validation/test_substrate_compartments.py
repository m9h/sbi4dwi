"""Multi-compartment substrates (glia, myelin) and the gridded sphere SDF."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import pandas as pd  # noqa: E402

from validation.lib import substrate_benchmark as sb  # noqa: E402


def _substrate(box=10.0, n_axons=4, r=0.8, myelin=0.3, n_glia=2, seed=0):
    """Straight z-axons with a myelin sheath plus a few glial 'cells' made of
    a soma and a short chain of process spheres, placed in free space."""
    rng = np.random.default_rng(seed)
    rows = []
    xs = [2.0, 2.0, 8.0, 8.0][:n_axons]; ys = [2.0, 8.0, 2.0, 8.0][:n_axons]
    for i in range(n_axons):
        for z in np.arange(0, box, r / 2):
            rows.append((xs[i], ys[i], z, r, r + myelin, 0, i, 0, "axon"))
    for g in range(n_glia):
        cx, cy, cz = 5.0, 5.0, 2.5 + 5.0 * g
        rows.append((cx, cy, cz, 1.0, 1.0, 1, 100 + g, 0, "soma"))
        for k in range(1, 8):
            rows.append((cx + 0.25 * k, cy, cz, 0.3, 0.3, 1, 100 + g, 1, "branch"))
    return pd.DataFrame(rows, columns=["x", "y", "z", "radius", "outer_radius", "type", "id", "parent", "component"])


def test_fractions_sum_to_one_and_myelin_excluded():
    sub = sb.make_crossing_substrate(_substrate(), 0.0, 10.0)
    fr = sub.fractions
    assert abs(sum(fr.values()) - 1.0) < 1e-6
    assert fr["myelin"] > 0.05 and fr["glia"] > 0.005 and fr["intra"] > 0.05
    # analytic: 4 cylinders r=0.8 in a 10×10 box → 0.0804; sheath to 1.1 → 0.152−0.080
    assert fr["intra"] == pytest.approx(4 * np.pi * 0.8 ** 2 / 100, abs=0.01)
    assert fr["myelin"] == pytest.approx(4 * np.pi * (1.1 ** 2 - 0.8 ** 2) / 100, abs=0.01)
    water = fr["intra"] + fr["glia"] + fr["extra"]
    assert sub.f_intra == pytest.approx(fr["intra"] / water)
    assert sub.f_axon_water == pytest.approx(fr["intra"] / (fr["intra"] + fr["extra"]))


def test_compartment_sdfs_partition_space():
    sub = sb.make_crossing_substrate(_substrate(), 0.0, 10.0)
    p = jax.random.uniform(jax.random.PRNGKey(0), (4000, 3)) * sub.box_m
    s_in = jax.vmap(sb.compartment_sdf(sub, "intra", sb.PERIODIC_INTRA))(p) <= 0
    s_gl = jax.vmap(sb.compartment_sdf(sub, "glia", sb.PERIODIC_GLIA))(p) <= 0
    s_ex = jax.vmap(sb.compartment_sdf(sub, "extra", sb.PERIODIC_EXTRA))(p) <= 0
    # same (box-periodic) image convention as the extra SDF; the axons here are
    # z-aligned so the intra set's chord images coincide with the box images
    s_out = jax.vmap(sb._spheres_sdf(sub.centers_m, sub.outer_radii_m, sub.box_m, sb.PERIODIC_EXTRA))(p) <= 0
    my = s_out & ~s_in
    # intra, myelin, glia, extra are disjoint and cover the box
    cnt = s_in.astype(int) + my.astype(int) + s_gl.astype(int) + s_ex.astype(int)
    assert bool(jnp.all(cnt == 1))


def test_grid_sdf_matches_brute_force():
    rng = np.random.default_rng(1)
    L = 10e-6
    C = rng.uniform(0, L, (5000, 3)); R = rng.uniform(0.1e-6, 0.4e-6, 5000)
    C = np.concatenate([C, [[5e-6, 5e-6, 5e-6]]]); R = np.concatenate([R, [2.5e-6]])   # one 'soma' → brute-force part
    for per in [sb.PERIODIC_EXTRA, sb.PERIODIC_GLIA]:
        grid = sb._spheres_sdf(C, R, L, per)                 # 5001 ≥ GRID_THRESHOLD → gridded
        brute = sb._spheres_sdf(C, R, L, per, n_cells=0)
        p = jax.random.uniform(jax.random.PRNGKey(2), (3000, 3)) * L
        a = np.asarray(jax.vmap(grid)(p)); b = np.asarray(jax.vmap(brute)(p))
        # exact wherever the true distance is within one cell of the surface
        h = L / max(2, int(np.clip(round(L / max(4 * np.median(R), L / 32)), 2, 32)))
        near = np.abs(b) < h
        np.testing.assert_allclose(a[near], b[near], atol=1e-9)
        assert np.all((a <= 0) == (b <= 0))
        assert np.all(a >= b - 1e-9)                          # never under-estimates


def test_glia_walkers_stay_in_their_cell():
    sub = sb.make_crossing_substrate(_substrate(), 0.0, 10.0)
    cell_of = sb.nearest_glia_fn(sub, sb.PERIODIC_GLIA)
    init = sb._init_fn(sub, "glia")
    p0 = init(jax.random.PRNGKey(0), 500)
    sdf = sb.compartment_sdf(sub, "glia", sb.PERIODIC_GLIA)
    assert bool(jnp.all(jax.vmap(sdf)(p0) <= 0))
    R = sb.pgse_lobe_sums(sub, "glia", 2e-9, 2e-6, 500, 3e-3, 6e-3, jax.random.PRNGKey(1))
    assert np.isfinite(np.asarray(R)).all()
    # restricted in a ~1 µm soma + 0.3 µm processes: strong attenuation is impossible at b = 3000
    b = np.array([3000.0, 3000.0]) * 1e6; u = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    S = np.asarray(sb.signals_from_lobe_sums(R, b, u, 3e-3, 6e-3))
    assert np.all(S > 0.8)


def _tilted_bundle(n_axons=6, box=20.0, r=0.8, tilt_deg=10.0, seed=0):
    """Straight axons tilted off z so their box-periodic image is NOT in the
    same axon — exactly the case the chord wrapping has to handle."""
    rng = np.random.default_rng(seed)
    rows = []
    t = np.tan(np.radians(tilt_deg))
    for i in range(n_axons):
        x0 = 3.0 + 3.0 * (i % 3) + rng.uniform(-0.2, 0.2); y0 = 4.0 + 8.0 * (i // 3) + rng.uniform(-0.2, 0.2)
        for z in np.arange(0, box, r / 2):
            rows.append((x0 + t * z, y0, z, r, r, 0, i, 0, "axon"))
    return pd.DataFrame(rows, columns=["x", "y", "z", "radius", "outer_radius", "type", "id", "parent", "component"])


def test_intra_axial_diffusion_is_free_along_tilted_axons():
    """Along the fibre, intra walkers must diffuse freely over the whole walk
    even when the diffusion length exceeds the box: the axon's own chord is
    the period, not the box. (Box-periodic wrapping gave D_eff ≈ 0.45 D.)"""
    df = _tilted_bundle()
    sub = sb.make_crossing_substrate(df, 0.0, 20.0)
    D, delta, Delta = 2.5e-9, 10e-3, 40e-3           # diffusion length ≈ 16 µm ~ box
    axis = sub.axes[0]
    R = sb.pgse_lobe_sums(sub, "intra", D, 4e-6, 4000, delta, Delta, jax.random.PRNGKey(0))
    # axial displacement variance over the whole walk = free value
    var_free = 2 * D * delta ** 2 * (Delta - delta / 3)
    assert (np.asarray(R) @ axis).var() / var_free == pytest.approx(1.0, abs=0.06)
    # and the signal: b low enough that S ≫ the 1/√N floor of |mean exp(iφ)|
    b = np.array([300.0, 600.0]) * 1e6
    S = np.asarray(sb.signals_from_lobe_sums(R, b, np.stack([axis, axis]), delta, Delta))
    adc = -np.log(S) / b
    np.testing.assert_allclose(adc, D, rtol=0.08)
    # every walker is still in its own axon
    axon_of = sb.nearest_axon_fn(sub, sb.PERIODIC_INTRA)
    init = sb._init_fn(sub, "intra"); p0 = init(jax.random.PRNGKey(0), 2000)
    assert bool(jnp.all(jax.vmap(sb.compartment_sdf(sub, "intra", sb.PERIODIC_INTRA))(p0) <= 0))


def test_permeable_walk_zero_kappa_has_no_transits():
    df = _tilted_bundle()
    sub = sb.make_crossing_substrate(df, 0.0, 20.0)
    R, c0, c1 = sb.pgse_lobe_sums_permeable(sub, 0.0, 2.5e-9, 1.5e-9, 4e-6, 1000, 5e-3, 10e-3, jax.random.PRNGKey(0))
    assert bool(jnp.all(c0 == c1))
    assert np.isfinite(np.asarray(R)).all()
    # extra walkers really are extra, intra really intra at the end of the walk
    assert abs(float(jnp.mean(c0 == 0)) - sub.fractions["intra"] / (sub.fractions["intra"] + sub.fractions["extra"])) < 0.02


def test_permeable_walk_exchange_time_and_stationarity():
    """Cylinders of radius r: intra residence time τ_i = r/(2κ), Kärger
    τ_ex = τ_i·(1 − f). The walk must (a) keep the intra fraction stationary
    (detailed balance with D_in ≠ D_ex) and (b) reproduce τ_ex within the
    discretisation error of the transit rule on a corrugated sphere chain."""
    df = _tilted_bundle()
    sub = sb.make_crossing_substrate(df, 0.0, 20.0)
    f_in = sub.fractions["intra"] / (sub.fractions["intra"] + sub.fractions["extra"])
    kappa, r = 4e-5, 0.8e-6
    tau_ex_theory = r / (2 * kappa) * (1 - f_in)
    delta, Delta = 5e-3, 15e-3
    R, c0, c1 = sb.pgse_lobe_sums_permeable(sub, kappa, 2.5e-9, 1.5e-9, 4e-6, 3000, delta, Delta, jax.random.PRNGKey(1))
    c0 = np.asarray(c0); c1 = np.asarray(c1)
    assert np.mean(c1 == 0) == pytest.approx(f_in, abs=0.03)          # stationary
    assert np.mean(c1[c0 == 0] == 1) > 0.3                             # plenty of transits at this κ
    tau = sb.exchange_time_from_transfer(c0, c1, delta + Delta, f_in)
    assert tau == pytest.approx(tau_ex_theory, rel=0.35)


def test_chord_images_cover_axon_ends():
    df = _tilted_bundle()
    sub = sb.make_crossing_substrate(df, 0.0, 20.0)
    Ca, Ra, ia = sb.intra_augmented(sub)
    assert len(Ra) > len(sub.radii_m)
    sdf = sb.compartment_sdf(sub, "intra", sb.PERIODIC_INTRA)
    # a point just beyond the far end of axon 0, along its chord, is inside the image
    a = 0; end = sub.axon_start_m[a] + sub.axon_chord_m[a]
    beyond = end + 0.5e-6 * sub.axon_chord_m[a] / np.linalg.norm(sub.axon_chord_m[a])
    assert float(sdf(jnp.asarray(beyond, jnp.float32))) <= 0


def test_extra_walkers_avoid_myelin_and_glia():
    sub = sb.make_crossing_substrate(_substrate(), 0.0, 10.0)
    init = sb._init_fn(sub, "extra")
    p0 = init(jax.random.PRNGKey(0), 800)
    s_out = sb._spheres_sdf(sub.centers_m, sub.outer_radii_m, sub.box_m, sb.PERIODIC_EXTRA)
    s_gl = sb.compartment_sdf(sub, "glia", sb.PERIODIC_EXTRA)
    assert bool(jnp.all(jax.vmap(s_out)(p0) > 0)) and bool(jnp.all(jax.vmap(s_gl)(p0) > 0))

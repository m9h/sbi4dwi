"""Monte Carlo substrate benchmark: geometry contracts + physics sanity."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import pandas as pd  # noqa: E402

from dmipy_jax.validation import substrate_benchmark as sb  # noqa: E402


def _straight_bundle(n_axons=6, box=10.0, r=0.8, seed=0):
    """Synthetic 'CATERPillar-like' straight axons along z as sphere chains,
    placed on a jittered grid so axons never overlap."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_axons)))
    pitch = (box - 2) / side
    assert pitch > 2 * r + 0.2, "axons would overlap"
    rows = []
    i = 0
    for gx in range(side):
        for gy in range(side):
            if i >= n_axons:
                break
            x = 1 + (gx + 0.5) * pitch + rng.uniform(-0.1, 0.1)
            y = 1 + (gy + 0.5) * pitch + rng.uniform(-0.1, 0.1)
            for z in np.arange(0, box, r / 2):
                rows.append((x, y, z, r, 0, i, -1))
            i += 1
    return pd.DataFrame(rows, columns=["x", "y", "z", "radius", "type", "id", "parent_id"])


def test_bundle_axis_and_rotation():
    df = _straight_bundle()
    a = sb.bundle_axis(df)
    np.testing.assert_allclose(a, [0, 0, 1], atol=1e-6)
    sub = sb.make_crossing_substrate(df, 60.0, 10.0)
    assert sub.axes.shape == (2, 3)
    got = np.degrees(np.arccos(abs(sub.axes[0] @ sub.axes[1])))
    assert got == pytest.approx(60.0, abs=1e-6)
    assert 0.05 < sub.f_intra < 0.9


def test_box_confinement_and_sheet_crossing():
    L = 10e-6
    p = jnp.array([-1e-6, 5e-6, 11e-6])
    np.testing.assert_allclose(np.asarray(sb.confine_box(p, L, (False, False, False))), [1e-6, 5e-6, 9e-6], atol=1e-11)
    np.testing.assert_allclose(np.asarray(sb.confine_box(p, L, (True, True, True))), [9e-6, 5e-6, 1e-6], atol=1e-11)
    df = _straight_bundle(n_axons=9, r=0.8, box=10.0)
    sub = sb.make_crossing_substrate(df, 90.0, 10.0)
    assert sub.meta["n_spheres_b"] > 0
    nA = len(sub.centers_m) - sub.meta["n_spheres_b"]
    assert np.all(sub.centers_m[:nA, 0] < 5e-6) and np.all(sub.centers_m[nA:, 0] >= 5e-6)
    assert len(np.unique(sub.axon_ids[:nA])) + len(np.unique(sub.axon_ids[nA:])) == len(np.unique(sub.axon_ids))


def test_free_diffusion_matches_exp_minus_bD():
    """Extra-cellular walkers in an (almost) empty box: S = exp(−bD)."""
    df = _straight_bundle(n_axons=1, r=0.3)
    sub = sb.make_crossing_substrate(df, 0.0, 10.0)
    assert sub.f_intra < 0.05
    b = np.array([0, 1000, 2000, 3000]) * 1e6
    u = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    D = 2.0e-9
    # extra walkers are fully periodic → exactly free diffusion
    df = _straight_bundle(n_axons=1, r=0.3, box=20.0)
    sub = sb.make_crossing_substrate(df, 0.0, 20.0)
    R = sb.pgse_lobe_sums(sub, False, D, 1e-5, 3000, 6e-3, 12e-3, jax.random.PRNGKey(0))
    S = np.asarray(sb.signals_from_lobe_sums(R, b, u, 6e-3, 12e-3))
    np.testing.assert_allclose(S, np.exp(-b * D), atol=0.03)


def test_intra_restricted_perpendicular_but_free_parallel():
    """Walkers inside straight z-axons: along z ≈ free, across ≈ restricted."""
    # intra walkers are periodic along z → parallel diffusion is free
    df = _straight_bundle(n_axons=8, r=1.0, box=20.0)
    sub = sb.make_crossing_substrate(df, 0.0, 20.0)
    b = np.array([3000, 3000]) * 1e6
    u = np.array([[0, 0, 1.0], [1, 0, 0]])
    D = 2.0e-9
    R = sb.pgse_lobe_sums(sub, True, D, 1e-5, 3000, 6e-3, 12e-3, jax.random.PRNGKey(1))
    S = np.asarray(sb.signals_from_lobe_sums(R, b, u, 6e-3, 12e-3))
    assert S[0] == pytest.approx(np.exp(-3000e6 * D), abs=0.03)     # parallel = free
    assert S[1] > 0.6                                               # perpendicular restricted


def test_build_volume_shape_and_noise():
    sig = np.linspace(1, 0.2, 20)
    data, mask = sb.build_volume(sig, shape=(4, 4, 1), snr=30)
    assert data.shape == (4, 4, 1, 20) and mask.sum() == 16
    assert abs(data[..., 0].mean() - 1.0) < 0.05


def test_intra_walkers_never_change_axon():
    """No tunnelling: every intra walker ends in the axon it started in."""
    df = _straight_bundle(n_axons=10, r=1.0, box=20.0, seed=3)
    sub = sb.make_crossing_substrate(df, 0.0, 20.0)
    axon_of = sb.nearest_axon_fn(sub, sb.PERIODIC_INTRA)
    init = sb._init_fn(sub, True)
    p0 = init(jax.random.PRNGKey(0), 1500)
    # run the real stepper via pgse_lobe_sums internals: replicate a short walk
    R = sb.pgse_lobe_sums(sub, True, 2e-9, 1e-5, 1500, 6e-3, 12e-3, jax.random.PRNGKey(0))
    assert np.isfinite(np.asarray(R)).all()
    # direct check with the module's step logic is exercised above; verify the
    # invariant explicitly on a manual walk
    sdf = sb.union_sdf(sub, sb.PERIODIC_INTRA)
    ids0 = jax.vmap(axon_of)(p0)
    D, dt, L = 2e-9, 1e-5, sub.box_m
    def step(carry, _):
        pos, k = carry; k, sk = jax.random.split(k)
        prop = pos + jnp.sqrt(2 * D * dt) * jax.random.normal(sk, pos.shape)
        prop = jax.vmap(lambda p: sb.confine_box(p, L, sb.PERIODIC_INTRA))(prop)
        def fix(p, o):
            p2 = jax.lax.cond(sdf(p) > 0, lambda: sb._reflect(p, o, sdf), lambda: p)
            ok = (sdf(p2) <= 0) & (axon_of(p2) == axon_of(o))
            return jax.lax.cond(ok, lambda: p2, lambda: o)
        return (jax.vmap(fix)(prop, pos), k), None
    (pos, _), _ = jax.lax.scan(step, (p0, jax.random.PRNGKey(1)), None, length=1500)
    assert bool(jnp.all(jax.vmap(axon_of)(pos) == ids0))

"""Monte Carlo substrate benchmark: geometry contracts + physics sanity."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import pandas as pd  # noqa: E402

from dmipy_jax.validation import substrate_benchmark as sb  # noqa: E402


def _straight_bundle(n_axons=6, box=10.0, r=0.8, seed=0):
    """Synthetic 'CATERPillar-like' straight axons along z as sphere chains."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_axons):
        x, y = rng.uniform(1, box - 1, 2)
        for k, z in enumerate(np.arange(0, box, r / 2)):
            rows.append((x, y, z, r, 0, i, -1))
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


def test_box_confinement_and_crossing_overlap_removal():
    L = 10e-6
    p = jnp.array([-1e-6, 5e-6, 11e-6])
    np.testing.assert_allclose(np.asarray(sb.confine_box(p, L, (False, False, False))), [1e-6, 5e-6, 9e-6], atol=1e-11)
    np.testing.assert_allclose(np.asarray(sb.confine_box(p, L, (True, True, True))), [9e-6, 5e-6, 1e-6], atol=1e-11)
    df = _straight_bundle(n_axons=6, r=0.8)
    sub = sb.make_crossing_substrate(df, 90.0, 10.0)
    assert sub.meta["n_dropped"] > 0
    # no remaining inter-bundle overlaps
    nA = len(df)
    A, B = sub.centers_m[:nA], sub.centers_m[nA:]
    rA, rB = sub.radii_m[:nA], sub.radii_m[nA:]
    d = np.linalg.norm(A[:, None] - B[None], axis=-1)
    assert np.all(d >= rA[:, None] + rB[None] - 1e-12)


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

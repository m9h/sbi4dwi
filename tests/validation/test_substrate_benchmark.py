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
    assert sub.centers_m.min() >= 0 and sub.centers_m.max() < 10e-6
    assert 0.05 < sub.f_intra < 0.9


def test_periodic_sdf_minimum_image():
    df = _straight_bundle(n_axons=1)
    sub = sb.make_crossing_substrate(df, 0.0, 10.0)
    sdf = sb.periodic_sdf(sub)
    c = sub.centers_m[0]
    assert float(sdf(jnp.asarray(c))) < 0
    assert float(sdf(jnp.asarray(c + np.array([10e-6, 0, 0])))) == pytest.approx(float(sdf(jnp.asarray(c))), abs=1e-12)


def test_free_diffusion_matches_exp_minus_bD():
    """Extra-cellular walkers in an (almost) empty box: S = exp(−bD)."""
    df = _straight_bundle(n_axons=1, r=0.3)
    sub = sb.make_crossing_substrate(df, 0.0, 10.0)
    assert sub.f_intra < 0.05
    b = np.array([0, 1000, 2000, 3000]) * 1e6
    u = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    D = 2.0e-9
    R = sb.pgse_lobe_sums(sub, False, D, 2e-5, 3000, 17.74e-3, 35.78e-3, jax.random.PRNGKey(0))
    S = np.asarray(sb.signals_from_lobe_sums(R, b, u, 17.74e-3, 35.78e-3))
    np.testing.assert_allclose(S, np.exp(-b * D), atol=0.03)


def test_intra_restricted_perpendicular_but_free_parallel():
    """Walkers inside straight z-axons: along z ≈ free, across ≈ restricted."""
    df = _straight_bundle(n_axons=8, r=1.0)
    sub = sb.make_crossing_substrate(df, 0.0, 10.0)
    b = np.array([3000, 3000]) * 1e6
    u = np.array([[0, 0, 1.0], [1, 0, 0]])
    D = 2.0e-9
    R = sb.pgse_lobe_sums(sub, True, D, 2e-5, 3000, 17.74e-3, 35.78e-3, jax.random.PRNGKey(1))
    S = np.asarray(sb.signals_from_lobe_sums(R, b, u, 17.74e-3, 35.78e-3))
    assert S[0] == pytest.approx(np.exp(-3000e6 * D), abs=0.05)     # parallel ≈ free
    assert S[1] > 0.6                                               # perpendicular restricted


def test_build_volume_shape_and_noise():
    sig = np.linspace(1, 0.2, 20)
    data, mask = sb.build_volume(sig, shape=(4, 4, 1), snr=30)
    assert data.shape == (4, 4, 1, 20) and mask.sum() == 16
    assert abs(data[..., 0].mean() - 1.0) < 0.05

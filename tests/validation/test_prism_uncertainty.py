"""Laplace fixel posterior: shapes, sane widths, calibration on synthetic data."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from dmipy_jax.validation import prism_jax as pj  # noqa: E402
from dmipy_jax.validation import prism_uncertainty as pu  # noqa: E402
from dmipy_jax.validation import prism_synthetic as ps  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    b = ps.make_benchmark(snr=30)
    # subset: 60° and 90° slices + singles, to keep the test fast
    sel = np.zeros(b["mask"].shape, bool); sel[[9, 15, 16]] = True
    cfg = pj.PrismConfig(n_fibres=2, n_iter=300, loss="nll")
    m = b["mask"] & sel
    gt = b["gt_dirs"][sel[b["mask"]]]
    ang = b["angle"][sel[b["mask"]]]
    fit = pj.fit_prism(b["data"], m, b["bvals"], b["bvecs"], cfg)
    post = pu.laplace_fixel_posterior(fit, b["data"], b["bvals"], b["bvecs"])
    return b, m, gt, ang, fit, post


def test_shapes_and_positive_widths(fitted):
    b, m, gt, ang, fit, post = fitted
    N = int(m.sum())
    assert post.cov.shape == (N, 2, 2, 2) and post.sigma_deg.shape == (N, 2)
    assert np.all(np.isfinite(post.sigma_deg))
    assert np.all(post.sigma_deg[:, 0] > 0)
    # at SNR 30 a well-determined fibre has ~1-5° uncertainty
    assert np.median(post.sigma_deg[ang > 0][:, 0]) < 8.0


def test_samples_are_unit_and_spread_matches_sigma(fitted):
    b, m, gt, ang, fit, post = fitted
    S = post.sample_dirs(jax.random.PRNGKey(0), 200)
    np.testing.assert_allclose(np.linalg.norm(S, axis=-1), 1.0, atol=1e-5)
    dev = np.degrees(np.arccos(np.clip(np.abs(np.einsum("snkj,nkj->snk", S, post.dirs)), 0, 1)))
    # angular sd of samples about the MAP tracks sigma_deg within a factor ~2
    ratio = dev.std(0)[:, 0] / np.maximum(post.sigma_deg[:, 0], 1e-3)
    assert 0.4 < np.median(ratio) < 2.5, np.median(ratio)


def test_cone_coverage_is_calibrated(fitted):
    """90 % cone must cover the truth in roughly 90 % of fixels (±15 pp:
    block-diagonal Laplace, Rician tails, 300-iter MAP)."""
    b, m, gt, ang, fit, post = fitted
    g = gt.copy(); g[ang == 0, 1] = 0.0
    cov = pu.cone_coverage(post, g, level=0.9)
    assert cov["n"] > 300
    assert 0.75 <= cov["coverage"] <= 1.0, cov["coverage"]


def test_single_fibre_second_slot_is_uncertain(fitted):
    """Where the second fibre has ~zero fraction its direction is
    unidentifiable: its Laplace width must be much larger than the main's."""
    b, m, gt, ang, fit, post = fitted
    single = ang == 0
    minor = post.wm_fracs[single, 1] < 0.05
    if minor.sum() > 5:
        assert np.median(post.sigma_deg[single][minor, 1]) > 3 * np.median(post.sigma_deg[single][:, 0])


def test_sigma_bounded_by_direction_prior(fitted):
    """No fixel's σ_θ may exceed the weak direction prior (~57°), including
    degenerate co-linear fibre pairs in single-fibre voxels."""
    b, m, gt, ang, fit, post = fitted
    assert np.all(post.sigma_deg <= 65.0), post.sigma_deg.max()

"""
Tests for PRISM-JAX (doc 006 §4): forward-model contracts, priors, and a
synthetic crossing-fibre recovery in PRISM's own metric.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from dmipy_jax.validation import prism_jax as pj  # noqa: E402


def _scheme(n_dirs=32, bvals_mm2=(1000, 2000, 3000), seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bv = [0.0] * 4 + [b for b in bvals_mm2 for _ in range(n_dirs)]
    gv = [np.zeros(3)] * 4 + [d for _ in bvals_mm2 for d in dirs]
    return np.array(bv) * 1e6, np.array(gv)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
# 1. Forward model contracts
# --------------------------------------------------------------------------- #

class TestForward:
    def test_pure_stick_collapse(self):
        """f_wm1=1, f_i=1 ⇒ S = S0·exp(−b D∥ (g·n)²)."""
        cfg = pj.PrismConfig(n_fibres=1)
        bvals, bvecs = _scheme()
        n = _unit([1, 1, 0])
        phys = {
            "s0": jnp.array([1.0]), "fracs": jnp.array([[0, 0, 1.0, 0]]),
            "dirs": jnp.array([[n]]), "fintra": jnp.array([1.0]),
            "d_par": jnp.asarray(cfg.d_par), "d_perp": jnp.asarray(cfg.d_perp),
            "sigma": None,
        }
        pred = np.asarray(pj.forward(phys, jnp.asarray(bvals), jnp.asarray(bvecs), cfg))[0]
        expect = np.exp(-bvals * cfg.d_par * (bvecs @ n) ** 2)
        np.testing.assert_allclose(pred, expect, atol=1e-6)

    def test_fractions_sum_to_one_and_s0_positive(self):
        cfg = pj.PrismConfig(n_fibres=3)
        p = pj.init_params(7, cfg, jax.random.PRNGKey(0))
        phys = pj.unpack(p, cfg)
        np.testing.assert_allclose(np.asarray(phys["fracs"]).sum(-1), 1.0, atol=1e-6)
        assert np.all(np.asarray(phys["s0"]) > 0)
        np.testing.assert_allclose(np.linalg.norm(np.asarray(phys["dirs"]), axis=-1), 1.0, atol=1e-6)

    def test_prism_default_diffusivities(self):
        """PRISM fixes D∥=1.7, D⊥=0.4, D_csf=3.0, D_gm=0.9, D_res=0.2 ×1e-3 mm²/s."""
        cfg = pj.PrismConfig()
        assert (cfg.d_par, cfg.d_perp) == (1.7e-9, 0.4e-9)
        assert (cfg.d_csf, cfg.d_gm, cfg.d_res) == (3.0e-9, 0.9e-9, 0.2e-9)

    def test_learnable_diffusivities_stay_in_band(self):
        cfg = pj.PrismConfig(learn_diffusivities=True)
        p = pj.init_params(3, cfg, jax.random.PRNGKey(0))
        p["dpar_logit"] = jnp.asarray(50.0)
        phys = pj.unpack(p, cfg)
        assert float(phys["d_par"]) <= cfg.d_par_range[1] + 1e-15
        assert float(phys["d_perp"]) == pytest.approx(cfg.d_perp, rel=1e-5)


# --------------------------------------------------------------------------- #
# 2. Priors
# --------------------------------------------------------------------------- #

class TestPriors:
    def test_neighbour_table_6conn(self):
        mask = np.ones((2, 2, 1), bool)
        nb = pj.build_neighbour_table(mask, 6)
        assert nb.shape == (4, 6)
        assert (nb >= 0).sum(axis=1).tolist() == [2, 2, 2, 2]

    def test_huber_laplacian_zero_on_constant_field(self):
        mask = np.ones((3, 3, 3), bool)
        nb = jnp.asarray(pj.build_neighbour_table(mask, 6))
        f = jnp.tile(jnp.array([[0.1, 0.2, 0.3, 0.4]]), (27, 1))
        assert float(pj.huber_laplacian(f, nb, 0.05)) == pytest.approx(0.0, abs=1e-8)

    def test_huber_laplacian_positive_on_step(self):
        mask = np.ones((2, 1, 1), bool)
        nb = jnp.asarray(pj.build_neighbour_table(mask, 6))
        f = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        assert float(pj.huber_laplacian(f, nb, 0.05)) > 0

    def test_repulsion_zero_orthogonal_max_parallel(self):
        f = jnp.array([[0.5, 0.5]])
        orth = jnp.array([[[1, 0, 0], [0, 1, 0]]], float)
        par = jnp.array([[[1, 0, 0], [1, 0, 0]]], float)
        assert float(pj.direction_repulsion(f, orth)) == pytest.approx(0.0)
        assert float(pj.direction_repulsion(f, par)) == pytest.approx(0.25)

    def test_sparsity_flat_above_tau(self):
        tau = 0.15
        a = jnp.array([[0.5, 0.5]])
        b = jnp.array([[0.9, 0.1]])
        assert float(pj.minor_fibre_sparsity(a, tau)) == pytest.approx(2 * tau)
        assert float(pj.minor_fibre_sparsity(b, tau)) == pytest.approx(tau + 0.1)

    def test_continuity_zero_when_neighbours_aligned(self):
        mask = np.ones((2, 1, 1), bool)
        nb = jnp.asarray(pj.build_neighbour_table(mask, 6))
        d = jnp.array([[[0, 0, 1.0]], [[0, 0, -1.0]]])   # antipodal = aligned
        f = jnp.array([[1.0], [1.0]])
        assert float(pj.directional_continuity(f, d, nb)) == pytest.approx(0.0, abs=1e-7)


# --------------------------------------------------------------------------- #
# 3. Synthetic recovery (PRISM's benchmark, miniature)
# --------------------------------------------------------------------------- #

def _crossing_volume(angle_deg, shape=(4, 4, 2), fintra=0.6, snr=None, seed=0):
    cfg = pj.PrismConfig(n_fibres=2)
    bvals, bvecs = _scheme()
    n1 = _unit([1, 0, 0])
    a = np.radians(angle_deg)
    n2 = _unit([np.cos(a), np.sin(a), 0])
    N = int(np.prod(shape))
    phys = {
        "s0": jnp.ones(N), "fracs": jnp.tile(jnp.array([[0, 0, 0.5, 0.5, 0]]), (N, 1)),
        "dirs": jnp.tile(jnp.array([[n1, n2]]), (N, 1, 1)), "fintra": jnp.full(N, fintra),
        "d_par": jnp.asarray(cfg.d_par), "d_perp": jnp.asarray(cfg.d_perp), "sigma": None,
    }
    clean = np.asarray(pj.forward(phys, jnp.asarray(bvals), jnp.asarray(bvecs), cfg))
    if snr is not None:
        rng = np.random.default_rng(seed)
        sig = 1.0 / snr
        clean = np.sqrt((clean + rng.normal(0, sig, clean.shape)) ** 2
                        + rng.normal(0, sig, clean.shape) ** 2)
    data = np.zeros(shape + (len(bvals),), np.float32)
    data[np.ones(shape, bool)] = clean
    gt = np.tile(np.array([[n1, n2]]), (N, 1, 1))
    return data, np.ones(shape, bool), bvals, bvecs, gt


@pytest.mark.parametrize("angle", [60, 90])
def test_recovers_crossing_mse(angle):
    data, mask, bvals, bvecs, gt = _crossing_volume(angle, snr=50)
    cfg = pj.PrismConfig(n_fibres=2, n_iter=400, loss="mse")
    fit = pj.fit_prism(data, mask, bvals, bvecs, cfg)
    err, recall = pj.angular_error_best_match(fit.dirs, fit.wm_fracs, gt)
    assert fit.loss_history[-1] < fit.loss_history[0]
    # Random init leaves a few voxels in a local minimum (PRISM §limitations
    # names exactly this); PRISM-MSE reports 95% recall. Warm start fixes it —
    # see test_warm_start_recovers_all_fibres.
    assert recall >= 0.9, recall
    assert err < 6.0, f"angular error {err:.2f}°"


def test_warm_start_recovers_all_fibres():
    """PRISM-plus lever: init directions from a coarse estimate (here GT
    perturbed by ~15°) must give recall 1.0 where random init does not."""
    data, mask, bvals, bvecs, gt = _crossing_volume(90, snr=50)
    rng = np.random.default_rng(1)
    init = gt + rng.normal(0, 0.25, gt.shape)
    init /= np.linalg.norm(init, axis=-1, keepdims=True)
    cfg = pj.PrismConfig(n_fibres=2, n_iter=400, loss="mse")
    fit = pj.fit_prism(data, mask, bvals, bvecs, cfg, init_dirs=init)
    err, recall = pj.angular_error_best_match(fit.dirs, fit.wm_fracs, gt)
    assert recall == 1.0
    assert err < 4.0, err


def test_recovers_crossing_nll_learns_sigma():
    data, mask, bvals, bvecs, gt = _crossing_volume(60, snr=30)
    cfg = pj.PrismConfig(n_fibres=2, n_iter=400, loss="nll", sigma_init=0.1)
    fit = pj.fit_prism(data, mask, bvals, bvecs, cfg)
    err, recall = pj.angular_error_best_match(fit.dirs, fit.wm_fracs, gt)
    assert recall == 1.0 and err < 6.0
    # σ learned back toward 1/SNR = 0.033 (in data units, S0≈1)
    assert 0.015 < fit.sigma < 0.08, fit.sigma


def test_single_fibre_minor_fraction_suppressed():
    """Sparsity prior must push the spurious second fibre to ~0."""
    cfg = pj.PrismConfig(n_fibres=2)
    bvals, bvecs = _scheme()
    shape = (3, 3, 2); N = 18
    n1 = _unit([0, 0, 1])
    phys = {
        "s0": jnp.ones(N), "fracs": jnp.tile(jnp.array([[0, 0, 1.0, 0, 0]]), (N, 1)),
        "dirs": jnp.tile(jnp.array([[n1, n1]]), (N, 1, 1)), "fintra": jnp.full(N, 0.6),
        "d_par": jnp.asarray(cfg.d_par), "d_perp": jnp.asarray(cfg.d_perp), "sigma": None,
    }
    clean = np.asarray(pj.forward(phys, jnp.asarray(bvals), jnp.asarray(bvecs), cfg))
    data = np.zeros(shape + (len(bvals),), np.float32); data[np.ones(shape, bool)] = clean
    fit = pj.fit_prism(data, np.ones(shape, bool), bvals, bvecs,
                       replace_cfg(cfg, n_iter=400))
    assert np.all(fit.wm_fracs[:, 1] < 0.1), fit.wm_fracs[:, 1].max()
    assert np.all(fit.wm_fracs[:, 0] > 0.8)


def replace_cfg(cfg, **kw):
    from dataclasses import replace
    return replace(cfg, **kw)


def test_pam_adapter_shapes_and_threshold():
    data, mask, bvals, bvecs, gt = _crossing_volume(90, snr=None)
    cfg = pj.PrismConfig(n_fibres=2, n_iter=200)
    fit = pj.fit_prism(data, mask, bvals, bvecs, cfg)
    from dipy.data import default_sphere
    pam = pj.prism_fit_to_pam(fit, default_sphere, peak_frac_min=0.05)
    assert pam.peak_dirs.shape == mask.shape + (5, 3)
    assert pam.peak_values.shape == mask.shape + (5,)
    pi = pam.peak_indices[mask]
    assert (pi[:, 0] >= 0).all()
    assert (pi[:, 1] >= 0).mean() > 0.9          # a stray local minimum is allowed
    assert (pi[:, 2:] == -1).all()
    assert (pam.peak_values[mask][:, 0] >= pam.peak_values[mask][:, 1]).all()

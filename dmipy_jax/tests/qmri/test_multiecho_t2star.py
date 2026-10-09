"""Tests for multi-echo T2* fitting."""
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, pytest
from dmipy_jax.signal_models.qmri.multiecho_t2star import (
    multiecho_signal, fit_multiecho_t2star, fit_multiecho_t2star_batch, MultiEchoResult)

# WAND-like: 7 echoes from ~2ms to ~30ms
TES = jnp.array([2.5, 6.5, 10.5, 14.5, 18.5, 22.5, 26.5]) * 1e-3  # seconds

class TestMultiechoSignal:
    def test_shape(self):
        assert multiecho_signal(1000.0, 0.03, TES).shape == (7,)
    def test_positive(self):
        assert jnp.all(multiecho_signal(1000.0, 0.03, TES) > 0)
    def test_decays(self):
        s = multiecho_signal(1000.0, 0.03, TES)
        assert jnp.all(jnp.diff(s) < 0)  # monotonically decreasing
    def test_known_value(self):
        # At TE=0, signal = S0
        s = multiecho_signal(500.0, 0.05, jnp.array([0.0]))
        assert float(s[0]) == pytest.approx(500.0)
    def test_differentiable(self):
        def loss(t2s): return jnp.sum(multiecho_signal(1000.0, t2s, TES)**2)
        g = jax.grad(loss)(0.03)
        assert jnp.isfinite(g)
    def test_jit(self):
        assert jax.jit(multiecho_signal)(1000.0, 0.03, TES).shape == (7,)
    def test_vmap(self):
        t2s = jnp.array([0.02, 0.03, 0.05])
        s = jax.vmap(multiecho_signal, in_axes=(None, 0, None))(1000.0, t2s, TES)
        assert s.shape == (3, 7)

class TestMultiechoFitting:
    def test_recovers_wm(self):
        data = multiecho_signal(1000.0, 0.030, TES)  # WM T2*=30ms
        r = fit_multiecho_t2star(data, TES)
        assert abs(float(r.T2star) - 0.030) < 0.003
    def test_recovers_gm(self):
        data = multiecho_signal(800.0, 0.050, TES)   # GM T2*=50ms
        r = fit_multiecho_t2star(data, TES)
        assert abs(float(r.T2star) - 0.050) < 0.005
    def test_recovers_S0(self):
        data = multiecho_signal(1500.0, 0.040, TES)
        r = fit_multiecho_t2star(data, TES)
        assert abs(float(r.S0) - 1500.0) / 1500.0 < 0.05
    def test_noisy(self):
        clean = multiecho_signal(1000.0, 0.035, TES)
        data = clean + jr.normal(jr.PRNGKey(0), clean.shape) * 5.0
        r = fit_multiecho_t2star(data, TES)
        assert abs(float(r.T2star) - 0.035) / 0.035 < 0.3
    def test_batch(self):
        t2s = jnp.array([0.025, 0.035, 0.050, 0.080])
        data = jax.vmap(multiecho_signal, in_axes=(None, 0, None))(1000.0, t2s, TES)
        results = fit_multiecho_t2star_batch(data, TES)
        np.testing.assert_allclose(results.T2star, t2s, rtol=0.05)

class TestMultiechoUncertainty:
    def test_returns_std(self):
        data = multiecho_signal(1000.0, 0.03, TES)
        r = fit_multiecho_t2star(data, TES)
        assert isinstance(r, MultiEchoResult)
        assert hasattr(r, 'T2star_std')
    def test_uncertainty_positive(self):
        data = multiecho_signal(1000.0, 0.03, TES)
        r = fit_multiecho_t2star(data, TES)
        assert float(r.T2star_std) > 0
    def test_noisy_wider_uncertainty(self):
        clean = multiecho_signal(1000.0, 0.04, TES)
        r_low = fit_multiecho_t2star(clean + jr.normal(jr.PRNGKey(0), clean.shape)*2, TES)
        r_high = fit_multiecho_t2star(clean + jr.normal(jr.PRNGKey(0), clean.shape)*20, TES)
        assert float(r_high.T2star_std) > float(r_low.T2star_std)

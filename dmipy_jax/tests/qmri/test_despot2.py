"""Tests for DESPOT2 SSFP T2 fitting."""
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, pytest
from dmipy_jax.signal_models.qmri.despot2 import (
    ssfp_signal, fit_despot2, fit_despot2_batch, DESPOT2Result)

FA = jnp.array([10, 20, 30, 40, 50]) * jnp.pi / 180
TR = 0.005  # 5ms typical bSSFP

class TestSSFPSignal:
    def test_shape(self):
        assert ssfp_signal(1.0, 0.08, 1000.0, TR, FA).shape == (5,)
    def test_positive(self):
        assert jnp.all(ssfp_signal(1.0, 0.08, 1000.0, TR, FA) > 0)
    def test_increases_with_T2(self):
        """Longer T2 → more signal (less decay during TR)."""
        s_short = ssfp_signal(1.0, 0.03, 1000.0, TR, FA)
        s_long = ssfp_signal(1.0, 0.15, 1000.0, TR, FA)
        assert float(jnp.mean(s_long)) > float(jnp.mean(s_short))
    def test_differentiable(self):
        def loss(t2): return jnp.sum(ssfp_signal(1.0, t2, 1000.0, TR, FA)**2)
        g = jax.grad(loss)(0.08)
        assert jnp.isfinite(g)
    def test_jit(self):
        assert jax.jit(ssfp_signal)(1.0, 0.08, 1000.0, TR, FA).shape == (5,)
    def test_vmap(self):
        t2s = jnp.array([0.03, 0.06, 0.1])
        s = jax.vmap(ssfp_signal, in_axes=(None, 0, None, None, None))(1.0, t2s, 1000.0, TR, FA)
        assert s.shape == (3, 5)

class TestDESPOT2Fitting:
    def test_recovers_wm_T2(self):
        T1, true_T2 = 0.8, 0.060  # WM
        data = ssfp_signal(T1, true_T2, 1000.0, TR, FA)
        r = fit_despot2(data, FA, TR, T1)
        assert abs(float(r.T2) - true_T2) / true_T2 < 0.1
    def test_recovers_gm_T2(self):
        T1, true_T2 = 1.3, 0.080  # GM
        data = ssfp_signal(T1, true_T2, 1000.0, TR, FA)
        r = fit_despot2(data, FA, TR, T1)
        assert abs(float(r.T2) - true_T2) / true_T2 < 0.1
    def test_recovers_M0(self):
        data = ssfp_signal(1.0, 0.07, 500.0, TR, FA)
        r = fit_despot2(data, FA, TR, 1.0)
        assert abs(float(r.M0) - 500.0) / 500.0 < 0.1
    def test_noisy(self):
        clean = ssfp_signal(1.0, 0.06, 1000.0, TR, FA)
        data = clean + jr.normal(jr.PRNGKey(0), clean.shape) * 5.0
        r = fit_despot2(data, FA, TR, 1.0)
        assert abs(float(r.T2) - 0.06) / 0.06 < 0.3
    def test_batch(self):
        T1s = jnp.array([0.8, 1.0, 1.3])
        T2s = jnp.array([0.06, 0.07, 0.08])
        data = jax.vmap(ssfp_signal, in_axes=(0, 0, None, None, None))(T1s, T2s, 1000.0, TR, FA)
        results = fit_despot2_batch(data, FA, TR, T1s)
        np.testing.assert_allclose(results.T2, T2s, rtol=0.1)

class TestDESPOT2Uncertainty:
    def test_returns_std(self):
        data = ssfp_signal(1.0, 0.07, 1000.0, TR, FA)
        r = fit_despot2(data, FA, TR, 1.0)
        assert hasattr(r, 'T2_std')
    def test_uncertainty_positive(self):
        data = ssfp_signal(1.0, 0.07, 1000.0, TR, FA)
        r = fit_despot2(data, FA, TR, 1.0)
        assert float(r.T2_std) > 0
    def test_noisy_wider(self):
        clean = ssfp_signal(1.0, 0.07, 1000.0, TR, FA)
        r_low = fit_despot2(clean + jr.normal(jr.PRNGKey(0), clean.shape)*2, FA, TR, 1.0)
        r_high = fit_despot2(clean + jr.normal(jr.PRNGKey(0), clean.shape)*20, FA, TR, 1.0)
        assert float(r_high.T2_std) > float(r_low.T2_std)

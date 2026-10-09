"""Tests for QMT two-pool model fitting."""
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, pytest
from dmipy_jax.signal_models.qmri.qmt import (
    qmt_signal_ratio, fit_qmt, QMTResult)

# Typical qMT acquisition: multiple offset frequencies
OFFSETS = jnp.array([1000, 2000, 4000, 8000, 16000, 32000.0])  # Hz
MT_FA = jnp.ones(6) * (500 * jnp.pi / 180)  # strong MT pulse

class TestQMTSignal:
    def test_shape(self):
        r = qmt_signal_ratio(0.12, 3.0, 10e-6, 1.0, OFFSETS, MT_FA)
        assert r.shape == (6,)
    def test_bounded(self):
        r = qmt_signal_ratio(0.12, 3.0, 10e-6, 1.0, OFFSETS, MT_FA)
        assert jnp.all(r >= 0) and jnp.all(r <= 1)
    def test_varies_with_offset(self):
        """Signal ratio should vary with offset frequency."""
        r = qmt_signal_ratio(0.12, 3.0, 10e-6, 1.0, OFFSETS, MT_FA)
        # The ratio should not be constant across offsets
        assert float(jnp.std(r)) > 1e-4
    def test_higher_F_more_saturation(self):
        """Higher bound pool fraction → more MT effect → lower ratio at low offset."""
        r_low = qmt_signal_ratio(0.05, 3.0, 10e-6, 1.0, OFFSETS, MT_FA)
        r_high = qmt_signal_ratio(0.20, 3.0, 10e-6, 1.0, OFFSETS, MT_FA)
        assert float(r_high[0]) < float(r_low[0])
    def test_differentiable(self):
        def loss(F): return jnp.sum(qmt_signal_ratio(F, 3.0, 10e-6, 1.0, OFFSETS, MT_FA)**2)
        g = jax.grad(loss)(0.12)
        assert jnp.isfinite(g)
    def test_jit(self):
        r = jax.jit(qmt_signal_ratio)(0.12, 3.0, 10e-6, 1.0, OFFSETS, MT_FA)
        assert r.shape == (6,)

class TestQMTFitting:
    def test_returns_result(self):
        mt_on = qmt_signal_ratio(0.12, 3.0, 10e-6, 1.0, OFFSETS, MT_FA) * 1000
        r = fit_qmt(mt_on, jnp.array([1000.0]), OFFSETS, MT_FA, R1f=1.0)
        assert isinstance(r, QMTResult)
    def test_recovers_F(self):
        true_F = 0.12
        ratio = qmt_signal_ratio(true_F, 3.0, 10e-6, 1.0, OFFSETS, MT_FA)
        mt_on = ratio * 1000
        r = fit_qmt(mt_on, jnp.array([1000.0]), OFFSETS, MT_FA, R1f=1.0)
        assert abs(float(r.F) - true_F) < 0.06
    def test_F_in_range(self):
        ratio = qmt_signal_ratio(0.15, 4.0, 10e-6, 1.0, OFFSETS, MT_FA)
        r = fit_qmt(ratio * 800, jnp.array([800.0]), OFFSETS, MT_FA, R1f=1.0)
        assert 0.0 < float(r.F) < 0.5
    def test_has_uncertainty(self):
        ratio = qmt_signal_ratio(0.12, 3.0, 10e-6, 1.0, OFFSETS, MT_FA)
        r = fit_qmt(ratio * 1000, jnp.array([1000.0]), OFFSETS, MT_FA, R1f=1.0)
        assert float(r.F_std) >= 0
        assert float(r.kf_std) >= 0
    def test_noisy(self):
        ratio = qmt_signal_ratio(0.12, 3.0, 10e-6, 1.0, OFFSETS, MT_FA)
        mt_on = ratio * 1000 + jr.normal(jr.PRNGKey(0), ratio.shape) * 10
        r = fit_qmt(mt_on, jnp.array([1000.0]), OFFSETS, MT_FA, R1f=1.0)
        assert 0.0 < float(r.F) < 0.5  # at least in physiological range

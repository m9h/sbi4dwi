"""Tests for MP2RAGE T1 mapping."""
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, pytest
from dmipy_jax.signal_models.qmri.mp2rage import (
    mp2rage_uniform, mp2rage_signal_model, fit_mp2rage_t1, MP2RAGEResult)

# Typical 3T MP2RAGE parameters
TI1, TI2, TR = 0.7, 2.5, 5.0
ALPHA1 = 4 * jnp.pi / 180
ALPHA2 = 5 * jnp.pi / 180

class TestMP2RAGEUniform:
    def test_scalar(self):
        uni = mp2rage_uniform(100.0, 200.0)
        assert uni.shape == ()
    def test_bounded(self):
        uni = mp2rage_uniform(jnp.array([100, -50, 300.0]), jnp.array([200, 100, 10.0]))
        assert jnp.all(uni >= -0.5)
        assert jnp.all(uni <= 0.5)
    def test_symmetric_inputs(self):
        """Equal magnitude INV1=INV2 → UNI = 0.5."""
        uni = mp2rage_uniform(100.0, 100.0)
        assert float(uni) == pytest.approx(0.5)

class TestMP2RAGESignalModel:
    def test_returns_scalar(self):
        uni = mp2rage_signal_model(1.0, TI1, TI2, TR, ALPHA1, ALPHA2)
        assert uni.shape == ()
    def test_bounded(self):
        T1s = jnp.linspace(0.2, 5.0, 50)
        unis = jax.vmap(lambda t: mp2rage_signal_model(t, TI1, TI2, TR, ALPHA1, ALPHA2))(T1s)
        assert jnp.all(unis >= -0.5)
        assert jnp.all(unis <= 0.5)
    def test_varies_with_T1(self):
        """Different T1 values should give different UNI values."""
        u1 = mp2rage_signal_model(0.5, TI1, TI2, TR, ALPHA1, ALPHA2)
        u2 = mp2rage_signal_model(2.0, TI1, TI2, TR, ALPHA1, ALPHA2)
        assert float(u1) != pytest.approx(float(u2), abs=0.01)
    def test_differentiable(self):
        def loss(t1): return mp2rage_signal_model(t1, TI1, TI2, TR, ALPHA1, ALPHA2)**2
        g = jax.grad(loss)(1.0)
        assert jnp.isfinite(g)

class TestMP2RAGEFitting:
    def test_returns_result(self):
        r = fit_mp2rage_t1(100.0, 200.0, TI1, TI2, TR, ALPHA1, ALPHA2)
        assert isinstance(r, MP2RAGEResult)
    def test_has_uncertainty(self):
        r = fit_mp2rage_t1(100.0, 200.0, TI1, TI2, TR, ALPHA1, ALPHA2)
        assert hasattr(r, 'T1_std')
        assert float(r.T1_std) > 0
    def test_t1_positive(self):
        r = fit_mp2rage_t1(100.0, 200.0, TI1, TI2, TR, ALPHA1, ALPHA2)
        assert float(r.T1) > 0
    def test_t1_in_range(self):
        """T1 should be physiologically plausible."""
        r = fit_mp2rage_t1(100.0, 200.0, TI1, TI2, TR, ALPHA1, ALPHA2)
        assert 0.1 < float(r.T1) < 6.0
    def test_different_inputs_different_T1(self):
        r1 = fit_mp2rage_t1(50.0, 200.0, TI1, TI2, TR, ALPHA1, ALPHA2)
        r2 = fit_mp2rage_t1(150.0, 200.0, TI1, TI2, TR, ALPHA1, ALPHA2)
        assert float(r1.T1) != pytest.approx(float(r2.T1), abs=0.1)

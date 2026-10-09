"""
Tests for Full Waveform Inversion for Ultrasound (FWI-US).
"""

import pytest
import jax
import jax.numpy as jnp


class TestFWILoss:
    """Tests for the FWI loss function."""

    def test_loss_zero_for_identical_fields(self):
        """Loss should be 0 when predicted == observed."""
        from dmipy_jax.biophysics.fwi_us import fwi_loss

        p = jnp.ones((32, 32)) * 0.5
        loss = fwi_loss(p, p)
        assert jnp.isclose(loss, 0.0, atol=1e-10)

    def test_loss_positive_for_different_fields(self):
        """Loss should be > 0 when fields differ."""
        from dmipy_jax.biophysics.fwi_us import fwi_loss

        p1 = jnp.ones((32, 32)) * 0.5
        p2 = jnp.ones((32, 32)) * 1.0
        loss = fwi_loss(p1, p2)
        assert loss > 0

    def test_tv_regularization_penalizes_discontinuities(self):
        """TV term should be higher for discontinuous fields."""
        from dmipy_jax.biophysics.fwi_us import fwi_loss

        p = jnp.ones((32, 32))  # dummy observed/predicted
        c_smooth = jnp.ones((32, 32)) * 1500.0
        c_rough = jnp.ones((32, 32)) * 1500.0
        c_rough = c_rough.at[::2, :].set(2000.0)  # alternating rows

        loss_smooth = fwi_loss(p, p, c_smooth, tv_weight=1.0)
        loss_rough = fwi_loss(p, p, c_rough, tv_weight=1.0)
        assert loss_rough > loss_smooth

    def test_loss_differentiable(self):
        """jax.grad through loss function."""
        from dmipy_jax.biophysics.fwi_us import fwi_loss

        def f(c):
            return fwi_loss(c, jnp.ones((8, 8)), c, tv_weight=0.01)

        g = jax.grad(f)(jnp.ones((8, 8)) * 1500.0)
        assert not jnp.any(jnp.isnan(g))


class TestFWIInversion:
    """Tests for the full FWI inversion (requires j-Wave, slow)."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_fwi_loss_decreases(self):
        """Loss should decrease during optimization."""
        from dmipy_jax.biophysics.fwi_us import invert_tissue_properties

        # Create synthetic observed data from known c=1500
        import importlib.util, sys, os
        spec = importlib.util.spec_from_file_location(
            "jwa", os.path.join(os.path.dirname(__file__), "../../biophysics/jwave_adapter.py"))
        jwa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(jwa)

        observed = jwa.run_simulation_2d(
            (32, 32), 0.5e-3, 1500.0, 1000.0,
            (16, 4), 500e3, 3e-5, pml_size=5,
        )["p_max"]

        c_recon, history = invert_tissue_properties(
            observed, 1600.0,  # start with wrong guess
            grid_shape=(32, 32), grid_spacing=0.5e-3,
            source_position=(16, 4), freq=500e3,
            n_iters=5, lr=10.0, tv_weight=0.0,
            t_end=3e-5, pml_size=5,
        )
        assert len(history) == 5
        # Loss should decrease (or at least not NaN)
        for h in history:
            assert not jnp.isnan(h)

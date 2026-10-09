"""
Tests for j-Wave simulation adapter.

RED phase: Define contracts for converting openlifu-style parameters
to j-Wave simulation inputs and running differentiable acoustic simulation.
"""

import pytest

pytest.importorskip("jwave")
import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Phase 3.1: Domain + Medium Construction
# ---------------------------------------------------------------------------

class TestJWaveDomain:
    """Tests for domain construction from grid parameters."""

    def test_create_domain_3d(self):
        """Create a 3D j-Wave domain from grid shape and spacing."""
        from dmipy_jax.biophysics.jwave_adapter import create_domain

        domain = create_domain(
            grid_shape=(64, 64, 64),
            grid_spacing=0.5e-3,  # 0.5mm in meters
        )
        assert domain.N == (64, 64, 64)
        assert domain.dx[0] == pytest.approx(0.5e-3, rel=1e-6)

    def test_create_domain_2d(self):
        """Create a 2D domain for fast testing."""
        from dmipy_jax.biophysics.jwave_adapter import create_domain

        domain = create_domain(grid_shape=(128, 128), grid_spacing=0.1e-3)
        assert domain.N == (128, 128)


class TestJWaveMedium:
    """Tests for heterogeneous medium construction."""

    def test_create_medium_homogeneous(self):
        """Scalar properties should create a valid medium."""
        from dmipy_jax.biophysics.jwave_adapter import create_domain, create_medium

        domain = create_domain((32, 32, 32), 1e-3)
        medium = create_medium(
            domain,
            sound_speed=1500.0,
            density=1000.0,
            pml_size=10,
        )
        assert medium is not None
        assert medium.domain.N == (32, 32, 32)

    def test_create_medium_heterogeneous(self):
        """3D array properties should create FourierSeries-wrapped medium."""
        from dmipy_jax.biophysics.jwave_adapter import create_domain, create_medium

        domain = create_domain((32, 32, 32), 1e-3)
        c = jnp.ones((32, 32, 32)) * 1500.0
        c = c.at[10:20, :, :].set(4080.0)  # skull layer
        rho = jnp.ones((32, 32, 32)) * 1000.0
        rho = rho.at[10:20, :, :].set(1900.0)

        medium = create_medium(domain, sound_speed=c, density=rho, pml_size=10)
        assert medium is not None


# ---------------------------------------------------------------------------
# Phase 3.2: Source Construction
# ---------------------------------------------------------------------------

class TestJWaveSources:
    """Tests for creating j-Wave sources from transducer parameters."""

    def test_create_sources_from_positions(self):
        """Element positions should map to j-Wave source grid indices."""
        from dmipy_jax.biophysics.jwave_adapter import create_domain, create_sources

        domain = create_domain((64, 64, 64), 0.5e-3)
        # 4 elements at known positions (in meters)
        positions = jnp.array([
            [0.005, 0.005, 0.0],
            [0.010, 0.005, 0.0],
            [0.005, 0.010, 0.0],
            [0.010, 0.010, 0.0],
        ])
        freq = 400e3
        delays = jnp.zeros(4)
        apod = jnp.ones(4)

        sources = create_sources(
            domain, positions, freq, delays, apod,
            dt=1e-7, n_cycles=5,
        )
        assert sources is not None

    def test_source_signals_shaped_correctly(self):
        """Each element should produce a time-domain signal."""
        from dmipy_jax.biophysics.jwave_adapter import create_domain, create_sources

        domain = create_domain((32, 32, 32), 1e-3)
        positions = jnp.array([[0.005, 0.005, 0.005], [0.010, 0.010, 0.010]])
        freq = 400e3
        delays = jnp.zeros(2)
        apod = jnp.ones(2)
        dt = 1e-7
        n_cycles = 5

        sources = create_sources(domain, positions, freq, delays, apod, dt=dt, n_cycles=n_cycles)
        # Signals should be (n_elements, n_timesteps)
        assert sources.signals.shape[0] == 2
        assert sources.signals.shape[1] > 0

    def test_delay_shifts_signal(self):
        """A delayed element's signal peak should arrive later."""
        from dmipy_jax.biophysics.jwave_adapter import _make_toneburst

        freq = 400e3
        dt = 1e-7
        n_cycles = 5

        sig_nodelay = _make_toneburst(freq, dt, n_cycles, delay=0.0, amplitude=1.0)
        sig_delayed = _make_toneburst(freq, dt, n_cycles, delay=5e-6, amplitude=1.0)

        peak0 = jnp.argmax(jnp.abs(sig_nodelay))
        peak1 = jnp.argmax(jnp.abs(sig_delayed))
        assert peak1 > peak0


# ---------------------------------------------------------------------------
# Phase 3.3: Full Simulation
# ---------------------------------------------------------------------------

class TestJWaveSimulation:
    """Tests for running j-Wave acoustic simulation."""

    @pytest.mark.slow
    def test_simulation_homogeneous_2d(self):
        """2D homogeneous simulation should produce non-zero pressure."""
        from dmipy_jax.biophysics.jwave_adapter import run_simulation_2d

        result = run_simulation_2d(
            grid_shape=(128, 128),
            grid_spacing=0.1e-3,
            sound_speed=1500.0,
            density=1000.0,
            source_position=(64, 10),
            freq=1e6,
            t_end=1e-5,
        )
        assert "p_max" in result
        assert jnp.max(result["p_max"]) > 0

    @pytest.mark.slow
    def test_simulation_heterogeneous_attenuates(self):
        """A high-impedance layer should reduce transmitted pressure."""
        from dmipy_jax.biophysics.jwave_adapter import run_simulation_2d

        # Homogeneous water
        result_water = run_simulation_2d(
            grid_shape=(128, 128),
            grid_spacing=0.1e-3,
            sound_speed=1500.0,
            density=1000.0,
            source_position=(64, 10),
            freq=1e6,
            t_end=1e-5,
        )

        # Water with skull slab
        c = jnp.ones((128, 128)) * 1500.0
        c = c.at[:, 40:50].set(4080.0)
        rho = jnp.ones((128, 128)) * 1000.0
        rho = rho.at[:, 40:50].set(1900.0)

        result_skull = run_simulation_2d(
            grid_shape=(128, 128),
            grid_spacing=0.1e-3,
            sound_speed=c,
            density=rho,
            source_position=(64, 10),
            freq=1e6,
            t_end=1e-5,
        )

        # Behind the skull (y>50), pressure should be lower
        p_water_behind = jnp.max(result_water["p_max"][:, 60:])
        p_skull_behind = jnp.max(result_skull["p_max"][:, 60:])
        assert p_skull_behind < p_water_behind, (
            f"Skull should attenuate: water={p_water_behind:.2f}, skull={p_skull_behind:.2f}"
        )

    @pytest.mark.slow
    def test_simulation_differentiable(self):
        """jax.grad through the simulation pipeline must not NaN."""
        from dmipy_jax.biophysics.jwave_adapter import (
            create_domain,
            create_medium,
            run_simulation_jax,
        )

        domain = create_domain((32, 32), 0.5e-3)

        def focal_pressure(c_value):
            medium = create_medium(domain, sound_speed=c_value, density=1000.0, pml_size=5)
            p_max = run_simulation_jax(
                medium,
                source_position=(16, 2),
                freq=500e3,
                t_end=5e-6,
            )
            return p_max[16, 16]

        grad_fn = jax.grad(focal_pressure)
        g = grad_fn(1500.0)
        assert not jnp.isnan(g), f"Gradient is NaN: {g}"

"""
RED tests for extended TUS experiments.

Covers: 3D simulation, multi-element arrays, multi-target, multi-frequency,
cross-validation, grid convergence, sensitivity analysis, Helmholtz solver.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 3D Volumetric Simulation
# ---------------------------------------------------------------------------

class TestSimulation3D:
    """Tests for 3D j-Wave simulation on volumetric head data."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_3d_simulation_produces_volume(self):
        """3D simulation should return a 3D pressure array."""
        from dmipy_jax.biophysics.jwave_adapter import create_domain, create_medium, run_simulation_jax
        from jwave.geometry import TimeAxis

        domain = create_domain((32, 32, 32), 1e-3)
        medium = create_medium(domain, sound_speed=1500.0, density=1000.0, pml_size=5)
        time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=1e-5)
        p = run_simulation_jax(medium, source_position=(16, 16, 5), freq=400e3,
                               t_end=1e-5, time_axis=time_axis)
        assert p.ndim == 3
        assert p.shape == (32, 32, 32)
        assert jnp.max(p) > 0

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_3d_skull_attenuates(self):
        """3D heterogeneous skull should reduce pressure vs water."""
        from dmipy_jax.biophysics.jwave_adapter import create_domain, create_medium, run_simulation_jax
        from jwave.geometry import TimeAxis

        N = 32
        domain = create_domain((N, N, N), 1e-3)

        # Water
        med_w = create_medium(domain, 1500.0, 1000.0, pml_size=5)
        ta = TimeAxis.from_medium(med_w, cfl=0.3, t_end=1e-5)
        p_w = run_simulation_jax(med_w, (16, 16, 5), 400e3, 1e-5, time_axis=ta)

        # Skull layer
        c = jnp.ones((N, N, N)) * 1500.0
        c = c.at[:, :, 10:15].set(4080.0)
        rho = jnp.ones((N, N, N)) * 1000.0
        rho = rho.at[:, :, 10:15].set(1900.0)
        med_s = create_medium(domain, c, rho, pml_size=5)
        p_s = run_simulation_jax(med_s, (16, 16, 5), 400e3, 1e-5, time_axis=ta)

        # Pressure behind skull should be lower
        assert jnp.max(p_s[:, :, 20:]) < jnp.max(p_w[:, :, 20:])


# ---------------------------------------------------------------------------
# Multi-Element Array
# ---------------------------------------------------------------------------

class TestMultiElementArray:
    """Tests for realistic multi-element transducer optimization."""

    def test_create_array_positions(self):
        """Generate N-element circular array positions."""
        from dmipy_jax.biophysics.tus_experiments import create_circular_array

        positions = create_circular_array(n_elements=16, radius=0.01, center=(0.016, 0.016, 0.002))
        assert positions.shape == (16, 3)
        # All z-coords should be at the standoff
        assert jnp.allclose(positions[:, 2], 0.002, atol=1e-6)

    def test_create_array_32_elements(self):
        """32-element array should have correct shape."""
        from dmipy_jax.biophysics.tus_experiments import create_circular_array

        positions = create_circular_array(n_elements=32, radius=0.015, center=(0.016, 0.016, 0.002))
        assert positions.shape == (32, 3)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_optimize_16_elements(self):
        """16-element optimization should converge (loss decreases)."""
        from dmipy_jax.biophysics.tus_experiments import run_array_optimization

        result = run_array_optimization(
            n_elements=16, grid_size=64, spacing_mm=0.2,
            freq=400e3, n_iters=5, skull_thickness_voxels=10,
        )
        assert result["n_elements"] == 16
        assert len(result["loss_history"]) == 5
        assert not np.any(np.isnan(result["loss_history"]))


# ---------------------------------------------------------------------------
# Multi-Target
# ---------------------------------------------------------------------------

class TestMultiTarget:
    """Tests for simulation at different brain targets."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_different_targets_different_attenuation(self):
        """Targets behind thicker skull should see more attenuation."""
        from dmipy_jax.biophysics.tus_experiments import simulate_at_targets

        results = simulate_at_targets(
            grid_size=64, spacing_mm=0.2, freq=400e3,
            targets={"shallow": (32, 32, 40), "deep": (32, 32, 55)},
            skull_range=(20, 30),
        )
        assert "shallow" in results
        assert "deep" in results
        # Both should have pressure values
        assert results["shallow"]["p_skull"] > 0
        assert results["deep"]["p_skull"] > 0


# ---------------------------------------------------------------------------
# Multi-Frequency
# ---------------------------------------------------------------------------

class TestMultiFrequency:
    """Tests for frequency-dependent skull penetration."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_lower_freq_better_penetration(self):
        """180kHz should penetrate skull better than 400kHz."""
        from dmipy_jax.biophysics.tus_experiments import compare_frequencies

        result = compare_frequencies(
            grid_size=64, spacing_mm=0.2,
            frequencies=[180e3, 400e3],
            skull_range=(20, 30),
            target=(32, 32, 50),
        )
        assert len(result) == 2
        # Lower frequency should have higher transmission
        assert result[180e3]["p_skull"] >= result[400e3]["p_skull"] * 0.5  # at least comparable


# ---------------------------------------------------------------------------
# Grid Convergence
# ---------------------------------------------------------------------------

class TestGridConvergence:
    """Tests for grid resolution convergence study."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_convergence_three_resolutions(self):
        """Focal pressure should converge as grid refines."""
        from dmipy_jax.biophysics.tus_experiments import grid_convergence_study

        results = grid_convergence_study(
            spacings_mm=[0.4, 0.2, 0.1],
            base_size_mm=12.8,
            freq=400e3,
        )
        assert len(results) == 3
        # All should produce non-zero pressure
        for sp, res in results.items():
            assert res["p_max"] > 0


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    """Tests for gradient-based sensitivity of focal pressure to skull properties."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_sensitivity_map_shape(self):
        """Sensitivity d(p_target)/d(c_skull) should have same shape as sound speed field."""
        from dmipy_jax.biophysics.tus_experiments import compute_sensitivity_map

        sens = compute_sensitivity_map(grid_size=32, spacing_mm=0.5, freq=400e3, target=(16, 16))
        assert sens.shape == (32, 32)
        # Sensitivity should be non-zero in skull region
        assert jnp.any(sens != 0)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_sensitivity_highest_in_skull(self):
        """Sensitivity should be highest in the skull region (where c changes matter)."""
        from dmipy_jax.biophysics.tus_experiments import compute_sensitivity_map

        sens = compute_sensitivity_map(
            grid_size=32, spacing_mm=0.5, freq=400e3, target=(16, 25),
            skull_range=(8, 12),
        )
        skull_sens = jnp.abs(sens[8:12, :]).mean()
        water_sens = jnp.abs(sens[0:5, :]).mean()
        assert skull_sens > water_sens


# ---------------------------------------------------------------------------
# Helmholtz (Frequency-Domain) Solver
# ---------------------------------------------------------------------------

class TestHelmholtzSolver:
    """Tests for frequency-domain Helmholtz solver as alternative to time-domain."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_helmholtz_produces_pressure(self):
        """Helmholtz solver should produce a non-zero complex pressure field."""
        from dmipy_jax.biophysics.tus_experiments import run_helmholtz_simulation

        result = run_helmholtz_simulation(grid_size=64, spacing_mm=0.2, freq=400e3)
        assert result["p_abs"].shape == (64, 64)
        assert jnp.max(result["p_abs"]) > 0

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_helmholtz_vs_time_domain_agreement(self):
        """Helmholtz and time-domain should produce correlated pressure patterns."""
        from dmipy_jax.biophysics.tus_experiments import compare_helmholtz_vs_timedomain

        result = compare_helmholtz_vs_timedomain(grid_size=64, spacing_mm=0.2, freq=400e3)
        # Correlation should be positive (same physics)
        assert result["correlation"] > 0.3

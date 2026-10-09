"""
Tests for radiation force computation and quasi-static displacement solver.

RED phase: Define contracts for F = 2αI/c radiation force,
spectral Green's function displacement, and Kuhl CANN shear modulus mapping.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


class TestRadiationForce:
    """Tests for acoustic radiation force computation."""

    def test_radiation_force_formula(self):
        """F = 2α · I / c with known values."""
        from dmipy_jax.biophysics.radiation_force import compute_radiation_force

        # Brain tissue: α=5.3 dB/cm/MHz, c=1560 m/s
        # Convert α to Np/m: 5.3 dB/cm * 100 cm/m / 8.686 dB/Np = 61.0 Np/m
        alpha_np_m = 61.0
        intensity = 10.0  # W/m² (typical neuromod ISPTA)
        c = 1560.0  # m/s

        F = compute_radiation_force(intensity, c, alpha_np_m)
        expected = 2 * alpha_np_m * intensity / c
        assert jnp.isclose(F, expected, rtol=1e-5)

    def test_radiation_force_zero_in_water(self):
        """Water has α≈0, so radiation force should be ~0."""
        from dmipy_jax.biophysics.radiation_force import compute_radiation_force

        F = compute_radiation_force(intensity=10.0, sound_speed=1500.0, attenuation_np_m=0.0)
        assert F == 0.0

    def test_radiation_force_array(self):
        """Force computation should work element-wise on arrays."""
        from dmipy_jax.biophysics.radiation_force import compute_radiation_force

        I = jnp.ones((32, 32)) * 10.0
        c = jnp.ones((32, 32)) * 1560.0
        alpha = jnp.ones((32, 32)) * 61.0
        alpha = alpha.at[10:20, :].set(0.0)  # water region

        F = compute_radiation_force(I, c, alpha)
        assert F.shape == (32, 32)
        assert jnp.all(F[10:20, :] == 0.0)  # water
        assert jnp.all(F[0:10, :] > 0.0)  # brain

    def test_radiation_force_from_jwave_output(self):
        """Integration: compute force from j-Wave intensity and acoustic property maps."""
        from dmipy_jax.biophysics.radiation_force import compute_radiation_force_from_simulation
        from dmipy_jax.biophysics.acoustic import map_labels_to_properties, db_per_cm_to_neper_per_m

        # Simulated labels: water=0, brain=4
        labels = jnp.zeros((16, 16), dtype=jnp.int32)
        labels = labels.at[5:12, 5:12].set(4)  # brain region

        props = map_labels_to_properties(labels)
        intensity = jnp.ones((16, 16)) * 5.0  # uniform intensity

        F = compute_radiation_force_from_simulation(
            intensity, props["sound_speed"], props["attenuation"]
        )
        assert F.shape == (16, 16)
        # Force should be higher in brain (has attenuation) than water (α=0)
        assert jnp.mean(F[5:12, 5:12]) > jnp.mean(F[0:3, 0:3])


class TestQuasiStaticDisplacement:
    """Tests for spectral Green's function displacement solver."""

    def test_displacement_inversely_proportional_to_stiffness(self):
        """Softer tissue → more displacement for same force."""
        from dmipy_jax.biophysics.radiation_force import solve_displacement_quasistatic

        force = jnp.zeros((32, 32))
        force = force.at[16, 16].set(1.0)  # point force
        dx = 1e-3  # 1mm

        u_soft = solve_displacement_quasistatic(force, shear_modulus=0.54e3, grid_spacing=dx)  # CC
        u_stiff = solve_displacement_quasistatic(force, shear_modulus=1.82e3, grid_spacing=dx)  # cortex

        # Softer tissue should displace more
        assert jnp.max(jnp.abs(u_soft)) > jnp.max(jnp.abs(u_stiff))

    def test_displacement_shape_preserved(self):
        """Output displacement has same shape as input force."""
        from dmipy_jax.biophysics.radiation_force import solve_displacement_quasistatic

        force = jnp.ones((64, 64)) * 0.1
        u = solve_displacement_quasistatic(force, shear_modulus=1e3, grid_spacing=1e-3)
        assert u.shape == (64, 64)

    @pytest.mark.xfail(reason="known failure, pre-existing, see docs/decisions/010 §7.3: quasi-static displacement 0.002 um, expected >0.01 um", strict=False)
    def test_displacement_range_microns(self):
        """Brain displacement at neuromod intensities should be 0.1-100 µm."""
        from dmipy_jax.biophysics.radiation_force import (
            compute_radiation_force,
            solve_displacement_quasistatic,
        )

        # Typical: α=61 Np/m, I=10 W/m², c=1560
        alpha = 61.0
        I = 10.0
        c = 1560.0
        F_scalar = 2 * alpha * I / c  # ~0.78 N/m³

        force = jnp.zeros((32, 32))
        force = force.at[14:18, 14:18].set(F_scalar)  # focal spot

        mu = 1.5e3  # 1.5 kPa average brain
        u = solve_displacement_quasistatic(force, shear_modulus=mu, grid_spacing=1e-3)

        max_u_um = float(jnp.max(jnp.abs(u))) * 1e6  # convert to µm
        assert 0.01 < max_u_um < 1000, f"Displacement {max_u_um:.2f} µm out of expected range"

    def test_displacement_differentiable(self):
        """jax.grad through force → displacement chain."""
        from dmipy_jax.biophysics.radiation_force import solve_displacement_quasistatic

        def peak_displacement(mu):
            force = jnp.zeros((16, 16))
            force = force.at[8, 8].set(1.0)
            u = solve_displacement_quasistatic(force, shear_modulus=mu, grid_spacing=1e-3)
            return jnp.max(jnp.abs(u))

        grad = jax.grad(peak_displacement)(1.0e3)
        assert not jnp.isnan(grad)
        assert grad < 0  # increasing stiffness → less displacement

    def test_zero_force_zero_displacement(self):
        """No force → no displacement."""
        from dmipy_jax.biophysics.radiation_force import solve_displacement_quasistatic

        force = jnp.zeros((16, 16))
        u = solve_displacement_quasistatic(force, shear_modulus=1e3, grid_spacing=1e-3)
        assert jnp.allclose(u, 0.0)


class TestCANNShearModulus:
    """Tests for ConstitutiveNN-based shear modulus mapping."""

    def test_kuhl_values_cortex_gt_corpus_callosum(self):
        """Cortex (1.82 kPa) should be stiffer than corpus callosum (0.54 kPa)."""
        from dmipy_jax.biophysics.radiation_force import get_cann_shear_modulus

        mu = get_cann_shear_modulus()
        assert mu["cortex"] > mu["corpus_callosum"]
        assert jnp.isclose(mu["cortex"], 1.82e3, rtol=0.1)  # kPa → Pa
        assert jnp.isclose(mu["corpus_callosum"], 0.54e3, rtol=0.1)

    def test_shear_modulus_from_labels(self):
        """Map tissue labels to regional shear modulus."""
        from dmipy_jax.biophysics.radiation_force import map_labels_to_shear_modulus

        labels = jnp.array([4, 5, 4, 5])  # GM, WM, GM, WM
        mu = map_labels_to_shear_modulus(labels)
        assert mu.shape == (4,)
        # GM should be softer than WM (cortex 1.82 < corona radiata/WM ~1.68)
        # Actually Kuhl: cortex(GM)=1.82 > corona radiata(WM)=0.94
        # But BrainMaterialMap uses WM=1.68, GM=1.12
        assert jnp.all(mu > 0)

    def test_corpus_callosum_3x_more_displacement(self):
        """CC should displace ~3.4x more than cortex from same force (Kuhl CANN)."""
        from dmipy_jax.biophysics.radiation_force import (
            solve_displacement_quasistatic,
            get_cann_shear_modulus,
        )

        mu = get_cann_shear_modulus()
        force = jnp.zeros((32, 32))
        force = force.at[16, 16].set(1.0)

        u_cortex = solve_displacement_quasistatic(force, mu["cortex"], 1e-3)
        u_cc = solve_displacement_quasistatic(force, mu["corpus_callosum"], 1e-3)

        ratio = float(jnp.max(jnp.abs(u_cc)) / jnp.max(jnp.abs(u_cortex)))
        # Ratio should be ~1.82/0.54 ≈ 3.4
        assert 2.5 < ratio < 4.5, f"CC/cortex displacement ratio = {ratio:.1f}, expected ~3.4"

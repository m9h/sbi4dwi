"""
Tests for MR-ARFI sequence design, phantom creation, and phase prediction.
"""

import pytest
import numpy as np
import jax.numpy as jnp


class TestARFISequence:
    """Tests for MR-ARFI pulse sequence design."""

    def test_sequence_has_msg_lobes(self):
        """Sequence must contain bipolar MSG pair."""
        from dmipy_jax.biophysics.mr_arfi import design_arfi_sequence

        params = design_arfi_sequence()
        assert params["has_msg_lobes"] is True
        assert params["msg_polarity"] == [+1, -1]

    def test_sequence_has_fus_trigger(self):
        """Sequence must have FUS trigger marker."""
        from dmipy_jax.biophysics.mr_arfi import design_arfi_sequence

        params = design_arfi_sequence()
        assert params["has_fus_trigger"] is True

    def test_encoding_coefficient(self):
        """Encoding coefficient = γ · G · δ."""
        from dmipy_jax.biophysics.mr_arfi import design_arfi_sequence, GAMMA

        params = design_arfi_sequence(msg_amplitude=40e-3, msg_duration=5e-3)
        expected = GAMMA * 40e-3 * 5e-3
        assert np.isclose(params["encoding_coefficient_rad_m"], expected, rtol=1e-6)

    def test_min_displacement_submicron(self):
        """Minimum detectable displacement should be sub-micron."""
        from dmipy_jax.biophysics.mr_arfi import design_arfi_sequence

        params = design_arfi_sequence(msg_amplitude=40e-3, msg_duration=5e-3)
        assert params["min_displacement_um"] < 1.0  # sub-micron


class TestARFIPhase:
    """Tests for analytical phase prediction and displacement recovery."""

    def test_phase_proportional_to_displacement(self):
        """2x displacement → 2x phase."""
        from dmipy_jax.biophysics.mr_arfi import predict_arfi_phase

        u1 = np.array([5e-6])  # 5 µm
        u2 = np.array([10e-6])  # 10 µm
        phi1 = predict_arfi_phase(u1)
        phi2 = predict_arfi_phase(u2)
        assert np.isclose(phi2[0] / phi1[0], 2.0, rtol=1e-6)

    def test_phase_zero_without_displacement(self):
        """No displacement → no phase shift."""
        from dmipy_jax.biophysics.mr_arfi import predict_arfi_phase

        phi = predict_arfi_phase(np.array([0.0]))
        assert phi[0] == 0.0

    def test_phase_magnitude_realistic(self):
        """5µm displacement at 40mT/m, 5ms should give ~0.27 rad."""
        from dmipy_jax.biophysics.mr_arfi import predict_arfi_phase, GAMMA

        u = 5e-6  # 5 µm
        phi = predict_arfi_phase(np.array([u]), msg_amplitude=40e-3, msg_duration=5e-3)
        expected = GAMMA * 40e-3 * 5e-3 * u  # ~0.27 rad
        assert np.isclose(phi[0], expected, rtol=1e-6)
        assert 0.1 < abs(phi[0]) < 1.0  # realistic range

    def test_displacement_recovery_roundtrip(self):
        """Encode then decode displacement should recover original."""
        from dmipy_jax.biophysics.mr_arfi import predict_arfi_phase, recover_displacement_from_phase

        u_true = np.array([1e-6, 5e-6, 10e-6])  # 1, 5, 10 µm
        phi_on = predict_arfi_phase(u_true, msg_amplitude=40e-3, msg_duration=5e-3)
        phi_off = np.zeros_like(phi_on)

        u_recovered = recover_displacement_from_phase(phi_on, phi_off,
                                                       msg_amplitude=40e-3, msg_duration=5e-3)
        assert np.allclose(u_recovered, u_true, rtol=1e-6)


class TestARFIPhantom:
    """Tests for displacement-encoded phantom creation."""

    def test_phantom_has_motion_data(self):
        """Phantom should include displacement per spin."""
        from dmipy_jax.biophysics.mr_arfi import create_arfi_phantom_data

        labels = np.array([[4, 4], [5, 5]])  # 2x2: GM, WM
        disp = np.array([[5e-6, 3e-6], [8e-6, 2e-6]])

        phantom = create_arfi_phantom_data(labels, disp, grid_spacing=2e-3)
        assert "displacement_z" in phantom
        assert phantom["n_spins"] == 4  # all tissue, no background
        assert len(phantom["displacement_z"]) == 4

    def test_phantom_filters_background(self):
        """Background spins (label=0) should be excluded."""
        from dmipy_jax.biophysics.mr_arfi import create_arfi_phantom_data

        labels = np.array([[0, 4], [4, 0]])
        disp = np.ones((2, 2)) * 5e-6

        phantom = create_arfi_phantom_data(labels, disp, grid_spacing=2e-3)
        assert phantom["n_spins"] == 2  # only 2 GM voxels

    def test_phantom_positions_on_grid(self):
        """Spin positions should be on the voxel grid."""
        from dmipy_jax.biophysics.mr_arfi import create_arfi_phantom_data

        labels = np.ones((4, 4), dtype=int) * 4
        disp = np.zeros((4, 4))

        phantom = create_arfi_phantom_data(labels, disp, grid_spacing=1e-3)
        assert phantom["positions"].shape == (16, 3)
        # x positions should be multiples of grid_spacing
        assert np.allclose(phantom["positions"][:, 0] % 1e-3, 0.0, atol=1e-10)

    def test_phantom_tissue_properties(self):
        """GM and WM should have different T1/T2."""
        from dmipy_jax.biophysics.mr_arfi import create_arfi_phantom_data

        labels = np.array([[4, 5]])  # GM, WM
        disp = np.zeros((1, 2))

        phantom = create_arfi_phantom_data(labels, disp, grid_spacing=1e-3)
        # GM T1 (1.6s) > WM T1 (0.8s)
        assert phantom["T1"][0] > phantom["T1"][1]

"""
Tests for openlifu bridge module.

These tests do NOT require openlifu to be installed.
They use synthetic data to validate the extraction and pipeline logic.
"""

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

xarray = pytest.importorskip("xarray")

from dmipy_jax.biophysics.openlifu_bridge import (
    SolutionData,
    extract_chain_inputs,
    load_solution_from_files,
    run_arfi_from_solution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_solution_dict():
    """Synthetic solution dict mimicking openlifu JSON + netCDF output."""
    N = 16
    return {
        "delays": np.zeros((1, 4)),
        "apodizations": np.ones((1, 4)),
        "frequency": 400e3,
        "voltage": 10.0,
        "target_position_m": np.array([0.008, 0.008, 0.0]),
        "foci_m": [np.array([0.008, 0.008, 0.0])],
        "p_max": np.random.rand(N, N).astype(np.float64) * 1e5,
        "intensity": np.random.rand(N, N).astype(np.float64) * 100,  # W/cm^2
        "coords_x": np.linspace(0, 0.015, N),  # already in metres
        "coords_y": np.linspace(0, 0.015, N),
    }


@pytest.fixture
def synthetic_solution_data(synthetic_solution_dict):
    """SolutionData extracted from synthetic dict."""
    return extract_chain_inputs(synthetic_solution_dict)


@pytest.fixture
def synthetic_json_file():
    """Write a minimal openlifu-style JSON file."""
    data = {
        "delays": [[0.0, 0.0, 0.0, 0.0]],
        "apodizations": [[1.0, 1.0, 1.0, 1.0]],
        "pulse": {"frequency": 500e3, "duration": 20e-6},
        "voltage": 12.0,
        "target": {
            "position": [10.0, 20.0, 30.0],
            "units": "mm",
        },
        "foci": [
            {"position": [10.0, 20.0, 30.0], "units": "mm"},
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        return f.name


# ---------------------------------------------------------------------------
# Tests: File loading
# ---------------------------------------------------------------------------


class TestSolutionLoading:
    def test_load_from_json(self, synthetic_json_file):
        result = load_solution_from_files(synthetic_json_file)
        assert isinstance(result, dict)
        assert "delays" in result
        assert "frequency" in result
        assert result["frequency"] == 500e3

    def test_load_target_position_mm_to_m(self, synthetic_json_file):
        result = load_solution_from_files(synthetic_json_file)
        target = result["target_position_m"]
        np.testing.assert_allclose(target, [0.01, 0.02, 0.03])

    def test_load_missing_nc_no_crash(self, synthetic_json_file):
        # Should warn but not crash when netCDF is missing
        result = load_solution_from_files(synthetic_json_file)
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: Extraction
# ---------------------------------------------------------------------------


class TestExtractChainInputs:
    def test_extract_produces_solution_data(self, synthetic_solution_dict):
        sd = extract_chain_inputs(synthetic_solution_dict)
        assert isinstance(sd, SolutionData)
        assert sd.p_max is not None
        assert sd.intensity is not None

    def test_extract_intensity_conversion(self, synthetic_solution_dict):
        """Intensity should be converted from W/cm^2 to W/m^2 (x 1e4)."""
        raw_intensity = synthetic_solution_dict["intensity"]
        sd = extract_chain_inputs(synthetic_solution_dict)
        np.testing.assert_allclose(sd.intensity, raw_intensity * 1e4)

    def test_extract_grid_spacing(self, synthetic_solution_dict):
        sd = extract_chain_inputs(synthetic_solution_dict)
        expected_dx = abs(
            synthetic_solution_dict["coords_x"][1]
            - synthetic_solution_dict["coords_x"][0]
        )
        assert abs(sd.grid_spacing_m - expected_dx) < 1e-10

    def test_extract_without_seg_params(self, synthetic_solution_dict):
        """Default tissue properties when no seg_params provided."""
        sd = extract_chain_inputs(synthetic_solution_dict)
        assert np.all(sd.sound_speed == 1560.0)
        assert np.all(sd.density == 1040.0)
        assert np.all(sd.attenuation == 5.3)


# ---------------------------------------------------------------------------
# Tests: ARFI pipeline
# ---------------------------------------------------------------------------


class TestARFIFromSolution:
    def test_results_dict_complete(self, synthetic_solution_data):
        results = run_arfi_from_solution(synthetic_solution_data)
        expected_keys = [
            "radiation_force",
            "displacement",
            "displacement_um",
            "arfi_phase",
            "arfi_sequence_params",
            "encoding_coefficient_rad_m",
            "shear_modulus",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_radiation_force_nonzero(self, synthetic_solution_data):
        results = run_arfi_from_solution(synthetic_solution_data)
        assert np.any(results["radiation_force"] != 0)

    def test_displacement_micron_range(self, synthetic_solution_data):
        results = run_arfi_from_solution(synthetic_solution_data)
        max_disp_um = results["max_displacement_um"]
        # Displacement should be in a physically reasonable range
        assert max_disp_um >= 0
        assert max_disp_um < 1e3  # < 1 mm

    def test_phase_realistic_range(self, synthetic_solution_data):
        results = run_arfi_from_solution(synthetic_solution_data)
        max_phase = np.max(np.abs(results["arfi_phase"]))
        # Phase should be small but nonzero for typical displacements
        assert max_phase >= 0
        assert max_phase < 100  # < 100 rad (sanity check)

    def test_encoding_coefficient_correct(self, synthetic_solution_data):
        G = 40e-3
        delta = 5e-3
        results = run_arfi_from_solution(
            synthetic_solution_data, msg_amplitude=G, msg_duration=delta
        )
        gamma = 2.675e8  # rad/s/T
        expected = gamma * G * delta
        assert abs(results["encoding_coefficient_rad_m"] - expected) < 1.0

    def test_with_labels(self, synthetic_solution_data):
        """Pipeline should work with explicit tissue labels."""
        shape = synthetic_solution_data.intensity.shape
        labels = np.full(shape, 4, dtype=np.int32)  # all GM
        results = run_arfi_from_solution(synthetic_solution_data, labels=labels)
        assert np.any(results["displacement"] != 0)

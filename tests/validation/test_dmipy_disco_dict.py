"""
Tests for the dmipy-JAX dictionary matcher on DiSCo.

This is the dmipy-JAX-vs-FORCE head-to-head on the scalar microstructure
recovery task (paper-aligned §16 metric). The §17-§19 connectivity work
applies to FORCE through dipy; dmipy-JAX has its own DictionaryMatcher
in `dmipy_jax/library/matcher.py` which has never been pointed at DiSCo
before this commit.

Tests pin:
  - build_disco_tuned_two_stick_simulator: 6-param simulator with
    DiSCo-aligned diffusivity priors per FORCE paper §3.2.
  - fit_dmipy_dict_on_disco: end-to-end smoke fit + library round-trip.
"""

from pathlib import Path

import numpy as np
import pytest


_helper = pytest.importorskip("validation.lib.dmipy_disco_dict")
DISCO_ROOT = Path.home() / ".dipy" / "disco" / "disco_1"


@pytest.fixture(scope="module")
def disco_ready():
    if not DISCO_ROOT.exists():
        pytest.skip(f"DiSCo not at {DISCO_ROOT}")


# --------------------------------------------------------------------------- #
# 1. DiSCo-tuned simulator
# --------------------------------------------------------------------------- #

class TestBuildDiSCoTunedSimulator:
    def test_parameter_ranges_match_paper(self):
        """Parallel diffusivity range must match FORCE paper §3.2:
        Uniform(0.54, 0.66) × 10⁻³ mm²/s = (0.54e-9, 0.66e-9) m²/s in SI."""
        from dmipy_jax.acquisition import JaxAcquisition
        import jax.numpy as jnp
        acq = JaxAcquisition(
            bvalues=jnp.array([0.0, 1.9e9]),
            gradient_directions=jnp.array([[1, 0, 0], [0, 0, 1]], dtype=float),
        )
        sim = _helper.build_disco_tuned_two_stick_simulator(acq)
        d_par_lo, d_par_hi = sim.parameter_ranges["d_par"]
        assert abs(d_par_lo - 0.54e-9) < 1e-12
        assert abs(d_par_hi - 0.66e-9) < 1e-12

    def test_f_iso_spans_extracellular_range(self):
        """DiSCo has sparse strands → GT Intra Volume Fraction mean ≈ 0.18,
        so per-voxel extracellular volume is roughly 0.4–0.95. Our 2-stick
        model has no extra-axonal zeppelin (FORCE does), so the `f_iso`
        parameter doubles as the extracellular water proxy and must span
        a wide range to recover GT NDI values across the dataset.

        (Original version of this test asserted ``f_iso_max ≤ 0.05`` based
        on a misreading of paper §3.2 "isotropic compartment was disabled".
        That phrase refers to FORCE's GM/CSF ball compartments — FORCE
        keeps its extra-axonal zeppelin. Our 2-stick model has nothing
        analogous, so we widen f_iso to cover the extracellular range
        directly.)"""
        from dmipy_jax.acquisition import JaxAcquisition
        import jax.numpy as jnp
        acq = JaxAcquisition(
            bvalues=jnp.array([0.0, 1.9e9]),
            gradient_directions=jnp.array([[1, 0, 0], [0, 0, 1]], dtype=float),
        )
        sim = _helper.build_disco_tuned_two_stick_simulator(acq)
        _, f_iso_max = sim.parameter_ranges["f_iso"]
        assert f_iso_max >= 0.8, (
            f"f_iso_max={f_iso_max} too narrow to recover DiSCo's GT "
            "Intra Volume Fraction range; 2-stick model needs f_iso as "
            "the extracellular proxy."
        )


# --------------------------------------------------------------------------- #
# 2. End-to-end smoke fit
# --------------------------------------------------------------------------- #

class TestFitDmipyDictOnDiSCo:
    def test_smoke_fit_recovers_ndi_per_voxel(self, disco_ready):
        """Run the dmipy-JAX dictionary fit on a small DiSCo ROI subset
        and verify the returned NDI map has consistent shape + range."""
        result = _helper.fit_dmipy_dict_on_disco(
            subject=1, snr=30, library_size=10_000,
            _smoke_roi_subset=(1, 2, 3),
        )
        for k in ("ndi", "fa_proxy", "mask"):
            assert k in result
        mask = result["mask"]
        ndi = result["ndi"]
        assert ndi.shape == mask.shape
        ndi_in = ndi[mask]
        assert (ndi_in >= 0).all() and (ndi_in <= 1.0 + 1e-3).all()


# --------------------------------------------------------------------------- #
# 3. 3D-orientation simulator (Option B2)
# --------------------------------------------------------------------------- #

class TestBuild3DTwoStickSimulator:
    def test_parameter_layout(self):
        """3D simulator has 8 params: d_par, θ1, φ1, θ2, φ2, ODI, f1, f_iso."""
        from dmipy_jax.acquisition import JaxAcquisition
        import jax.numpy as jnp
        acq = JaxAcquisition(
            bvalues=jnp.array([0.0, 1.9e9]),
            gradient_directions=jnp.array([[1, 0, 0], [0, 0, 1]], dtype=float),
        )
        sim = _helper.build_disco_tuned_3d_two_stick_simulator(acq)
        assert sim.parameter_names == [
            "d_par", "theta1", "phi1", "theta2", "phi2", "odi", "f1", "f_iso",
        ]

    def test_phi_spans_full_azimuth(self):
        """phi1/phi2 must cover [0, 2π] for full sphere coverage."""
        from dmipy_jax.acquisition import JaxAcquisition
        import jax.numpy as jnp
        import math
        acq = JaxAcquisition(
            bvalues=jnp.array([0.0, 1.9e9]),
            gradient_directions=jnp.array([[1, 0, 0], [0, 0, 1]], dtype=float),
        )
        sim = _helper.build_disco_tuned_3d_two_stick_simulator(acq)
        for axis in ("phi1", "phi2"):
            lo, hi = sim.parameter_ranges[axis]
            assert abs(lo - 0.0) < 1e-9
            assert abs(hi - 2 * math.pi) < 1e-6


# --------------------------------------------------------------------------- #
# 4. params → PeaksAndMetrics adapter
# --------------------------------------------------------------------------- #

class TestDmipyParamsToPAM:
    def test_unit_norm_peaks_from_2_fibre_params(self):
        """Convert a single voxel's [d_par, θ1, φ1, θ2, φ2, ODI, f1, f_iso]
        to a PAM-compatible (peak_dirs, peak_values, peak_indices) triple
        and assert peak directions are unit norm."""
        import numpy as np
        from dipy.data import default_sphere

        params = np.array([0.6e-9, np.pi / 2, 0.0, np.pi / 2, np.pi / 2,
                           0.1, 0.5, 0.1])  # 2 orthogonal in-plane fibres
        peak_dirs, peak_values, peak_indices = _helper.dmipy_params_to_pam_single(
            params, sphere=default_sphere,
        )
        # 5 slots in dipy convention, but only first 2 should be populated
        assert peak_dirs.shape == (5, 3)
        assert peak_values.shape == (5,)
        assert peak_indices.shape == (5,)
        # Norms: first 2 should be ~1, rest 0
        norms = np.linalg.norm(peak_dirs, axis=-1)
        assert abs(norms[0] - 1.0) < 1e-6
        assert abs(norms[1] - 1.0) < 1e-6
        assert (norms[2:] == 0).all()

    def test_in_plane_fibres_at_z_axis(self):
        """θ=0 → mu = +z regardless of φ."""
        import numpy as np
        from dipy.data import default_sphere

        params = np.array([0.6e-9, 0.0, 0.0, 0.0, 1.5, 0.1, 0.5, 0.1])
        peak_dirs, _, _ = _helper.dmipy_params_to_pam_single(
            params, sphere=default_sphere,
        )
        # Both peaks should point along z (or -z under antipodal)
        for k in range(2):
            assert abs(abs(peak_dirs[k, 2]) - 1.0) < 1e-6

    def test_phi_pi_over_2_gives_y_axis(self):
        """θ=π/2, φ=π/2 → mu = +y. Verifies 3D parameterisation actually
        uses the φ parameter (the §20 bug was φ-collapsed-to-zero)."""
        import math
        import numpy as np
        from dipy.data import default_sphere

        params = np.array([0.6e-9, math.pi / 2, math.pi / 2,
                           math.pi / 2, 0.0, 0.1, 0.5, 0.1])
        peak_dirs, _, _ = _helper.dmipy_params_to_pam_single(
            params, sphere=default_sphere,
        )
        # First peak should be along +y
        assert abs(peak_dirs[0, 1] - 1.0) < 1e-3, f"got {peak_dirs[0]}"
        # Second peak should be along +x (φ=0, θ=π/2)
        assert abs(peak_dirs[1, 0] - 1.0) < 1e-3, f"got {peak_dirs[1]}"


# --------------------------------------------------------------------------- #
# 5. 3D-orientation stick + zeppelin simulator (§22)
# --------------------------------------------------------------------------- #

class TestBuild3DStickZeppelinSimulator:
    """§22 extension: adds an extra-axonal zeppelin per fibre with
    NODDI-style tortuosity coupling (lambda_perp = lambda_par·(1-v_ic))."""

    def _acq(self):
        from dmipy_jax.acquisition import JaxAcquisition
        import jax.numpy as jnp
        return JaxAcquisition(
            bvalues=jnp.array([0.0, 1.9e9]),
            gradient_directions=jnp.array([[1, 0, 0], [0, 0, 1]], dtype=float),
        )

    def test_parameter_layout_adds_v_ic(self):
        """9 params: d_par, θ1, φ1, θ2, φ2, ODI, v_ic, f1, f_iso."""
        sim = _helper.build_disco_tuned_3d_stick_zeppelin_simulator(self._acq())
        assert sim.parameter_names == [
            "d_par", "theta1", "phi1", "theta2", "phi2",
            "odi", "v_ic", "f1", "f_iso",
        ]

    def test_v_ic_spans_physiological_range(self):
        """v_ic must span a wide range: NODDI papers report 0.3-0.7 in
        WM, and DiSCo strands may go higher. Range should at least
        cover (0.3, 0.95)."""
        sim = _helper.build_disco_tuned_3d_stick_zeppelin_simulator(self._acq())
        v_ic_lo, v_ic_hi = sim.parameter_ranges["v_ic"]
        assert v_ic_lo <= 0.3 and v_ic_hi >= 0.9, \
            f"v_ic range ({v_ic_lo}, {v_ic_hi}) too narrow"

    def test_v_ic_equal_one_reduces_to_stick_only(self):
        """When v_ic = 1, the zeppelin compartment has zero weight and
        the signal must match the §21 3D-stick-only simulator's output
        for the same parameters. This validates the new compartment
        is layered additively, not replacing the stick."""
        import numpy as np
        import jax.numpy as jnp
        acq = self._acq()
        sim_zs = _helper.build_disco_tuned_3d_stick_zeppelin_simulator(acq)
        sim_s = _helper.build_disco_tuned_3d_two_stick_simulator(acq)
        # Stick-only params: d_par, θ1, φ1, θ2, φ2, ODI, f1, f_iso
        p_stick = jnp.array([0.6e-9, 1.0, 0.5, 0.8, 1.0, 0.1, 0.5, 0.1])
        # Stick+zeppelin params: d_par, θ1, φ1, θ2, φ2, ODI, v_ic, f1, f_iso
        # with v_ic = 1.0
        p_zs = jnp.array([0.6e-9, 1.0, 0.5, 0.8, 1.0, 0.1, 1.0, 0.5, 0.1])
        s_stick = sim_s.forward_fn(p_stick, acq)
        s_zs = sim_zs.forward_fn(p_zs, acq)
        np.testing.assert_allclose(np.asarray(s_zs), np.asarray(s_stick),
                                    atol=1e-6, rtol=1e-4)

    def test_extra_axonal_lowers_attenuation_vs_pure_stick(self):
        """With v_ic = 0.5 (half intra, half extra), the perpendicular
        diffusivity is d_par/2 — non-zero — so signal perpendicular to
        the fibre should *decay more* than with pure sticks (where
        d_perp = 0). Validates that the zeppelin compartment is doing
        the perpendicular-attenuation work."""
        import jax.numpy as jnp
        import numpy as np
        from dmipy_jax.acquisition import JaxAcquisition
        # Two measurements: b=0 and one b=1900 along +x (perpendicular
        # to a +z fibre).
        acq = JaxAcquisition(
            bvalues=jnp.array([0.0, 1.9e9]),
            gradient_directions=jnp.array([[1, 0, 0], [1, 0, 0]], dtype=float),
        )
        sim_zs = _helper.build_disco_tuned_3d_stick_zeppelin_simulator(acq)
        sim_s = _helper.build_disco_tuned_3d_two_stick_simulator(acq)
        # Single fibre along +z (θ=0); f1 dominates.
        # ODI low → sharp fibre.
        p_s = jnp.array([0.6e-9, 0.0, 0.0, 0.0, 0.0, 0.01, 0.99, 0.0])
        p_zs = jnp.array([0.6e-9, 0.0, 0.0, 0.0, 0.0, 0.01, 0.5, 0.99, 0.0])
        s_stick = np.asarray(sim_s.forward_fn(p_s, acq))[1]
        s_zs = np.asarray(sim_zs.forward_fn(p_zs, acq))[1]
        # The zeppelin compartment with v_ic=0.5 has d_perp = 0.3e-9, so
        # the perp-to-fibre signal must decay below the stick-only value.
        assert s_zs < s_stick, (
            f"zeppelin must add perpendicular attenuation: "
            f"s_zs={s_zs} should be < s_stick={s_stick}"
        )

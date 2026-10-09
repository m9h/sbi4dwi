"""
Tests for acoustic tissue property mapping.

RED phase: These tests define the contract for tissue label → acoustic property
conversion, HU-based continuous mapping, and unit conversions.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Phase 1.1: Tissue Lookup Table + Label Mapping
# ---------------------------------------------------------------------------

class TestTissueAcousticProperties:
    """Tests for the TISSUE_ACOUSTIC_PROPERTIES table."""

    def test_tissue_table_has_required_tissues(self):
        """All five SCI head model tissues must be present."""
        from dmipy_jax.biophysics.acoustic import TISSUE_ACOUSTIC_PROPERTIES

        required = {1, 2, 3, 4, 5}  # scalp, skull, CSF, GM, WM
        assert required.issubset(set(TISSUE_ACOUSTIC_PROPERTIES.keys()))

    def test_tissue_table_has_background(self):
        """Background label 0 (air/water) must be present."""
        from dmipy_jax.biophysics.acoustic import TISSUE_ACOUSTIC_PROPERTIES

        assert 0 in TISSUE_ACOUSTIC_PROPERTIES

    def test_tissue_table_has_required_params(self):
        """Each tissue must define all five acoustic parameters."""
        from dmipy_jax.biophysics.acoustic import TISSUE_ACOUSTIC_PROPERTIES

        params = {
            "sound_speed",
            "density",
            "attenuation",
            "specific_heat",
            "thermal_conductivity",
        }
        for label, props in TISSUE_ACOUSTIC_PROPERTIES.items():
            assert params.issubset(set(props.keys())), (
                f"Tissue label {label} missing params: "
                f"{params - set(props.keys())}"
            )

    def test_tissue_table_physical_ranges(self):
        """Sound speed in [300, 5000] m/s, density in [1, 2500] kg/m3."""
        from dmipy_jax.biophysics.acoustic import TISSUE_ACOUSTIC_PROPERTIES

        for label, props in TISSUE_ACOUSTIC_PROPERTIES.items():
            assert 300 <= props["sound_speed"] <= 5000, (
                f"Tissue {label}: c={props['sound_speed']} out of range"
            )
            assert 1 <= props["density"] <= 2500, (
                f"Tissue {label}: rho={props['density']} out of range"
            )
            assert props["attenuation"] >= 0, (
                f"Tissue {label}: alpha={props['attenuation']} negative"
            )

    def test_skull_sound_speed_above_3000(self):
        """Cortical bone sound speed should be well above soft tissue."""
        from dmipy_jax.biophysics.acoustic import TISSUE_ACOUSTIC_PROPERTIES

        assert TISSUE_ACOUSTIC_PROPERTIES[2]["sound_speed"] > 3000

    def test_csf_matches_water(self):
        """CSF acoustic properties should be close to water."""
        from dmipy_jax.biophysics.acoustic import TISSUE_ACOUSTIC_PROPERTIES

        csf = TISSUE_ACOUSTIC_PROPERTIES[3]
        assert abs(csf["sound_speed"] - 1500) < 50
        assert abs(csf["density"] - 1000) < 50


class TestMapLabelsToProperties:
    """Tests for map_labels_to_properties()."""

    def test_map_labels_to_sound_speed(self):
        """Integer label array -> sound speed array with correct values."""
        from dmipy_jax.biophysics.acoustic import (
            TISSUE_ACOUSTIC_PROPERTIES,
            map_labels_to_properties,
        )

        labels = jnp.array([[[1, 2], [3, 4]]])  # (1, 2, 2)
        props = map_labels_to_properties(labels)

        assert "sound_speed" in props
        # Scalp (1)
        assert jnp.isclose(
            props["sound_speed"][0, 0, 0],
            TISSUE_ACOUSTIC_PROPERTIES[1]["sound_speed"],
            atol=1.0,
        )
        # Skull (2)
        assert jnp.isclose(
            props["sound_speed"][0, 0, 1],
            TISSUE_ACOUSTIC_PROPERTIES[2]["sound_speed"],
            atol=1.0,
        )
        # CSF (3)
        assert jnp.isclose(
            props["sound_speed"][0, 1, 0],
            TISSUE_ACOUSTIC_PROPERTIES[3]["sound_speed"],
            atol=1.0,
        )
        # GM (4)
        assert jnp.isclose(
            props["sound_speed"][0, 1, 1],
            TISSUE_ACOUSTIC_PROPERTIES[4]["sound_speed"],
            atol=1.0,
        )

    def test_map_labels_preserves_shape(self):
        """Output arrays must have same spatial shape as input labels."""
        from dmipy_jax.biophysics.acoustic import map_labels_to_properties

        labels = jnp.ones((32, 32, 32), dtype=jnp.int32)
        props = map_labels_to_properties(labels)
        for key in props:
            assert props[key].shape == (32, 32, 32), (
                f"{key} shape {props[key].shape} != (32, 32, 32)"
            )

    def test_map_labels_returns_all_five_params(self):
        """All five acoustic parameters must be in the output dict."""
        from dmipy_jax.biophysics.acoustic import map_labels_to_properties

        labels = jnp.zeros((4, 4, 4), dtype=jnp.int32)
        props = map_labels_to_properties(labels)
        expected = {
            "sound_speed",
            "density",
            "attenuation",
            "specific_heat",
            "thermal_conductivity",
        }
        assert expected == set(props.keys())

    def test_map_labels_background_is_water(self):
        """Label 0 (background) should map to water-like properties."""
        from dmipy_jax.biophysics.acoustic import map_labels_to_properties

        labels = jnp.zeros((2, 2, 2), dtype=jnp.int32)
        props = map_labels_to_properties(labels)
        # Water: c ~1500, rho ~1000
        assert jnp.allclose(props["sound_speed"], 1500.0, atol=100.0)
        assert jnp.allclose(props["density"], 1000.0, atol=100.0)

    def test_map_labels_output_dtype_float(self):
        """Output arrays should be float, not int."""
        from dmipy_jax.biophysics.acoustic import map_labels_to_properties

        labels = jnp.array([1, 2, 3], dtype=jnp.int32)
        props = map_labels_to_properties(labels)
        for key in props:
            assert jnp.issubdtype(props[key].dtype, jnp.floating)


# ---------------------------------------------------------------------------
# Phase 1.2: HU-Based Continuous Skull Properties
# ---------------------------------------------------------------------------

class TestHUToAcousticProperties:
    """Tests for Hounsfield unit to acoustic property conversion."""

    def test_hu_to_density_range(self):
        """HU 300-2000 should map to density 1000-2200 kg/m3."""
        from dmipy_jax.biophysics.acoustic import hu_to_acoustic_properties

        hu = jnp.linspace(300, 2000, 50)
        props = hu_to_acoustic_properties(hu)
        assert jnp.all(props["density"] >= 1000)
        assert jnp.all(props["density"] <= 2200)

    def test_hu_to_sound_speed_range(self):
        """Bone HU range should produce sound speeds in [1500, 4500] m/s."""
        from dmipy_jax.biophysics.acoustic import hu_to_acoustic_properties

        hu = jnp.linspace(300, 2000, 50)
        props = hu_to_acoustic_properties(hu)
        assert jnp.all(props["sound_speed"] >= 1500)
        assert jnp.all(props["sound_speed"] <= 4500)

    def test_hu_to_sound_speed_monotonic(self):
        """Higher HU (denser bone) -> higher sound speed."""
        from dmipy_jax.biophysics.acoustic import hu_to_acoustic_properties

        hu = jnp.linspace(300, 2000, 50)
        props = hu_to_acoustic_properties(hu)
        diffs = jnp.diff(props["sound_speed"])
        assert jnp.all(diffs >= 0), "Sound speed must be monotonically non-decreasing with HU"

    def test_hu_to_density_monotonic(self):
        """Higher HU -> higher density."""
        from dmipy_jax.biophysics.acoustic import hu_to_acoustic_properties

        hu = jnp.linspace(300, 2000, 50)
        props = hu_to_acoustic_properties(hu)
        diffs = jnp.diff(props["density"])
        assert jnp.all(diffs >= 0), "Density must be monotonically non-decreasing with HU"

    def test_hu_water_baseline(self):
        """HU=0 (water) should give approximately water properties."""
        from dmipy_jax.biophysics.acoustic import hu_to_acoustic_properties

        hu = jnp.array([0.0])
        props = hu_to_acoustic_properties(hu)
        assert jnp.isclose(props["density"][0], 1000.0, atol=50.0)
        assert jnp.isclose(props["sound_speed"][0], 1500.0, atol=100.0)

    def test_hu_returns_all_params(self):
        """Must return sound_speed, density, and attenuation at minimum."""
        from dmipy_jax.biophysics.acoustic import hu_to_acoustic_properties

        hu = jnp.array([500.0, 1000.0, 1500.0])
        props = hu_to_acoustic_properties(hu)
        assert "sound_speed" in props
        assert "density" in props
        assert "attenuation" in props

    def test_hu_pipeline_differentiable(self):
        """End-to-end HU->acoustic must be JAX-differentiable."""
        from dmipy_jax.biophysics.acoustic import hu_to_acoustic_properties

        def pipeline(hu_val):
            props = hu_to_acoustic_properties(hu_val)
            return props["sound_speed"].sum()

        grad_fn = jax.grad(pipeline)
        g = grad_fn(jnp.array([500.0, 1000.0]))
        assert not jnp.any(jnp.isnan(g))
        assert g.shape == (2,)


class TestAttenuationUnitConversion:
    """Tests for attenuation unit conversion utilities."""

    def test_neper_to_db_conversion(self):
        """1 Np/m = 8.686 dB/m."""
        from dmipy_jax.biophysics.acoustic import neper_per_m_to_db_per_cm

        # 1 Np/m = 8.686 dB/m = 0.08686 dB/cm
        result = neper_per_m_to_db_per_cm(1.0)
        assert jnp.isclose(result, 0.08686, atol=0.001)

    def test_cortical_bone_attenuation_conversion(self):
        """NDK cortical bone: 54.553 Np/m/MHz should be ~4.74 dB/cm/MHz."""
        from dmipy_jax.biophysics.acoustic import neper_per_m_to_db_per_cm

        # 54.553 Np/m * 8.686 dB/Np / 100 cm/m = 4.738 dB/cm
        result = neper_per_m_to_db_per_cm(54.553)
        assert jnp.isclose(result, 4.738, atol=0.1)

    def test_db_to_neper_roundtrip(self):
        """Convert Np/m -> dB/cm -> Np/m should be identity."""
        from dmipy_jax.biophysics.acoustic import (
            neper_per_m_to_db_per_cm,
            db_per_cm_to_neper_per_m,
        )

        original = 54.553
        db_cm = neper_per_m_to_db_per_cm(original)
        restored = db_per_cm_to_neper_per_m(db_cm)
        assert jnp.isclose(restored, original, atol=0.01)


# ---------------------------------------------------------------------------
# Integration: PseudoCT -> Acoustic Properties
# ---------------------------------------------------------------------------

class TestPseudoCTAcousticIntegration:
    """Tests for combining PseudoCTMapper with acoustic property mapping."""

    def test_pseudoct_to_acoustic_pipeline(self):
        """MRI intensity -> PseudoCT -> HU -> acoustic properties end-to-end."""
        from dmipy_jax.biophysics.pseudo_ct import PseudoCTMapper
        from dmipy_jax.biophysics.acoustic import hu_to_acoustic_properties

        mapper = PseudoCTMapper(method="plymouth")
        # Dark T1 region (skull-like): intensity ~0.2
        mri_skull = jnp.array([0.2])
        hu_skull = mapper.mri_to_hu(mri_skull)
        props_skull = hu_to_acoustic_properties(hu_skull)

        # Bright T1 region (brain-like): intensity ~0.8
        mri_brain = jnp.array([0.8])
        hu_brain = mapper.mri_to_hu(mri_brain)
        props_brain = hu_to_acoustic_properties(hu_brain)

        # Skull should have higher sound speed than brain
        assert props_skull["sound_speed"][0] > props_brain["sound_speed"][0]
        # Skull should have higher density than brain
        assert props_skull["density"][0] > props_brain["density"][0]

    def test_pseudoct_pipeline_fully_differentiable(self):
        """Full MRI -> HU -> acoustic pipeline must support jax.grad."""
        from dmipy_jax.biophysics.pseudo_ct import PseudoCTMapper
        from dmipy_jax.biophysics.acoustic import hu_to_acoustic_properties

        mapper = PseudoCTMapper(method="plymouth")

        def full_pipeline(intensity):
            hu = mapper.mri_to_hu(intensity)
            props = hu_to_acoustic_properties(hu)
            return props["sound_speed"].sum()

        grad_fn = jax.grad(full_pipeline)
        g = grad_fn(jnp.array([0.3, 0.5, 0.7]))
        assert not jnp.any(jnp.isnan(g))
        assert g.shape == (3,)

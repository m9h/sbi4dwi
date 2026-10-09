"""
Acoustic tissue property mapping for transcranial focused ultrasound simulation.

Maps integer tissue labels (from segmentation) or Hounsfield units (from CT/pseudo-CT)
to acoustic properties (sound speed, density, attenuation) suitable for k-Wave or j-Wave.

References:
    - Aubry et al. (2022). ITRUSST benchmark. JASA 152(2):1003-1019.
    - Connor et al. (2002). Bio-acoustic thermal lensing. JASA.
    - Schneider et al. (2000). HU to density. Phys. Med. Biol. 45:459-478.
    - Mast (2000). Empirical relationships for acoustic properties. JASA 108(3):1023-1029.
    - NDK (AE Studio) material constants.
"""

import jax.numpy as jnp
from typing import Dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEPER_TO_DB = 8.685889638  # 1 Np = 20*log10(e) dB


# ---------------------------------------------------------------------------
# Tissue Acoustic Property Table
# ---------------------------------------------------------------------------

# Label convention matches SCI Institute head model:
#   0 = background (water/air)
#   1 = scalp
#   2 = skull (cortical bone)
#   3 = CSF
#   4 = gray matter
#   5 = white matter
#
# Sound speed (m/s), density (kg/m3), attenuation (dB/cm/MHz),
# specific heat (J/kg/K), thermal conductivity (W/m/K)
#
# Sources: ITRUSST benchmark (Aubry 2022), Connor 2002, NDK constants

TISSUE_ACOUSTIC_PROPERTIES: Dict[int, Dict[str, float]] = {
    0: {
        "sound_speed": 1500.0,
        "density": 1000.0,
        "attenuation": 0.0,
        "specific_heat": 4182.0,
        "thermal_conductivity": 0.598,
    },  # water (background)
    1: {
        "sound_speed": 1610.0,
        "density": 1090.0,
        "attenuation": 3.5,
        "specific_heat": 3391.0,
        "thermal_conductivity": 0.37,
    },  # scalp
    2: {
        "sound_speed": 4080.0,
        "density": 1900.0,
        "attenuation": 4.74,
        "specific_heat": 1100.0,
        "thermal_conductivity": 0.30,
    },  # skull (cortical bone) — 54.553 Np/m/MHz converted
    3: {
        "sound_speed": 1500.0,
        "density": 1000.0,
        "attenuation": 0.0,
        "specific_heat": 4182.0,
        "thermal_conductivity": 0.598,
    },  # CSF
    4: {
        "sound_speed": 1560.0,
        "density": 1040.0,
        "attenuation": 5.3,
        "specific_heat": 3630.0,
        "thermal_conductivity": 0.51,
    },  # gray matter
    5: {
        "sound_speed": 1560.0,
        "density": 1040.0,
        "attenuation": 5.3,
        "specific_heat": 3600.0,
        "thermal_conductivity": 0.50,
    },  # white matter
}

# Parameter names in consistent order
_PARAM_NAMES = [
    "sound_speed",
    "density",
    "attenuation",
    "specific_heat",
    "thermal_conductivity",
]

# Build lookup arrays (label index -> property value) for fast JAX indexing
_MAX_LABEL = max(TISSUE_ACOUSTIC_PROPERTIES.keys())
_LOOKUP_ARRAYS = {}
for _param in _PARAM_NAMES:
    _vals = []
    for _i in range(_MAX_LABEL + 1):
        _props = TISSUE_ACOUSTIC_PROPERTIES.get(_i, TISSUE_ACOUSTIC_PROPERTIES[0])
        _vals.append(_props[_param])
    _LOOKUP_ARRAYS[_param] = jnp.array(_vals, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Label-based mapping
# ---------------------------------------------------------------------------

def map_labels_to_properties(
    labels: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Map integer tissue labels to acoustic property arrays.

    Args:
        labels: Integer array of any shape. Values should be in [0, 5].
                0=background/water, 1=scalp, 2=skull, 3=CSF, 4=GM, 5=WM.

    Returns:
        Dict with keys: 'sound_speed', 'density', 'attenuation',
        'specific_heat', 'thermal_conductivity'. Each value is a float32
        array with the same shape as labels.
    """
    # Clip to valid range (out-of-range labels map to background)
    safe_labels = jnp.clip(labels, 0, _MAX_LABEL).astype(jnp.int32)
    return {
        param: lookup[safe_labels]
        for param, lookup in _LOOKUP_ARRAYS.items()
    }


# ---------------------------------------------------------------------------
# Attenuation unit conversions
# ---------------------------------------------------------------------------

def neper_per_m_to_db_per_cm(alpha_np_m: float) -> float:
    """
    Convert attenuation from Np/m to dB/cm.

    1 Np/m = 8.686 dB/m = 0.08686 dB/cm

    Args:
        alpha_np_m: Attenuation in Nepers per metre (per MHz).

    Returns:
        Attenuation in dB per centimetre (per MHz).
    """
    return alpha_np_m * NEPER_TO_DB / 100.0


def db_per_cm_to_neper_per_m(alpha_db_cm: float) -> float:
    """
    Convert attenuation from dB/cm to Np/m.

    Args:
        alpha_db_cm: Attenuation in dB per centimetre (per MHz).

    Returns:
        Attenuation in Nepers per metre (per MHz).
    """
    return alpha_db_cm * 100.0 / NEPER_TO_DB


# ---------------------------------------------------------------------------
# Hounsfield unit based continuous property mapping
# ---------------------------------------------------------------------------

def hu_to_acoustic_properties(
    hu: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Convert Hounsfield units to acoustic properties using empirical relations.

    Uses piecewise linear fits from Schneider et al. (2000) for density
    and Mast (2000) for sound speed, adapted for JAX differentiability.

    Args:
        hu: Array of Hounsfield unit values (any shape). HU=0 is water,
            HU>300 is bone.

    Returns:
        Dict with keys: 'sound_speed', 'density', 'attenuation'.
        Each value has the same shape as hu.
    """
    density = _hu_to_density(hu)
    sound_speed = _density_to_sound_speed(density)
    attenuation = _hu_to_attenuation(hu)
    return {
        "sound_speed": sound_speed,
        "density": density,
        "attenuation": attenuation,
    }


def _hu_to_density(hu: jnp.ndarray) -> jnp.ndarray:
    """
    Hounsfield units to density using Schneider et al. (2000) piecewise model.

    Simplified to a smooth approximation for JAX differentiability:
    - HU <= 0: rho = 1000 + 1.0 * HU  (soft tissue / water)
    - HU > 0 and HU <= 1000: rho = 1000 + 0.5 * HU  (soft tissue to bone transition)
    - HU > 1000: rho = 1000 + 0.5 * 1000 + 0.6 * (HU - 1000)  (dense bone)

    Uses softplus-based blending for smooth transitions.
    """
    # Smooth piecewise with soft transitions for differentiability
    # Region 1: HU <= 0 (water/soft tissue)
    rho_low = 1000.0 + 1.0 * hu

    # Region 2: 0 < HU <= 1000 (soft tissue to bone)
    rho_mid = 1000.0 + 0.5 * hu

    # Region 3: HU > 1000 (dense bone)
    rho_high = 1500.0 + 0.6 * (hu - 1000.0)

    # Smooth blending using sigmoid-like transitions
    # Transition at HU=0
    w1 = jax_sigmoid(-hu, sharpness=0.1)  # weight for low region
    # Transition at HU=1000
    w3 = jax_sigmoid(hu - 1000.0, sharpness=0.01)  # weight for high region
    w2 = 1.0 - w1 - w3  # weight for mid region
    w2 = jnp.maximum(w2, 0.0)  # numerical safety

    rho = w1 * rho_low + w2 * rho_mid + w3 * rho_high
    return jnp.clip(rho, 1.0, 2500.0)


def _density_to_sound_speed(rho: jnp.ndarray) -> jnp.ndarray:
    """
    Density to sound speed using Mast (2000) empirical relationship.

    c = 1.33 * rho + 167   (simplified linear fit for bone range)

    Clipped to physical range [1480, 4500] m/s.
    """
    c = 1.33 * rho + 167.0
    return jnp.clip(c, 1480.0, 4500.0)


def _hu_to_attenuation(hu: jnp.ndarray) -> jnp.ndarray:
    """
    Estimate attenuation from HU using empirical scaling.

    For bone (HU > 300): linearly interpolate between soft tissue (~0.5 dB/cm/MHz)
    and cortical bone (~4.74 dB/cm/MHz) based on HU.

    Returns attenuation in dB/cm/MHz (openlifu convention).
    """
    # Soft tissue baseline
    alpha_soft = 0.5  # dB/cm/MHz
    # Cortical bone peak
    alpha_bone = 4.74  # dB/cm/MHz (54.553 Np/m/MHz converted)

    # Smooth interpolation based on HU
    # At HU=0: ~soft tissue. At HU=2000: ~cortical bone.
    bone_fraction = jnp.clip(hu / 2000.0, 0.0, 1.0)
    alpha = alpha_soft + (alpha_bone - alpha_soft) * bone_fraction
    return jnp.maximum(alpha, 0.0)


def jax_sigmoid(x: jnp.ndarray, sharpness: float = 1.0) -> jnp.ndarray:
    """Smooth sigmoid for differentiable piecewise blending."""
    return jax.nn.sigmoid(x * sharpness)


# Needed for jax_sigmoid
import jax

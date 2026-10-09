"""
Acoustic radiation force computation and quasi-static tissue displacement solver.

Connects j-Wave acoustic simulation output to tissue displacement using
region-specific shear moduli from Kuhl's Constitutive Artificial Neural
Networks (Linka, St Pierre, Kuhl — Acta Biomaterialia 2023).

The radiation force F = 2αI/c creates micron-scale displacement in brain
tissue, which depends on the local shear modulus (softer tissue displaces more).
The displacement is solved via a spectral Green's function (FFT-based),
fully differentiable in JAX.

References:
    - Linka, St Pierre, Kuhl (2023). Acta Biomaterialia 160:134-151.
    - Budday et al. (2017). Acta Biomaterialia 48:319-340.
    - Rieke, Butts Pauly (2008). JMRI 27(2):376-390.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional, Union


# ---------------------------------------------------------------------------
# Kuhl CANN region-specific shear moduli (Pa)
# From Linka, St Pierre, Kuhl (2023) Acta Biomaterialia Table 3
# ---------------------------------------------------------------------------

CANN_SHEAR_MODULI = {
    "cortex": 1.82e3,           # 1.82 kPa — gray matter (cortical surface)
    "basal_ganglia": 0.88e3,    # 0.88 kPa — gray matter (deep nuclei)
    "corona_radiata": 0.94e3,   # 0.94 kPa — white matter (projection fibers)
    "corpus_callosum": 0.54e3,  # 0.54 kPa — white matter (commissural fibers)
}

# Simplified label-to-shear-modulus mapping for SCI head model labels
# Uses averaged Kuhl CANN values per tissue class
_LABEL_SHEAR_MODULUS = {
    0: 0.0,       # background (no tissue)
    1: 1.1e3,     # scalp (~skin, estimated)
    2: 5.0e3,     # skull (bone — very stiff, minimal displacement)
    3: 0.0,       # CSF (fluid — no shear modulus)
    4: 1.35e3,    # gray matter (average of cortex 1.82 and BG 0.88)
    5: 0.74e3,    # white matter (average of CR 0.94 and CC 0.54)
    6: 1.35e3,    # additional GM region
    7: 0.74e3,    # additional WM region
    8: 0.0,       # air/cavity
}

_MAX_LABEL = max(_LABEL_SHEAR_MODULUS.keys())
_SHEAR_LOOKUP = jnp.array(
    [_LABEL_SHEAR_MODULUS.get(i, 0.0) for i in range(_MAX_LABEL + 1)],
    dtype=jnp.float32,
)


# ---------------------------------------------------------------------------
# Radiation Force
# ---------------------------------------------------------------------------

def compute_radiation_force(
    intensity: Union[float, jnp.ndarray],
    sound_speed: Union[float, jnp.ndarray],
    attenuation_np_m: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """
    Compute acoustic radiation force density from intensity field.

    F(r) = 2 · α(r) · I(r) / c(r)

    Args:
        intensity: Acoustic intensity in W/m².
        sound_speed: Speed of sound in m/s.
        attenuation_np_m: Absorption coefficient in Nepers/metre.

    Returns:
        Radiation force density in N/m³.
    """
    return 2.0 * attenuation_np_m * intensity / sound_speed


def compute_radiation_force_from_simulation(
    intensity: jnp.ndarray,
    sound_speed: jnp.ndarray,
    attenuation_db_cm: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute radiation force from j-Wave output and acoustic property maps.

    Converts attenuation from dB/cm/MHz (openlifu convention) to Np/m,
    then applies F = 2αI/c.

    Args:
        intensity: Intensity field from j-Wave (W/m²).
        sound_speed: Sound speed map (m/s).
        attenuation_db_cm: Attenuation in dB/cm/MHz (from acoustic.py).

    Returns:
        Radiation force density (N/m³).
    """
    # Convert dB/cm → Np/m: multiply by 100/8.686
    alpha_np_m = attenuation_db_cm * 100.0 / 8.686
    return compute_radiation_force(intensity, sound_speed, alpha_np_m)


# ---------------------------------------------------------------------------
# Quasi-Static Displacement (Spectral Green's Function)
# ---------------------------------------------------------------------------

def solve_displacement_quasistatic(
    force: jnp.ndarray,
    shear_modulus: Union[float, jnp.ndarray],
    grid_spacing: float,
) -> jnp.ndarray:
    """
    Solve for quasi-static tissue displacement from radiation force.

    Uses a spectral method: in Fourier space, the displacement is
    u_hat(k) = F_hat(k) / (µ · |k|²), then IFFT back to real space.

    This is the Green's function solution for the Poisson-like equation:
    µ · ∇²u = -F (quasi-static limit of the elastic wave equation).

    Fully differentiable in JAX (FFT/IFFT are JAX-traceable).

    Args:
        force: Force density field (N/m³), any spatial shape.
        shear_modulus: Shear modulus in Pa (scalar or array matching force shape).
            If array, uses the spatial mean for the spectral solve.
        grid_spacing: Grid spacing in metres.

    Returns:
        Displacement field (metres), same shape as force.
    """
    ndim = force.ndim

    # Use mean shear modulus for spectral solve (heterogeneous µ would need
    # iterative solver; mean is a good first approximation)
    if isinstance(shear_modulus, (jnp.ndarray,)) and shear_modulus.ndim > 0:
        mu = jnp.mean(shear_modulus)
    else:
        mu = shear_modulus

    # Avoid division by zero
    mu = jnp.maximum(mu, 1e-6)

    # FFT of force field
    F_hat = jnp.fft.fftn(force)

    # Build |k|² in Fourier space
    k_components = []
    for i in range(ndim):
        n = force.shape[i]
        ki = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=grid_spacing)
        shape = [1] * ndim
        shape[i] = n
        k_components.append(ki.reshape(shape))

    k_sq = sum(ki**2 for ki in k_components)

    # Avoid division by zero at k=0 (DC component)
    k_sq = jnp.where(k_sq == 0, 1.0, k_sq)

    # Spectral Green's function: u_hat = F_hat / (µ · k²)
    u_hat = F_hat / (mu * k_sq)

    # Zero the DC component (no net displacement)
    dc_idx = tuple([0] * ndim)
    u_hat = u_hat.at[dc_idx].set(0.0)

    # IFFT back to real space
    u = jnp.fft.ifftn(u_hat).real

    return u


# ---------------------------------------------------------------------------
# ConstitutiveNN Shear Modulus Mapping
# ---------------------------------------------------------------------------

def get_cann_shear_modulus() -> Dict[str, float]:
    """
    Get Kuhl CANN region-specific shear moduli.

    From Linka, St Pierre, Kuhl (2023) Acta Biomaterialia 160:134-151.
    Values are in Pascals.

    Returns:
        Dict mapping region names to shear modulus (Pa).
    """
    return CANN_SHEAR_MODULI.copy()


def map_labels_to_shear_modulus(
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """
    Map integer tissue labels to shear modulus values.

    Uses averaged Kuhl CANN values per tissue class.
    0=background, 1=scalp, 2=skull, 3=CSF, 4=GM, 5=WM.

    Args:
        labels: Integer label array (any shape).

    Returns:
        Shear modulus array (Pa), same shape as labels.
    """
    safe_labels = jnp.clip(labels, 0, _MAX_LABEL).astype(jnp.int32)
    return _SHEAR_LOOKUP[safe_labels]


# ---------------------------------------------------------------------------
# PRF Thermometry (Rieke & Butts Pauly 2008)
# ---------------------------------------------------------------------------

def compute_prf_temperature(
    phase_current: jnp.ndarray,
    phase_baseline: jnp.ndarray,
    b0: float = 3.0,
    te: float = 20e-3,
    alpha_prf: float = -0.01e-6,
) -> jnp.ndarray:
    """
    PRF shift MR thermometry.

    ΔT = Δφ / (γ · α_PRF · B0 · TE)

    Args:
        phase_current: Phase image during heating (radians).
        phase_baseline: Phase image before heating (radians).
        b0: Main field strength (Tesla).
        te: Echo time (seconds).
        alpha_prf: PRF temperature coefficient (-0.01 ppm/°C = -0.01e-6 /°C).

    Returns:
        Temperature change map (degrees Celsius).
    """
    gamma = 2.675e8  # rad/s/T (proton gyromagnetic ratio)
    delta_phi = phase_current - phase_baseline
    delta_T = delta_phi / (gamma * alpha_prf * b0 * te)
    return delta_T

"""
Export optimized TUS solutions to openlifu-compatible format.

Converts JAX-optimized delays, apodizations, and pressure fields into
xarray Datasets matching the format produced by openlifu's kwave_if.run_simulation().
"""

import numpy as np
import xarray as xa
from typing import Dict, Optional, Tuple, Union


def export_simulation_result(
    p_max: np.ndarray,
    coords: dict,
    density: Union[float, np.ndarray] = 1000.0,
    sound_speed: Union[float, np.ndarray] = 1500.0,
) -> xa.Dataset:
    """
    Convert simulation output to openlifu-compatible xarray Dataset.

    Matches the format from openlifu's kwave_if.run_simulation():
    - p_max: Peak positive pressure (Pa)
    - p_min: Peak negative pressure (Pa)
    - intensity: I = p^2 / (2*Z) in W/cm^2

    Args:
        p_max: Peak pressure array from j-Wave (numpy).
        coords: xarray-compatible coordinates dict.
        density: Density for impedance calculation.
        sound_speed: Sound speed for impedance calculation.

    Returns:
        xarray.Dataset with p_max, p_min, intensity.
    """
    p_max_np = np.asarray(p_max, dtype=np.float32)
    p_min_np = p_max_np.copy()  # Symmetric approximation

    Z = np.asarray(density, dtype=np.float32) * np.asarray(sound_speed, dtype=np.float32)
    intensity_np = 1e-4 * p_min_np**2 / (2 * Z)

    p_max_da = xa.DataArray(
        p_max_np, coords=coords, name="p_max",
        attrs={"units": "Pa", "long_name": "PPP"},
    )
    p_min_da = xa.DataArray(
        p_min_np, coords=coords, name="p_min",
        attrs={"units": "Pa", "long_name": "PNP"},
    )
    intensity_da = xa.DataArray(
        intensity_np, coords=coords, name="I",
        attrs={"units": "W/cm^2", "long_name": "Intensity"},
    )

    return xa.Dataset({"p_max": p_max_da, "p_min": p_min_da, "intensity": intensity_da})


def build_solution_dict(
    delays: np.ndarray,
    apodizations: np.ndarray,
    simulation_result: xa.Dataset,
    freq: float,
    voltage: float = 1.0,
    target: Optional[Tuple[float, float, float]] = None,
) -> Dict:
    """
    Build a dict compatible with openlifu Solution.from_dict().

    This can be loaded into openlifu's Solution object for safety analysis,
    Slicer visualization, and hardware programming.

    Args:
        delays: (num_foci, num_elements) delay array in seconds.
        apodizations: (num_foci, num_elements) amplitude weights.
        simulation_result: xarray Dataset from export_simulation_result().
        freq: Transducer frequency (Hz).
        voltage: Applied voltage (V).
        target: (x, y, z) target coordinates.

    Returns:
        Dict with Solution-compatible fields.
    """
    return {
        "delays": np.asarray(delays),
        "apodizations": np.asarray(apodizations),
        "voltage": voltage,
        "pulse": {
            "frequency": freq,
            "amplitude": 1.0,
            "duration": 20.0 / freq,
        },
        "simulation_result": simulation_result,
        "target": target,
        "approved": False,
    }

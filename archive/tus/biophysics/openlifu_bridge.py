"""
Bridge module connecting openlifu-python treatment planning output
to sbi4dwi's differentiable MR-ARFI simulation chain.

Accepts an openlifu Solution (from files or live object), extracts
pressure fields and tissue properties, and feeds them into the existing
radiation_force -> displacement -> MR-ARFI phase pipeline.

openlifu-python is an optional dependency. When not installed, only
file-based loading (JSON + netCDF) is supported via xarray directly.

Usage:
    # From files (no openlifu needed):
    results = run_arfi_pipeline_from_files("solution.json", "sim_result.nc")

    # From live openlifu Solution:
    results = run_arfi_pipeline_from_solution(solution, seg_params)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import xarray as xa

try:
    import openlifu
    from openlifu.plan.solution import Solution as OpenLIFUSolution

    _HAS_OPENLIFU = True
except ImportError:
    _HAS_OPENLIFU = False

from . import radiation_force, mr_arfi


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class SolutionData:
    """Extracted data from an openlifu Solution, ready for the ARFI chain.

    All arrays are numpy/jnp. All spatial quantities are in SI (metres, Pa, W/m^2).
    """

    p_max: np.ndarray  # Peak pressure field (Pa)
    intensity: np.ndarray  # Acoustic intensity (W/m^2)
    sound_speed: np.ndarray  # Sound speed map (m/s)
    density: np.ndarray  # Density map (kg/m^3)
    attenuation: np.ndarray  # Attenuation (dB/cm/MHz)
    grid_spacing_m: float  # Uniform grid spacing in metres
    freq: float  # Ultrasound frequency (Hz)
    target_position_m: np.ndarray  # (3,) target xyz in metres
    target_voxel: Tuple[int, ...]  # Target grid indices
    delays: np.ndarray  # (num_foci, num_elements) seconds
    apodizations: np.ndarray  # (num_foci, num_elements)
    element_positions_m: Optional[np.ndarray] = None
    focus_index: int = 0


# ---------------------------------------------------------------------------
# File-based loading (no openlifu dependency)
# ---------------------------------------------------------------------------


def load_solution_from_files(
    json_path: Union[str, Path],
    nc_path: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Load openlifu Solution data from JSON + netCDF files.

    Does NOT require openlifu to be installed. Parses the JSON for
    delays, apodizations, frequency, target. Loads simulation_result
    from netCDF via xarray.

    Args:
        json_path: Path to solution .json file.
        nc_path: Path to simulation result .nc file. If None, derived
                 from json_path by changing extension.

    Returns:
        Dict with raw solution data.
    """
    json_path = Path(json_path)

    with open(json_path) as f:
        data = json.load(f)

    # Extract beamforming parameters
    delays = np.array(data.get("delays", [[0.0]]))
    apodizations = np.array(data.get("apodizations", [[1.0]]))

    pulse = data.get("pulse", {})
    frequency = float(pulse.get("frequency", 400e3))
    voltage = float(data.get("voltage", 0.0))

    # Target position (mm -> m)
    target = data.get("target", {})
    target_pos = np.array(target.get("position", [0, 0, 0]), dtype=np.float64)
    target_units = target.get("units", "mm")
    if target_units == "mm":
        target_pos *= 1e-3

    # Foci
    foci = data.get("foci", [])
    foci_m = []
    for f_data in foci:
        pos = np.array(f_data.get("position", [0, 0, 0]), dtype=np.float64)
        if f_data.get("units", "mm") == "mm":
            pos *= 1e-3
        foci_m.append(pos)

    result = {
        "delays": delays,
        "apodizations": apodizations,
        "frequency": frequency,
        "voltage": voltage,
        "target_position_m": target_pos,
        "foci_m": foci_m,
    }

    # Load netCDF simulation result
    if nc_path is None:
        nc_path = json_path.with_suffix(".nc")
        if not nc_path.exists():
            nc_path = json_path.parent / (json_path.stem + "_simulation_result.nc")

    nc_path = Path(nc_path)
    if nc_path.exists():
        ds = xa.open_dataset(nc_path).load()
        result["p_max"] = ds["p_max"].values if "p_max" in ds else None
        result["intensity"] = ds["intensity"].values if "intensity" in ds else None

        # Extract grid info
        for dim in ["x", "y", "z"]:
            if dim in ds.coords:
                vals = ds.coords[dim].values.astype(np.float64)
                units = ds.coords[dim].attrs.get("units", "mm")
                if units == "mm":
                    vals *= 1e-3
                result[f"coords_{dim}"] = vals

        ds.close()

    return result


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_chain_inputs(
    solution: Union[dict, "OpenLIFUSolution"],
    seg_params: Optional[xa.Dataset] = None,
    focus_index: int = 0,
) -> SolutionData:
    """
    Extract arrays needed by the MR-ARFI chain from an openlifu Solution.

    Handles two modes:
    1. Live openlifu Solution object (when openlifu is installed)
    2. Dict from load_solution_from_files()

    Args:
        solution: openlifu Solution object or dict from file loading.
        seg_params: Optional xarray Dataset with sound_speed, density,
                    attenuation from SegmentationMethod.seg_params().
        focus_index: Which focal point to extract for multi-focus solutions.

    Returns:
        SolutionData with all arrays in SI units.
    """
    if _HAS_OPENLIFU and isinstance(solution, OpenLIFUSolution):
        return _extract_from_openlifu(solution, seg_params, focus_index)
    elif isinstance(solution, dict):
        return _extract_from_dict(solution, seg_params, focus_index)
    else:
        raise TypeError(f"Expected dict or OpenLIFUSolution, got {type(solution)}")


def _extract_from_dict(
    data: dict,
    seg_params: Optional[xa.Dataset],
    focus_index: int,
) -> SolutionData:
    """Extract from file-loaded dict."""
    p_max = data.get("p_max")
    intensity = data.get("intensity")

    # Intensity conversion: W/cm^2 -> W/m^2
    if intensity is not None:
        intensity = intensity.astype(np.float64) * 1e4

    # If no intensity, estimate from pressure
    if intensity is None and p_max is not None:
        Z_water = 1.5e6  # acoustic impedance of water
        intensity = p_max.astype(np.float64) ** 2 / (2.0 * Z_water)

    if p_max is None:
        p_max = np.zeros(1)
    if intensity is None:
        intensity = np.zeros(1)

    # Grid spacing
    grid_spacing = 1e-3  # default 1mm
    for dim in ["x", "y", "z"]:
        key = f"coords_{dim}"
        if key in data and len(data[key]) > 1:
            grid_spacing = abs(float(data[key][1] - data[key][0]))
            break

    # Tissue properties
    if seg_params is not None:
        sound_speed = seg_params["sound_speed"].values.astype(np.float64)
        density = seg_params["density"].values.astype(np.float64)
        attenuation = seg_params["attenuation"].values.astype(np.float64)
    else:
        # Default: uniform brain tissue
        sound_speed = np.full_like(intensity, 1560.0)
        density = np.full_like(intensity, 1040.0)
        attenuation = np.full_like(intensity, 5.3)

    target_pos = data.get("target_position_m", np.zeros(3))

    # Find target voxel (nearest grid point)
    target_voxel = (0,) * p_max.ndim
    if "coords_x" in data and p_max.ndim >= 1:
        coords = [data.get(f"coords_{d}", np.array([0])) for d in ["x", "y", "z"]]
        idx = []
        for i, c in enumerate(coords[: p_max.ndim]):
            if len(c) > 0 and i < len(target_pos):
                idx.append(int(np.argmin(np.abs(c - target_pos[i]))))
            else:
                idx.append(0)
        target_voxel = tuple(idx)

    return SolutionData(
        p_max=p_max.astype(np.float64),
        intensity=intensity,
        sound_speed=sound_speed,
        density=density,
        attenuation=attenuation,
        grid_spacing_m=grid_spacing,
        freq=data.get("frequency", 400e3),
        target_position_m=target_pos,
        target_voxel=target_voxel,
        delays=data.get("delays", np.array([[0.0]])),
        apodizations=data.get("apodizations", np.array([[1.0]])),
        focus_index=focus_index,
    )


def _extract_from_openlifu(solution, seg_params, focus_index):
    """Extract from live openlifu Solution object."""
    if not _HAS_OPENLIFU:
        raise ImportError("openlifu-python required for live Solution extraction")

    sim = solution.simulation_result
    p_max = sim["p_max"].values.astype(np.float64)

    if "intensity" in sim:
        intensity = sim["intensity"].values.astype(np.float64) * 1e4
    else:
        intensity = p_max**2 / (2.0 * 1.5e6)

    # Handle focal_point_index dimension
    if "focal_point_index" in sim.dims:
        p_max = p_max[focus_index]
        intensity = intensity[focus_index]

    # Grid spacing from coordinates
    grid_spacing = 1e-3
    for dim in ["x", "y", "z"]:
        if dim in sim.coords and len(sim.coords[dim]) > 1:
            vals = sim.coords[dim].values.astype(np.float64)
            units = sim.coords[dim].attrs.get("units", "mm")
            if units == "mm":
                vals *= 1e-3
            grid_spacing = abs(float(vals[1] - vals[0]))
            break

    # Tissue properties
    if seg_params is not None:
        sound_speed = seg_params["sound_speed"].values.astype(np.float64)
        density = seg_params["density"].values.astype(np.float64)
        attenuation = seg_params["attenuation"].values.astype(np.float64)
    else:
        sound_speed = np.full_like(intensity, 1560.0)
        density = np.full_like(intensity, 1040.0)
        attenuation = np.full_like(intensity, 5.3)

    target_pos = np.array(
        solution.target.get_position(units="m"), dtype=np.float64
    )

    delays = solution.delays if solution.delays is not None else np.array([[0.0]])
    apod = (
        solution.apodizations
        if solution.apodizations is not None
        else np.ones_like(delays)
    )

    element_positions = None
    if solution.transducer is not None:
        element_positions = solution.transducer.get_positions(units="m")

    target_voxel = tuple([0] * intensity.ndim)

    return SolutionData(
        p_max=p_max,
        intensity=intensity,
        sound_speed=sound_speed,
        density=density,
        attenuation=attenuation,
        grid_spacing_m=grid_spacing,
        freq=solution.pulse.frequency,
        target_position_m=target_pos,
        target_voxel=target_voxel,
        delays=np.array(delays),
        apodizations=np.array(apod),
        element_positions_m=element_positions,
        focus_index=focus_index,
    )


# ---------------------------------------------------------------------------
# MR-ARFI pipeline
# ---------------------------------------------------------------------------


def run_arfi_from_solution(
    solution_data: SolutionData,
    msg_amplitude: float = 40e-3,
    msg_duration: float = 5e-3,
    labels: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run MR-ARFI simulation chain from openlifu Solution data.

    Reuses existing functions from radiation_force.py and mr_arfi.py.
    Skips acoustic simulation (already done by openlifu) and starts
    from the radiation force computation.

    Args:
        solution_data: SolutionData from extract_chain_inputs().
        msg_amplitude: Motion-sensitizing gradient amplitude (T/m).
        msg_duration: MSG lobe duration (seconds).
        labels: Optional integer tissue label array. If None, infers
                tissue type from sound speed.

    Returns:
        Dict with all intermediate and final results.
    """
    intensity = jnp.array(solution_data.intensity)
    c = jnp.array(solution_data.sound_speed)
    alpha_db = jnp.array(solution_data.attenuation)
    dx = solution_data.grid_spacing_m

    # Radiation force
    force = radiation_force.compute_radiation_force_from_simulation(
        intensity, c, alpha_db
    )

    # Shear modulus
    if labels is not None:
        mu = radiation_force.map_labels_to_shear_modulus(jnp.array(labels))
    else:
        # Infer from sound speed: brain tissue ≈ 1.35 kPa
        mu = jnp.where(c > 3000, 5.0e3, 1.35e3)  # skull vs brain
        mu = jnp.where(c < 1510, 0.0, mu)  # water/CSF

    # Displacement
    displacement = radiation_force.solve_displacement_quasistatic(force, mu, dx)

    # MR-ARFI phase
    arfi_phase = mr_arfi.predict_arfi_phase(
        np.array(displacement), msg_amplitude, msg_duration
    )

    # Sequence parameters
    seq_params = mr_arfi.design_arfi_sequence(
        msg_amplitude=msg_amplitude, msg_duration=msg_duration
    )

    # Target metrics
    tv = solution_data.target_voxel
    results = {
        "radiation_force": np.array(force),
        "shear_modulus": np.array(mu),
        "displacement": np.array(displacement),
        "displacement_um": np.array(displacement) * 1e6,
        "max_displacement_um": float(np.max(np.abs(displacement))) * 1e6,
        "arfi_phase": np.array(arfi_phase),
        "arfi_sequence_params": seq_params,
        "msg_amplitude": msg_amplitude,
        "msg_duration": msg_duration,
        "encoding_coefficient_rad_m": seq_params["encoding_coefficient_rad_m"],
        "target_voxel": tv,
        "grid_spacing_m": dx,
    }

    # Target-specific metrics
    try:
        results["force_at_target"] = float(force[tv])
        results["displacement_at_target_um"] = float(displacement[tv]) * 1e6
        results["phase_at_target_rad"] = float(arfi_phase[tv])
    except (IndexError, TypeError):
        pass

    return results


# ---------------------------------------------------------------------------
# Convenience top-level functions
# ---------------------------------------------------------------------------


def run_arfi_pipeline_from_files(
    json_path: Union[str, Path],
    nc_path: Optional[Union[str, Path]] = None,
    msg_amplitude: float = 40e-3,
    msg_duration: float = 5e-3,
    focus_index: int = 0,
) -> Dict:
    """
    End-to-end: load openlifu Solution files -> MR-ARFI results.

    Does not require openlifu to be installed.
    """
    raw = load_solution_from_files(json_path, nc_path)
    solution_data = extract_chain_inputs(raw, focus_index=focus_index)
    return run_arfi_from_solution(solution_data, msg_amplitude, msg_duration)


def run_arfi_pipeline_from_solution(
    solution: "OpenLIFUSolution",
    seg_params: Optional[xa.Dataset] = None,
    msg_amplitude: float = 40e-3,
    msg_duration: float = 5e-3,
    focus_index: int = 0,
) -> Dict:
    """
    End-to-end: live openlifu Solution object -> MR-ARFI results.

    Requires openlifu to be installed.
    """
    if not _HAS_OPENLIFU:
        raise ImportError("openlifu-python is required for this function")
    solution_data = extract_chain_inputs(solution, seg_params, focus_index)
    return run_arfi_from_solution(solution_data, msg_amplitude, msg_duration)

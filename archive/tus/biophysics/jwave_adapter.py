"""
j-Wave simulation adapter for transcranial focused ultrasound.

Provides the differentiable acoustic simulation layer that bridges sbi4dwi's
tissue property estimation to j-Wave's pseudospectral wave solver. Produces
pressure fields compatible with openlifu's Solution format.

This module requires j-Wave (pip install jwave) and its dependency jaxdf.
All imports are local to avoid hard dependencies.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Domain construction
# ---------------------------------------------------------------------------

def create_domain(
    grid_shape: Tuple[int, ...],
    grid_spacing: float,
):
    """
    Create a j-Wave computational domain.

    Args:
        grid_shape: (nx, ny) or (nx, ny, nz) grid dimensions.
        grid_spacing: Uniform spacing in metres.

    Returns:
        jaxdf.geometry.Domain instance.
    """
    from jaxdf.geometry import Domain

    ndim = len(grid_shape)
    dx = tuple([grid_spacing] * ndim)
    return Domain(grid_shape, dx)


# ---------------------------------------------------------------------------
# Medium construction
# ---------------------------------------------------------------------------

def create_medium(
    domain,
    sound_speed: Union[float, jnp.ndarray],
    density: Union[float, jnp.ndarray],
    attenuation: Union[float, jnp.ndarray] = 0.0,
    pml_size: float = 20.0,
):
    """
    Create a j-Wave medium with optional heterogeneous properties.

    Args:
        domain: jaxdf Domain.
        sound_speed: Scalar (homogeneous) or array matching domain.N (heterogeneous). m/s.
        density: Scalar or array. kg/m^3.
        attenuation: Scalar or array. Absorption coefficient.
        pml_size: PML layer thickness in grid points.

    Returns:
        jwave.geometry.Medium instance.
    """
    from jwave import FourierSeries
    from jwave.geometry import Medium

    def _to_field(value, domain):
        if isinstance(value, (int, float)):
            return value
        # Array — wrap in FourierSeries with trailing dimension
        arr = jnp.asarray(value)
        if arr.ndim == len(domain.N):
            arr = arr[..., jnp.newaxis]  # add trailing dim for FourierSeries
        return FourierSeries(arr, domain)

    return Medium(
        domain=domain,
        sound_speed=_to_field(sound_speed, domain),
        density=_to_field(density, domain),
        attenuation=_to_field(attenuation, domain),
        pml_size=pml_size,
    )


# ---------------------------------------------------------------------------
# Source / toneburst generation
# ---------------------------------------------------------------------------

def _make_toneburst(
    freq: float,
    dt: float,
    n_cycles: int,
    delay: float = 0.0,
    amplitude: float = 1.0,
) -> jnp.ndarray:
    """
    Generate a delayed sinusoidal toneburst.

    Args:
        freq: Frequency in Hz.
        dt: Time step in seconds.
        n_cycles: Number of cycles.
        delay: Time delay in seconds.
        amplitude: Peak amplitude.

    Returns:
        1D array of the toneburst signal.
    """
    duration = n_cycles / freq
    # Pad for delay + duration + some extra
    total_time = delay + duration + 2.0 / freq
    n_samples = int(total_time / dt) + 1
    t = jnp.arange(n_samples) * dt

    # Sinusoidal burst starting at t=delay
    signal = amplitude * jnp.sin(2.0 * jnp.pi * freq * (t - delay))
    # Window: active only during [delay, delay+duration]
    window = jnp.where((t >= delay) & (t <= delay + duration), 1.0, 0.0)
    return signal * window


def create_sources(
    domain,
    positions: jnp.ndarray,
    freq: float,
    delays: jnp.ndarray,
    apodizations: jnp.ndarray,
    dt: float,
    n_cycles: int = 5,
):
    """
    Create j-Wave Sources from transducer element parameters.

    This function is JAX-trace-compatible w.r.t. delays and apodizations,
    so it can be used inside jax.grad.

    Args:
        domain: jaxdf Domain.
        positions: (n_elements, ndim) physical coordinates in metres.
        freq: Frequency in Hz.
        delays: (n_elements,) time delays in seconds (can be traced).
        apodizations: (n_elements,) amplitude weights (can be traced).
        dt: Simulation time step in seconds.
        n_cycles: Number of toneburst cycles.

    Returns:
        jwave.geometry.Sources instance.
    """
    from jwave.geometry import Sources

    n_elements = positions.shape[0]
    ndim = positions.shape[1]

    # Convert physical positions to grid indices (not traced — positions are fixed)
    dx = domain.dx[0]
    grid_indices = jnp.round(positions / dx).astype(jnp.int32)
    for d in range(ndim):
        grid_indices = grid_indices.at[:, d].set(
            jnp.clip(grid_indices[:, d], 0, domain.N[d] - 1)
        )

    # Generate signals in a JAX-traceable way (vectorized, no Python float() calls)
    signals = _make_tonebursts_vectorized(freq, dt, n_cycles, delays, apodizations)

    # Build position tuples (concrete — positions aren't traced)
    gi_np = np.array(grid_indices)
    pos_tuple = tuple(
        list(int(gi_np[j, d]) for j in range(n_elements))
        for d in range(ndim)
    )

    return Sources(
        positions=pos_tuple,
        signals=signals,
        dt=dt,
        domain=domain,
    )


def _make_tonebursts_vectorized(
    freq: float,
    dt: float,
    n_cycles: int,
    delays: jnp.ndarray,
    amplitudes: jnp.ndarray,
) -> jnp.ndarray:
    """
    Generate delayed tonebursts for all elements simultaneously.

    JAX-traceable w.r.t. delays and amplitudes.

    Args:
        freq: Frequency in Hz.
        dt: Time step in seconds.
        n_cycles: Number of cycles.
        delays: (n_elements,) time delays in seconds.
        amplitudes: (n_elements,) amplitude weights.

    Returns:
        (n_elements, n_samples) array of toneburst signals.
    """
    duration = n_cycles / freq
    # Use the max possible delay to size the signal array
    max_delay = 50e-6  # generous upper bound for delays
    total_time = max_delay + duration + 2.0 / freq
    n_samples = int(total_time / dt) + 1
    t = jnp.arange(n_samples) * dt  # (n_samples,)

    # Broadcast: (n_elements, 1) delays against (n_samples,) time
    t_shifted = t[jnp.newaxis, :] - delays[:, jnp.newaxis]  # (n_elements, n_samples)

    # Sinusoidal signal
    signal = jnp.sin(2.0 * jnp.pi * freq * t_shifted)

    # Window: active only during [0, duration] of shifted time
    window = jnp.where((t_shifted >= 0) & (t_shifted <= duration), 1.0, 0.0)

    # Apply amplitude and window
    return amplitudes[:, jnp.newaxis] * signal * window


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def run_simulation_2d(
    grid_shape: Tuple[int, int],
    grid_spacing: float,
    sound_speed: Union[float, jnp.ndarray],
    density: Union[float, jnp.ndarray],
    source_position: Tuple[int, int],
    freq: float,
    t_end: float,
    n_cycles: int = 5,
    pml_size: float = 10.0,
) -> Dict[str, jnp.ndarray]:
    """
    Run a 2D j-Wave simulation (fast, for testing).

    Args:
        grid_shape: (nx, ny).
        grid_spacing: Spacing in metres.
        sound_speed: Scalar or 2D array (m/s).
        density: Scalar or 2D array (kg/m^3).
        source_position: (x, y) grid index of point source.
        freq: Source frequency (Hz).
        t_end: Simulation end time (seconds).
        n_cycles: Toneburst cycles.
        pml_size: PML thickness in grid points.

    Returns:
        Dict with 'p_max' (2D array of peak positive pressure).
    """
    from jwave.acoustics.time_varying import simulate_wave_propagation
    from jwave.geometry import TimeAxis

    domain = create_domain(grid_shape, grid_spacing)
    medium = create_medium(domain, sound_speed, density, pml_size=pml_size)
    time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=t_end)
    dt = float(time_axis.dt)

    # Single point source
    positions = jnp.array([[source_position[0] * grid_spacing,
                            source_position[1] * grid_spacing]])
    sources = create_sources(
        domain, positions, freq,
        delays=jnp.zeros(1), apodizations=jnp.ones(1),
        dt=dt, n_cycles=n_cycles,
    )

    # Run simulation
    pressure = simulate_wave_propagation(medium, time_axis, sources=sources)

    # Extract pressure data from j-Wave output
    p_data = _extract_pressure_timeseries(pressure)
    p_max = jnp.max(jnp.abs(p_data), axis=0)

    return {"p_max": p_max}


def _extract_pressure_timeseries(pressure) -> jnp.ndarray:
    """
    Extract raw JAX array from j-Wave simulation output.

    j-Wave returns a list of FourierSeries (one per timestep) or a stacked array.
    This function handles both cases.

    Returns:
        (Nt, *spatial_dims) array of pressure values.
    """
    if isinstance(pressure, (list, tuple)):
        # List of FourierSeries — extract .on_grid from each and stack
        frames = []
        for frame in pressure:
            if hasattr(frame, "on_grid"):
                data = frame.on_grid
            elif hasattr(frame, "params"):
                data = frame.params
            else:
                data = jnp.asarray(frame)
            # Remove trailing singleton from FourierSeries (N, 1) -> (N,)
            if data.ndim > 0 and data.shape[-1] == 1:
                data = data[..., 0]
            frames.append(data)
        return jnp.stack(frames, axis=0)
    elif hasattr(pressure, "on_grid"):
        # Single FourierSeries
        data = pressure.on_grid
        if data.shape[-1] == 1:
            data = data[..., 0]
        return data
    else:
        # Already a raw array
        data = jnp.asarray(pressure)
        if data.ndim > 1 and data.shape[-1] == 1:
            data = data[..., 0]
        return data


def run_simulation_jax(
    medium,
    source_position: Tuple[int, ...],
    freq: float,
    t_end: float,
    n_cycles: int = 3,
    time_axis=None,
    sources=None,
) -> jnp.ndarray:
    """
    Run j-Wave simulation returning pure JAX array (for differentiability).

    Note: When using inside jax.grad, the medium must be the only traced
    argument. Construct time_axis and sources OUTSIDE the traced function
    and pass them in, since TimeAxis.from_medium() calls float() which
    is not compatible with JAX tracing.

    Args:
        medium: j-Wave Medium (already constructed).
        source_position: Grid indices for a single point source.
        freq: Source frequency (Hz).
        t_end: End time (seconds).
        n_cycles: Toneburst cycles.
        time_axis: Pre-computed TimeAxis (required for differentiable use).
        sources: Pre-computed Sources (optional, built from source_position if None).

    Returns:
        2D or 3D array of peak absolute pressure.
    """
    from jwave.acoustics.time_varying import simulate_wave_propagation
    from jwave.geometry import TimeAxis as TA

    if time_axis is None:
        time_axis = TA.from_medium(medium, cfl=0.3, t_end=t_end)

    domain = medium.domain
    ndim = len(domain.N)

    if sources is None:
        dt = float(time_axis.dt)
        pos = jnp.array([[source_position[d] * domain.dx[d] for d in range(ndim)]])
        sources = create_sources(
            domain, pos, freq,
            delays=jnp.zeros(1), apodizations=jnp.ones(1),
            dt=dt, n_cycles=n_cycles,
        )

    pressure = simulate_wave_propagation(medium, time_axis, sources=sources)
    p_data = _extract_pressure_timeseries(pressure)
    return jnp.max(jnp.abs(p_data), axis=0)


def simulation_result_to_xarray(
    p_max: jnp.ndarray,
    coords: dict,
    density: Union[float, jnp.ndarray] = 1000.0,
    sound_speed: Union[float, jnp.ndarray] = 1500.0,
):
    """
    Convert j-Wave simulation output to openlifu-compatible xarray Dataset.

    Produces the same format as openlifu's kwave_if.run_simulation():
    - p_max: Peak positive pressure (Pa)
    - p_min: Peak negative pressure (Pa) — approximated as p_max for time-domain
    - intensity: I = p^2 / (2 * Z) in W/cm^2

    Args:
        p_max: Peak pressure array from j-Wave.
        coords: Dict of xarray coordinates (or xarray.Coordinates).
        density: Density for impedance calculation.
        sound_speed: Sound speed for impedance calculation.

    Returns:
        xarray.Dataset compatible with openlifu Solution.
    """
    import xarray as xa
    import numpy as np

    p_max_np = np.array(p_max)
    p_min_np = p_max_np  # Symmetric approximation for toneburst

    Z = np.asarray(density) * np.asarray(sound_speed)
    intensity_np = 1e-4 * p_min_np**2 / (2 * Z)

    p_max_da = xa.DataArray(p_max_np, coords=coords, name="p_max",
                            attrs={"units": "Pa", "long_name": "PPP"})
    p_min_da = xa.DataArray(p_min_np, coords=coords, name="p_min",
                            attrs={"units": "Pa", "long_name": "PNP"})
    intensity_da = xa.DataArray(intensity_np, coords=coords, name="I",
                                attrs={"units": "W/cm^2", "long_name": "Intensity"})

    return xa.Dataset({"p_max": p_max_da, "p_min": p_min_da, "intensity": intensity_da})

"""
Extended TUS experiments: 3D simulation, multi-element arrays, multi-target,
multi-frequency, grid convergence, sensitivity analysis, Helmholtz solver.

All functions are designed to run on Modal A100 or DGX Spark GPUs.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Array geometry helpers
# ---------------------------------------------------------------------------

def create_circular_array(
    n_elements: int,
    radius: float,
    center: Tuple[float, float, float] = (0.016, 0.016, 0.002),
) -> jnp.ndarray:
    """
    Generate positions for a circular transducer array.

    Args:
        n_elements: Number of elements.
        radius: Array radius in metres.
        center: (x, y, z) center position in metres.

    Returns:
        (n_elements, 3) array of element positions.
    """
    angles = jnp.linspace(0, 2 * jnp.pi, n_elements, endpoint=False)
    x = center[0] + radius * jnp.cos(angles)
    y = center[1] + radius * jnp.sin(angles)
    z = jnp.full(n_elements, center[2])
    return jnp.stack([x, y, z], axis=1)


def _make_skull_phantom_2d(
    grid_size: int,
    skull_range: Tuple[int, int] = (20, 30),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create a 2D phantom with a skull layer."""
    c = jnp.ones((grid_size, grid_size)) * 1500.0
    c = c.at[:, skull_range[0]:skull_range[1]].set(4080.0)
    rho = jnp.ones((grid_size, grid_size)) * 1000.0
    rho = rho.at[:, skull_range[0]:skull_range[1]].set(1900.0)
    return c, rho


def _make_skull_phantom_3d(
    grid_size: int,
    skull_range: Tuple[int, int] = (10, 15),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create a 3D phantom with a skull shell (slab in z)."""
    c = jnp.ones((grid_size, grid_size, grid_size)) * 1500.0
    c = c.at[:, :, skull_range[0]:skull_range[1]].set(4080.0)
    rho = jnp.ones((grid_size, grid_size, grid_size)) * 1000.0
    rho = rho.at[:, :, skull_range[0]:skull_range[1]].set(1900.0)
    return c, rho


# ---------------------------------------------------------------------------
# Multi-element array optimization
# ---------------------------------------------------------------------------

def run_array_optimization(
    n_elements: int = 16,
    grid_size: int = 64,
    spacing_mm: float = 0.2,
    freq: float = 400e3,
    n_iters: int = 10,
    lr: float = 1e-7,
    skull_thickness_voxels: int = 10,
) -> Dict:
    """
    Optimize delays for an N-element circular array through skull.

    Returns dict with delays, loss_history, timing.
    """
    import time
    # Import adapter modules directly to avoid __init__ chain
    import importlib.util, sys, os
    biophys = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("jwa", os.path.join(biophys, "jwave_adapter.py"))
    jwa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jwa)

    from jwave.geometry import TimeAxis
    import optax

    spacing_m = spacing_mm * 1e-3
    domain = jwa.create_domain((grid_size, grid_size), spacing_m)

    skull_start = grid_size // 3
    skull_end = skull_start + skull_thickness_voxels
    c, rho = _make_skull_phantom_2d(grid_size, (skull_start, skull_end))

    medium = jwa.create_medium(domain, c, rho, pml_size=8)
    time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=5e-5)
    dt = float(time_axis.dt)

    # Array on the source side of skull
    array_center = (grid_size // 2 * spacing_m, 5 * spacing_m)
    positions = create_circular_array(n_elements, 0.005, (array_center[0], array_center[1], 0.0))
    positions_2d = positions[:, :2]  # 2D simulation

    target = (grid_size // 2, grid_size * 3 // 4)  # behind skull

    initial_delays = jnp.zeros(n_elements)

    def loss_fn(delays):
        sources = jwa.create_sources(
            domain, positions_2d, freq,
            delays=delays, apodizations=jnp.ones(n_elements),
            dt=dt, n_cycles=5,
        )
        p_max = jwa.run_simulation_jax(medium, target, freq, 5e-5,
                                        time_axis=time_axis, sources=sources)
        return -p_max[target]

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(initial_delays)
    delays = initial_delays
    loss_history = []

    t0 = time.time()
    for i in range(n_iters):
        loss_val, grads = jax.value_and_grad(loss_fn)(delays)
        updates, opt_state = optimizer.update(grads, opt_state)
        delays = optax.apply_updates(delays, updates)
        loss_history.append(float(loss_val))
    elapsed = time.time() - t0

    return {
        "n_elements": n_elements,
        "optimized_delays": np.array(delays).tolist(),
        "loss_history": loss_history,
        "time_s": elapsed,
        "time_per_iter": elapsed / n_iters,
        "improvement": abs(loss_history[-1] / loss_history[0]) if loss_history[0] != 0 else 0,
    }


# ---------------------------------------------------------------------------
# Multi-target simulation
# ---------------------------------------------------------------------------

def simulate_at_targets(
    grid_size: int = 64,
    spacing_mm: float = 0.2,
    freq: float = 400e3,
    targets: Dict[str, Tuple[int, ...]] = None,
    skull_range: Tuple[int, int] = (20, 30),
) -> Dict:
    """Simulate at multiple brain targets through skull."""
    import importlib.util, os
    biophys = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("jwa", os.path.join(biophys, "jwave_adapter.py"))
    jwa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jwa)

    spacing_m = spacing_mm * 1e-3
    source_pos = (grid_size // 2, 5)

    if targets is None:
        targets = {"shallow": (32, 40), "deep": (32, 55)}

    # Water baseline
    r_water = jwa.run_simulation_2d(
        (grid_size, grid_size), spacing_m, 1500.0, 1000.0,
        source_pos, freq, 5e-5, pml_size=8,
    )

    # Skull
    c, rho = _make_skull_phantom_2d(grid_size, skull_range)
    r_skull = jwa.run_simulation_2d(
        (grid_size, grid_size), spacing_m, c, rho,
        source_pos, freq, 5e-5, pml_size=8,
    )

    results = {}
    for name, target in targets.items():
        p_w = float(r_water["p_max"][target])
        p_s = float(r_skull["p_max"][target])
        atten = 100 * (1 - p_s / p_w) if p_w > 0 else 0
        results[name] = {
            "target": target,
            "p_water": p_w,
            "p_skull": p_s,
            "attenuation_pct": atten,
        }

    return results


# ---------------------------------------------------------------------------
# Multi-frequency comparison
# ---------------------------------------------------------------------------

def compare_frequencies(
    grid_size: int = 64,
    spacing_mm: float = 0.2,
    frequencies: List[float] = None,
    skull_range: Tuple[int, int] = (20, 30),
    target: Tuple[int, int] = (32, 50),
) -> Dict:
    """Compare skull penetration at different frequencies."""
    import importlib.util, os
    biophys = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("jwa", os.path.join(biophys, "jwave_adapter.py"))
    jwa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jwa)

    if frequencies is None:
        frequencies = [180e3, 400e3]

    spacing_m = spacing_mm * 1e-3
    source_pos = (grid_size // 2, 5)
    c, rho = _make_skull_phantom_2d(grid_size, skull_range)

    results = {}
    for freq in frequencies:
        r_w = jwa.run_simulation_2d(
            (grid_size, grid_size), spacing_m, 1500.0, 1000.0,
            source_pos, freq, 5e-5, pml_size=8,
        )
        r_s = jwa.run_simulation_2d(
            (grid_size, grid_size), spacing_m, c, rho,
            source_pos, freq, 5e-5, pml_size=8,
        )
        p_w = float(r_w["p_max"][target])
        p_s = float(r_s["p_max"][target])
        results[freq] = {
            "p_water": p_w,
            "p_skull": p_s,
            "attenuation_pct": 100 * (1 - p_s / p_w) if p_w > 0 else 0,
        }

    return results


# ---------------------------------------------------------------------------
# Grid convergence study
# ---------------------------------------------------------------------------

def grid_convergence_study(
    spacings_mm: List[float] = None,
    base_size_mm: float = 12.8,
    freq: float = 400e3,
) -> Dict:
    """Run simulations at different grid resolutions to check convergence."""
    import importlib.util, os
    biophys = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("jwa", os.path.join(biophys, "jwave_adapter.py"))
    jwa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jwa)

    if spacings_mm is None:
        spacings_mm = [0.4, 0.2, 0.1]

    results = {}
    for sp_mm in spacings_mm:
        grid_size = int(base_size_mm / sp_mm)
        sp_m = sp_mm * 1e-3
        source_pos = (grid_size // 2, grid_size // 8)
        target = (grid_size // 2, grid_size * 3 // 4)

        r = jwa.run_simulation_2d(
            (grid_size, grid_size), sp_m, 1500.0, 1000.0,
            source_pos, freq, 5e-5, pml_size=max(5, grid_size // 10),
        )
        results[sp_mm] = {
            "grid_size": grid_size,
            "p_max": float(jnp.max(r["p_max"])),
            "p_target": float(r["p_max"][target]),
        }

    return results


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def compute_sensitivity_map(
    grid_size: int = 32,
    spacing_mm: float = 0.5,
    freq: float = 400e3,
    target: Tuple[int, int] = (16, 25),
    skull_range: Tuple[int, int] = (8, 12),
) -> jnp.ndarray:
    """
    Compute d(p_target)/d(c) — sensitivity of focal pressure to sound speed field.

    Uses jax.grad through the simulation.

    Returns:
        2D array of sensitivities (same shape as simulation grid).
    """
    import importlib.util, os
    biophys = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("jwa", os.path.join(biophys, "jwave_adapter.py"))
    jwa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jwa)
    from jwave.geometry import TimeAxis

    spacing_m = spacing_mm * 1e-3
    domain = jwa.create_domain((grid_size, grid_size), spacing_m)
    source_pos = (grid_size // 2, 3)

    # Build reference medium for time axis
    c_ref, rho_ref = _make_skull_phantom_2d(grid_size, skull_range)
    medium_ref = jwa.create_medium(domain, c_ref, rho_ref, pml_size=5)
    time_axis = TimeAxis.from_medium(medium_ref, cfl=0.3, t_end=3e-5)
    dt = float(time_axis.dt)

    # Pre-build sources
    pos = jnp.array([[source_pos[0] * spacing_m, source_pos[1] * spacing_m]])
    sources = jwa.create_sources(domain, pos, freq, jnp.zeros(1), jnp.ones(1), dt=dt, n_cycles=3)

    def pressure_at_target(c_field):
        medium = jwa.create_medium(domain, c_field, rho_ref, pml_size=5)
        p = jwa.run_simulation_jax(medium, source_pos, freq, 3e-5,
                                    time_axis=time_axis, sources=sources)
        return p[target]

    sensitivity = jax.grad(pressure_at_target)(c_ref)
    return sensitivity


# ---------------------------------------------------------------------------
# Helmholtz (frequency-domain) solver
# ---------------------------------------------------------------------------

def run_helmholtz_simulation(
    grid_size: int = 64,
    spacing_mm: float = 0.2,
    freq: float = 400e3,
    source_pos: Tuple[int, int] = None,
) -> Dict:
    """
    Run frequency-domain Helmholtz solver.

    Returns dict with complex pressure field and absolute pressure.
    """
    from jwave.acoustics.time_harmonic import helmholtz_solver
    from jwave.geometry import Medium
    from jaxdf.discretization import FourierSeries
    from jaxdf.geometry import Domain

    spacing_m = spacing_mm * 1e-3
    domain = Domain((grid_size, grid_size), (spacing_m, spacing_m))

    if source_pos is None:
        source_pos = (grid_size // 2, grid_size // 8)

    # Medium — PML size must be small enough relative to grid
    # helmholtz_solver internally uses PML arrays that must broadcast with the grid
    pml = min(10, grid_size // 8)
    medium = Medium(domain=domain, sound_speed=1500.0, density=1000.0, pml_size=pml)

    # Angular frequency
    omega = 2 * np.pi * freq

    # Source: point source as FourierSeries (required by helmholtz_solver's
    # internal gradient operator dispatch — OnGrid is not supported)
    src_data = jnp.zeros((grid_size, grid_size, 1), dtype=jnp.complex64)
    src_data = src_data.at[source_pos[0], source_pos[1], 0].set(1.0 + 0j)
    source = FourierSeries(src_data, domain)

    # Solve
    p_complex = helmholtz_solver(medium, omega, source, tol=1e-2, maxiter=200)

    # Extract pressure
    p_data = p_complex.on_grid
    if p_data.shape[-1] == 1:
        p_data = p_data[..., 0]
    p_abs = jnp.abs(p_data)

    return {
        "p_complex": p_data,
        "p_abs": p_abs,
        "p_max": float(jnp.max(p_abs)),
    }


def compare_helmholtz_vs_timedomain(
    grid_size: int = 64,
    spacing_mm: float = 0.2,
    freq: float = 400e3,
) -> Dict:
    """Compare Helmholtz and time-domain solvers on same homogeneous problem."""
    import importlib.util, os
    biophys = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("jwa", os.path.join(biophys, "jwave_adapter.py"))
    jwa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jwa)

    source_pos = (grid_size // 2, grid_size // 8)
    spacing_m = spacing_mm * 1e-3

    # Time-domain
    td = jwa.run_simulation_2d(
        (grid_size, grid_size), spacing_m, 1500.0, 1000.0,
        source_pos, freq, 5e-5, pml_size=10,
    )

    # Helmholtz
    hz = run_helmholtz_simulation(grid_size, spacing_mm, freq, source_pos)

    # Normalize and compare
    td_norm = np.array(td["p_max"]) / np.max(td["p_max"]) if np.max(td["p_max"]) > 0 else np.array(td["p_max"])
    hz_norm = np.array(hz["p_abs"]) / np.max(hz["p_abs"]) if np.max(hz["p_abs"]) > 0 else np.array(hz["p_abs"])

    corr = float(np.corrcoef(td_norm.ravel(), hz_norm.ravel())[0, 1])

    return {
        "td_p_max": float(np.max(td["p_max"])),
        "hz_p_max": hz["p_max"],
        "correlation": corr,
    }

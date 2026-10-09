"""
Gradient-based delay optimization for transcranial focused ultrasound.

Uses j-Wave's differentiable acoustic simulation to optimize per-element
delays that compensate for skull-induced phase aberration, maximizing
focal pressure at the target while respecting safety constraints.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple


def focal_pressure_loss(
    p_field: jnp.ndarray,
    target_voxel: Tuple[int, ...],
) -> float:
    """
    Loss function: negative pressure at the target (minimize to maximize pressure).

    Args:
        p_field: Peak pressure array (any spatial shape).
        target_voxel: Grid indices of the target point.

    Returns:
        Scalar loss = -p_field[target_voxel].
    """
    return -p_field[target_voxel]


def optimize_delays(
    domain,
    medium_params: Dict[str, jnp.ndarray],
    element_positions: jnp.ndarray,
    target_voxel: Tuple[int, ...],
    freq: float,
    initial_delays: Optional[jnp.ndarray] = None,
    n_iters: int = 50,
    lr: float = 1e-7,
    n_cycles: int = 5,
    t_end: float = 2e-5,
    pml_size: float = 10.0,
) -> Tuple[jnp.ndarray, List[float]]:
    """
    Optimize transducer element delays to maximize focal pressure at target.

    Uses Adam optimizer with jax.grad through j-Wave simulation.

    Args:
        domain: j-Wave Domain.
        medium_params: Dict with 'sound_speed' and 'density' arrays.
        element_positions: (n_elements, ndim) physical coords in metres.
        target_voxel: Grid indices of focal target.
        freq: Transducer frequency (Hz).
        initial_delays: (n_elements,) starting delays. Defaults to zeros.
        n_iters: Optimization iterations.
        lr: Learning rate.
        n_cycles: Toneburst cycles.
        t_end: Simulation end time (seconds).
        pml_size: PML thickness.

    Returns:
        (optimized_delays, loss_history) tuple.
    """
    import optax
    # Import from sibling module — use relative import if running as package,
    # or importlib for standalone execution
    try:
        from dmipy_jax.biophysics.jwave_adapter import (
            create_medium,
            create_sources,
            run_simulation_jax,
        )
    except (ImportError, ModuleNotFoundError):
        import importlib.util as _ilu
        import os as _os
        _spec = _ilu.spec_from_file_location(
            "jwave_adapter",
            _os.path.join(_os.path.dirname(__file__), "jwave_adapter.py"),
        )
        _jwa = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_jwa)
        create_medium = _jwa.create_medium
        create_sources = _jwa.create_sources
        run_simulation_jax = _jwa.run_simulation_jax
    from jwave.geometry import TimeAxis

    n_elements = element_positions.shape[0]
    if initial_delays is None:
        initial_delays = jnp.zeros(n_elements)

    # Pre-compute medium, time_axis (not traced)
    medium = create_medium(
        domain,
        sound_speed=medium_params["sound_speed"],
        density=medium_params["density"],
        pml_size=pml_size,
    )
    time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=t_end)
    dt = float(time_axis.dt)

    def loss_fn(delays):
        sources = create_sources(
            domain, element_positions, freq,
            delays=delays, apodizations=jnp.ones(n_elements),
            dt=dt, n_cycles=n_cycles,
        )
        p_max = run_simulation_jax(
            medium,
            source_position=target_voxel,  # not used when sources provided
            freq=freq,
            t_end=t_end,
            time_axis=time_axis,
            sources=sources,
        )
        return focal_pressure_loss(p_max, target_voxel)

    # Adam optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(initial_delays)
    delays = initial_delays
    loss_history = []

    for i in range(n_iters):
        loss_val, grads = jax.value_and_grad(loss_fn)(delays)
        updates, opt_state = optimizer.update(grads, opt_state)
        delays = optax.apply_updates(delays, updates)
        loss_history.append(float(loss_val))

    return delays, loss_history

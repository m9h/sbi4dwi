"""
2-fibre forward model + acquisition shared by validate_force_replication_v2.py
and validate_force_snr_sweep.py.

Pure functions extracted from validate_force_replication_v2.py for the same
reason as three_fiber.py: independent testability and reuse.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator

# Re-export so call sites only have to import from one place
from validation.lib.three_fiber import (  # noqa: F401
    acq_to_gtab_si,
    make_multishell_acquisition,
)


def two_stick_signal(
    acq: JaxAcquisition,
    mu1: jnp.ndarray,
    mu2: jnp.ndarray,
    f1: float,
    f_iso: float = 0.05,
    d_par: float = 1.7e-9,
    d_iso: float = 3.0e-9,
) -> jnp.ndarray:
    """Two-stick + isotropic mixture signal under SI b-values.

    f2 = 1 - f1 - f_iso (implicit). At b=0 every term collapses to 1, so
    the weighted sum is exactly f1 + f2 + f_iso = 1.
    """
    f2 = 1.0 - f1 - f_iso
    cos1 = acq.gradient_directions @ mu1
    cos2 = acq.gradient_directions @ mu2
    s1 = jnp.exp(-acq.bvalues * d_par * cos1 ** 2)
    s2 = jnp.exp(-acq.bvalues * d_par * cos2 ** 2)
    s_iso = jnp.exp(-acq.bvalues * d_iso)
    return f1 * s1 + f2 * s2 + f_iso * s_iso


def build_two_stick_simulator(acq: JaxAcquisition) -> ModelSimulator:
    """5-param planar 2-stick simulator for dmipy-JAX library generation.

    params layout: ``[d_par, theta1, theta2, f1, f_iso]``
    Both sticks lie in the +x/+z plane: ``mu_i = [sin θ_i, 0, cos θ_i]``.
    """
    def forward_fn(params, acq):
        d_par = params[0]
        t1, t2 = params[1], params[2]
        f1, f_iso = params[3], params[4]
        f2 = 1.0 - f1 - f_iso
        mu1 = jnp.array([jnp.sin(t1), 0.0, jnp.cos(t1)])
        mu2 = jnp.array([jnp.sin(t2), 0.0, jnp.cos(t2)])
        cos1 = acq.gradient_directions @ mu1
        cos2 = acq.gradient_directions @ mu2
        s1 = jnp.exp(-acq.bvalues * d_par * cos1 ** 2)
        s2 = jnp.exp(-acq.bvalues * d_par * cos2 ** 2)
        s_iso = jnp.exp(-acq.bvalues * 3.0e-9)
        return f1 * s1 + f2 * s2 + f_iso * s_iso

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_par", "theta1", "theta2", "f1", "f_iso"],
        parameter_ranges={
            "d_par": (1.0e-9, 2.5e-9),
            "theta1": (0.0, float(jnp.pi)),
            "theta2": (0.0, float(jnp.pi)),
            "f1": (0.1, 0.8),
            "f_iso": (0.0, 0.2),
        },
        acquisition=acq,
    )


def add_rician_noise(
    signal: Float[Array, "M"],
    snr: float,
    key: PRNGKeyArray,
) -> Float[Array, "M"]:
    """Add Rician noise targeting *snr* on the b=0 = 1.0 reference.

    Equivalent to ``sqrt((S + n1)^2 + n2^2)`` with n1, n2 ~ N(0, 1/snr).
    Pure function — same key + same snr always produces the same noise.
    """
    sigma = 1.0 / snr
    k1, k2 = jax.random.split(key)
    n1 = jax.random.normal(k1, signal.shape) * sigma
    n2 = jax.random.normal(k2, signal.shape) * sigma
    return jnp.sqrt((signal + n1) ** 2 + n2 ** 2)

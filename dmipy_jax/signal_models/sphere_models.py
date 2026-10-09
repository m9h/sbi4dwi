import jax
import jax
import jax.numpy as jnp
from jax import jit, lax
import scipy.special as ssp
from jax.scipy import special as jsp
from dmipy_jax.constants import SPHERE_ROOTS, GYRO_MAGNETIC_RATIO, spherical_bessel_derivative_roots
import functools
import equinox as eqx
from typing import Any
from jaxtyping import Array

class S1Dot(eqx.Module):
    r"""
    The Dot model [1]_ - an non-diffusing compartment.
    It has no parameters and returns 1 no matter the input.

    References
    ----------
    .. [1] Panagiotaki et al.
           "Compartment models of the diffusion MR signal in brain white
            matter: a taxonomy and comparison". NeuroImage (2012)
    """

    parameter_names = []
    parameter_cardinality = {}
    parameter_ranges = {}
        
    def __call__(self, bvals, gradient_directions, **kwargs):
        # Return ones matching the shape of bvals
        return jnp.ones_like(bvals, dtype=float)


@jit
def g2_sphere_stejskal_tanner(q, diameter):
    """
    The Stejskal Tanner signal approximation of a sphere model.
    """
    radius = diameter / 2.0
    factor = 2 * jnp.pi * q * radius
    
    # Handle singularity at factor=0
    # sin(x)/x -> 1, cos(x) -> 1. (1-1) -> 0.
    # But prefactor 3/x^2 (singularity).
    # Expansion: sin(x)/x approx 1 - x^2/6. cos(x) approx 1 - x^2/2.
    # Bracket: (1 - x^2/6) - (1 - x^2/2) = x^2/3.
    # Result: 3/x^2 * (x^2/3) = 1.
    
    # Use safe division
    is_nonzero = jnp.abs(factor) > 1e-6
    safe_factor = jnp.where(is_nonzero, factor, 1.0)
    
    term = (jnp.sin(safe_factor) / safe_factor) - jnp.cos(safe_factor)
    
    # E = [3 * j1(z) / z]^2
    # term = z * j1(z)
    # We want 3 * (term/z) / z = 3 * term / z^2
    
    E = (3 * term / (safe_factor ** 2)) ** 2
    
    return jnp.where(is_nonzero, E, 1.0)


class SphereStejskalTanner(eqx.Module):
    r"""
    The Stejskal Tanner signal approximation of a sphere model.

    Parameters
    ----------
    diameter : float
        sphere diameter in meters.
    """

    parameter_names = ['diameter']
    parameter_cardinality = {'diameter': 1}
    parameter_ranges = {'diameter': (1e-6, 20e-6)}

    diameter: Any = None

    def __call__(self, bvals, gradient_directions, **kwargs):
        diameter = kwargs.get('diameter', self.diameter)
        
        # We need qvalues.
        # Check if 'qvalues' in kwargs, else derive from bvals/tau approx?
        # Stejskal Tanner theory strictly uses q.
        # q = 1/(2pi) * sqrt(b/tau) ?
        # Assuming acquisition provides 'qvalues' or we calculate q_mag.
        
        if 'qvalues' in kwargs:
             q = kwargs['qvalues']
        elif 'bvals' in kwargs or bvals is not None:
             # Try to derive q from bvals if tau provided
             # But bvals not passed if we use **kwargs? bvals passed explicitly in __call__ usually.
             
             # Fallback to approximating q from bvals if tau known is dangerous without G info.
             # However, typically in dMipy acquisition scheme, q is precalculated.
             # OptimistixFitter wrapper usually unpacks acquisition.
             # Let's try to calculate q if tau (or big/small delta) is present.
             
             if 'tau' in kwargs:
                 tau = kwargs['tau']
                 q = jnp.sqrt(bvals / (tau + 1e-12)) / (2 * jnp.pi)
             elif 'big_delta' in kwargs and 'small_delta' in kwargs:
                 tau = kwargs['big_delta'] - kwargs['small_delta']/3.0
                 q = jnp.sqrt(bvals / (tau + 1e-12)) / (2 * jnp.pi)
             else:
                 # Last resort: if bvals are 0, q is 0.
                 # If bvals > 0, we need timing.
                 # Raise error if timing missing.
                 raise ValueError("SphereStejskalTanner requires 'qvalues' or timing info ('tau' or 'big_delta'/'small_delta') to derive q.")
        else:
             raise ValueError("SphereStejskalTanner requires 'qvalues' in kwargs.")
             
        return g2_sphere_stejskal_tanner(q, diameter)



# Helper for spherical bessel using pure_callback to scipy
# This avoids the issue of bessel_jn requiring integer order in JAX
@jax.custom_jvp
def spherical_jn_jax(n, z):
    # Host callback
    # Result shape is same as z
    result_shape = jax.ShapeDtypeStruct(z.shape, z.dtype)
    
    def host_fn(n, z):
        return ssp.spherical_jn(int(n), z)
        
    return jax.pure_callback(host_fn, result_shape, n, z)

@spherical_jn_jax.defjvp
def spherical_jn_jvp(primals, tangents):
    n, z = primals
    _, z_dot = tangents
    
    val = spherical_jn_jax(n, z)
    
    # Derivative w.r.t z
    # j_n'(z)
    # Recursion: j_n'(z) = j_{n-1}(z) - (n+1)/z * j_n(z)
    
    safe_z = jnp.where(jnp.abs(z) < 1e-10, 1e-10, z)
    
    # Check n==0 case statically (n is python int usually if unrolled, or tracer?)
    # If unrolled via python loop, n is int.
    # If dynamic, we use lax.cond?
    # But for SphereCallaghan we forced python loop so n is static int.
    
    if isinstance(n, int):
        if n == 0:
            # j0'(z) = -j1(z)
            der = -spherical_jn_jax(1, z)
        else:
            jn_minus = spherical_jn_jax(n - 1, z)
            der = jn_minus - (n + 1) / safe_z * val
    else:
        # Dynamic n (fallback if used elsewhere)
        # Use lax.cond or where
        # We assume n is integer tracer
        jn_minus = spherical_jn_jax(n - 1, z)
        der_generic = jn_minus - (n + 1) / safe_z * val
        
        j1 = spherical_jn_jax(1, z)
        der_0 = -j1
        
        der = jax.lax.cond(n == 0, lambda: der_0, lambda: der_generic)

    return val, der * z_dot

def spherical_jn_derivative_jax(n, z):
    # Wrapper to get derivative directly if needed, or we can just use autodiff of spherical_jn_jax
    # But legacy code calls this explicit helper.
    # We can use the same logic as JVP.
    # Or just jax.grad? No, z is array.
    
    val = spherical_jn_jax(n, z)
    safe_z = jnp.where(jnp.abs(z) < 1e-10, 1e-10, z)
    
    if isinstance(n, int):
        if n == 0:
            return -spherical_jn_jax(1, z)
        else:
            return spherical_jn_jax(n - 1, z) - (n + 1) / safe_z * val
    else:
        return jax.lax.cond(
            n == 0, 
            lambda: -spherical_jn_jax(1, z), 
            lambda: spherical_jn_jax(n - 1, z) - (n + 1) / safe_z * val
        )


def _spherical_jn_stack_host(z, n_max):
    import numpy as np
    orders = np.arange(n_max + 1, dtype=np.float64)
    return ssp.spherical_jn(orders[:, None].astype(int), np.asarray(z, dtype=np.float64)[None, :]).astype(z.dtype)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def spherical_jn_stack(z, n_max):
    """``j_n(z)`` for all orders ``n = 0 .. n_max`` in one host callback; shape ``(n_max+1,) + z.shape``.

    Differentiable in ``z`` through ``j_n'(z) = j_{n-1}(z) - (n+1)/z j_n(z)`` (``j_0' = -j_1``).
    """
    z = jnp.asarray(z)
    out = jax.ShapeDtypeStruct((n_max + 1,) + z.shape, z.dtype)
    return jax.pure_callback(
        lambda z_in: _spherical_jn_stack_host(z_in.reshape(-1), n_max).reshape((n_max + 1,) + z_in.shape),
        out, z, vmap_method="sequential",
    )


@spherical_jn_stack.defjvp
def _spherical_jn_stack_jvp(n_max, primals, tangents):
    (z,) = primals
    (dz,) = tangents
    full = spherical_jn_stack(z, n_max + 1)
    val = full[:-1]
    return val, _spherical_jn_derivative_from_stack(full, z) * dz[None]


def _spherical_jn_derivative_from_stack(j, z):
    """``j_n'(z)`` for n = 0 .. len(j)-2 given ``j[n] = j_n(z)`` for n = 0 .. len(j)-1."""
    n_max = j.shape[0] - 2
    n = jnp.arange(1, n_max + 1, dtype=z.dtype)
    safe_z = jnp.where(jnp.abs(z) < jnp.finfo(z.dtype).tiny, 1.0, z)
    der_pos = j[0:n_max] - (n[:, None] + 1.0) / safe_z[None, :] * j[1:n_max + 1]
    der_0 = -j[1:2]
    return jnp.concatenate([der_0, der_pos], axis=0)


def g3_sphere_callaghan(q, tau, diameter, diffusion_constant, alpha):
    r"""Callaghan (1995) finite-time attenuation for diffusion inside a sphere
    with reflecting walls (narrow-pulse limit).

    .. math::

        E(q,\tau) = \frac{9 j_1(x)^2}{x^2}
          + \sum_{(n,k)\neq(0,0)} 6(2n+1)\, e^{-\alpha_{nk}^2 D\tau/R^2}
            \frac{\alpha_{nk}^2}{\alpha_{nk}^2 - n(n+1)}
            \frac{(x\, j_n'(x))^2}{(x^2-\alpha_{nk}^2)^2},
        \qquad x = 2\pi q R,

    where ``alpha[k, n]`` are the roots of ``j_n'`` from
    :func:`dmipy_jax.constants.spherical_bessel_derivative_roots`. The first
    term is the ``alpha = 0`` eigenmode and equals the Stejskal-Tanner sphere
    ``(3 j_1(x)/x)^2``, which the series reduces to as ``tau -> inf``; as
    ``D tau << R^2`` it tends to free diffusion ``exp(-b D)``.

    This is derived from the eigenfunction expansion (``psi_nkm = j_n(alpha r/R) Y_nm``,
    ``|psi_nk|^2`` normalisation ``R^3 j_n(alpha)^2 (1 - n(n+1)/alpha^2)/2``). The
    original dmipy ``S3SphereCallaghanApproximation`` used the cylindrical roots
    ``J_m'`` with weight ``alpha^2 - n(n-1)`` and an invalid
    ``spherical_jn`` call, so it is not usable as a reference; the limit tests in
    ``dmipy_jax/tests/test_callaghan_reference.py`` pin this implementation instead.

    Args:
        q: (N,) q-values in m^-1 (``q > 0``; the caller masks ``q = 0`` to 1).
        tau: scalar or (N,) diffusion time in s.
        diameter: sphere diameter in m.
        diffusion_constant: intra-sphere diffusivity in m^2/s.
        alpha: (n_roots, n_functions) root table.
    """
    q = jnp.asarray(q)
    tau = jnp.broadcast_to(jnp.asarray(tau, dtype=q.dtype), q.shape)
    radius = diameter / 2.0
    x = 2.0 * jnp.pi * q * radius
    x2 = x ** 2
    n_roots, n_functions = alpha.shape

    j = spherical_jn_stack(x, n_functions)                  # j_0 .. j_{n_functions}, (M+1, N)
    jp = _spherical_jn_derivative_from_stack(j, x)          # j_0' .. j_{M-1}', (M, N)

    alpha2 = alpha ** 2                                     # (K, M)
    n = jnp.arange(n_functions, dtype=x.dtype)
    nonzero_root = alpha2 > 0
    weight_denom = jnp.where(nonzero_root, alpha2 - n[None, :] * (n[None, :] + 1.0), 1.0)
    weight = jnp.where(nonzero_root, 6.0 * (2.0 * n[None, :] + 1.0) * alpha2 / weight_denom, 0.0)  # (K, M)

    decay = jnp.exp(-alpha2[None] * diffusion_constant * tau[:, None, None] / radius ** 2)  # (N, K, M)
    denom = (x2[:, None, None] - alpha2[None]) ** 2
    denom = jnp.maximum(denom, jnp.finfo(x.dtype).tiny)
    xjp2 = (x[None, :] * jp) ** 2                           # (M, N)
    series = jnp.sum(decay * weight[None] * xjp2.T[:, None, :] / denom, axis=(1, 2))

    zero_mode = 9.0 * j[1] ** 2 / x2
    return zero_mode + series


@jit
def g3_sphere(bvals, bvecs, diameter, diffusion_constant, big_delta, small_delta):
    """
    Computes signal for restricted diffusion in a Sphere (Soma) using
    the Gaussian Phase Distribution (GPD) approximation (Murday & Cotts, 1968).
    """
    radius = diameter / 2.0
    
    # 1. Gradients
    tau = big_delta - small_delta / 3.0
    G_mag = jnp.sqrt(bvals / (tau + 1e-9)) / (GYRO_MAGNETIC_RATIO * small_delta)
    
    # 2. Roots (Dimensionless mu)
    # alpha_sq_broad = mu^2
    alpha_sq = SPHERE_ROOTS ** 2
    alpha_sq_broad = alpha_sq[None, :] 
    
    # 3. Time Constants (Dm = D * mu^2 / R^2)
    # This part was correct
    Dm_alpha2 = diffusion_constant * alpha_sq_broad / (radius**2)
    
    # 4. The Denominator (The Fix)
    # We use dimensionless mu for the check (mu^2 - 2)
    # The physical 1/alpha^2 term contributes an R^2 factor to the numerator
    # because 1/alpha_phys^2 = R^2 / mu^2
    denom_dimensionless = alpha_sq_broad * (alpha_sq_broad - 2)
    
    # 5. Time Terms
    # Ensure deltas broadcast against roots (axis 1)
    # small_delta/big_delta: (N,) or scalar -> expand to (N, 1) or (1, 1) if not already
    sd_expanded = jnp.expand_dims(small_delta, -1) if jnp.ndim(small_delta) > 0 else small_delta
    bd_expanded = jnp.expand_dims(big_delta, -1) if jnp.ndim(big_delta) > 0 else big_delta
    
    exp_1 = jnp.exp(-Dm_alpha2 * sd_expanded)
    exp_2 = jnp.exp(-Dm_alpha2 * bd_expanded)
    exp_3 = jnp.exp(-Dm_alpha2 * (bd_expanded - sd_expanded))
    exp_4 = jnp.exp(-Dm_alpha2 * (bd_expanded + sd_expanded))
    
    time_term = (
        2 * sd_expanded 
        - (2 + exp_3 - 2 * exp_2 - 2 * exp_1 + exp_4) / Dm_alpha2
    )
    
    # 6. Summation
    # We multiply by radius**2 because of the conversion from physical alpha to mu
    # We also need to divide by Dm_alpha2 ($D \alpha^2$) as per the formula
    sum_term = jnp.sum((time_term / denom_dimensionless / Dm_alpha2), axis=1) * (radius ** 2)
    
    # 7. Final Signal
    prefactor = 2 * (GYRO_MAGNETIC_RATIO * G_mag) ** 2
    log_E = -prefactor * sum_term
    
    return jnp.exp(log_E)


class SphereGPD(eqx.Module):
    r"""
    The Gaussian Phase Distribution (GPD) approximation of the Sphere model [1]_.
    Also known as the Murday-Cotts model.

    Parameters
    ----------
    diameter : float
        sphere diameter in meters.
    diffusion_constant : float
        diffusion constant in m^2/s.

    References
    ----------
    .. [1] Murday, J. S., and R. M. Cotts. "Self-diffusion coefficient of
            liquid lithium." The Journal of Chemical Physics 48.11 (1968):
            4938-4945.
    """
    
    diameter: Any = None
    diffusion_constant: Any = None

    parameter_names = ('diameter', 'diffusion_constant')
    parameter_cardinality = {'diameter': 1, 'diffusion_constant': 1}
    parameter_ranges = {
        'diameter': (1e-6, 20e-6),
        'diffusion_constant': (0.1e-9, 3e-9)
    }

    def __init__(self, diameter=None, diffusion_constant=None):
        self.diameter = diameter
        self.diffusion_constant = diffusion_constant

    def __call__(self, bvals, gradient_directions, **kwargs):
        diameter = kwargs.get('diameter', self.diameter)
        diffusion_constant = kwargs.get('diffusion_constant', self.diffusion_constant)
        
        big_delta = kwargs.get('big_delta')
        small_delta = kwargs.get('small_delta')
        
        if big_delta is None or small_delta is None:
             raise ValueError("SphereGPD requires 'big_delta' and 'small_delta' in kwargs.")

        return g3_sphere(bvals, gradient_directions, diameter, diffusion_constant, big_delta, small_delta)



class SphereCallaghan(eqx.Module):
    r"""
    The Callaghan model [1]_ of diffusion inside a sphere.
    """
    
    diameter: Any = None
    diffusion_constant: Any = None
    number_of_roots: int = eqx.field(static=True, default=20)
    number_of_functions: int = eqx.field(static=True, default=50)
    alpha: Array = eqx.field(init=False)

    parameter_names = ('diameter', 'diffusion_constant')
    parameter_cardinality = {'diameter': 1, 'diffusion_constant': 1}
    parameter_ranges = {
        'diameter': (1e-6, 20e-6),
        'diffusion_constant': (0.1e-9, 3e-9)
    }

    def __init__(self, diameter=None, diffusion_constant=None, number_of_roots=20, number_of_functions=50):
        self.diameter = diameter
        self.diffusion_constant = diffusion_constant
        self.number_of_roots = number_of_roots
        self.number_of_functions = number_of_functions
        
        self.alpha = spherical_bessel_derivative_roots(number_of_roots, number_of_functions)

    def __call__(self, bvals, gradient_directions, **kwargs):
        diameter = kwargs.get('diameter', self.diameter)
        diff_const = kwargs.get('diffusion_constant', self.diffusion_constant)
        
        # Need q and tau.
        if 'qvalues' in kwargs:
            q = kwargs['qvalues']
        elif bvals is not None and 'big_delta' in kwargs and 'small_delta' in kwargs:
             tau = kwargs['big_delta'] - kwargs['small_delta']/3.0
             q = jnp.sqrt(bvals / (tau + 1e-12)) / (2 * jnp.pi)
        elif bvals is not None and 'tau' in kwargs:
             # If tau provided but not big/small
             q = jnp.sqrt(bvals / (kwargs['tau'] + 1e-12)) / (2 * jnp.pi)
        else:
             raise ValueError("SphereCallaghan requires 'qvalues' or bvals+timing.")
             
        if 'tau' in kwargs:
            tau = kwargs['tau']
        elif 'big_delta' in kwargs and 'small_delta' in kwargs:
            tau = kwargs['big_delta'] - kwargs['small_delta']/3.0
        else:
             raise ValueError("SphereCallaghan requires 'tau' or 'big_delta'/'small_delta'.")

        q = jnp.asarray(q)
        nonzero = q * diameter > 1e-12
        safe_q = jnp.where(nonzero, q, 1.0 / (jnp.pi * diameter))   # x = 2 where masked
        e = g3_sphere_callaghan(safe_q, tau, diameter, diff_const, self.alpha)
        return jnp.where(nonzero, e, 1.0)
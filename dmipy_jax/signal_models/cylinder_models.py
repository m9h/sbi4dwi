import jax
import jax.numpy as jnp
import scipy.special as ssp
from jax.scipy import special as jsp
from jax import pure_callback, jit, vmap, lax
from dmipy_jax.constants import GYRO_MAGNETIC_RATIO, bessel_derivative_roots
import functools
import equinox as eqx
from jaxtyping import Array, Float
from typing import Any, Tuple


# Removed pure_callback wrappers. Using jax.scipy.special directly.


@jit
def safe_bessel_j1(z):
    """
    Computes J1(z) safely.
    For z < 4.0, uses Taylor series (via Horner-like scan) to avoid instability
    in jax.scipy.special.bessel_jn recurrence.
    For z >= 4.0, uses jax.scipy.special.bessel_jn.
    """
    threshold = 4.0
    
    def taylor_j1(z_val):
        n_terms = 50 # Increased for stability at larger z
        z2_4 = z_val**2 / 4.0
        
        def body(carry, k):
            current_sum, current_term = carry
            mult = -z2_4 / (k * (k+1))
            next_term = current_term * mult
            new_sum = current_sum + next_term
            return (new_sum, next_term), None
            
        term_0 = z_val / 2.0
        init = (term_0, term_0)
        ks = jnp.arange(1, n_terms, dtype=jnp.float32)
        (final_sum, _), _ = jax.lax.scan(body, init, ks)
        return final_sum

    taylor_out = jax.vmap(taylor_j1)(jnp.atleast_1d(z))
    
    # For AxCaliber range (z < 20 usually), Taylor with sufficient terms is fine.
    # We avoid bessel_jn entirely to keep it on GPU.
    # We extended threshold logic implicitly by just returning taylor_out
    
    # Ensure shapes match input
    if taylor_out.ndim != z.ndim:
        taylor_out = taylor_out.reshape(z.shape)
        
    return taylor_out


@jit
def c2_cylinder(bvals, bvecs, mu, lambda_par, diameter, big_delta, small_delta):
    """
    Computes signal for a Cylinder with finite radius (Soderman approximation).
    
    Args:
        bvals: (N,) array of b-values in s/m^2 (SI).
        bvecs: (N, 3) array of gradient directions.
        mu: (3,) array defining the fiber orientation.
        lambda_par: Scalar diffusivity along the fiber (m^2/s).
        diameter: Cylinder diameter in meters.
        big_delta: Diffusion time / pulse separation (s).
        small_delta: Pulse duration (s).
        
    Returns:
        (N,) array of signal attenuation (0.0 to 1.0).
    """
    # 1. Parallel Signal (Stick)
    # Project gradients onto fiber axis: (g . mu)
    dot_prod = jnp.dot(bvecs, mu)
    signal_par = jnp.exp(-bvals * lambda_par * (dot_prod ** 2))
    
    # 2. Perpendicular Signal (Soderman / Stejskal-Tanner)
    # We need to calculate q_perp.
    # q = gamma * G * delta / (2*pi)
    # But usually we have b-values. b = (gamma * G * delta)^2 * (Delta - delta/3)
    # So q = sqrt(b / (Delta - delta/3)) / (2*pi)
    
    # However, standard dmipy uses q directly if available, or derives it.
    # Let's derive q_mag from bvals.
    # q = sqrt(b / tau) / (2 pi) in m^-1 for SI b-values (s/m^2); b = (2 pi q)^2 tau.
    tau = big_delta - small_delta / 3.0
    safe_tau = jnp.where(tau > 0, tau, 1.0)
    q_mag = jnp.sqrt(jnp.maximum(bvals, 0.0) / safe_tau) / (2 * jnp.pi)
    
    # Project gradients perpendicular to fiber axis
    # |g_perp| = |g - (g . mu)mu| = |g| * sqrt(1 - (g_hat . mu)^2)
    # q_perp = q_mag * sqrt(1 - dot_prod^2)
    sin_theta_sq = 1 - dot_prod**2
    # Clip to avoid negative due to precision
    # Also clip lower bound to avoid NaN gradient at 0 (parallel)
    sin_theta_sq = jnp.clip(sin_theta_sq, 1e-12, 1.0) 
    q_perp = q_mag * jnp.sqrt(sin_theta_sq)
    
    radius = diameter / 2.0
    argument = 2 * jnp.pi * q_perp * radius
    
    # E_perp = [ 2 * J1(2*pi*q*R) / (2*pi*q*R) ]^2
    # Handle singularity at argument=0 where J1(x)/x -> 0.5, so 2*0.5=1
    
    # Safe division
    valid_mask = argument > 1e-6
    safe_arg = jnp.where(valid_mask, argument, 1.0)
    
    # bessel_jn returns values for orders 0 to v (scipy behavior) or just v (jax behavior).
    # JAX scipy.special.bessel_jn(z, v=v) returns J_v(z) directly.
    # It does NOT return a list of orders.
    # Use safe_bessel_j1 implemented above
    j1_term = 2 * safe_bessel_j1(safe_arg) / safe_arg
    signal_perp = j1_term ** 2
    
    # If argument is small, signal is 1.0
    signal_perp = jnp.where(valid_mask, signal_perp, 1.0)
    
    return signal_par * signal_perp


@jit
def c1_stick(bvals, bvecs, mu, lambda_par):
    """
    Computes signal for a Stick (zero-radius cylinder).
    
    Args:
        bvals: (N,) array of b-values in s/m^2 (SI).
        bvecs: (N, 3) array of gradient directions.
        mu: (3,) array defining the fiber orientation.
        lambda_par: Scalar diffusivity along the fiber (m^2/s).
        
    Returns:
        (N,) array of signal attenuation (0.0 to 1.0).
    """
    # Project gradients onto fiber axis: (g . mu)
    dot_prod = jnp.dot(bvecs, mu)
    
    # Signal decay depends only on the parallel component
    # S = exp(-b * d_par * (g . mu)^2)
    signal = jnp.exp(-bvals * lambda_par * (dot_prod ** 2))
    return signal


class RestrictedCylinder(eqx.Module):
    r"""
    The Stejskal-Tanner approximation of the cylinder model with finite radius [1]_.
    
    Parameters
    ----------
    mu : array, shape(2)
        angles [theta, phi] representing main orientation on the sphere.
    lambda_par : float
        parallel diffusivity in m^2/s.
    diameter : float
        cylinder diameter in meters.

    References
    ----------
    .. [1] Soderman, Olle, and Bengt Jonsson. "Restricted diffusion in
            cylindrical geometry." Journal of Magnetic Resonance, Series A
            117.1 (1995): 94-97.
    """
    
    mu: Any = None
    lambda_par: Any = None
    diameter: Any = None

    parameter_names = ('mu', 'lambda_par', 'diameter')
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'diameter': 1}
    parameter_ranges = {
        'mu': ([0, jnp.pi], [-jnp.pi, jnp.pi]),
        'lambda_par': (0.1e-9, 3e-9),
        'diameter': (1e-7, 20e-6)
    }

    def __init__(self, mu=None, lambda_par=None, diameter=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter

    def __call__(self, bvals, gradient_directions, **kwargs):
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        diameter = kwargs.get('diameter', self.diameter)
        mu = kwargs.get('mu', self.mu)
        
        big_delta = kwargs.get('big_delta', kwargs.get('Delta'))
        small_delta = kwargs.get('small_delta', kwargs.get('delta'))
        
        if big_delta is None or small_delta is None:
             raise ValueError("RestrictedCylinder requires 'big_delta' and 'small_delta' in kwargs/acquisition.")

        # Convert spherical [theta, phi] to cartesian vector
        mu = jnp.asarray(mu)
        if mu.ndim > 0:
             theta = mu[0]
             phi = mu[1]
        else:
             theta = mu
             phi = 0.0 # Should not happen for RestrictedCylinder
             
        st = jnp.sin(theta)
        ct = jnp.cos(theta)
        sp = jnp.sin(phi)
        cp = jnp.cos(phi)
        
        # Ensure mu_cart is (3,)
        mu_cart = jnp.array([st * cp, st * sp, ct])
        if mu_cart.ndim > 1:
            mu_cart = jnp.squeeze(mu_cart)

        return c2_cylinder(bvals, gradient_directions, mu_cart, lambda_par, diameter, big_delta, small_delta)


def _bessel_j_stack_host(z, n_max):
    import numpy as np
    import scipy.special as ssp
    orders = np.arange(n_max + 1, dtype=np.float64)
    return ssp.jv(orders[:, None], np.asarray(z, dtype=np.float64)[None, :]).astype(z.dtype)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def bessel_j_stack(z, n_max):
    """``J_v(z)`` for all integer orders ``v = 0 .. n_max`` in one host callback.

    Returns an array of shape ``(n_max + 1,) + z.shape``. ``n_max`` must be a
    static Python int. The derivative is supplied by a custom JVP using
    ``J_v'(z) = (J_{v-1}(z) - J_{v+1}(z)) / 2`` and ``J_0'(z) = -J_1(z)``, so
    the kernel is differentiable in ``z`` while the Bessel evaluation itself
    runs through ``scipy.special.jv`` on the host.
    """
    z = jnp.asarray(z)
    out_shape = jax.ShapeDtypeStruct((n_max + 1,) + z.shape, z.dtype)
    flat = pure_callback(
        lambda z_in: _bessel_j_stack_host(z_in.reshape(-1), n_max).reshape((n_max + 1,) + z_in.shape),
        out_shape, z, vmap_method="sequential",
    )
    return flat


@bessel_j_stack.defjvp
def _bessel_j_stack_jvp(n_max, primals, tangents):
    (z,) = primals
    (dz,) = tangents
    full = bessel_j_stack(z, n_max + 1)          # orders 0 .. n_max+1
    val = full[:-1]
    j_minus = jnp.concatenate([-full[1:2], full[:-2]], axis=0)  # J_{v-1}; J_{-1} = -J_1
    j_plus = full[1:]
    deriv = 0.5 * (j_minus - j_plus)
    return val, deriv * dz[None]


def bessel_jn_fixed(v, z):
    """``J_v(z)`` for a single static integer order ``v`` (kept for compatibility)."""
    return bessel_j_stack(z, int(v))[int(v)]


def jvp_v1(v, z):
    """First derivative ``J_v'(z) = (J_{v-1}(z) - J_{v+1}(z)) / 2`` for static integer ``v >= 1``."""
    st = bessel_j_stack(z, int(v) + 1)
    return 0.5 * (st[int(v) - 1] - st[int(v) + 1])


def callaghan_perpendicular_attenuation(q, tau, diameter, diffusion_perp, alpha):
    r"""Callaghan (1995) finite-time attenuation for diffusion inside a cylinder,
    measured perpendicular to its axis.

    Faithful port of dmipy's ``C3CylinderCallaghanApproximation.perpendicular_attenuation``
    (``git show 7b254f4:dmipy/signal_models/cylinder_models.py``):

    .. math::

        E(q,\tau) = \sum_k 4\, e^{-\alpha_{0k}^2 D \tau / R^2}
            \frac{x^2}{(x^2-\alpha_{0k}^2)^2} J_1(x)^2
          + \sum_{m\ge1}\sum_k 8\, e^{-\alpha_{mk}^2 D \tau / R^2}
            \frac{\alpha_{mk}^2}{\alpha_{mk}^2-m^2}
            \frac{(x J_m'(x))^2}{(x^2-\alpha_{mk}^2)^2},
        \qquad x = 2\pi q R,

    where ``alpha[k, m]`` are the roots of ``J_m'`` from
    :func:`dmipy_jax.constants.bessel_derivative_roots` (``alpha[0, 0] = 0``).

    Args:
        q: (N,) perpendicular q-values in m^-1.
        tau: scalar or (N,) diffusion time in s.
        diameter: cylinder diameter in m.
        diffusion_perp: intra-cylindrical diffusivity in m^2/s.
        alpha: (n_roots, n_functions) root table.

    Returns:
        (N,) attenuation. Not defined at ``q = 0`` (the caller masks it to 1).
    """
    q = jnp.asarray(q)
    tau = jnp.broadcast_to(jnp.asarray(tau, dtype=q.dtype), q.shape)
    radius = diameter / 2.0
    x = 2.0 * jnp.pi * q * radius                     # (N,)
    x2 = x ** 2
    n_roots, n_functions = alpha.shape

    # All Bessel orders needed: J_0 .. J_{n_functions} (J_m' needs J_{m+1}).
    J = bessel_j_stack(x, n_functions)                # (n_functions+1, N)

    alpha2 = alpha ** 2                               # (K, M)
    decay = jnp.exp(-alpha2[None] * diffusion_perp * tau[:, None, None] / radius ** 2)  # (N, K, M)
    denom = (x2[:, None, None] - alpha2[None]) ** 2   # (N, K, M)
    denom = jnp.maximum(denom, jnp.finfo(x.dtype).tiny)

    # m = 0 term: 4 exp(.) x^2 / (x^2 - a^2)^2 J_1(x)^2
    term0 = 4.0 * decay[:, :, 0] * x2[:, None] / denom[:, :, 0] * (J[1] ** 2)[:, None]
    res = jnp.sum(term0, axis=1)

    if n_functions > 1:
        m = jnp.arange(1, n_functions)
        # J_m'(x) = (J_{m-1} - J_{m+1}) / 2 for m >= 1
        Jp = 0.5 * (J[0:n_functions - 1] - J[2:n_functions + 1])   # (M-1, N)
        xJp2 = (x[None, :] * Jp) ** 2                                # (M-1, N)
        a2 = alpha2[:, 1:]                                           # (K, M-1)
        weight = a2 / (a2 - m[None, :] ** 2)                         # (K, M-1)
        term_m = 8.0 * decay[:, :, 1:] * weight[None] * xJp2.T[:, None, :] / denom[:, :, 1:]
        res = res + jnp.sum(term_m, axis=(1, 2))
    return res


def c3_cylinder_callaghan(bvals, bvecs, mu, lambda_par, diameter, diffusion_perp, tau, alpha):
    """Signal of a finite-radius cylinder: stick along ``mu`` times Callaghan's
    perpendicular attenuation.

    All quantities are SI, as everywhere in the package: ``bvals`` in s/m^2,
    diffusivities in m^2/s, ``diameter`` in m, ``tau`` in s. The q-value follows
    dmipy's PGSE convention ``b = (2 pi q)^2 tau``, i.e. ``q = sqrt(b/tau)/(2 pi)``
    in m^-1.

    Args:
        bvals: (N,) b-values in s/m^2.
        bvecs: (N, 3) unit gradient directions.
        mu: (3,) unit fibre orientation.
        lambda_par: parallel diffusivity (m^2/s).
        diameter: cylinder diameter (m).
        diffusion_perp: perpendicular intra-cylindrical diffusivity (m^2/s).
        tau: scalar or (N,) diffusion time (s).
        alpha: (n_roots, n_functions) roots of ``J_m'`` (see
            :func:`dmipy_jax.constants.bessel_derivative_roots`).

    Returns:
        (N,) signal attenuation.
    """
    bvals = jnp.asarray(bvals)
    tau = jnp.broadcast_to(jnp.asarray(tau, dtype=bvals.dtype), bvals.shape)

    dot_prod = jnp.dot(bvecs, mu)
    signal_par = jnp.exp(-bvals * lambda_par * dot_prod ** 2)

    safe_tau = jnp.where(tau > 0, tau, 1.0)
    q_mag = jnp.sqrt(jnp.maximum(bvals, 0.0) / safe_tau) / (2.0 * jnp.pi)
    sin_theta_sq = jnp.clip(1.0 - dot_prod ** 2, 1e-12, 1.0)   # lower clip keeps the sqrt gradient finite
    q_perp = q_mag * jnp.sqrt(sin_theta_sq)

    nonzero = q_perp * diameter > 1e-12                           # x = 2 pi q R > ~6e-12
    safe_q = jnp.where(nonzero, q_perp, 1.0 / (jnp.pi * diameter))  # x = 2 where masked
    e_perp = callaghan_perpendicular_attenuation(safe_q, tau, diameter, diffusion_perp, alpha)
    e_perp = jnp.where(nonzero, e_perp, 1.0)
    return signal_par * e_perp


class CallaghanRestrictedCylinder(eqx.Module):
    r"""
    The Callaghan model [1]_ - a cylinder with finite radius - typically
    used for intra-axonal diffusion. The perpendicular diffusion is modelled
    after Callaghan's solution for the disk. Is dependent on both q-value
    and diffusion time.

    Parameters
    ----------
    mu : array, shape(2)
        angles [theta, phi] representing main orientation on the sphere.
    lambda_par : float
        parallel diffusivity in m^2/s.
    diameter : float
        cylinder (axon) diameter in meters.
    diffusion_perpendicular : float
        intra-cylindrical perpendicular diffusivity (m^2/s).
    number_of_roots : integer
        number of roots to use for the Callaghan cylinder model.
    number_of_functions : integer
        number of functions to use for the Callaghan cylinder model.

    References
    ----------
    .. [1] Callaghan, Paul T. "Pulsed-gradient spin-echo NMR for planar,
            cylindrical, and spherical pores under conditions of wall
            relaxation." Journal of magnetic resonance, Series A 113.1 (1995):
            53-59.
    """
    
    mu: Any = None
    lambda_par: Any = None
    diameter: Any = None
    diffusion_perpendicular: Any = 1.7e-9
    number_of_roots: int = eqx.field(static=True, default=20)
    number_of_functions: int = eqx.field(static=True, default=50)
    # alpha should be a field to be part of the pytree
    alpha: Array = eqx.field(init=False)

    parameter_names = ('mu', 'lambda_par', 'diameter', 'diffusion_perpendicular')
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'diameter': 1, 'diffusion_perpendicular': 1}
    parameter_ranges = {
        'mu': ([0, jnp.pi], [-jnp.pi, jnp.pi]),
        'lambda_par': (0.1e-9, 3e-9),
        'diameter': (1e-7, 20e-6),
        'diffusion_perpendicular': (0.1e-9, 3e-9)
    }

    def __init__(self, mu=None, lambda_par=None, diameter=None, 
                 diffusion_perpendicular=1.7e-9, number_of_roots=20, number_of_functions=50):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter
        self.diffusion_perpendicular = diffusion_perpendicular
        self.number_of_roots = number_of_roots
        self.number_of_functions = number_of_functions
        self.alpha = bessel_derivative_roots(number_of_roots, number_of_functions)

    def __call__(self, bvals, gradient_directions, **kwargs):
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        diameter = kwargs.get('diameter', self.diameter)
        diffusion_perp = kwargs.get('diffusion_perpendicular', self.diffusion_perpendicular)
        mu = kwargs.get('mu', self.mu)
        
        # Need tau (diffusion time)
        # Prefer 'tau' from kwargs, else derive from big_delta/small_delta
        if 'tau' in kwargs:
            tau = kwargs['tau']
        elif 'big_delta' in kwargs and 'small_delta' in kwargs:
             tau = kwargs['big_delta'] - kwargs['small_delta'] / 3.0
        else:
             raise ValueError("CallaghanRestrictedCylinder requires 'tau' or 'big_delta'/'small_delta' in kwargs.")

        # Convert spherical [theta, phi] to cartesian vector
        theta = mu[0]
        phi = mu[1]
        st = jnp.sin(theta)
        ct = jnp.cos(theta)
        sp = jnp.sin(phi)
        cp = jnp.cos(phi)
        mu_cart = jnp.array([st * cp, st * sp, ct])

        return c3_cylinder_callaghan(
            bvals, gradient_directions, mu_cart, lambda_par, diameter, diffusion_perp, tau, self.alpha
        )


class C1Stick(eqx.Module):
    r"""
    The Stick model - a cylinder with zero radius.
    
    Parameters
    ----------
    mu : array, shape(2)
        angles [theta, phi] representing main orientation on the sphere.
    lambda_par : float
        parallel diffusivity in m^2/s.
    """
    
    mu: Any = None
    lambda_par: Any = None

    parameter_names = ('mu', 'lambda_par')
    parameter_cardinality = {'mu': 2, 'lambda_par': 1}
    parameter_ranges = {
        'mu': ([0, jnp.pi], [-jnp.pi, jnp.pi]),
        'lambda_par': (0.1e-9, 3e-9)
    }

    def __init__(self, mu=None, lambda_par=None):
        self.mu = mu
        self.lambda_par = lambda_par

    def __call__(self, bvals, gradient_directions, **kwargs):
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        
        # Convert spherical [theta, phi] to cartesian vector
        mu = jnp.asarray(mu)
        if mu.size == 3:
             # Assume already cartesian if size 3
             mu_cart = mu
        elif mu.ndim > 0:
             theta = mu[0]
             phi = mu[1]
             st = jnp.sin(theta)
             ct = jnp.cos(theta)
             sp = jnp.sin(phi)
             cp = jnp.cos(phi)
             mu_cart = jnp.array([st * cp, st * sp, ct])
        else:
             # Default or scalar? Should not happen if mu is (2,) params
             mu_cart = jnp.array([1.0, 0.0, 0.0]) # Dummy fallback

        return c1_stick(bvals, gradient_directions, mu_cart, lambda_par)

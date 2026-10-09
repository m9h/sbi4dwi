"""Physical constants and pre-computed numerical tables for dMRI models.

Stores physical constants and expensive pre-computed values as JAX arrays
to avoid repeated computation on the GPU.

Constants:

* ``SPHERE_ROOTS`` -- the 100 roots of the transcendental equation
  ``1/(alpha R) J_{3/2}(alpha R) = J_{5/2}(alpha R)`` (R = 1), i.e. the
  roots of the derivative of the order-3/2 spherical Bessel function, used
  by the Sphere GPD model (Murday-Cotts / Balinov) in
  :mod:`dmipy_jax.signal_models.sphere_models` and
  :mod:`dmipy_jax.signal_models.sandi`. Identical to dmipy's
  ``SPHERE_TRASCENDENTAL_ROOTS`` (``git show 7b254f4:dmipy/signal_models/sphere_models.py``).
* ``GYRO_MAGNETIC_RATIO`` -- proton gyromagnetic ratio
  (2.6752e8 rad s^-1 T^-1), used by all simulation backends that convert
  gradient amplitude to spin phase.

Functions:

* :func:`bessel_derivative_roots` -- table of roots of ``J_m'(x) = 0`` shared
  by the Callaghan cylinder and sphere models.
"""

import jax.numpy as jnp
import numpy as np

SPHERE_ROOTS = jnp.array([
    2.081575978, 5.940369990, 9.205840145,
    12.40444502, 15.57923641, 18.74264558, 21.89969648,
    25.05282528, 28.20336100, 31.35209173, 34.49951492,
    37.64596032, 40.79165523, 43.93676147, 47.08139741,
    50.22565165, 53.36959180, 56.51327045, 59.65672900,
    62.80000055, 65.94311190, 69.08608495, 72.22893775,
    75.37168540, 78.51434055, 81.65691380, 84.79941440,
    87.94185005, 91.08422750, 94.22655255, 97.36883035,
    100.5110653, 103.6532613, 106.7954217, 109.9375497,
    113.0796480, 116.2217188, 119.3637645, 122.5057870,
    125.6477880, 128.7897690, 131.9317315, 135.0736768,
    138.2156061, 141.3575204, 144.4994207, 147.6413080,
    150.7831829, 153.9250463, 157.0668989, 160.2087413,
    163.3505741, 166.4923978, 169.6342129, 172.7760200,
    175.9178194, 179.0596116, 182.2013968, 185.3431756,
    188.4849481, 191.6267147, 194.7684757, 197.9102314,
    201.0519820, 204.1937277, 207.3354688, 210.4772054,
    213.6189378, 216.7606662, 219.9023907, 223.0441114,
    226.1858287, 229.3275425, 232.4692530, 235.6109603,
    238.7526647, 241.8943662, 245.0360648, 248.1777608,
    251.3194542, 254.4611451, 257.6028336, 260.7445198,
    263.8862038, 267.0278856, 270.1695654, 273.3112431,
    276.4529189, 279.5945929, 282.7362650, 285.8779354,
    289.0196041, 292.1612712, 295.3029367, 298.4446006,
    301.5862631, 304.7279241, 307.8695837, 311.0112420,
    314.1528990
])

# Backwards-compatible alias (dmipy's name).
SPHERE_TRASCENDENTAL_ROOTS = SPHERE_ROOTS

# Gyromagnetic ratio for Hydrogen (rad * Hz / T)
# Or simplified: 2.675987E8 rad/s/T. 
# In dmipy we often work with 'q' directly, but if we need G, we need Gamma.
GYRO_MAGNETIC_RATIO = 267.5152549e6


def bessel_derivative_roots(n_roots: int, n_functions: int) -> jnp.ndarray:
    """Roots ``alpha[k, m]`` of ``J_m'(alpha) = 0`` as used by Callaghan (1995).

    Column ``m`` holds the first ``n_roots`` positive roots of the derivative
    of the Bessel function of order ``m``, except that column 0 starts with
    the trivial root ``alpha[0, 0] = 0`` (``J_0'(0) = 0``) followed by the
    first ``n_roots - 1`` non-trivial roots. This reproduces dmipy's
    ``C3CylinderCallaghanApproximation`` / ``S3SphereCallaghanApproximation``
    tables exactly; the zero root carries the long-time (Soderman) limit of
    the cylinder series and must not be omitted.

    Computed on the host with ``scipy.special.jnp_zeros`` and returned as a
    JAX array of shape ``(n_roots, n_functions)``.
    """
    import scipy.special as ssp

    alpha = np.empty((n_roots, n_functions))
    alpha[0, 0] = 0.0
    if n_roots > 1:
        alpha[1:, 0] = ssp.jnp_zeros(0, n_roots - 1)
    for m in range(1, n_functions):
        alpha[:, m] = ssp.jnp_zeros(m, n_roots)
    return jnp.asarray(alpha)


def spherical_bessel_derivative_roots(n_roots: int, n_functions: int) -> jnp.ndarray:
    """Roots ``alpha[k, n]`` of ``j_n'(alpha) = 0`` (spherical Bessel functions).

    These are the eigenvalues for diffusion inside a sphere with reflecting
    walls (Callaghan 1995, sphere case). Column 0 starts with the trivial root
    ``alpha[0, 0] = 0`` followed by the roots of ``j_1`` (4.4934, 7.7253, ...);
    column 1 is the classic Murday-Cotts table ``SPHERE_ROOTS`` (2.0816, 5.9404,
    ...). Found by sign scanning plus Brent refinement on the host; returned as
    a JAX array of shape ``(n_roots, n_functions)``.
    """
    import scipy.special as ssp
    from scipy import optimize

    alpha = np.zeros((n_roots, n_functions))
    step = 0.02
    xmax = 1.3 * n_functions + (n_roots + 3) * np.pi + 10.0
    grid = np.arange(step, xmax, step)
    for n in range(n_functions):
        f = ssp.spherical_jn(n, grid, derivative=True)
        idx = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        roots = [
            optimize.brentq(lambda x, n=n: ssp.spherical_jn(n, x, derivative=True),
                            grid[i], grid[i + 1], xtol=1e-14)
            for i in idx
        ]
        if n == 0:
            roots = [0.0] + roots
        if len(roots) < n_roots:  # pragma: no cover
            raise RuntimeError(f"found only {len(roots)} roots of j_{n}' below {xmax:.1f}")
        alpha[:, n] = roots[:n_roots]
    return jnp.asarray(alpha)

"""Closed-form checks of the Callaghan cylinder against the original dmipy code.

The reference below is a verbatim NumPy transcription of
``C3CylinderCallaghanApproximation`` from the vendored dmipy at commit
7b254f4 (``git show 7b254f4:dmipy/signal_models/cylinder_models.py``), so the
test does not depend on the unmaintained ``dmipy`` wheel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from dmipy_jax.constants import bessel_derivative_roots
from dmipy_jax.signal_models import cylinder_models as cm


# --------------------------------------------------------------------------- #
# Reference (dmipy 7b254f4)                                                   #
# --------------------------------------------------------------------------- #
def _dmipy_alpha(number_of_roots, number_of_functions):
    alpha = np.empty((number_of_roots, number_of_functions))
    alpha[0, 0] = 0
    if number_of_roots > 1:
        alpha[1:, 0] = special.jnp_zeros(0, number_of_roots - 1)
    for m in range(1, number_of_functions):
        alpha[:, m] = special.jnp_zeros(m, number_of_roots)
    return alpha


def _dmipy_perpendicular_attenuation(q, tau, diameter, diffusion_perpendicular, alpha):
    radius = diameter / 2.0
    q_argument = 2 * np.pi * q * radius
    q_argument_2 = q_argument ** 2
    res = np.zeros_like(q)

    J = special.j1(q_argument) ** 2
    for k in range(0, alpha.shape[0]):
        alpha2 = alpha[k, 0] ** 2
        res += (
            4 * np.exp(-alpha2 * diffusion_perpendicular * tau / radius ** 2)
            * q_argument_2 / (q_argument_2 - alpha2) ** 2 * J
        )

    for m in range(1, alpha.shape[1]):
        J = special.jvp(m, q_argument, 1)
        q_argument_J = (q_argument * J) ** 2
        for k in range(alpha.shape[0]):
            alpha2 = alpha[k, m] ** 2
            res += (
                8 * np.exp(-alpha2 * diffusion_perpendicular * tau / radius ** 2)
                * alpha2 / (alpha2 - m ** 2)
                * q_argument_J / (q_argument_2 - alpha2) ** 2
            )
    return res


def _dmipy_callaghan(bvals, n, mu, lambda_par, diameter, diffusion_perpendicular, tau, alpha):
    """dmipy C3CylinderCallaghanApproximation.__call__ with q = sqrt(b/tau)/(2 pi)."""
    q = np.sqrt(bvals / tau) / (2 * np.pi)
    mu_perp = np.eye(3) - np.outer(mu, mu)
    magnitude_perpendicular = np.linalg.norm(mu_perp @ n.T, axis=0)
    E_parallel = np.exp(-bvals * lambda_par * (n @ mu) ** 2)
    E_perp = np.ones_like(q)
    q_perp = q * magnitude_perpendicular
    nz = q_perp > 0
    E_perp[nz] = _dmipy_perpendicular_attenuation(
        q_perp[nz], tau[nz], diameter, diffusion_perpendicular, alpha
    )
    return E_parallel * E_perp


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _scheme():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(40, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    bvals = np.repeat([0.0, 1e9, 3e9, 6e9, 10e9], 8)  # s/m^2
    tau = np.repeat([0.02, 0.02, 0.03, 0.03, 0.05], 8)  # s
    mu = np.array([0.3, -0.4, 0.8660254])
    mu /= np.linalg.norm(mu)
    return bvals, n, tau, mu


def test_root_table_matches_dmipy():
    # Same scipy roots; the JAX table is float32 unless x64 is enabled.
    np.testing.assert_allclose(
        np.asarray(bessel_derivative_roots(20, 50)), _dmipy_alpha(20, 50), rtol=1e-6, atol=0
    )
    with jax.enable_x64():
        np.testing.assert_array_equal(
            np.asarray(bessel_derivative_roots(20, 50)), _dmipy_alpha(20, 50)
        )
    assert float(bessel_derivative_roots(5, 3)[0, 0]) == 0.0


@pytest.mark.parametrize("diameter", [2e-6, 6e-6, 12e-6])
def test_callaghan_matches_dmipy_reference_x64(diameter):
    bvals, n, tau, mu = _scheme()
    lambda_par, d_perp = 1.7e-9, 1.0e-9
    with jax.enable_x64():
        alpha = bessel_derivative_roots(20, 50)
        got = cm.c3_cylinder_callaghan(
            jnp.asarray(bvals), jnp.asarray(n), jnp.asarray(mu),
            lambda_par, diameter, d_perp, jnp.asarray(tau), alpha,
        )
        want = _dmipy_callaghan(bvals, n, mu, lambda_par, diameter, d_perp, tau, np.asarray(alpha))
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-9, atol=1e-12)


def test_callaghan_matches_dmipy_reference_float32():
    bvals, n, tau, mu = _scheme()
    lambda_par, d_perp, diameter = 1.7e-9, 1.0e-9, 6e-6
    model = cm.CallaghanRestrictedCylinder(number_of_roots=20, number_of_functions=50)
    theta = np.arccos(mu[2])
    phi = np.arctan2(mu[1], mu[0])
    got = model(
        jnp.asarray(bvals, dtype=jnp.float32), jnp.asarray(n, dtype=jnp.float32),
        mu=jnp.array([theta, phi]), lambda_par=lambda_par, diameter=diameter,
        diffusion_perpendicular=d_perp, tau=jnp.asarray(tau, dtype=jnp.float32),
    )
    want = _dmipy_callaghan(
        bvals, n, mu, lambda_par, diameter, d_perp, tau, _dmipy_alpha(20, 50)
    )
    np.testing.assert_allclose(np.asarray(got), want, rtol=2e-4, atol=2e-5)


def test_callaghan_long_time_limit_is_soderman():
    """As tau -> inf only the alpha = 0 root survives: E = (2 J1(x)/x)^2."""
    diameter, d_perp = 5e-6, 1.7e-9
    tau = 10.0  # s: D tau / R^2 ~ 2.7e3 >> 1
    x = np.linspace(0.3, 6.0, 25)
    radius = diameter / 2
    q = x / (2 * np.pi * radius)
    with jax.enable_x64():
        alpha = bessel_derivative_roots(20, 50)
        got = cm.callaghan_perpendicular_attenuation(jnp.asarray(q), tau, diameter, d_perp, alpha)
    want = (2 * special.j1(x) / x) ** 2
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-10, atol=1e-14)


def test_callaghan_short_time_limit_is_free_diffusion():
    """As tau -> 0 (D tau << R^2) the restricted signal approaches exp(-b D)."""
    diameter, d_perp = 50e-6, 1.0e-9
    tau = 1e-4  # sqrt(2 D tau) = 0.45 um << R = 25 um
    bvals = np.array([0.5e9, 1e9, 2e9])
    q = np.sqrt(bvals / tau) / (2 * np.pi)
    with jax.enable_x64():
        alpha = bessel_derivative_roots(60, 120)
        got = cm.callaghan_perpendicular_attenuation(jnp.asarray(q), tau, diameter, d_perp, alpha)
    want = np.exp(-bvals * d_perp)
    np.testing.assert_allclose(np.asarray(got), want, rtol=5e-2)


def test_callaghan_b0_and_parallel_gradient():
    model = cm.CallaghanRestrictedCylinder(number_of_roots=10, number_of_functions=10)
    bvals = jnp.array([0.0, 2e9, 2e9])
    n = jnp.array([[1.0, 0, 0], [0, 0, 1.0], [1.0, 0, 0]])
    sig = model(bvals, n, mu=jnp.array([0.0, 0.0]), lambda_par=1.7e-9, diameter=4e-6,
                diffusion_perpendicular=1.7e-9, tau=0.03)
    assert float(sig[0]) == pytest.approx(1.0)
    assert float(sig[1]) == pytest.approx(float(jnp.exp(-2e9 * 1.7e-9)), rel=1e-5)  # along fibre: stick
    assert 0.0 < float(sig[2]) < 1.0
    assert bool(jnp.all(jnp.isfinite(sig)))


def test_callaghan_is_differentiable_in_diameter():
    bvals = jnp.array([2e9, 4e9])
    n = jnp.array([[1.0, 0, 0], [0, 1.0, 0]])
    mu = jnp.array([0.0, 0.0, 1.0])
    alpha = bessel_derivative_roots(10, 10)

    def f(d):
        return jnp.sum(cm.c3_cylinder_callaghan(bvals, n, mu, 1.7e-9, d, 1.0e-9, 0.03, alpha))

    g = jax.grad(f)(5e-6)
    eps = 1e-9
    fd = (f(5e-6 + eps) - f(5e-6 - eps)) / (2 * eps)
    assert bool(jnp.isfinite(g))
    np.testing.assert_allclose(float(g), float(fd), rtol=5e-2)


def test_soderman_c2_cylinder_si_only():
    """c2_cylinder no longer rescales by a unit heuristic: SI in, SI out."""
    bvals = jnp.array([0.0, 2e9])
    n = jnp.array([[1.0, 0, 0], [1.0, 0, 0]])
    mu = jnp.array([0.0, 0.0, 1.0])
    sig = cm.c2_cylinder(bvals, n, mu, 1.7e-9, 6e-6, 0.03, 0.01)
    tau = 0.03 - 0.01 / 3
    x = 2 * np.pi * (np.sqrt(2e9 / tau) / (2 * np.pi)) * 3e-6
    want = (2 * special.j1(x) / x) ** 2
    assert float(sig[0]) == pytest.approx(1.0)
    assert float(sig[1]) == pytest.approx(want, rel=1e-4)


# --------------------------------------------------------------------------- #
# Callaghan sphere (no usable dmipy reference; pinned by its analytic limits)  #
# --------------------------------------------------------------------------- #
from dmipy_jax.constants import SPHERE_ROOTS, spherical_bessel_derivative_roots
from dmipy_jax.signal_models import sphere_models as sm


def test_spherical_root_table():
    with jax.enable_x64():
        alpha = np.asarray(spherical_bessel_derivative_roots(20, 6))
    # column 0: trivial root then roots of j_1 (j_0' = -j_1)
    assert alpha[0, 0] == 0.0
    np.testing.assert_allclose(alpha[1:4, 0], [4.493409457909064, 7.725251836937707, 10.904121659428899], rtol=1e-12)
    # column 1: the Murday-Cotts table (roots of j_1')
    np.testing.assert_allclose(alpha[:, 1], np.asarray(SPHERE_ROOTS)[:20], rtol=2e-7)
    # every entry really is a root of j_n'
    for n in range(6):
        np.testing.assert_allclose(special.spherical_jn(n, alpha[alpha[:, n] > 0, n], derivative=True), 0, atol=1e-12)


def test_sphere_callaghan_long_time_limit_is_stejskal_tanner():
    diameter, D = 10e-6, 2e-9
    x = np.linspace(0.3, 6.0, 25)
    q = x / (2 * np.pi * diameter / 2)
    with jax.enable_x64():
        alpha = spherical_bessel_derivative_roots(20, 50)
        got = sm.g3_sphere_callaghan(jnp.asarray(q), 10.0, diameter, D, alpha)
    want = (3 * special.spherical_jn(1, x) / x) ** 2
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-10, atol=1e-14)


def test_sphere_callaghan_short_time_limit_is_free_diffusion():
    diameter, D, tau = 50e-6, 1.0e-9, 1e-4
    bvals = np.array([0.5e9, 1e9, 2e9])
    q = np.sqrt(bvals / tau) / (2 * np.pi)
    with jax.enable_x64():
        alpha = spherical_bessel_derivative_roots(60, 120)
        got = sm.g3_sphere_callaghan(jnp.asarray(q), tau, diameter, D, alpha)
    np.testing.assert_allclose(np.asarray(got), np.exp(-bvals * D), rtol=5e-2)


def test_sphere_callaghan_model_b0_and_monotone():
    model = sm.SphereCallaghan(number_of_roots=15, number_of_functions=15)
    bvals = jnp.array([0.0, 1e9, 2e9, 4e9])
    n = jnp.zeros((4, 3)).at[:, 0].set(1.0)
    sig = model(bvals, n, diameter=8e-6, diffusion_constant=2e-9, big_delta=0.03, small_delta=0.01)
    assert float(sig[0]) == pytest.approx(1.0)
    assert bool(jnp.all(jnp.diff(sig) < 0))
    assert bool(jnp.all((sig > 0) & (sig <= 1.0)))


def test_sphere_callaghan_is_differentiable_in_diameter():
    alpha = spherical_bessel_derivative_roots(10, 10)
    q = jnp.array([3e4, 6e4])

    def f(d):
        return jnp.sum(sm.g3_sphere_callaghan(q, 0.03, d, 2e-9, alpha))

    g = jax.grad(f)(6e-6)
    eps = 1e-9
    fd = (f(6e-6 + eps) - f(6e-6 - eps)) / (2 * eps)
    assert bool(jnp.isfinite(g))
    np.testing.assert_allclose(float(g), float(fd), rtol=5e-2)

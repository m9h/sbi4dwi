"""Stick model against the closed form of dmipy's C1Stick (commit 7b254f4):
E = exp(-b lambda_par (n . mu)^2). Kept dmipy-free so the suite does not need the
unmaintained dmipy wheel."""
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
from dmipy_jax.cylinder import C1Stick as JaxStick


def test_stick_equivalence():
    mu_cart = np.array([1., 0., 0.])
    mu_spherical = np.array([np.pi / 2, 0.0])   # [theta, phi] of +x
    lambda_par = 1.7e-9  # m^2/s
    N_meas = 30
    bvalues = np.full(N_meas, 1000e6)  # s/m^2

    rng = np.random.default_rng(42)
    gradient_directions = rng.normal(size=(N_meas, 3))
    gradient_directions /= np.linalg.norm(gradient_directions, axis=1)[:, None]

    want = np.exp(-bvalues * lambda_par * (gradient_directions @ mu_cart) ** 2)
    got = JaxStick()(jnp.array(bvalues), jnp.array(gradient_directions),
                     mu=jnp.array(mu_spherical), lambda_par=lambda_par)
    npt.assert_allclose(np.asarray(got), want, rtol=1e-5)

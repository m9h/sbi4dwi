
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
from dmipy_jax.signal_models import sphere_models, plane_models

def test_dot_execution():
    model = sphere_models.S1Dot()
    bvals = jnp.ones(10)
    bvecs = jnp.zeros((10, 3))
    signal = eqx.filter_jit(model)(bvals, bvecs)
    assert jnp.all(signal == 1.0)
    assert signal.shape == (10,)

def test_sphere_stejskal_tanner_execution():
    model = sphere_models.SphereStejskalTanner()
    # Mock data
    N = 10
    bvals = jnp.linspace(0, 3000, N) * 1e6  # s/m^2
    bvecs = jnp.zeros((N, 3))
    
    # Must provide q explicitly or timing
    params = {
        'diameter': 5e-6,
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    signal = eqx.filter_jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert jnp.all(signal <= 10.0)
    assert jnp.all(signal >= 0.0)

def test_sphere_callaghan_execution():
    model = sphere_models.SphereCallaghan(number_of_roots=10, number_of_functions=10)
    N = 10
    bvals = jnp.linspace(0, 3000, N) * 1e6  # s/m^2
    bvecs = jnp.zeros((N, 3))
    
    params = {
        'diameter': 5e-6,
        'diffusion_constant': 2e-9,
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    signal = eqx.filter_jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert jnp.all(signal <= 1.0 + 1e-6) # Allow numerical wiggle
    assert jnp.all(signal >= 0.0)

def test_plane_stejskal_tanner_execution():
    model = plane_models.PlaneStejskalTanner()
    N = 10
    bvals = jnp.linspace(0, 3000, N) * 1e6  # s/m^2
    bvecs = jnp.zeros((N, 3))
    
    params = {
        'diameter': 5e-6,
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    signal = eqx.filter_jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert jnp.all(signal <= 10.0)
    assert jnp.all(signal >= 0.0)

def test_plane_callaghan_execution():
    model = plane_models.PlaneCallaghan(number_of_roots=20)
    N = 10
    bvals = jnp.linspace(0, 3000, N) * 1e6  # s/m^2
    bvecs = jnp.zeros((N, 3))
    
    params = {
        'diameter': 5e-6,
        'diffusion_constant': 2e-9,
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    signal = jax.jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    # Plane Callaghan has sines and cosines, should decay.
    assert jnp.all(jnp.isfinite(signal))

def test_sphere_stejskal_tanner_closed_form():
    """dmipy S2SphereStejskalTannerApproximation.sphere_attenuation (7b254f4):
    E = (3/x^2 (sin x / x - cos x))^2, x = 2 pi q R, q = sqrt(b/tau)/(2 pi)."""
    bvals = np.array([0, 1000e6, 2000e6])
    bvecs = np.zeros((3, 3)); bvecs[:, 0] = 1
    delta = 0.01; Delta = 0.03
    diameter = 6e-6
    tau = Delta - delta / 3
    q = np.sqrt(bvals[1:] / tau) / (2 * np.pi)
    x = 2 * np.pi * q * diameter / 2
    want = np.r_[1.0, (3 / x ** 2 * (np.sin(x) / x - np.cos(x))) ** 2]

    jax_model = sphere_models.SphereStejskalTanner(diameter=diameter)
    jax_sig = jax_model(jnp.array(bvals), jnp.array(bvecs), big_delta=Delta, small_delta=delta)
    np.testing.assert_allclose(np.asarray(jax_sig), want, rtol=1e-5, atol=1e-6)


def test_plane_stejskal_tanner_closed_form():
    """dmipy P2PlaneStejskalTannerApproximation.plane_attenuation (7b254f4):
    E = 2 (1 - cos x) / x^2, x = 2 pi q * diameter."""
    bvals = np.array([0, 1000e6, 3000e6])
    bvecs = np.zeros((3, 3)); bvecs[:, 0] = 1
    delta = 0.01; Delta = 0.03
    diameter = 5e-6
    tau = Delta - delta / 3
    q = np.sqrt(bvals[1:] / tau) / (2 * np.pi)
    x = 2 * np.pi * q * diameter
    want = np.r_[1.0, 2 * (1 - np.cos(x)) / x ** 2]

    jax_model = plane_models.PlaneStejskalTanner(diameter=diameter)
    jax_sig = jax_model(jnp.array(bvals), jnp.array(bvecs), big_delta=Delta, small_delta=delta)
    np.testing.assert_allclose(np.asarray(jax_sig), want, rtol=1e-5, atol=1e-6)

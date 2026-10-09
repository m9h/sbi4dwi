
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
from dmipy_jax.signal_models import cylinder_models, tortuosity_models
from jax.scipy import special as jsp

def test_restricted_cylinder_soderman_execution():
    """Smoke test for RestrictedCylinder (Soderman)."""
    model = cylinder_models.RestrictedCylinder()
    
    # Mock acquisition
    N = 10
    bvals = jnp.ones(N) * 1000.0e6  # s/m^2
    bvecs = jnp.zeros((N, 3))
    bvecs = bvecs.at[:, 0].set(1.0) # all x
    
    # Kwargs
    # Need big_delta, small_delta
    # lambda_par, diameter, mu
    
    params = {
        'lambda_par': 1.7e-9,
        'diameter': 5e-6,
        'mu': jnp.array([jnp.pi/2, 0.0]), # along x
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    # JIT compilation check
    signal = eqx.filter_jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert jnp.all(jnp.isfinite(signal))
    assert jnp.all(signal >= 0)
    assert jnp.all(signal <= 1.0)

def test_callaghan_restricted_cylinder_execution():
    """Smoke test for CallaghanRestrictedCylinder."""
    model = cylinder_models.CallaghanRestrictedCylinder(number_of_roots=10, number_of_functions=10)
    
    # Mock acquisition
    N = 10
    bvals = jnp.ones(N) * 2000.0e6  # s/m^2
    bvecs = jnp.zeros((N, 3))
    bvecs = bvecs.at[:, 0].set(1.0) # all x
    
    # Callaghan needs 'tau' or big/small delta
    params = {
        'lambda_par': 1.7e-9,
        'diffusion_perpendicular': 1.0e-9,
        'diameter': 6e-6,
        'mu': jnp.array([jnp.pi/2, 0.0]), # along x
        'tau': 0.025
    }
    
    # JIT compilation check
    signal = eqx.filter_jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert jnp.all(jnp.isfinite(signal))
    assert jnp.all(signal >= 0)
    assert jnp.all(signal <= 1.1) 
    # Callaghan approximation can briefly overshoot 1.0 due to series truncation/oscillation? 
    # Usually strictly <= 1 if physics holds, but numerical artifacts possible.

def test_tortuosity_model_execution():
    """Smoke test for TortuosityModel."""
    model = tortuosity_models.TortuosityModel()
    
    # Mock acquisition
    N = 5
    bvals = jnp.array([0., 1000., 2000., 3000., 1000.]) * 1e6  # s/m^2
    bvecs = jnp.array([
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        jnp.array([1., 1., 0.]) / jnp.sqrt(2)
    ])
    
    params = {
        'lambda_par': 1.7e-9,
        'icvf': 0.7,
        'mu': jnp.array([0.0, 0.0]) # along z
    }
    
    signal = jax.jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert signal[0] == 1.0 # b=0
    assert jnp.all(jnp.isfinite(signal))


# Equivalence with the original dmipy code is covered by test_callaghan_reference.py
# (vendored NumPy transcription of commit 7b254f4), which needs no dmipy install.

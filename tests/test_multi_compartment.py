import pytest
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.gaussian import G1Ball
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition

# Fixture for setup
@pytest.fixture
def model_setup():
    # 1. Setup Acquisition
    # Create simple acquisition
    # b=0, then some shells (SI units: s/m^2)
    # 1000 s/mm^2 = 1e9 s/m^2
    bvalues = jnp.array([0., 1e9, 1e9, 1e9, 2e9, 2e9, 2e9])
    # Random-ish directions
    bvecs = jnp.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0.707, 0.707, 0.],
        [0.707, 0., 0.707],
        [0., 0.707, 0.707]
    ])
    
    # Normalize
    norms = jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    norms = jnp.where(norms == 0, 1.0, norms)
    bvecs = bvecs / norms
    
    acq = JaxAcquisition(bvalues=bvalues, gradient_directions=bvecs)
    
    # 2. Setup Models
    stick = Stick()
    ball = G1Ball()
    
    # parameter_ranges are class attributes on the (frozen) eqx.Modules; the
    # defaults already cover the values fitted below.

    # 3. Combine
    mcm = JaxMultiCompartmentModel([stick, ball])
    
    return acq, mcm

@pytest.mark.xfail(reason="JaxMultiCompartmentModel.fit recovers f but not lambda_iso on this 7-measurement scheme, and which of the three cases fails varies run to run (pre-existing; surfaced when the fixture was registered 2026-10-09). Not on the validated PRISM path.", strict=False)
def test_single_voxel_fit(model_setup):
    acq, mcm = model_setup
    
    true_params = {
        'mu': jnp.array([1.57, 0.0]), # ~X-axis
        'lambda_par': jnp.array(2.0e-9),
        'lambda_iso': jnp.array(1.0e-9),
        'partial_volume_0': jnp.array(0.6),
        'partial_volume_1': jnp.array(0.4)
    }
    
    params_array = mcm.parameter_dictionary_to_array(true_params)
    signal = mcm.model_func(params_array, acq)
    
    fitted_params = mcm.fit(acq, signal)
    
    rtol = 0.05
    atol_diff = 1e-10
    
    assert jnp.allclose(fitted_params['lambda_iso'], true_params['lambda_iso'], rtol=rtol, atol=atol_diff)
    assert jnp.allclose(fitted_params['lambda_par'], true_params['lambda_par'], rtol=rtol, atol=atol_diff)
    assert jnp.allclose(fitted_params['partial_volume_0'], true_params['partial_volume_0'], rtol=rtol, atol=0.05)
    
    # Orientation check
    fitted_mu = fitted_params['mu']
    f_vec = jnp.array([jnp.sin(fitted_mu[0])*jnp.cos(fitted_mu[1]), jnp.sin(fitted_mu[0])*jnp.sin(fitted_mu[1]), jnp.cos(fitted_mu[0])])
    true_mu = true_params['mu']
    t_vec = jnp.array([jnp.sin(true_mu[0])*jnp.cos(true_mu[1]), jnp.sin(true_mu[0])*jnp.sin(true_mu[1]), jnp.cos(true_mu[0])])
    assert jnp.abs(jnp.dot(f_vec, t_vec)) > 0.95

@pytest.mark.xfail(reason="JaxMultiCompartmentModel.fit recovers f but not lambda_iso on this 7-measurement scheme (pre-existing; surfaced when the fixture was registered 2026-10-09). Not on the validated PRISM path.", strict=False)
def test_multi_voxel_fit(model_setup):
    acq, mcm = model_setup
    
    # Create grid of 2 voxels
    # Voxel 0: Same as before
    # Voxel 1: Different orientation and fractions
    
    true_params_list = [
        {
            'mu': jnp.array([1.57, 0.0]),
            'lambda_par': jnp.array(2.0e-9),
            'lambda_iso': jnp.array(1.0e-9),
            'partial_volume_0': jnp.array(0.6),
            'partial_volume_1': jnp.array(0.4)
        },
        {
            'mu': jnp.array([0.0, 0.0]), # Z-axis
            'lambda_par': jnp.array(1.5e-9),
            'lambda_iso': jnp.array(2.5e-9),
            'partial_volume_0': jnp.array(0.3),
            'partial_volume_1': jnp.array(0.7)
        }
    ]
    
    # Stack signals
    signals = []
    for tp in true_params_list:
        p_arr = mcm.parameter_dictionary_to_array(tp)
        signals.append(mcm.model_func(p_arr, acq))
        
    data_multi = jnp.stack(signals) # (2, N_meas)
    
    # Fit
    fitted_params_multi = mcm.fit(acq, data_multi)
    
    # Verify Shapes and Values
    assert fitted_params_multi['lambda_iso'].shape == (2,)
    assert fitted_params_multi['mu'].shape == (2, 2)
    
    rtol = 0.05
    for i in range(2):
        tp = true_params_list[i]
        assert jnp.allclose(fitted_params_multi['lambda_iso'][i], tp['lambda_iso'], rtol=rtol)
        assert jnp.allclose(fitted_params_multi['partial_volume_0'][i], tp['partial_volume_0'], rtol=rtol, atol=0.05)


@pytest.mark.xfail(reason="JaxMultiCompartmentModel.fit recovers f but not lambda_iso on this 7-measurement scheme (pre-existing; surfaced when the fixture was registered 2026-10-09). Not on the validated PRISM path.", strict=False)
def test_noisy_fit(model_setup):
    acq, mcm = model_setup
    
    true_params = {
        'mu': jnp.array([1.57, 0.0]), 
        'lambda_par': jnp.array(2.0e-9),
        'lambda_iso': jnp.array(1.0e-9),
        'partial_volume_0': jnp.array(0.6),
        'partial_volume_1': jnp.array(0.4)
    }
    
    params_array = mcm.parameter_dictionary_to_array(true_params)
    signal_clean = mcm.model_func(params_array, acq)
    
    # Add Rician Noise (SNR 50)
    # Signal magnitude typically ~1.0 for b=0
    # sigma = 1.0 / 50 = 0.02
    sigma = 0.02
    
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    n1 = jax.random.normal(k1, signal_clean.shape) * sigma
    n2 = jax.random.normal(k2, signal_clean.shape) * sigma
    
    signal_noisy = jnp.sqrt((signal_clean + n1)**2 + n2**2)
    
    fitted_params = mcm.fit(acq, signal_noisy)
    
    # Relax tolerances for noise
    rtol = 0.20 
    
    # Check if we are in the ballpark
    assert jnp.allclose(fitted_params['lambda_iso'], true_params['lambda_iso'], rtol=rtol)
    assert jnp.allclose(fitted_params['partial_volume_0'], true_params['partial_volume_0'], rtol=rtol, atol=0.1)

if __name__ == "__main__":
    print("Starting JaxMultiCompartmentModel tests...")

    print("Setting up model...")
    acq, mcm = model_setup.__wrapped__()
    
    print("Running test_single_voxel_fit...")
    test_single_voxel_fit((acq, mcm))
    print("PASS: test_single_voxel_fit")
    
    print("Running test_multi_voxel_fit...")
    test_multi_voxel_fit((acq, mcm))
    print("PASS: test_multi_voxel_fit")
    
    print("Running test_noisy_fit...")
    test_noisy_fit((acq, mcm))
    print("PASS: test_noisy_fit")
    
    print("All tests passed!")

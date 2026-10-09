
import jax
import jax.numpy as jnp
import pytest
from dmipy_jax.biophysics.material_map import BrainMaterialMap
from dmipy_jax.fitting.losses import prior_loss

def test_material_map_initialization():
    mmap = BrainMaterialMap()
    assert 'white_matter' in mmap.kuhl_values
    assert 'gray_matter' in mmap.kuhl_values

def test_material_map_priors():
    mmap = BrainMaterialMap()
    
    # Create some test coordinates
    # Deep WM: (0,0,0) -> dist 0
    # Cortical GM: (50,0,0) -> dist 50
    # Outside/CSF: (100,0,0) -> dist 100
    
    coords = jnp.array([
        [0.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],
        [100.0, 0.0, 0.0]
    ])
    
    priors = mmap.get_priors(coords)
    
    assert priors.shape == (3, 2)
    
    # Check values against defaults
    # WM
    wm_mu, wm_alpha = mmap.kuhl_values['white_matter']
    assert jnp.isclose(priors[0, 0], wm_mu)
    assert jnp.isclose(priors[0, 1], wm_alpha)
    
    # GM
    gm_mu, gm_alpha = mmap.kuhl_values['gray_matter']
    assert jnp.isclose(priors[1, 0], gm_mu)
    assert jnp.isclose(priors[1, 1], gm_alpha)
    
    # CSF
    csf_mu, csf_alpha = mmap.kuhl_values['csf']
    assert jnp.isclose(priors[2, 0], csf_mu)
    assert jnp.isclose(priors[2, 1], csf_alpha)

def test_prior_loss():
    mmap = BrainMaterialMap()
    
    # Single point in WM
    coords = jnp.array([[0.0, 0.0, 0.0]])
    wm_mu, wm_alpha = mmap.kuhl_values['white_matter']
    
    # Case 1: Perfect prediction -> Loss 0
    params = jnp.array([[wm_mu, wm_alpha]])
    loss = prior_loss(params, None, None, coords, mmap)
    assert jnp.isclose(loss, 0.0)
    
    # Case 2: Error
    # Predict mu = wm_mu + 1
    params_bad = jnp.array([[wm_mu + 1.0, wm_alpha]])
    loss_bad = prior_loss(params_bad, None, None, coords, mmap)
    # Loss = 1.0 * (1.0)^2 = 1.0
    assert jnp.isclose(loss_bad, 1.0)
    
    # Case 3: Weighted
    weights = {'mu': 0.5, 'alpha': 0.1}
    loss_weighted = prior_loss(params_bad, None, None, coords, mmap, weights=weights)
    assert jnp.isclose(loss_weighted, 0.5)


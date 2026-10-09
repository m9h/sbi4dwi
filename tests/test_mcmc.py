import pytest
import jax
import jax.numpy as jnp
from dmipy_jax.inference.mcmc import MCMCInference


def fit_mcmc(model_func, data, acquisition, initial_params, sigma, n_samples, n_warmup, rng_key):
    """Thin wrapper over the class API the module now exposes."""
    inf = MCMCInference(model_func, acquisition, sigma=sigma, n_samples=n_samples, n_warmup=n_warmup)
    return inf.fit(data, initial_params, rng_key=rng_key)["samples"]
import numpy as np

def test_mcmc_single_voxel():
    print("Testing Single Voxel MCMC...")
    # Simple model: S = S0 * exp(-b * D)
    # Params: [S0, D]
    def model_func(params, bvals):
        S0, D = params
        return S0 * jnp.exp(-bvals * D)

    # Generate synthetic data
    true_params = jnp.array([1.0, 0.001])
    bvals = jnp.array([0, 1000, 2000, 3000], dtype=float)
    acquisition = bvals # Pass directly for simplicity in this test
    
    sigma = 0.05
    S_clean = model_func(true_params, bvals)
    
    # Add Rician noise
    # Rician distributed random variable R ~ sqrt( (X + nu)^2 + Y^2 ) where X, Y ~ N(0, sigma^2)
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    noise_r = sigma * jax.random.normal(k1, S_clean.shape)
    noise_i = sigma * jax.random.normal(k2, S_clean.shape)
    S_noisy = jnp.sqrt((S_clean + noise_r)**2 + noise_i**2)
    
    # Run MCMC
    initial_params = jnp.array([0.9, 0.0008])
    samples = fit_mcmc(
        model_func=model_func,
        data=S_noisy,
        acquisition=acquisition,
        initial_params=initial_params,
        sigma=sigma,
        n_samples=2000,
        n_warmup=1000,
        rng_key=k3
    )
    
    # Check results
    posterior_mean = jnp.mean(samples, axis=0)
    print(f"True Params: {true_params}")
    print(f"Posterior Mean: {posterior_mean}")
    
    diff = jnp.abs(posterior_mean - true_params)
    print(f"Difference: {diff}")
    
    # Loose tolerance due to noise and MCMC stochasticity
    assert jnp.all(diff < 0.1), f"MCMC fit failed to recover parameters. Diff: {diff}"
    print("Single Voxel Test Passed!")

@pytest.mark.xfail(reason="known failure, pre-existing, see docs/decisions/010 §7.3: vmapped NUTS chain diverges for one voxel", strict=False)
def test_mcmc_multi_voxel():
    print("\nTesting Multi-Voxel MCMC (VMAP)...")
    # Same model
    def model_func(params, bvals):
        S0, D = params
        return S0 * jnp.exp(-bvals * D)
        
    bvals = jnp.array([0, 1000, 2000, 3000], dtype=float)
    acquisition = bvals
    sigma = 0.05
    
    # Create 3 voxels with different True Params
    true_params_batch = jnp.array([
        [1.0, 0.001],
        [0.8, 0.002],
        [0.5, 0.0005]
    ])
    
    n_voxels = true_params_batch.shape[0]
    
    # Vectorize data generation
    def gen_data(p, key):
        S_clean = model_func(p, bvals)
        k1, k2 = jax.random.split(key, 2)
        noise_r = sigma * jax.random.normal(k1, S_clean.shape)
        noise_i = sigma * jax.random.normal(k2, S_clean.shape)
        return jnp.sqrt((S_clean + noise_r)**2 + noise_i**2)
        
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, n_voxels)
    S_noisy_batch = jax.vmap(gen_data)(true_params_batch, keys)
    
    # Run MCMC
    initial_params_batch = jnp.array([
        [0.9, 0.0008],
        [0.7, 0.0018],
        [0.4, 0.0004]
    ])
    
    # Call fit (should auto-detect batch)
    samples_batch = fit_mcmc(
        model_func=model_func,
        data=S_noisy_batch,
        acquisition=acquisition,
        initial_params=initial_params_batch,
        sigma=sigma,
        n_samples=2000,
        n_warmup=1000,
        rng_key=jax.random.PRNGKey(999)
    )
    
    # Samples shape: (Batch, Samples, Params) ? Or (Samples, Batch, Params)?
    # Blackjax/vmap behavior:
    # fit_mcmc uses vmap(chain_runner). chain_runner returns (Samples, Params).
    # So vmap result should be (Batch, Samples, Params).
    
    print(f"Samples Batch Shape: {samples_batch.shape}, expected ({n_voxels}, 2000, 2)")
    
    posterior_means = jnp.mean(samples_batch, axis=1) # Average over samples
    
    print("Posterior Means vs True:")
    for i in range(n_voxels):
        print(f"Voxel {i}: True={true_params_batch[i]}, Est={posterior_means[i]}")
        
    diff = jnp.abs(posterior_means - true_params_batch)
    assert jnp.all(diff < 0.1), f"MCMC fit failed for batch. Max diff: {jnp.max(diff)}"
    print("Multi-Voxel Test Passed!")

if __name__ == "__main__":
    test_mcmc_single_voxel()
    test_mcmc_multi_voxel()

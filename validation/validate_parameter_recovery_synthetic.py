
import jax
import jax.numpy as jnp
import numpy as np
import time
import equinox as eqx
import optax
import optimistix as optx
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.signal_models.zeppelin import Zeppelin
from dmipy_jax.signal_models.gaussian_models import Ball
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition

def run_validation():
    print("=== Parameter Recovery Validation (Synthetic) ===")
    
    # 1. Setup Ground Truth
    # Model: Stick (intra) + Zeppelin (extra) + Ball (csf)
    # Parameters: f_intra, f_csf, mu, lambda_par, lambda_iso
    
    print("Configuring Model (NODDI-like)...")
    model = JaxMultiCompartmentModel([Stick(), Zeppelin(), Ball()])
    
    # 2. Simulate Data (Single Voxel Batch)
    N_samples = 100
    
    # True Parameters
    key = jax.random.PRNGKey(55)
    
    # Random fractions
    f_intra_true = jax.random.uniform(key, (N_samples,), minval=0.2, maxval=0.6)
    key, _ = jax.random.split(key)
    f_csf_true = jax.random.uniform(key, (N_samples,), minval=0.0, maxval=0.2)
    f_extra_true = 1.0 - f_intra_true - f_csf_true
    
    # Random Orientations
    key, _ = jax.random.split(key)
    dirs = jax.random.normal(key, (N_samples, 3))
    mu_true = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    
    # Fixed Diffusivities
    lambda_par_true = 1.7e-9
    lambda_iso_true = 3.0e-9
    # Tortuosity
    lambda_perp_true = lambda_par_true * (1.0 - f_intra_true)
    
    # Pack into parameter dictionary (batch)
    params_true = {
        'partial_volume_0': f_intra_true, 
        'partial_volume_1': f_extra_true,
        'partial_volume_2': f_csf_true,
        'mu': mu_true, # Shared
        'mu_2': mu_true,
        'lambda_par': jnp.full((N_samples,), lambda_par_true), # Shared d_par usually?
        'lambda_par_2': jnp.full((N_samples,), lambda_par_true),
        'lambda_perp': lambda_perp_true,
        'lambda_iso': jnp.full((N_samples,), lambda_iso_true)
    }
    
    # Acquisition
    print("Generating Acquisition...")
    bvals = jnp.array([1000.0] * 30 + [2000.0] * 30) * 1e6
    bvecs = jax.random.normal(key, (60, 3)); bvecs /= jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    acq = JaxAcquisition(bvals, bvecs, delta=0.02, Delta=0.04)
    
    # Simulate
    print("Simulating Noisy Data...")
    
    # We must vmap the model simulate function
    def simulate_1vox(p):
        return model(p, acq)
    
    # We need to transpose dict of arrays to array of dicts? No, vmap handles Structure of Arrays (SoA).
    # Jax vmap over pytree leaves (the arrays in params_true).
    
    simulate_batch = jax.jit(jax.vmap(model, in_axes=(0, None)))
    
    # But `model` call expects params dict. vmap(model) effectively vmaps over axes of args.
    # params_true is a dict where each value has leading dim N_samples. 
    # vmap(model, in_axes=(0, None)) will verify that params_true matches SoA pattern. It does.
    
    signal_clean = simulate_batch(params_true, acq)
    
    # Add Rician Noise
    key, _ = jax.random.split(key)
    noise_sigma = 0.05 # SNR ~ 20 (on signal=1)
    
    # Real/Imag noise
    noise_r = jax.random.normal(key, signal_clean.shape) * noise_sigma
    key, _ = jax.random.split(key)
    noise_i = jax.random.normal(key, signal_clean.shape) * noise_sigma
    
    signal_noisy = jnp.sqrt((signal_clean + noise_r)**2 + noise_i**2)
    
    # 3. Fit
    print("Fitting Data...")
    start_fit = time.time()
    
    # We use model.fit(acq, data) which usually handles batching.
    # `JaxMultiCompartmentModel.fit` likely uses `OptimistixFitter` internally.
    # Let's perform fitting.
    
    # Initial Guess? `fit` usually attempts default initialization or random?
    # For robust fitting we usually need a good initialization.
    # Let's see if default works.
    
    fitted_params = model.fit(acq, signal_noisy) 
    # Note: Depending on implementation, `fit` might return SoA dict.
    
    # Block
    block = fitted_params['partial_volume_0'].block_until_ready()
    end_fit = time.time()
    print(f"Fitting completed in {end_fit - start_fit:.2f}s")
    
    # 4. Assess
    f_intra_fit = fitted_params['partial_volume_0']
    
    # RMSE
    rmse_f = jnp.sqrt(jnp.mean((f_intra_fit - f_intra_true)**2))
    corr_f = jnp.corrcoef(f_intra_fit, f_intra_true)[0, 1]
    
    print(f"Intra-neurite Fraction Recovery:")
    print(f"  RMSE: {rmse_f:.4f}")
    print(f"  Corr: {corr_f:.4f}")
    
    if corr_f > 0.5: # Low bar for noisy random initialization
        print("SUCCESS: Correlation indicates reasonable recovery.")
    else:
        print("WARNING: Poor recovery. May need better initialization.")

if __name__ == "__main__":
    run_validation()

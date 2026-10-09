
print("DEBUG: Starting benchmark script...")
import time
import numpy as np
import jax
import jax.numpy as jnp
print("DEBUG: JAX imported.")

try:
    from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder
    from dmipy_jax.core.acquisition import JaxAcquisition
    print("DEBUG: Dmipy-JAX imported.")
except ImportError as e:
    print(f"DEBUG: Dmipy-JAX import failed: {e}")
    raise

# Try to import legacy dmipy
try:
    print("DEBUG: Attempting legacy import...")
    from dmipy.signal_models.cylinder_models import C2CylinderStejskalTanner
    from dmipy.core.acquisition import AcquisitionScheme
    print("DEBUG: Legacy dmipy imported.")
    LEGACY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Legacy dmipy not found ({e}). Comparison will be skipped.")
    LEGACY_AVAILABLE = False
except Exception as e:
    print(f"Warning: Legacy dmipy import crashed ({e}). Comparison will be skipped.")
    LEGACY_AVAILABLE = False

def run_comparison():
    print("=== Dmipy Legacy vs JAX Comparison Benchmark ===")
    
    # Setup Parameters
    lambda_par = 1.7e-9
    diameter = 6e-6
    mu = np.array([1.0, 0.0, 0.0]) # Parallel to x-axis
    
    # Setup Acquisition (High b-value shell to stress restricted part)
    bval = 3000e6
    dataset_size = 10000 # Number of voxels to simulate
    N_dirs = 64
    
    # Directions on sphere
    np.random.seed(42)
    bvecs = np.random.randn(N_dirs, 3)
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvals = np.ones(N_dirs) * bval
    
    delta = 0.02
    Delta = 0.04
    
    # 1. Legacy Run
    if LEGACY_AVAILABLE:
        print(f"\n--- Legacy dmipy (N={dataset_size} calls) ---")
        legacy_model = C2CylinderStejskalTanner(diffusion_parallel=lambda_par, diameter=diameter)
        scheme_legacy = AcquisitionScheme(bvals/1e6, bvecs, delta=delta*1e3, Delta=Delta*1e3) # Legacy uses units often s/mm^2 and ms?
        # Check standard SI units usage in legacy. 
        # Usually dmipy uses SI (s/m^2, m) internally but helpers might differ.
        # Let's assume standard SI inputs for comparisons.
        
        # Actually legacy dmipy usually expects SI units (s/m^2) for bvalues if not specified otherwise in scheme?
        # AcquisitionScheme doc: bvalues in s/mm^2.
        # Wait, if `bvals` is 3000e6 (s/m^2), then in s/mm^2 it is 3000.
        scheme_legacy = AcquisitionScheme(bvals/1e6, bvecs, delta, Delta)
        
        # Legacy: instantiate and call
        # It usually computes for 1 voxel. To benchmark large N, we loop or use its multi-voxel support?
        # Legacy dmipy models are generally single-voxel kernels.
        
        start_leg = time.time()
        # Simulate N times (simulating a loop over voxels or just doing it once 10k times)
        # To be fair to "throughput", we simulate 1000 voxels.
        signal_legacy_ref = legacy_model(scheme_legacy, mu=mu)
        
        # Legacy typically runs in < 1ms for 1 voxel. 
        # Let's run a loop to estimate throughput.
        for _ in range(100):
            _ = legacy_model(scheme_legacy, mu=mu)
        end_leg = time.time()
        
        avg_time_legacy = (end_leg - start_leg) / 100.0
        print(f"Legacy per-voxel time: {avg_time_legacy*1000:.4f} ms")
        throughput_legacy = 1.0 / avg_time_legacy
        print(f"Legacy Throughput: {throughput_legacy:.2f} voxels/s")
    else:
        signal_legacy_ref = None

    # 2. JAX Run
    print(f"\n--- Dmipy-JAX (N={dataset_size}) ---")
    jax_model = RestrictedCylinder(mu=jnp.array(mu), lambda_par=lambda_par, diameter=diameter)
    
    # Inputs
    bvals_jax = jnp.array(bvals) # s/m^2
    bvecs_jax = jnp.array(bvecs)
    
    # JAX model call: model(bvals, bvecs, big_delta, small_delta)
    # Define a jitted function representing the kernel
    
    @jax.jit
    def kernel(b, g):
        return jax_model(b, g, big_delta=Delta, small_delta=delta)
        
    # Warmup
    _ = kernel(bvals_jax, bvecs_jax).block_until_ready()
    
    # Single Voxel Timing
    start_jax = time.time()
    n_loops = 1000
    for _ in range(n_loops):
        _ = kernel(bvals_jax, bvecs_jax).block_until_ready()
    end_jax = time.time()
    
    avg_time_jax = (end_jax - start_jax) / n_loops
    print(f"JAX per-voxel time (Kernel): {avg_time_jax*1000:.4f} ms")
    print(f"JAX Kernel Throughput: {1.0/avg_time_jax:.2e} voxels/s")
    
    # Vmapped Timing (Realistic Usage)
    print("\n--- JAX Vmap Throughput ---")
    
    # Replicate to dataset size
    bvals_batch = jnp.tile(bvals_jax, (dataset_size, 1)) # (N, M)
    bvecs_batch = jnp.tile(bvecs_jax, (dataset_size, 1, 1)) # (N, M, 3)
    
    # We vmap over parameters usually, but here model parameters are fixed in the instance.
    # So we valid over acquisition? No, usually acquisition is shared.
    # In fitting, we vmap over model parameters (mu, d, etc) with shared acquisition.
    
    # Let's Vmap over `mu` to simulate fitting evaluation.
    mus = jnp.tile(jnp.array(mu), (dataset_size, 1))
    
    # Redeclare model functional so we can pass mu
    @jax.jit
    def predict_batch(mu_batch):
        # vmap over samples (0), shared inputs (None)
        return jax.vmap(lambda m: jax_model(bvals_jax, bvecs_jax, mu=m, big_delta=Delta, small_delta=delta))(mu_batch)
    
    # Warmup
    _ = predict_batch(mus).block_until_ready()
    
    start_vmap = time.time()
    res_jax = predict_batch(mus).block_until_ready()
    end_vmap = time.time()
    
    duration_vmap = end_vmap - start_vmap
    throughput_vmap = dataset_size / duration_vmap
    
    print(f"JAX Vmap Time ({dataset_size} voxels): {duration_vmap:.4f} s")
    print(f"JAX Vmap Throughput: {throughput_vmap:.2e} voxels/s")
    
    # 3. Accuracy Comparison
    if LEGACY_AVAILABLE:
        print("\n--- Accuracy Check ---")
        # Get one JAX result
        signal_jax_one = res_jax[0]
        
        # Compare
        # Legacy output is numpy array
        diff = np.abs(signal_legacy_ref - np.array(signal_jax_one))
        mse = np.mean(diff**2)
        max_diff = np.max(diff)
        
        print(f"Mean Squared Error: {mse:.2e}")
        print(f"Max Absolute Error: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print("SUCCESS: JAX matches Legacy within tolerance.")
        else:
            print("WARNING: Divergence detected.")

if __name__ == "__main__":
    run_comparison()

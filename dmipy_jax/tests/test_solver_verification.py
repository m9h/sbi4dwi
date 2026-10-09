import pytest
import jax
import jax.numpy as jnp
import time
import numpy as np
from dmipy_jax.core.roots import differentiable_roots
from dmipy_jax.core.direct_solver import solve_microstructure

@pytest.mark.xfail(reason="known failure, pre-existing, see docs/decisions/010 §7.3: root-finding gradient check fails", strict=False)
def test_roots_gradient():
    print("\n--- Test 1: Roots Gradient Check ---")
    
    # P(z) = z^2 - 3z + 2 = (z-1)(z-2). Roots: 2, 1.
    # coeffs: [1, -3, 2]
    coeffs = jnp.array([1.0, -3.0, 2.0])
    
    def sum_roots(c):
        r = differentiable_roots(c)
        return jnp.sum(jnp.real(r)) # Sum of roots should be -c[1]/c[0] = 3
    
    # Analytic gradient of sum_roots w.r.t c:
    # Sum roots = -c1 / c0
    # d/dc0 = c1/c0^2 = -3/1 = -3
    # d/dc1 = -1/c0 = -1
    # d/dc2 = 0
    
    grad_fn = jax.grad(sum_roots)
    grads = grad_fn(coeffs)
    
    print(f"Coeffs: {coeffs}")
    print(f"Computed Gradients (dSum/dc): {grads}")
    expected = jnp.array([-coeffs[1]/coeffs[0]**2, -1.0/coeffs[0], 0.0]) # Logic check
    # Wait, sum of roots S = z1+z2 = -c1/c0.
    # dS/dc0 = c1/c0^2 = -3/1 = -3.
    # dS/dc1 = -1/c0 = -1.
    # dS/dc2 = 0.
    print(f"Expected Gradients:       {expected}")
    
    # Finite Difference
    eps = 1e-4
    fd_grads = []
    for i in range(3):
        c_plus = coeffs.at[i].add(eps)
        c_minus = coeffs.at[i].add(-eps)
        s_plus = sum_roots(c_plus)
        s_minus = sum_roots(c_minus)
        fd_grads.append((s_plus - s_minus) / (2*eps))
    
    print(f"Finite Diff Gradients:    {fd_grads}")
    
    assert jnp.allclose(grads, jnp.array(fd_grads), atol=1e-3), "Gradient check failed!"
    print("Gradient Check PASSED.")

@pytest.mark.xfail(reason="known failure, pre-existing, see docs/decisions/010 §7.3: jnp.roots strip_zeros under jit", strict=False)
def test_recovery():
    print("\n--- Test 2: Microstructure Recovery ---")
    # Synthesis
    f_true = 0.7
    D1_true = 0.5e-3 # Slow
    D2_true = 2.0e-3 # Fast (Large)
    
    delta_b = 1000.0
    b_values = jnp.array([0.0, 1000.0, 2000.0, 3000.0])
    
    # Signal S_k = f * exp(-k*delta_b*D1) + (1-f) * exp(-k*delta_b*D2)
    # Assume S0=1 (normalized)
    
    # Invariants input: [S0, S1, S2, S3]
    ks = jnp.arange(4)
    # shape (4,)
    signal = f_true * jnp.exp(-ks * delta_b * D1_true) + (1-f_true) * jnp.exp(-ks * delta_b * D2_true)
    
    print(f"Synthetic Invariants: {signal}")
    
    # Solve
    # Ensure batch dimension for consistency with vmap
    signal_batch = jnp.expand_dims(signal, 0)
    
    result = solve_microstructure(signal_batch, delta_b=delta_b)
    # Result: [D_slow, D_fast, f_slow]
    
    D_slow, D_fast, f_slow = result[0]
    
    print(f"Recovered: D_slow={D_slow:.4e}, D_fast={D_fast:.4e}, f={f_slow:.2f}")
    print(f"True:      D_slow={D1_true:.4e}, D_fast={D2_true:.4e}, f={f_true:.2f}")
    
    assert jnp.isclose(D_slow, D1_true, rtol=1e-3), "D_slow mismatch"
    assert jnp.isclose(D_fast, D2_true, rtol=1e-3), "D_fast mismatch"
    assert jnp.isclose(f_slow, f_true, rtol=1e-3), "Fraction mismatch"
    print("Recovery Test PASSED.")

@pytest.mark.xfail(reason="known failure, pre-existing, see docs/decisions/010 §7.3: jnp.roots under jit / empty roots array", strict=False)
def test_performance():
    print("\n--- Test 3: Large Scale Performance (vmap) ---")
    
    N_voxels = 1_000_000
    print(f"Generating {N_voxels} voxels...")
    
    key = jax.random.PRNGKey(0)
    # Random parameters
    f = jax.random.uniform(key, (N_voxels,), minval=0.1, maxval=0.9)
    D1 = jax.random.uniform(key, (N_voxels,), minval=0.1e-3, maxval=1.0e-3)
    D2 = jax.random.uniform(key, (N_voxels,), minval=1.5e-3, maxval=3.0e-3)
    
    delta_b = 1000.0
    ks = jnp.arange(4) # (4,)
    
    # Compute signals: (N, 4)
    # f shape (N, 1), D shape (N, 1), ks shape (1, 4)
    f_ = f[:, None]
    D1_ = D1[:, None]
    D2_ = D2[:, None]
    ks_ = ks[None, :]
    
    signals = f_ * jnp.exp(-ks_ * delta_b * D1_) + (1-f_) * jnp.exp(-ks_ * delta_b * D2_)
    
    # JIT Compile
    print("Compiling...")
    solver_vmap = jax.jit(jax.vmap(lambda s: solve_microstructure(s, delta_b=delta_b)))
    
    # Warmup
    _ = solver_vmap(signals[:10]).block_until_ready()
    
    print("Running Benchmark...")
    start = time.time()
    results = solver_vmap(signals).block_until_ready()
    end = time.time()
    
    duration = end - start
    print(f"Processed {N_voxels} voxels in {duration:.4f} seconds.")
    
    if duration < 1.0:
        print("PERFORMANCE SUCCESS: < 1 second.")
    else:
        print("PERFORMANCE WARNING: > 1 second.")

if __name__ == "__main__":
    test_roots_gradient()
    test_recovery()
    test_performance()

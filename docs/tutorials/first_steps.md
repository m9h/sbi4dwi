# First Steps with SBI4DWI

This tutorial demonstrates the core advantages of SBI4DWI (import name `dmipy_jax`): high-performance signal prediction using JAX's compilation and vectorization capabilities. We will compare a standard Python loop against JAX's `vmap` and `jit`.

## 1. Setup

First, let's import the necessary modules. We use `dmipy_jax` for the model and acquisition, and `jax` for the computation.

```python
import time
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.acquisition import JaxAcquisition
```

## 2. Defining the Acquisition

In SBI4DWI, we use `JaxAcquisition` to handle the experimental scheme. It converts inputs to JAX arrays automatically. b-values are stored in **SI units (s/m^2)**: a clinical shell at b = 1000 s/mm^2 is `1000e6`.

```python
# Setup Parameters
N_VOXELS = 50000
N_GRADIENTS = 60

# Shell at b=1000 s/mm^2 (= 1000e6 s/m^2 in SI) with 60 gradient directions
bvals = jnp.ones(N_GRADIENTS) * 1000e6

# Random gradients on sphere
grads_np = np.random.randn(N_GRADIENTS, 3)
grads_np /= np.linalg.norm(grads_np, axis=1, keepdims=True)
grads = jnp.array(grads_np)

# Create JaxAcquisition object
acq = JaxAcquisition(bvalues=bvals, gradient_directions=grads)
```

## 3. Creating Mock Data

We generate synthetic microstructure parameters for 50,000 voxels to simulate a large dataset.

```python
# Random Model Parameters
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)

# mu: (N, 2) angles [theta, phi]
mu = jax.random.uniform(k1, shape=(N_VOXELS, 2), minval=0.0, maxval=np.pi)

# lambda_par: (N,) diffusivity [0.1e-9, 3.0e-9]
lambda_par = jax.random.uniform(k2, shape=(N_VOXELS,), minval=0.1e-9, maxval=3.0e-9)
```

## 4. The Model: Stick

We instantiate the `C1Stick` model. JAX models in SBI4DWI are stateless configuration objects; the state (parameters) is passed at call time.

```python
model = C1Stick()

# Define a prediction wrapper for a single voxel
def predict_one(mu_val, lambda_val):
    return model(acq.bvalues, acq.gradient_directions, mu=mu_val, lambda_par=lambda_val)
```

## 5. JAX Acceleration: Vmap and JIT

### Vectorization with `vmap`
Instead of writing a `for` loop, we use `jax.vmap` (vectorizing map) to automatically batch the computation over the voxel dimension.

```python
# Maps over axis 0 of mu and lambda_par
vmapped_predict = jax.vmap(predict_one, in_axes=(0, 0))
```

### Compilation with `jit`
We use `jax.jit` (Just-In-Time compilation) to compile the vectorised function into XLA (Accelerated Linear Algebra) code, optimized for your CPU or GPU.

```python
jit_vmapped_predict = jax.jit(vmapped_predict)
```

### Why `block_until_ready()`?

JAX operations are asynchronous. When you call a JAX function, it returns a placeholder (a future) immediately, while the computation happens in the background. To measure the *actual* computation time, we must call `.block_until_ready()` on the result.

```python
# Warmup (compilation happens here)
_ = jit_vmapped_predict(mu[:10], lambda_par[:10]).block_until_ready()

print(f"Running Vmap on {N_VOXELS} voxels...")
start_time = time.perf_counter()

# Run on full dataset
res = jit_vmapped_predict(mu, lambda_par)
res.block_until_ready() # Wait for completion

end_time = time.perf_counter()
print(f"JAX Time: {end_time - start_time:.5f} s")
```

## Comparisons

On a typical machine, the JAX implementation is orders of magnitude faster than a Python loop, especially for complex composition models.

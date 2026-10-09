# Physics Simulation with SBI4DWI

SBI4DWI includes a Monte Carlo diffusion simulator written in JAX
(`dmipy_jax.simulation.monte_carlo`). It lets you simulate restricted
diffusion in geometries described by signed distance functions, and the same
code runs on CPU or GPU.

The sizes in this tutorial (a few thousand particles, a few hundred time
steps) are chosen so that the whole page runs in seconds on a CPU. For
production-quality signals scale `N_particles` and the time resolution up
and run on a GPU.

## 1. Setup

Import the necessary modules from `dmipy_jax.simulation`.

```python
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.simulation.monte_carlo import simulate_ground_truth, cylinder_sdf
```

## 2. Define Geometry

The simulator uses Signed Distance Functions (SDFs) to define geometry.
Negative values are inside the fluid, positive values are walls. All lengths
are in metres.

```python
# Radius of 5 microns
radius = 5.0e-6

# Define SDF for a cylinder (axis along z)
# We use partial application to bake in the radius
from functools import partial
geometry_sdf = partial(cylinder_sdf, radius=radius)
```

## 3. Initialize Particles

We need a function to generate initial particle positions. For a cylinder, we
can initialize them uniformly inside.

```python
def initialization_func(key, n_particles):
    # Simplified initialization: a cube inside the cylinder for the demo.
    # In practice, use rejection sampling to fill the circle uniformly.
    return jax.random.uniform(key, (n_particles, 3), minval=-radius/2, maxval=radius/2)
```

## 4. Compile Simulator

Generate the optimized simulation function for this geometry.

```python
simulator = simulate_ground_truth(geometry_sdf, initialization_func)
```

## 5. Define Gradient Waveform

Create a simple pulsed-gradient waveform: a single gradient lobe along x for
the first 10 ms of a 50 ms window. (A full PGSE experiment would add the
refocused second lobe; the simulator accumulates phase from whatever
waveform you pass.)

```python
# Time resolution
dt = 1e-4
duration = 0.05 # 50 ms
n_steps = int(duration / dt)   # 500 steps

# Simple gradient pulse
G_max = 0.04 # 40 mT/m
gradients = jnp.zeros((n_steps, 3))
# Apply gradient in X direction for the first 10 ms
gradients = gradients.at[:100, 0].set(G_max)
```

## 6. Run Simulation

Run the simulation. The returned value is the magnitude of the ensemble
average of the particle phases, i.e. the signal attenuation in [0, 1].

```python
key = jax.random.PRNGKey(0)
D_water = 3.0e-9  # Diffusivity (m^2/s)
N_particles = 2_000

# First run compiles (may take a few seconds)
signal = simulator(gradients, D_water, dt, N_particles, key)

print(f"Simulated Signal Attenuation: {float(signal):.4f}")
assert 0.0 <= float(signal) <= 1.0
```

## 7. Performance

Because the simulator is JIT-compiled, subsequent runs with different
parameters (e.g., different `D` or `gradients` of the same shape) reuse the
compiled kernel and are much faster, especially on GPU:

```python
signal_slow = simulator(gradients, 1.0e-9, dt, N_particles, key)
print(f"Attenuation with D = 1e-9 m^2/s: {float(signal_slow):.4f}")
```

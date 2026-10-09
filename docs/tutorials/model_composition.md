# Multi-Compartment Model Composition and Classical Fitting

This tutorial walks through building multi-compartment diffusion MRI signal
models from individual compartments and fitting them to data using the
classical (optimisation-based) pipeline in SBI4DWI. By the end you will
know how to:

* Set up an acquisition scheme with correct SI units
* Instantiate and compose compartment models
* Generate synthetic signals from ground-truth parameters
* Fit single-voxel and multi-voxel data
* Evaluate parameter recovery under Rician noise

The code patterns here are drawn directly from the project test suite
(`tests/test_jax_full_workflow.py` and `tests/test_multi_compartment.py`).

---

## 1. Multi-compartment diffusion models in brief

Biological tissue is not a single homogeneous medium. White matter, for
example, contains water trapped inside axons (*intra-axonal*), water
diffusing between axons (*extra-axonal*), and free water (*CSF*). A
multi-compartment model (MCM) represents the measured signal as a
volume-fraction-weighted sum of independent compartment signals:

$$
S(\mathbf{q}, b) = \sum_{i=1}^{N} f_i \, S_i(\mathbf{q}, b)
$$

where $f_i$ are the partial volume fractions ($\sum f_i = 1$) and each
$S_i$ is an analytical signal model describing one tissue type.

SBI4DWI lets you compose these models like building blocks: instantiate
each compartment, pass them as a list to `JaxMultiCompartmentModel`, and the
framework handles parameter bookkeeping, signal composition, and fitting.

---

## 2. Setting up `JaxAcquisition` with proper SI units

The most common gotcha in diffusion MRI software is **b-value units**.
Clinical scanners and tools like FSL and DIPY report b-values in
**s/mm^2**, but SBI4DWI stores them internally in **SI units (s/m^2)**.
The conversion factor is:

$$
b_{\text{SI}} = b_{\text{scanner}} \times 10^6
$$

So a shell at b = 1000 s/mm^2 becomes 1e9 s/m^2 in code.

```python
import jax
import jax.numpy as jnp
from dmipy_jax.acquisition import JaxAcquisition

# Two-shell acquisition: b=0, b=1000, b=2000 (in s/mm^2)
# IMPORTANT: convert to SI (s/m^2) before passing to JaxAcquisition
bvalues = jnp.array([
    0.,                           # b=0 volume
    1e9, 1e9, 1e9,                # b=1000 s/mm^2 = 1e9 s/m^2
    2e9, 2e9, 2e9                 # b=2000 s/mm^2 = 2e9 s/m^2
])

# Gradient directions (unit vectors). b=0 gets a zero vector.
bvecs = jnp.array([
    [0., 0., 0.],                 # b=0 — direction is irrelevant
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [0.707, 0.707, 0.],
    [0.707, 0., 0.707],
    [0., 0.707, 0.707]
])

# Normalise non-zero vectors to unit length
norms = jnp.linalg.norm(bvecs, axis=1, keepdims=True)
norms = jnp.where(norms == 0, 1.0, norms)
bvecs = bvecs / norms

acq = JaxAcquisition(bvalues=bvalues, gradient_directions=bvecs)
```

```{tip}
If you load b-values from an FSL `.bval` file (which uses s/mm^2), multiply
by 1e6 before constructing `JaxAcquisition`. Forgetting this will produce
nearly flat signals because the effective diffusion weighting is a million
times too small.
```

---

## 3. Building compartment models

SBI4DWI provides several analytical compartment models as Equinox
modules. The three most common building blocks are:

| Model | Class | Parameters | Tissue interpretation |
|-------|-------|------------|-----------------------|
| **Stick** | `Stick()` | `mu` (orientation), `lambda_par` (axial diffusivity) | Intra-axonal (zero-radius cylinder) |
| **Ball** | `G1Ball()` | `lambda_iso` (isotropic diffusivity) | CSF / free water |
| **Zeppelin** | `G2Zeppelin()` | `mu`, `lambda_par`, `lambda_perp` | Extra-axonal (axially symmetric tensor) |

Each model is a stateless `eqx.Module`. Parameters can be set at
construction time or passed as keyword arguments at call time. Parameter
ranges and cardinalities are stored as class attributes.

```python
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.gaussian import G1Ball, G2Zeppelin

stick = Stick()
ball = G1Ball()

# Inspect parameter metadata
print(stick.parameter_names)        # ['mu', 'lambda_par']
print(stick.parameter_cardinality)  # {'mu': 2, 'lambda_par': 1}
print(ball.parameter_ranges)        # {'lambda_iso': (0.1e-9, 3e-9)}
```

The Stick model uses spherical coordinates for orientation: `mu = [theta,
phi]` where theta is the polar angle (0 to pi) and phi is the azimuthal
angle (-pi to pi). Internally it converts to a Cartesian unit vector before
computing the signal:

$$
S_{\text{stick}} = \exp\!\bigl(-b \cdot \lambda_\parallel \cdot (\mathbf{g} \cdot \hat{\mu})^2\bigr)
$$

The Ball model is isotropic (direction-independent):

$$
S_{\text{ball}} = \exp\!\bigl(-b \cdot \lambda_{\text{iso}}\bigr)
$$

---

## 4. Composing a multi-compartment model

Pass a list of compartments to `JaxMultiCompartmentModel`. The framework
automatically:

1. Collects parameter names from all models
2. Resolves name collisions (appending `_1`, `_2`, etc. if needed)
3. Adds `partial_volume_0`, `partial_volume_1`, ... for volume fractions
4. Builds a composed forward function via `compose_models()`

```python
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

model = JaxMultiCompartmentModel([stick, ball])

# The model now manages all parameters from both compartments
print(model.parameter_names)
# ['mu', 'lambda_par', 'lambda_iso', 'partial_volume_0', 'partial_volume_1']
```

The composed signal is:

$$
S = f_0 \cdot S_{\text{stick}} + f_1 \cdot S_{\text{ball}}
$$

where $f_0$ = `partial_volume_0` and $f_1$ = `partial_volume_1` (and
$f_0 + f_1 = 1$).

---

## 5. Generating synthetic signals from ground truth parameters

To test fitting, we first generate noiseless data from known parameters.
Parameters are specified as a dictionary and then converted to a flat array
using `parameter_dictionary_to_array()`:

```python
# Ground truth: 60% stick (axonal) + 40% ball (CSF)
true_params = {
    'mu': jnp.array([1.57, 0.0]),          # orientation ~ x-axis
    'lambda_par': jnp.array(2.0e-9),       # axial diffusivity (m^2/s)
    'lambda_iso': jnp.array(1.0e-9),       # isotropic diffusivity (m^2/s)
    'partial_volume_0': jnp.array(0.6),    # stick fraction
    'partial_volume_1': jnp.array(0.4)     # ball fraction
}

# Convert dict -> flat JAX array (model knows the ordering)
params_array = model.parameter_dictionary_to_array(true_params)

# Forward model: parameters + acquisition -> signal
signal = model.model_func(params_array, acq)
print(f"Signal shape: {signal.shape}")   # (7,) — one value per measurement
print(f"b=0 signal:   {signal[0]:.4f}")  # Should be ~1.0 (normalised)
```

The `parameter_dictionary_to_array` method handles the mapping from named
parameters to the flat vector that the composed forward function expects.
This is essential because the optimiser works on flat arrays internally.

---

## 6. Single-voxel fitting

The `model.fit()` method runs gradient-based optimisation (via Optimistix)
to recover parameters from a signal vector. For a 1-D signal input, it
performs single-voxel fitting:

```python
fitted_params = model.fit(acq, signal)

# Compare fitted vs true
print(f"True  lambda_par: {true_params['lambda_par']:.2e}")
print(f"Fit   lambda_par: {fitted_params['lambda_par']:.2e}")

print(f"True  lambda_iso: {true_params['lambda_iso']:.2e}")
print(f"Fit   lambda_iso: {fitted_params['lambda_iso']:.2e}")

print(f"True  f_stick:    {true_params['partial_volume_0']:.3f}")
print(f"Fit   f_stick:    {fitted_params['partial_volume_0']:.3f}")
```

For noiseless data, you should see near-exact recovery (errors < 1e-10 for
diffusivities, < 1e-2 for volume fractions).

To verify orientation recovery, convert the fitted spherical angles back to
a Cartesian vector and check the dot product with the true orientation:

```python
def mu_to_cartesian(mu):
    """Convert spherical (theta, phi) to unit Cartesian vector."""
    theta, phi = mu[0], mu[1]
    return jnp.array([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(theta)
    ])

fitted_vec = mu_to_cartesian(fitted_params['mu'])
true_vec = mu_to_cartesian(true_params['mu'])

# Stick is antipodally symmetric, so check |dot product|
alignment = jnp.abs(jnp.dot(fitted_vec, true_vec))
print(f"Orientation alignment: {alignment:.4f}")  # Should be > 0.95
```

---

## 7. Multi-voxel batch fitting

When you pass a 2-D array of shape `(N_voxels, N_measurements)` to
`model.fit()`, it automatically vectorises the fitting across voxels:

```python
# Define two voxels with different microstructure
voxel_params = [
    {
        'mu': jnp.array([1.57, 0.0]),       # x-axis
        'lambda_par': jnp.array(2.0e-9),
        'lambda_iso': jnp.array(1.0e-9),
        'partial_volume_0': jnp.array(0.6),
        'partial_volume_1': jnp.array(0.4)
    },
    {
        'mu': jnp.array([0.0, 0.0]),        # z-axis
        'lambda_par': jnp.array(1.5e-9),
        'lambda_iso': jnp.array(2.5e-9),
        'partial_volume_0': jnp.array(0.3),
        'partial_volume_1': jnp.array(0.7)
    }
]

# Generate signals and stack into (2, N_measurements)
signals = []
for tp in voxel_params:
    p_arr = model.parameter_dictionary_to_array(tp)
    signals.append(model.model_func(p_arr, acq))
data_multi = jnp.stack(signals)

print(f"Multi-voxel data shape: {data_multi.shape}")  # (2, 7)

# Fit all voxels at once
fitted_multi = model.fit(acq, data_multi)

# Fitted parameters are now arrays with a leading voxel dimension
print(f"lambda_iso shape: {fitted_multi['lambda_iso'].shape}")  # (2,)
print(f"mu shape:         {fitted_multi['mu'].shape}")           # (2, 2)

# Verify each voxel
for i in range(2):
    err = jnp.abs(fitted_multi['lambda_iso'][i] - voxel_params[i]['lambda_iso'])
    print(f"Voxel {i} lambda_iso error: {err:.2e}")
```

This vectorised fitting is the key to scaling SBI4DWI to whole-brain
volumes with hundreds of thousands of voxels.

---

## 8. Adding Rician noise and verifying parameter recovery

Real MRI data is corrupted by Rician noise. The magnitude signal from a
complex Gaussian channel is:

$$
S_{\text{noisy}} = \sqrt{(S + n_1)^2 + n_2^2}
$$

where $n_1, n_2 \sim \mathcal{N}(0, \sigma^2)$ and the signal-to-noise
ratio (SNR) is defined as $\text{SNR} = S_{b=0} / \sigma$.

```python
# Generate clean signal
params_array = model.parameter_dictionary_to_array(true_params)
signal_clean = model.model_func(params_array, acq)

# Add Rician noise at SNR = 50
# Since S(b=0) ~ 1.0 for normalised signals, sigma = 1/SNR
snr = 50
sigma = 1.0 / snr

key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)
n1 = jax.random.normal(k1, signal_clean.shape) * sigma
n2 = jax.random.normal(k2, signal_clean.shape) * sigma

signal_noisy = jnp.sqrt((signal_clean + n1)**2 + n2**2)
```

Now fit the noisy data and check that parameters are recovered within
reasonable tolerances:

```python
fitted_noisy = model.fit(acq, signal_noisy)

# With noise, we relax tolerances
rtol = 0.20  # 20% relative tolerance

print(f"lambda_iso: true={true_params['lambda_iso']:.2e}  "
      f"fit={fitted_noisy['lambda_iso']:.2e}")
print(f"f_stick:    true={true_params['partial_volume_0']:.3f}  "
      f"fit={fitted_noisy['partial_volume_0']:.3f}")

assert jnp.allclose(
    fitted_noisy['lambda_iso'], true_params['lambda_iso'],
    rtol=rtol
)
assert jnp.allclose(
    fitted_noisy['partial_volume_0'], true_params['partial_volume_0'],
    rtol=rtol, atol=0.1
)
```

```{note}
At SNR = 50 you should see parameter errors well within 20%. At lower SNR
(e.g., 10--20), the classical optimisation-based fitting starts to struggle
-- this is where simulation-based inference (SBI) can offer substantial
improvements. See the {doc}`sbi_dti` tutorial for the SBI approach.
```

---

## 9. Parameter dictionaries and array conversion

The `JaxMultiCompartmentModel` provides two-way conversion between
human-readable parameter dictionaries and the flat arrays used internally:

```python
# Dict -> Array
params_array = model.parameter_dictionary_to_array(true_params)
print(f"Flat array length: {len(params_array)}")
# mu(2) + lambda_par(1) + lambda_iso(1) + f0(1) + f1(1) = 6

# The model.fit() method returns a dictionary directly. With the default
# compute_uncertainty=True it also carries a '<name>_std' entry per parameter
# (Cramer-Rao lower-bound standard deviations).
fitted = model.fit(acq, signal)
print(type(fitted))  # dict
print(sorted(fitted.keys()))
# ['lambda_iso', 'lambda_iso_std', 'lambda_par', 'lambda_par_std', 'mu', 'mu_std',
#  'partial_volume_0', 'partial_volume_0_std', 'partial_volume_1', 'partial_volume_1_std']
assert set(model.parameter_names) <= set(fitted.keys())
```

The parameter ordering in the flat array is determined by the order in which
models are passed to `JaxMultiCompartmentModel`, followed by the volume
fractions. You generally do not need to know this ordering -- use the
dictionary interface for clarity and let the framework handle the mapping.

---

## Summary

| Step | Function / Class | Purpose |
|------|-----------------|---------|
| Acquisition | `JaxAcquisition(bvalues=..., gradient_directions=...)` | Define measurement scheme (SI units!) |
| Compartments | `Stick()`, `G1Ball()`, `G2Zeppelin()` | Individual signal models |
| Composition | `JaxMultiCompartmentModel([...])` | Combine into MCM |
| Forward model | `model.model_func(params_array, acq)` | Predict signal |
| Dict/array | `model.parameter_dictionary_to_array(dict)` | Convert for internal use |
| Fitting | `model.fit(acq, data)` | Optimisation-based parameter recovery |

The classical fitting pipeline is ideal for quick prototyping and for
noiseless or high-SNR data. For noisy, high-throughput, or
uncertainty-aware inference, proceed to the SBI tutorials.

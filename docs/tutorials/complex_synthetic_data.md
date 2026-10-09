# Creating Complex Synthetic Datasets

For testing and benchmarking microstructure imaging algorithms, simple
homogeneous voxel simulations are often insufficient. SBI4DWI lets you
generate synthetic datasets with known ground truth properties, including:
- Complex multi-compartment compositions (e.g., NODDI-like, SANDI).
- Spatially varying microstructure parameters (phantoms).
- Realistic spatial noise distributions (Gaussian random fields).

This tutorial walks through creating a small "phantom" dataset: a
three-compartment Stick + Zeppelin + Ball model evaluated on a 3-D grid of
voxels, with spatially correlated Rician noise.

## 1. Defining the Microstructure Model

First, we define the physics model we want to simulate.
`JaxMultiCompartmentModel` composes arbitrary compartment models into a
volume-fraction-weighted sum.

```python
import jax
import jax.numpy as jnp
import numpy as np

from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G2Zeppelin, G1Ball

# Define NODDI-like components (no orientation dispersion in this demo)
stick = C1Stick()       # Intra-neurite
zeppelin = G2Zeppelin() # Extra-neurite
ball = G1Ball()         # CSF

# Combine models
noddi = JaxMultiCompartmentModel([stick, zeppelin, ball])
print(noddi.parameter_names)
```

Where two compartments share a parameter name the framework appends a
numeric suffix (`mu_2`, `lambda_par_2` belong to the Zeppelin). The
original dmipy `set_equal_parameter` / `set_tortuous_parameter` constraints
are not objects in SBI4DWI: constraints are applied **when you build the
parameter arrays**, as we do below for the tortuosity relation
$\lambda_\perp = \lambda_\parallel (1 - f_\text{stick})$.

## 2. Generating Spatially Varying Parameters

Instead of a single voxel, we create "parameter maps" where each parameter is
a 3-D numpy array.

```python
dimensions = (10, 10, 10)
x = np.linspace(0, 1, dimensions[0])
y = np.linspace(0, 1, dimensions[1])
z = np.linspace(0, 1, dimensions[2])
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Generate varying volume fractions
# Stick (intra) fraction increases along X
f_stick = 0.3 + 0.4 * X
# CSF fraction increases along Y
f_csf = 0.2 * Y

# Ensure sum == 1
f_zeppelin = 1.0 - f_stick - f_csf

# Define orientations as spherical angles (theta, phi): fibres in the
# x-y plane (theta = pi/2) rotating with Z
theta = np.full(dimensions, np.pi / 2)
phi = Z * np.pi
mu = np.stack([theta, phi], axis=-1)          # (..., 2)

# Fixed diffusivities (m^2/s)
lambda_par = 1.7e-9
lambda_iso = 3.0e-9
# Tortuosity constraint for the extra-neurite compartment
lambda_perp = lambda_par * (1.0 - f_stick)

n_vox = int(np.prod(dimensions))
const = lambda v: np.full(n_vox, v)

# Build parameter dictionary with one row per voxel
parameters = {
    'mu': mu.reshape(n_vox, 2),
    'lambda_par': const(lambda_par),
    'mu_2': mu.reshape(n_vox, 2),                   # aligned with the stick
    'lambda_par_2': const(lambda_par),              # equal to the stick's
    'lambda_perp': lambda_perp.reshape(n_vox),
    'lambda_iso': const(lambda_iso),
    'partial_volume_0': f_stick.reshape(n_vox),
    'partial_volume_1': f_zeppelin.reshape(n_vox),
    'partial_volume_2': f_csf.reshape(n_vox),
}
parameters = {k: jnp.asarray(v) for k, v in parameters.items()}
```

## 3. Simulating Signal

Define a three-shell acquisition (b-values in **SI units**, s/m^2) and call
the model with the dictionary of per-voxel arrays. The model detects the
leading voxel dimension and vectorises over it with `jax.vmap`.

```python
from dmipy_jax.acquisition import JaxAcquisition

rng = np.random.default_rng(0)
n_dirs = 20
shells = [1000e6, 2000e6, 3000e6]      # 1000, 2000, 3000 s/mm^2

bvals = np.concatenate([np.zeros(2)] + [np.full(n_dirs, b) for b in shells])
bvecs = rng.standard_normal((len(bvals), 3))
bvecs[:2] = 0.0
bvecs /= np.maximum(np.linalg.norm(bvecs, axis=1, keepdims=True), 1e-8)

scheme = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

signal = noddi(parameters, scheme)           # (n_vox, N_measurements)
signal = np.asarray(signal).reshape(dimensions + (len(bvals),))
print(signal.shape)                           # (10, 10, 10, 62)
assert np.allclose(signal[..., :2], 1.0)      # b=0 volumes are unattenuated
```

## 4. Adding Realistic Spatial Noise

Simple white noise doesn't capture realistic MRI artefacts like coil
sensitivity profiles or tissue heterogeneity. We can use **Gaussian random
fields (GRF)** to add spatially correlated noise.

### Spatially Varying SNR Map
Simulate coil sensitivity by varying SNR across the image.

```python
import scipy.ndimage

def generate_grf(shape, fwhm=5.0):
    noise = np.random.normal(0, 1, shape)
    sigma = fwhm / 2.355
    smooth = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
    smooth -= smooth.mean()
    smooth /= smooth.std()
    return smooth

# Base SNR 30, varying by +/- 10
snr_map = 30 + 10 * generate_grf(dimensions, fwhm=5.0)
snr_map = np.maximum(snr_map, 5.0) # Avoid negative SNR
```

### Adding Rician Noise
Add noise based on the local SNR map.

```python
sigma_map = 1.0 / snr_map
sigma_map = sigma_map[..., None] # Broadcast to measurements

noise_r = np.random.normal(0, 1, signal.shape) * sigma_map
noise_i = np.random.normal(0, 1, signal.shape) * sigma_map

noisy_signal = np.sqrt((signal + noise_r)**2 + noise_i**2)
print(f"Noisy b=0 mean: {noisy_signal[..., :2].mean():.3f} (Rician bias pushes it above 1)")
```

## 5. Going Further

The same pattern -- a parameter dictionary of per-voxel arrays plus a
`JaxAcquisition` -- works for any composition of SBI4DWI compartments. For
phantoms with orientation dispersion, wrap the Stick and Zeppelin in a
`DistributedModel` with an `SD1Watson` distribution as shown in the
{doc}`sbi_noddi` tutorial.

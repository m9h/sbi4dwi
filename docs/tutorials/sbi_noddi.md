# Advanced Microstructure: SBI for NODDI

Neurite Orientation Dispersion and Density Imaging (NODDI) is one of the most
widely used biophysical models in clinical diffusion MRI. It decomposes the
white-matter signal into three tissue compartments -- intra-neurite sticks,
extra-neurite hindered diffusion, and free-water isotropic diffusion -- and
returns clinically interpretable maps of neurite density, orientation
dispersion, and CSF contamination.

This tutorial demonstrates how to set up a full NODDI forward model in
SBI4DWI, define physiologically informed priors, simulate training data with
the tortuosity constraint, and train a neural posterior estimator (NPE) that
returns the complete posterior distribution over NODDI parameters in a single
forward pass.

Network and training sizes are kept small so the page runs in under a minute
on CPU; production settings are noted where they differ.

```{contents}
:depth: 3
:local:
```

## Prerequisites

- SBI4DWI installed (`uv sync` from the repository root)
- Familiarity with the SBI training pipeline (see {doc}`training_to_deployment`)
- Basic understanding of diffusion MRI signal models

---

## 1. Introduction to NODDI

The NODDI model {cite}`zhang2012noddi` explains the normalised diffusion signal
$E(\mathbf{q})$ as a three-compartment mixture:

$$
E(\mathbf{q}) \;=\; f_\text{iso}\, E_\text{iso}(\mathbf{q})
  \;+\; (1 - f_\text{iso})\!\left[
    f_\text{ic}\, E_\text{ic}(\mathbf{q})
    \;+\; (1 - f_\text{ic})\, E_\text{ec}(\mathbf{q})
  \right]
$$

where:

| Symbol | Compartment | Biophysical model |
|--------|-------------|-------------------|
| $E_\text{ic}$ | Intra-cellular (neurites) | Stick convolved with Watson distribution |
| $E_\text{ec}$ | Extra-cellular (hindered) | Zeppelin convolved with Watson distribution |
| $E_\text{iso}$ | Isotropic (CSF / free water) | Ball (isotropic Gaussian) |

The free parameters of interest are:

- **$f_\text{ic}$** (neurite density index, NDI) -- fraction of tissue signal
  from neurites.
- **$f_\text{iso}$** -- volume fraction of free water.
- **$\kappa$** -- Watson concentration parameter controlling orientation
  dispersion (higher $\kappa$ = more aligned neurites).

The extra-cellular perpendicular diffusivity is not free but is constrained by
the **tortuosity approximation** (Section 6).

---

## 2. Setting Up Distributed Models

NODDI accounts for fibre dispersion by convolving each compartment signal with
an orientation distribution function (ODF) on the sphere. In SBI4DWI this is
handled by {class}`~dmipy_jax.distributions.distribute_models.DistributedModel`,
which numerically integrates a base signal model over a spherical distribution
of the orientation parameter `mu`.

```python
from dmipy_jax.distributions.distribute_models import DistributedModel
from dmipy_jax.distributions.sphere_distributions import SD1Watson
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G2Zeppelin, G1Ball
```

The intra-cellular compartment is a Stick (zero-radius cylinder) dispersed
with a Watson distribution:

```python
stick = C1Stick()
watson_ic = SD1Watson(grid_size=200)

intra = DistributedModel(stick, watson_ic, target_parameter='mu')
```

`DistributedModel` removes `mu` from the Stick parameter list and replaces it
with the Watson parameters (`mu`, `kappa`). The signal is computed by
evaluating the Stick at each grid point on the sphere, weighting by the Watson
PDF, and summing -- a numerical convolution performed efficiently under
`jax.jit`.

The extra-cellular compartment uses the same pattern with a Zeppelin:

```python
zeppelin = G2Zeppelin()
watson_ec = SD1Watson(grid_size=200)

extra = DistributedModel(zeppelin, watson_ec, target_parameter='mu')
```

The isotropic compartment needs no dispersion:

```python
ball = G1Ball()
```

---

## 3. Composing the Multi-Compartment Model

The three compartments are combined into a single forward model using
{class}`~dmipy_jax.core.modeling_framework.JaxMultiCompartmentModel`:

```python
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

model = JaxMultiCompartmentModel([intra, extra, ball])
print(model.parameter_names)
```

The composite model collects all sub-model parameters into a single namespace.
When parameter names collide across compartments, the framework appends a
numeric suffix (`_2` for the second compartment). For NODDI the resulting
parameters are:

| Parameter | Compartment | Description |
|-----------|-------------|-------------|
| `lambda_par` | Intra-cellular | Parallel diffusivity |
| `mu` | Intra-cellular | Fibre orientation $(\theta, \phi)$ |
| `kappa` | Intra-cellular | Watson concentration |
| `lambda_par_2` | Extra-cellular | Parallel diffusivity |
| `lambda_perp` | Extra-cellular | Perpendicular diffusivity (tortuosity-constrained) |
| `mu_2` | Extra-cellular | Fibre orientation (shared with intra) |
| `kappa_2` | Extra-cellular | Watson concentration (shared with intra) |
| `lambda_iso` | Ball | Isotropic diffusivity |
| `partial_volume_0` | -- | Signal fraction for intra |
| `partial_volume_1` | -- | Signal fraction for extra |
| `partial_volume_2` | -- | Signal fraction for ball |

```python
assert set(model.parameter_names) == {
    'lambda_par', 'mu', 'kappa',
    'lambda_par_2', 'lambda_perp', 'mu_2', 'kappa_2',
    'lambda_iso',
    'partial_volume_0', 'partial_volume_1', 'partial_volume_2',
}
```

---

## 4. Multi-Shell Acquisition Design

NODDI requires at least two non-zero b-value shells to disentangle the three
compartments. The standard clinical protocol uses $b = 700$ and
$b = 2000\;\text{s/mm}^2$, each with 32 gradient directions, plus two $b = 0$
volumes:

```python
import jax
import jax.numpy as jnp
from dmipy_jax.acquisition import JaxAcquisition

def get_acquisition():
    """Create a standard two-shell NODDI acquisition."""
    # b-values in SI units (s/m^2).  700 s/mm^2 = 7e8 s/m^2.
    b1 = 700e6
    b2 = 2000e6

    bvals = jnp.concatenate([
        jnp.zeros(2),          # 2 x b=0
        jnp.full(32, b1),      # 32 dirs at b=700
        jnp.full(32, b2),      # 32 dirs at b=2000
    ])

    # Random gradient directions on the unit sphere
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    def rand_unit_vectors(k, n):
        z = jax.random.normal(k, (n, 3))
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    v0 = jnp.array([[1.0, 0.0, 0.0]] * 2)   # b=0 directions (unused)
    v1 = rand_unit_vectors(k1, 32)
    v2 = rand_unit_vectors(k2, 32)

    bvecs = jnp.concatenate([v0, v1, v2], axis=0)

    return JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

acquisition = get_acquisition()
N_MEAS = len(acquisition.bvalues)  # 66 measurements
assert N_MEAS == 66
```

```{note}
SBI4DWI stores b-values in SI units ($\text{s/m}^2$). Clinical scanners and
DIPY report b-values in $\text{s/mm}^2$. Multiply by $10^6$ when converting
from clinical to SI.
```

The lower shell ($b = 700$) is sensitive to the extra-cellular compartment and
orientation, while the higher shell ($b = 2000$) provides the contrast needed
to separate intra-cellular signal (sticks) from the extra-cellular hindered
component.

---

## 5. Parameter Priors for NODDI

Choosing informative but broad priors is critical for SBI. The priors should
cover the physiological range of each parameter while concentrating simulation
effort where the parameters are most likely to occur in vivo.

The snippets in this section draw a handful of samples from each prior to
show the sampling calls; Section 8 assembles them into the batch simulator.

```python
key = jax.random.PRNGKey(0)
batch_size = 8
```

### 5.1 Neurite Density Index ($f_\text{ic}$)

In healthy white matter $f_\text{ic}$ typically falls in the range 0.3--0.7. A
$\text{Beta}(5, 5)$ distribution captures this:

```python
f_intra = jax.random.beta(key, 5.0, 5.0, shape=(batch_size,))
assert f_intra.shape == (batch_size,)
```

The Beta(5, 5) distribution has mean 0.5 and places 95% of its mass between
approximately 0.2 and 0.8 -- wide enough to cover pathological values while
focusing training samples in the physiological range.

### 5.2 Isotropic Volume Fraction ($f_\text{iso}$)

CSF contamination is typically small in white matter but can be substantial in
cortical or periventricular voxels. A $\text{Beta}(0.5, 5.0)$ distribution
places most mass near zero while maintaining support up to 1:

```python
f_iso = jax.random.beta(key, 0.5, 5.0, shape=(batch_size,))
```

### 5.3 Watson Concentration ($\kappa$) -- Log-Uniform Prior

The Watson concentration parameter $\kappa$ spans several orders of magnitude:
$\kappa \approx 0.1$ for nearly isotropic dispersion and $\kappa > 30$ for
highly aligned fibres. A **log-uniform** prior ensures that the network sees
adequate training examples across the full dynamic range:

$$
\log \kappa \sim \mathcal{U}(\log 0.1,\; \log 32)
\quad\Longleftrightarrow\quad
p(\kappa) \propto \frac{1}{\kappa}
$$

```python
min_log_k = jnp.log(0.1)
max_log_k = jnp.log(32.0)
kappa = jnp.exp(
    jax.random.uniform(key, (batch_size,), minval=min_log_k, maxval=max_log_k)
)
```

```{tip}
A uniform prior on $\kappa$ would over-represent the high-$\kappa$ regime
where the signal is already well determined by a single fibre direction.
The log-uniform prior balances training effort across the dispersion spectrum.
```

### 5.4 Fibre Orientation ($\mu$)

The orientation $\mu = (\theta, \phi)$ is sampled uniformly on the sphere.
Drawing a 3D Gaussian vector and normalising gives a uniform distribution on
$S^2$, from which the polar angles are extracted:

```python
z = jax.random.normal(key, (batch_size, 3))
z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
theta = jnp.arccos(jnp.clip(z[:, 2], -1.0, 1.0))
phi = jnp.arctan2(z[:, 1], z[:, 0])
mu = jnp.stack([theta, phi], axis=1)  # (batch_size, 2)
```

---

## 6. The Watson Distribution and Orientation Dispersion

The Watson distribution is the spherical analogue of a Gaussian for
axially symmetric orientations. Its probability density function on the unit
sphere is:

$$
f(\mathbf{n};\, \boldsymbol{\mu},\, \kappa)
  = \frac{1}{C(\kappa)}\,
    \exp\!\bigl(\kappa\,(\mathbf{n} \cdot \boldsymbol{\mu})^2\bigr)
$$

where $C(\kappa) = {}_1F_1\!\bigl(\tfrac{1}{2};\, \tfrac{3}{2};\, \kappa\bigr)$
is the confluent hypergeometric normalisation constant.

The concentration parameter $\kappa$ controls dispersion:

| $\kappa$ | Regime | Biological interpretation |
|----------|--------|--------------------------|
| $\approx 0$ | Isotropic | Crossing or fanning fibres |
| $1$--$4$ | Moderate dispersion | Typical cortex / pathology |
| $> 16$ | Highly concentrated | Coherent white-matter bundles |

In SBI4DWI, {class}`~dmipy_jax.distributions.sphere_distributions.SD1Watson`
evaluates the Watson PDF on a Fibonacci sphere grid (default 200 points) and
uses it as integration weights inside `DistributedModel`.

---

## 7. Tortuosity Constraint (Extra-Cellular Compartment)

A key feature of the NODDI model is the **tortuosity constraint** that links
the extra-cellular perpendicular diffusivity to the neurite density:

$$
d_\perp = d_\parallel\,(1 - f_\text{ic})
$$

This relationship emerges from the long-time limit of diffusion around
randomly packed cylinders (Szafer et al., 1995). It reduces the number of
free parameters by one and regularises the fit.

In SBI4DWI, the tortuosity constraint is enforced **at simulation time** --
when sampling training pairs, $d_\perp$ is computed from $d_\parallel$ and
$f_\text{ic}$ rather than sampled independently:

```python
d_par = 1.7e-9   # m^2/s  (fixed intra-neurite parallel diffusivity)
d_iso = 3.0e-9   # m^2/s  (free-water diffusivity)

# Tortuosity constraint
d_perp = d_par * (1.0 - f_intra)   # (batch_size,)
```

The volume fractions for each compartment are derived from the physical
fractions:

```python
# S = f_iso * S_ball + (1 - f_iso) * [f_ic * S_stick + (1 - f_ic) * S_zepp]
pv_ball  = f_iso
pv_stick = (1.0 - f_iso) * f_intra
pv_zepp  = (1.0 - f_iso) * (1.0 - f_intra)
```

These are passed directly to the multi-compartment model as
`partial_volume_0`, `partial_volume_1`, and `partial_volume_2`.

---

## 8. Simulating Training Data

The `get_batch` function brings together the prior, forward model, tortuosity
constraint, and noise model to generate training pairs
$(\boldsymbol{\theta}, \mathbf{x})$ on the fly:

```python
import functools

@functools.partial(jax.jit, static_argnames=('batch_size',))
def get_batch(key, batch_size=128):
    """Generate (noisy_signal, parameters) training pairs."""
    k_fi, k_fiso, k_kap, k_mu, k_noise = jax.random.split(key, 5)

    # --- Sample priors ---
    f_intra = jax.random.beta(k_fi, 5.0, 5.0, shape=(batch_size,))
    f_iso   = jax.random.beta(k_fiso, 0.5, 5.0, shape=(batch_size,))

    min_lk, max_lk = jnp.log(0.1), jnp.log(32.0)
    kappa = jnp.exp(jax.random.uniform(k_kap, (batch_size,),
                                        minval=min_lk, maxval=max_lk))

    # Orientation -- uniform on S^2
    z = jax.random.normal(k_mu, (batch_size, 3))
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    theta = jnp.arccos(jnp.clip(z[:, 2], -1.0, 1.0))
    phi   = jnp.arctan2(z[:, 1], z[:, 0])
    mu    = jnp.stack([theta, phi], axis=1)

    # --- Derived quantities ---
    d_par  = 1.7e-9
    d_iso  = 3.0e-9
    d_perp = d_par * (1.0 - f_intra)            # tortuosity

    pv_stick = (1.0 - f_iso) * f_intra
    pv_zepp  = (1.0 - f_iso) * (1.0 - f_intra)
    pv_ball  = f_iso

    # --- Forward simulation ---
    params_dict = {
        'mu': mu, 'kappa': kappa,
        'lambda_par': jnp.full((batch_size,), d_par),
        'mu_2': mu, 'kappa_2': kappa,               # shared orientation
        'lambda_par_2': jnp.full((batch_size,), d_par),
        'lambda_perp': d_perp,                       # tortuosity-constrained
        'lambda_iso': jnp.full((batch_size,), d_iso),
        'partial_volume_0': pv_stick,
        'partial_volume_1': pv_zepp,
        'partial_volume_2': pv_ball,
    }
    signals = model(params_dict, acquisition)        # (batch_size, N_MEAS)

    # --- Rician noise at SNR ~ 30 ---
    sigma = 1.0 / 30.0
    k1, k2 = jax.random.split(k_noise)
    n1 = jax.random.normal(k1, signals.shape) * sigma
    n2 = jax.random.normal(k2, signals.shape) * sigma
    signals_noisy = jnp.sqrt((signals + n1)**2 + n2**2)

    # --- Target parameters ---
    targets = jnp.stack([f_intra, f_iso, kappa], axis=-1)
    return signals_noisy, targets

x_demo, y_demo = get_batch(jax.random.PRNGKey(1), batch_size=4)
print(x_demo.shape, y_demo.shape)   # (4, 66) (4, 3)
```

```{important}
The noise model is **Rician**, not Gaussian. In magnitude MRI the noise
envelope follows a Rice distribution: $S_\text{noisy} = |S + n_1 + i\,n_2|$
where $n_1, n_2 \sim \mathcal{N}(0, \sigma^2)$. This is critical for
accurate SBI training, especially at low SNR where the Rician bias
(rectified noise floor) is significant.
```

---

## 9. Training SBI for NODDI Parameters

With the simulator in place, we train a Mixture Density Network (MDN) to
approximate the posterior $p(f_\text{ic}, f_\text{iso}, \kappa \mid \mathbf{x})$.

### 9.1 Network Architecture

The MDN maps the 66-dimensional signal vector to the parameters of a Gaussian
mixture with $K$ components. Each component predicts a 3D mean and diagonal
covariance for the three NODDI parameters:

```python
import equinox as eqx
import optax
from dmipy_jax.inference.mdn import MixtureDensityNetwork, mdn_loss

key = jax.random.PRNGKey(123)
k_net, k_train = jax.random.split(key)

network = MixtureDensityNetwork(
    in_features=N_MEAS,       # 66 measurements
    out_features=3,            # f_intra, f_iso, kappa
    num_components=4,          # Gaussian mixture components
    width_size=64,             # production: 128
    depth=2,                   # production: 4
    key=k_net,
    activation=jax.nn.gelu,
)
```

Four mixture components are sufficient for NODDI because the posterior is
typically unimodal (the model is identifiable with two b-shells). More
components can be used when orientation ambiguities or low SNR create
multimodality.

### 9.2 Training Loop

Training proceeds by sampling fresh batches from the simulator at each
iteration -- an "online" training strategy that avoids storing a large
dataset:

```python
optimizer = optax.adam(5e-4)
opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

@eqx.filter_jit
def step(net, opt_state, x, y):
    def batch_loss(model, signals, params):
        per_sample = jax.vmap(mdn_loss, in_axes=(None, 0, 0))(model, signals, params)
        return jnp.mean(per_sample)

    loss, grads = eqx.filter_value_and_grad(batch_loss)(net, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state, net)
    new_net = eqx.apply_updates(net, updates)
    return new_net, new_opt_state, loss

# Train for 300 iterations with batch size 128 (production: 500+ at 256)
N_ITERS = 300
k_iter = k_train
losses = []
for i in range(N_ITERS):
    k_iter, k_batch = jax.random.split(k_iter)
    x_batch, y_batch = get_batch(k_batch, batch_size=128)
    network, opt_state, loss = step(network, opt_state, x_batch, y_batch)
    losses.append(float(loss))

    if i % 100 == 0:
        print(f"Iteration {i:4d} | Loss: {loss:.4f}")

assert losses[-1] < losses[0]
```

The loss is the negative log-likelihood of the true parameters under the
predicted Gaussian mixture (see {func}`~dmipy_jax.inference.mdn.mdn_loss`).
A few hundred iterations are enough to see the loss fall steadily; a
production run continues until it plateaus.

---

## 10. Posterior Sampling and Visualisation

After training, the MDN yields a full posterior distribution for any observed
signal vector. Sampling is straightforward:

```python
import numpy as np
from dmipy_jax.inference.mdn import sample_posterior

# Generate a test observation
k_test = jax.random.PRNGKey(999)
x_test, y_test = get_batch(k_test, 5)

# Pick a single voxel
signal = x_test[0]
true_params = y_test[0]  # (f_intra, f_iso, kappa)

# Draw 2000 posterior samples
samples = sample_posterior(network, signal, k_test, n_samples=2000)
assert samples.shape == (2000, 3)
print("posterior mean:", np.round(np.asarray(samples.mean(0)), 3), " truth:", np.round(np.asarray(true_params), 3))
```

A corner plot reveals the joint and marginal posteriors:

```python
import os, tempfile
import matplotlib
matplotlib.use("Agg")
import corner

labels = [r"$f_\mathrm{ic}$", r"$f_\mathrm{iso}$", r"$\kappa$"]

fig = corner.corner(
    np.array(samples),
    labels=labels,
    truths=np.array(true_params),
    truth_color='red',
    show_titles=True,
)
fig.savefig(os.path.join(tempfile.mkdtemp(), "noddi_posterior.png"), dpi=150)
```

---

## 11. Interpreting Results

### Neurite Density Index (NDI)

The marginal posterior over $f_\text{ic}$ gives the neurite density index.
The posterior mean is a natural point estimate, and the 90% highest density
interval (HDI) quantifies uncertainty. In healthy white matter expect
$f_\text{ic} \approx 0.4$--$0.7$; demyelination or oedema reduce it.

### Orientation Dispersion Index (ODI)

The ODI is a monotonic transform of $\kappa$:

$$
\text{ODI} = \frac{2}{\pi} \arctan\!\left(\frac{1}{\kappa}\right)
$$

| ODI | $\kappa$ | Interpretation |
|-----|----------|----------------|
| $\approx 1$ | $\approx 0$ | Isotropic dispersion |
| $0.2$--$0.3$ | $2$--$5$ | Moderate (cortex, crossings) |
| $< 0.1$ | $> 16$ | Highly coherent fibres |

The SBI posterior over $\kappa$ can be transformed to an ODI posterior by
applying the formula to each sample -- a convenient advantage of having
samples rather than a point estimate.

### CSF Fraction ($f_\text{iso}$)

Values of $f_\text{iso} > 0.1$ typically indicate partial volume with
cerebrospinal fluid. The posterior often shows a sharp peak near zero in
white matter, which is well captured by the Beta(0.5, 5) prior.

---

## 12. Practical Guidance

**When to use more mixture components.** If the posterior is multimodal
(e.g. in crossing-fibre voxels or low-SNR regimes), increase
`num_components` from 4 to 8. Alternatively, switch to a normalising flow
backend (see {doc}`normalizing_flows`) for maximum flexibility.

**Prior sensitivity.** The SBI posterior is influenced by the training prior.
If your population differs from the Beta(5, 5) / Beta(0.5, 5) assumptions
(e.g. neonatal tissue), adjust the prior parameters and retrain.

**Multi-shell design.** Adding a third shell (e.g. $b = 300$) improves
estimation of $f_\text{iso}$. The acquisition design functions in
SBI4DWI make it straightforward to experiment with different protocols.

**Scaling to whole-brain.** Once trained, the MDN can be deployed to
whole-brain NIfTI volumes via
{class}`~dmipy_jax.pipeline.deploy.SBIPredictor`. A 66-measurement volume
with 100k voxels runs in under 5 seconds on GPU.

---

## References

- Zhang, H. et al. "NODDI: Practical in vivo neurite orientation dispersion
  and density imaging of the human brain." *NeuroImage* 61(4), 2012.
- Szafer, A. et al. "Theoretical model for water diffusion in tissues."
  *Magnetic Resonance in Medicine* 33(5), 1995.
- Papamakarios, G. & Murray, I. "Fast epsilon-free inference of simulation
  models with Bayesian conditional density estimation." NeurIPS, 2016.

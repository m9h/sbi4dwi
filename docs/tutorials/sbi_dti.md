# SBI for DTI: From Tensor Model to FA/MD Maps

This tutorial demonstrates how to use **simulation-based inference (SBI)**
to estimate diffusion tensor imaging (DTI) parameters -- specifically
Fractional Anisotropy (FA) and Mean Diffusivity (MD) -- from noisy
diffusion-weighted signals. Rather than fitting each voxel independently
with a classical optimiser, we train a neural density estimator on
simulated data and then amortise inference across all voxels in a single
forward pass.

The code patterns here are adapted from
`dmipy_jax/examples/sbi/train_dti.py`. Network and training sizes have been
shrunk so that the whole page runs in well under a minute on a CPU; the
production settings are noted where they differ.

---

## 1. Why SBI beats voxelwise fitting for DTI

Classical DTI fitting (e.g., least-squares tensor estimation) processes
each voxel independently. This has two major limitations:

1. **Speed**: fitting a whole brain (~200,000 white-matter voxels) requires
   200,000 independent optimisations.
2. **Noise sensitivity**: at low SNR the per-voxel likelihood surface
   becomes flat or multimodal, and point estimates can be badly biased.

SBI inverts this workflow. We:

1. **Simulate** a large training set of (parameter, signal) pairs from the
   forward model with realistic noise.
2. **Train** a neural density estimator (here, a Mixture Density Network)
   to predict the posterior $p(\theta \mid x)$ given a signal vector $x$.
3. **Deploy** the trained network: a single forward pass per voxel yields
   the full posterior, not just a point estimate.

Because the network is trained *once* and amortised over all voxels,
inference on a whole brain takes seconds on a GPU.

---

## 2. The DTI forward model

The diffusion tensor $\mathbf{D}$ is a 3x3 positive-definite symmetric
matrix characterised by three eigenvalues $(\lambda_1, \lambda_2,
\lambda_3)$ and an orientation given by three Euler angles
$(\alpha, \beta, \gamma)$ in the Z-Y-Z convention. The signal attenuation
for a single measurement with gradient direction $\mathbf{g}$ and b-value
$b$ is:

$$
S = \exp\!\bigl(-b \, \mathbf{g}^T \mathbf{D} \, \mathbf{g}\bigr)
  = \exp\!\Bigl(-b \sum_{k=1}^{3} \lambda_k \, (\mathbf{g} \cdot \mathbf{e}_k)^2 \Bigr)
$$

where $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ are the eigenvectors
obtained from the Euler-angle rotation matrix.

In SBI4DWI, this is implemented by the `Tensor` class in
`dmipy_jax.signal_models.gaussian_models`:

```python
from dmipy_jax.signal_models.gaussian_models import Tensor

# Instantiate with specific parameters
tensor = Tensor(
    lambda_1=2.0e-9,   # m^2/s
    lambda_2=0.5e-9,   # m^2/s
    lambda_3=0.5e-9,   # m^2/s
    alpha=0.0,          # radians
    beta=0.0,           # radians
    gamma=0.0           # radians
)
```

The biologically plausible range for eigenvalues is 0.1--3.0 um^2/ms, which
in SI units is **0.1e-9 to 3.0e-9 m^2/s**.

---

## 3. Setting up the acquisition

For this tutorial we use a single-shell acquisition with 32 gradient
directions at b = 1000 s/mm^2. In SI units that is **1e9 s/m^2**.

We generate uniformly distributed directions on the sphere:

```python
import jax
import jax.numpy as jnp

def sample_on_sphere(key, shape):
    """Sample uniform random unit vectors on the sphere."""
    z = jax.random.normal(key, shape + (3,))
    return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

# Fixed acquisition protocol
N_DIRS = 32
B_VALUE = 1e9  # 1000 s/mm^2 in SI

key_acq = jax.random.PRNGKey(42)
BVECS = sample_on_sphere(key_acq, (N_DIRS,))   # (32, 3)
BVALS = jnp.full((N_DIRS,), B_VALUE)            # (32,)
```

```{important}
The gradient directions are **fixed** for the entire training and inference
process. The neural network learns the mapping from signal to parameters
*for this specific protocol*. If you change the acquisition (e.g., different
number of directions or b-value), you must retrain.
```

---

## 4. Synthetic data generation with Rician noise

Each training sample is produced by:

1. Sampling random eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$
   from the biologically plausible range.
2. Sampling random Euler angles for orientation.
3. Computing the noiseless signal via the Tensor forward model.
4. Corrupting with Rician noise at a chosen SNR.

### Rician noise model

MRI magnitude images follow a Rician distribution. For a true signal $S$
and noise standard deviation $\sigma$:

$$
S_{\text{noisy}} = \sqrt{(S + n_1)^2 + n_2^2}, \quad
n_1, n_2 \sim \mathcal{N}(0, \sigma^2)
$$

With $S_{b=0} \approx 1.0$ (normalised), setting $\sigma = 1/\text{SNR}$
gives the desired noise level.

### Generating a single sample

```python
def forward_single(l1, l2, l3, alpha, beta, gamma):
    """Compute DTI signal for one set of parameters."""
    model = Tensor(
        lambda_1=l1, lambda_2=l2, lambda_3=l3,
        alpha=alpha, beta=beta, gamma=gamma
    )
    return model(BVALS, BVECS)

# Example: a single highly anisotropic tensor
signal = forward_single(
    l1=2.5e-9, l2=0.3e-9, l3=0.3e-9,
    alpha=0.0, beta=0.0, gamma=0.0
)
print(f"Signal shape: {signal.shape}")  # (32,)
```

---

## 5. Batch data generation with `jax.vmap`

To train a neural network we need thousands of samples. Rather than looping
in Python, we use `jax.vmap` to vectorise the forward model and
`jax.jit` to compile the entire batch-generation function:

```python
import functools

@functools.partial(jax.jit, static_argnames=['batch_size'])
def get_batch(key, batch_size=128):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # 1. Sample eigenvalues (ordered: l1 >= l2 >= l3)
    l_min, l_max = 0.1e-9, 3.0e-9
    l1 = jax.random.uniform(k1, (batch_size,), minval=l_min, maxval=l_max)
    l2 = jax.random.uniform(k2, (batch_size,), minval=l_min, maxval=l1)
    l3 = jax.random.uniform(k3, (batch_size,), minval=l_min, maxval=l2)

    # 2. Sample orientations (Euler angles, Z-Y-Z convention)
    alpha = jax.random.uniform(k4, (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    beta  = jax.random.uniform(k5, (batch_size,), minval=0.0,     maxval=jnp.pi)
    gamma = jax.random.uniform(k1, (batch_size,), minval=-jnp.pi, maxval=jnp.pi)

    # 3. Vectorised forward model
    signals = jax.vmap(forward_single)(l1, l2, l3, alpha, beta, gamma)

    # 4. Add Rician noise (SNR = 30)
    sigma = 1.0 / 30.0
    k_n1, k_n2 = jax.random.split(k2, 2)
    n1 = jax.random.normal(k_n1, signals.shape) * sigma
    n2 = jax.random.normal(k_n2, signals.shape) * sigma
    signals_noisy = jnp.sqrt((signals + n1)**2 + n2**2)

    # 5. Compute target metrics (FA, MD)
    fa, md = compute_fa_md(l1, l2, l3)

    # Scale MD by 1e9 so both targets are O(1) for training stability
    targets = jnp.stack([fa, md * 1e9], axis=-1)

    return signals_noisy, targets
```

A single call to `get_batch(key, batch_size=256)` produces 256 noisy
signal vectors and their corresponding (FA, MD) targets in one compiled
XLA kernel -- no Python-level loops involved. (`get_batch` refers to
`compute_fa_md`, defined in the next section; because the function body is
only traced when it is first *called*, the definition order does not
matter.)

---

## 6. Computing derived metrics: FA and MD

Fractional Anisotropy and Mean Diffusivity are the two most widely reported
DTI scalar maps. They are computed directly from the tensor eigenvalues:

**Mean Diffusivity (MD):**
$$
\text{MD} = \frac{\lambda_1 + \lambda_2 + \lambda_3}{3}
$$

**Fractional Anisotropy (FA):**
$$
\text{FA} = \sqrt{\frac{3}{2}} \cdot
\frac{\sqrt{(\lambda_1 - \text{MD})^2 + (\lambda_2 - \text{MD})^2 + (\lambda_3 - \text{MD})^2}}
     {\sqrt{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}}
$$

FA ranges from 0 (isotropic, e.g. CSF) to 1 (perfectly anisotropic, e.g.
a single coherent fibre bundle). In healthy white matter, FA is typically
0.4--0.8.

```python
def compute_fa_md(lambda_1, lambda_2, lambda_3):
    """Compute FA and MD from tensor eigenvalues."""
    md = (lambda_1 + lambda_2 + lambda_3) / 3.0

    num = (lambda_1 - md)**2 + (lambda_2 - md)**2 + (lambda_3 - md)**2
    denom = lambda_1**2 + lambda_2**2 + lambda_3**2

    # Guard against division by zero for isotropic tensors.  Eigenvalues
    # are ~1e-9 m^2/s, so denom ~ 1e-18: the guard must be far smaller than
    # that (an absolute 1e-9 here would silently drive FA to zero).
    fa_sq = 1.5 * num / jnp.maximum(denom, 1e-30)
    fa = jnp.sqrt(jnp.clip(fa_sq, 0.0, 1.0))

    return fa, md

# Sanity check the batch generator
x_demo, y_demo = get_batch(jax.random.PRNGKey(1), batch_size=256)
print(x_demo.shape, y_demo.shape)   # (256, 32) (256, 2)
assert jnp.all((y_demo[:, 0] >= 0) & (y_demo[:, 0] <= 1))
assert float(y_demo[:, 0].std()) > 0.1    # FA actually varies across the prior
```

---

## 7. Training a Mixture Density Network

The inference network is a **Mixture Density Network (MDN)** -- an MLP
whose output layer parameterises a Gaussian mixture model (GMM). This
allows the network to represent multimodal posteriors, which arise when
different tissue configurations produce similar signals.

### Network architecture

The MDN outputs three sets of parameters for $K$ Gaussian components:

* **Mixing logits** $\pi_k$ -- converted to weights via softmax
* **Means** $\mu_k \in \mathbb{R}^D$ -- component centres
* **Log-standard-deviations** $\log \sigma_k$ -- component widths

For DTI with FA and MD as targets, $D = 2$.

```python
import equinox as eqx
import optax

class MixtureDensityNetwork(eqx.Module):
    mlp: eqx.nn.MLP
    n_components: int
    n_outputs: int

    def __init__(self, key, in_size, out_size, n_components=8,
                 width=128, depth=3):
        self.n_components = n_components
        self.n_outputs = out_size
        # Output: K logits + K*D means + K*D log_sigmas
        total_out = n_components * (1 + 2 * out_size)
        self.mlp = eqx.nn.MLP(
            in_size=in_size, out_size=total_out,
            width_size=width, depth=depth,
            activation=jax.nn.gelu, key=key
        )

    def __call__(self, x):
        raw = self.mlp(x)
        nc, no = self.n_components, self.n_outputs
        logits     = raw[:nc]
        means      = raw[nc : nc + nc*no].reshape(nc, no)
        log_sigmas = raw[nc + nc*no :].reshape(nc, no)
        sigmas = jnp.exp(log_sigmas)
        return logits, means, sigmas
```

### Loss function

The MDN is trained to maximise the log-likelihood of the true parameters
under the predicted mixture:

$$
\mathcal{L} = -\log \sum_{k=1}^{K} \pi_k \,
\mathcal{N}(\theta_{\text{true}} \mid \mu_k, \text{diag}(\sigma_k^2))
$$

We compute this in log-space using `logsumexp` for numerical stability:

```python
def mdn_loss_fn(model, x, y):
    """Negative log-likelihood of y under the predicted mixture."""
    logits, means, sigmas = model(x)

    # Per-component log-probabilities (diagonal covariance)
    z = (y - means) / sigmas
    log_prob_comps = (
        -0.5 * jnp.sum(z**2, axis=-1)
        - jnp.sum(jnp.log(sigmas), axis=-1)
        - 0.5 * means.shape[1] * jnp.log(2 * jnp.pi)
    )

    # Mixture log-likelihood via logsumexp
    log_pis = jax.nn.log_softmax(logits)
    return -jax.scipy.special.logsumexp(log_pis + log_prob_comps)

def batch_loss(model, x_batch, y_batch):
    """Average NLL over a batch."""
    per_sample = jax.vmap(mdn_loss_fn, in_axes=(None, 0, 0))(
        model, x_batch, y_batch
    )
    return jnp.mean(per_sample)
```

### Training loop

For a production run use `n_components=8, width=128, depth=3` and
5000 iterations at batch size 256; here we use a small network and 1500
iterations so the tutorial completes quickly on CPU.

```python
N_ITERS = 1500
BATCH = 128

def train():
    key = jax.random.PRNGKey(0)
    key_net, key_data = jax.random.split(key)

    # Initialise network: 32 signal inputs -> 2 outputs (FA, MD)
    model = MixtureDensityNetwork(
        key_net, in_size=N_DIRS, out_size=2,
        n_components=4, width=64, depth=2
    )

    # Adam optimiser
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Train with on-the-fly data generation
    key_iter = key_data
    for i in range(N_ITERS):
        key_iter, k = jax.random.split(key_iter)
        x_batch, y_batch = get_batch(k, batch_size=BATCH)
        model, opt_state, loss = step(model, opt_state, x_batch, y_batch)

        if (i + 1) % 500 == 0:
            print(f"Iter {i+1}/{N_ITERS}  Loss: {loss:.4f}")

    return model

model = train()
```

```{tip}
Because `get_batch` generates fresh random data every iteration, the
network never sees the same sample twice. This eliminates overfitting and
means we do not need a separate validation set -- the training loss *is*
the generalisation loss.
```

---

## 8. Validating parameter recovery

After training, we evaluate the network on held-out synthetic data to
verify that it recovers FA and MD accurately.

### Prediction via expected value

For each test signal, the MDN outputs a full mixture posterior. The
simplest point estimate is the expected value (mean of the mixture):

$$
\hat{\theta} = \sum_{k=1}^{K} \pi_k \, \mu_k
$$

```python
@eqx.filter_jit
def predict_mean(model, x):
    """Compute the expected value of the MDN posterior."""
    logits, means, _ = model(x)
    weights = jax.nn.softmax(logits)[:, None]  # (K, 1)
    return jnp.sum(weights * means, axis=0)    # (D,)

# Generate a test batch
key_test = jax.random.PRNGKey(999)
x_test, y_test = get_batch(key_test, batch_size=1000)

# Vectorised prediction
preds = jax.vmap(predict_mean, in_axes=(None, 0))(model, x_test)

# Quantify recovery: correlation between true and predicted FA / MD
corr_fa = float(jnp.corrcoef(y_test[:, 0], preds[:, 0])[0, 1])
corr_md = float(jnp.corrcoef(y_test[:, 1], preds[:, 1])[0, 1])
print(f"Correlation  FA: {corr_fa:.3f}   MD: {corr_md:.3f}")
assert corr_md > 0.9     # MD is easy: it is set by the mean attenuation
assert corr_fa > 0.4     # FA needs the directional pattern; improves with training
```

### Scatter plot validation

A good model should produce points clustered tightly along the identity
line (predicted = true):

```python
import os, tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# FA
axes[0].scatter(y_test[:, 0], preds[:, 0], alpha=0.3, s=5)
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel("True FA")
axes[0].set_ylabel("Predicted FA")
axes[0].set_title("Fractional Anisotropy")
axes[0].grid(True)

# MD (scaled by 1e9 during training, so axis is in um^2/ms)
axes[1].scatter(y_test[:, 1], preds[:, 1], alpha=0.3, s=5)
axes[1].plot([0, 3], [0, 3], 'k--', lw=1)
axes[1].set_xlabel("True MD (um^2/ms)")
axes[1].set_ylabel("Predicted MD (um^2/ms)")
axes[1].set_title("Mean Diffusivity")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(tempfile.mkdtemp(), "dti_sbi_validation.png"))
plt.close(fig)
```

With the short training run above MD is recovered almost perfectly
(correlation ~0.99 -- it is set by the mean attenuation over directions),
while FA reaches a correlation of roughly 0.6: it depends on the angular
pattern across the 32 directions, which the small network is still learning.
At SNR = 30 with the production settings (5000 iterations, width 128) you
should see tight correlation for both FA and MD, with slightly more scatter
at extreme values (very high FA near 1.0, or very low MD).

---

## Summary

| Stage | What happens | Key function |
|-------|-------------|--------------|
| Forward model | Tensor eigenvalues + Euler angles -> signal | `Tensor(...)()` |
| Batch generation | vmap + Rician noise -> (signals, targets) | `get_batch()` |
| Derived metrics | Eigenvalues -> FA, MD | `compute_fa_md()` |
| Training | MDN learns $p(\text{FA}, \text{MD} \mid \text{signal})$ | `batch_loss()` + `optax.adam` |
| Inference | Single forward pass per voxel | `predict_mean()` |

### When to use SBI vs classical fitting

| Criterion | Classical (`model.fit`) | SBI (MDN / Flow) |
|-----------|------------------------|-------------------|
| Speed (whole brain) | Minutes--hours | Seconds |
| Uncertainty estimates | No (point estimate) | Yes (full posterior) |
| Low-SNR robustness | Poor | Good (learned prior) |
| Setup cost | None | Training time |
| Acquisition change | No retraining | Must retrain |

For production DTI pipelines where speed and uncertainty quantification
matter, SBI is the recommended approach. For quick exploratory fits on
small ROIs, the classical pipeline from the
{doc}`model_composition` tutorial may be more convenient.

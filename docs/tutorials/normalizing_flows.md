# Flexible Posteriors with Normalizing Flows

Mixture Density Networks (MDNs) are the default posterior estimator in
SBI4DWI, but they assume that the posterior can be decomposed into a small
number of Gaussian blobs. When the true posterior is skewed, heavy-tailed,
banana-shaped, or exhibits complex correlations between parameters, a
**normalizing flow** provides a more expressive alternative.

This tutorial introduces the normalizing flow architecture implemented in
SBI4DWI, walks through a toy training example, and provides practical
guidance on when to reach for flows instead of MDNs.

```{contents}
:depth: 3
:local:
```

## Prerequisites

- SBI4DWI installed (`uv sync` from the repository root)
- Familiarity with the SBI training pipeline (see {doc}`training_to_deployment`)
- Basic probability and change-of-variables concepts

---

## 1. Why Flows Instead of MDNs?

A Mixture Density Network represents the posterior as a mixture of $K$
Gaussians with diagonal covariance. This works well when the posterior is
roughly elliptical and unimodal (or has a few well-separated modes), but it
struggles with:

- **Curved degeneracies** -- e.g. the banana-shaped posterior that arises when
  two parameters are correlated through a nonlinear forward model.
- **Heavy tails** -- Gaussian tails decay as $\exp(-x^2)$; tissue parameters
  near physical boundaries (fraction $\to 0$ or $\to 1$) often have
  asymmetric, slower-decaying tails.
- **High-dimensional multimodality** -- representing $M$ modes in $D$
  dimensions requires $\mathcal{O}(M)$ components, each with $D^2/2$
  covariance parameters. Flows scale more gracefully.

A **normalizing flow** is a learned invertible transformation
$f_\phi: \mathbb{R}^D \to \mathbb{R}^D$ that maps a simple base distribution
(standard Gaussian) to an arbitrarily complex target distribution. The key
insight is the **change-of-variables formula**:

$$
p_\phi(\boldsymbol{\theta} \mid \mathbf{x})
  = p_\text{base}\!\bigl(f_\phi^{-1}(\boldsymbol{\theta};\, \mathbf{x})\bigr)
    \;\left|\det \frac{\partial f_\phi^{-1}}{\partial \boldsymbol{\theta}}\right|
$$

where $\mathbf{x}$ is the conditioning context (the observed signal). By
stacking multiple invertible layers, the flow learns to warp the Gaussian base
into an accurate approximation of the posterior.

---

## 2. Rational Quadratic Spline (RQS) Transforms

The expressiveness of a normalizing flow depends on the choice of bijection.
SBI4DWI uses the **Rational Quadratic Spline** (RQS) transform from
Durkan et al. (2019), implemented in
{class}`~dmipy_jax.inference.flows.RationalQuadraticSpline`.

### 2.1 How RQS Works

An RQS transform is a monotonic piecewise-rational function defined by $K$
bins. Within each bin, the mapping from input $x$ to output $y$ is a
rational quadratic function determined by three sets of learnable
parameters:

- **Widths** $w_k$ -- the extent of each bin along the input axis (softmax-normalised)
- **Heights** $h_k$ -- the extent along the output axis (softmax-normalised)
- **Derivatives** $d_k$ -- the slope at each knot boundary (softplus-positive)

These are chosen such that the resulting function is:

1. **Monotonic** -- guaranteed by construction (positive widths, heights, and
   derivatives).
2. **Invertible** -- the inverse is computed analytically by solving a
   quadratic equation.
3. **Smooth** -- $C^1$ continuity at every knot.

The log-determinant of the Jacobian has a closed-form expression, making
density evaluation efficient.

### 2.2 Verifying Invertibility

The following test (adapted from `dmipy_jax/tests/test_flows.py`) confirms
that the forward and inverse transforms are consistent:

```python
import jax
import jax.numpy as jnp
from dmipy_jax.inference.flows import RationalQuadraticSpline

key = jax.random.PRNGKey(42)
K = 4   # number of bins
D = 1   # single dimension

# Random spline parameters: widths, heights, derivatives
params = jax.random.normal(key, (D, 3 * K + 1))

rqs = RationalQuadraticSpline(num_bins=K)

# Forward pass
x = jnp.array([1.5])
z, log_det_fwd = rqs(x, params, inverse=False)

# Inverse pass
x_reconstructed, log_det_inv = rqs(z, params, inverse=True)

# Verify round-trip
assert jnp.allclose(x, x_reconstructed, atol=1e-4)

# Forward and inverse log-dets must cancel
assert jnp.allclose(log_det_fwd + log_det_inv, 0.0, atol=1e-4)
```

The `params` tensor has shape `(D, 3*K + 1)` -- for each dimension, $K$
unconstrained widths, $K$ unconstrained heights, and $K + 1$ unconstrained
derivatives (one per knot, including both endpoints). Inside the RQS, these
are split and constrained:

```python
# Internal parameter layout per dimension:
c_w = params[..., :K]          # widths  -> softmax -> positive, sum to 1
c_h = params[..., K:2*K]       # heights -> softmax -> positive, sum to 1
c_d = params[..., 2*K:]        # derivatives -> softplus -> positive
```

---

## 3. Building a FlowNetwork with Conditional Context

The {class}`~dmipy_jax.inference.flows.FlowNetwork` stacks multiple
{class}`~dmipy_jax.inference.flows.CouplingLayer` modules, each containing:

1. A **conditioner MLP** that takes the unchanged dimensions plus the
   conditioning context (signal vector) and predicts RQS parameters for the
   remaining dimensions.
2. An **RQS bijector** that transforms those dimensions.

Between layers, the dimension ordering is reversed so that every dimension
gets transformed by at least one layer.

```python
from dmipy_jax.inference.flows import FlowNetwork

key = jax.random.PRNGKey(0)

flow = FlowNetwork(
    key,
    n_layers=3,     # number of coupling layers
    n_dim=2,        # dimensionality of theta (parameter space)
    n_context=5,    # dimensionality of x (signal / conditioning)
)
```

### Architecture Summary

```
Input theta (D,) ──┐
                    │  reverse dims
                    ▼
              CouplingLayer 1
                │  split: theta[:D//2] unchanged
                │         theta[D//2:] transformed by RQS
                │  conditioner MLP: [theta[:D//2], context] -> RQS params
                    │  reverse dims
                    ▼
              CouplingLayer 2
                    │  ...
                    ▼
              CouplingLayer L
                    │
                    ▼
              z ~ N(0, I)   (base distribution)
```

The `FlowNetwork` exposes two core methods:

- **`log_prob(theta, context)`** -- evaluates $\log p(\theta \mid \text{context})$
  by running the forward (normalising) direction and summing base log-prob
  with log-det-Jacobian contributions from each layer.
- **`sample(key, context, n_samples)`** -- generates posterior samples by
  drawing $z \sim \mathcal{N}(0, I)$ and running the inverse (generative)
  direction through the layers in reverse order.

### Shape Verification

```python
theta = jnp.ones((2,))
context = jnp.ones((5,))

# Log probability -- scalar output
lp = flow.log_prob(theta, context)
assert lp.shape == ()

# Sampling -- (n_samples, n_dim)
samples = flow.sample(key, context, n_samples=10)
assert samples.shape == (10, 2)
```

---

## 4. Training with Maximum Likelihood

Training a conditional normalizing flow amounts to **maximising the
log-probability** of the true parameters under the flow, conditioned on
simulated signals. Equivalently, we minimise the negative log-likelihood:

$$
\mathcal{L}(\phi) = -\mathbb{E}_{(\boldsymbol{\theta}, \mathbf{x}) \sim p_\text{sim}}
  \bigl[\log p_\phi(\boldsymbol{\theta} \mid \mathbf{x})\bigr]
$$

This is identical in spirit to the MDN loss but the density is parameterised
by the flow rather than a Gaussian mixture.

```python
import optax
import equinox as eqx

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))

@eqx.filter_jit
def train_step(flow, opt_state, batch_theta, batch_context):
    def loss_fn(model):
        log_probs = jax.vmap(model.log_prob)(batch_theta, batch_context)
        return -jnp.mean(log_probs)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
    updates, new_opt_state = optimizer.update(grads, opt_state, flow)
    new_flow = eqx.apply_updates(flow, updates)
    return new_flow, new_opt_state, loss
```

Each call to `train_step` performs one gradient update. The `jax.vmap` over
`log_prob` vectorises density evaluation across the batch -- each sample
independently passes through the coupling layers, and gradients flow back
through the RQS inverse and the conditioner MLPs.

---

## 5. Sampling from the Trained Flow

Once trained, generating posterior samples is a single call:

```python
test_context = jnp.array([1.0, 0.5, -0.3, 0.8, 0.2])  # observed signal
samples = flow.sample(key, test_context, n_samples=5000)
# samples.shape == (5000, 2)
```

Under the hood, the flow:

1. Draws `n_samples` vectors from $\mathcal{N}(0, I_D)$.
2. Passes each through the coupling layers **in reverse order**, applying the
   **inverse** RQS transform at each layer.
3. Returns the resulting parameter vectors in the original $\theta$ space.

Because each layer is analytically invertible, sampling is as fast as density
evaluation -- no iterative procedures or MCMC chains are needed.

---

## 6. Toy Example: Learning a Conditional Gaussian

To build intuition, let us train a flow on a simple conditional distribution
where the ground truth is known. Consider:

$$
p(\mathbf{x} \mid c) = \mathcal{N}\!\bigl(\mathbf{x};\; [c, c],\; 0.1\,\mathbf{I}\bigr)
$$

The flow must learn to shift its base distribution by the context value $c$.

```python
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from dmipy_jax.inference.flows import FlowNetwork

key = jax.random.PRNGKey(101)

# 2D flow conditioned on 1D context
flow = FlowNetwork(key, n_layers=3, n_dim=2, n_context=1)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))

@eqx.filter_jit
def make_step(flow, opt_state, batch_x, batch_c):
    def loss_fn(model):
        lp = jax.vmap(model.log_prob)(batch_x, batch_c)
        return -jnp.mean(lp)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
    updates, opt_state = optimizer.update(grads, opt_state, flow)
    flow = eqx.apply_updates(flow, updates)
    return flow, opt_state, loss

# Training loop
batch_size = 128
for step in range(500):
    key, subkey = jax.random.split(key)

    # Context: c ~ Uniform(-2, 2)
    c = jax.random.uniform(subkey, (batch_size, 1), minval=-2, maxval=2)

    # Target: x ~ N([c, c], 0.1 I)
    noise = jax.random.normal(subkey, (batch_size, 2)) * 0.1
    x = c + noise   # broadcasts (B, 1) + (B, 2) via JAX broadcasting

    flow, opt_state, loss = make_step(flow, opt_state, x, c)

    if step % 100 == 0:
        print(f"Step {step:4d} | Loss: {loss:.4f}")
```

### Verification

After training, we verify that samples from the flow match the target
distribution:

```python
# Condition on c = 1.0
test_c = jnp.array([1.0])
samples = flow.sample(key, test_c, n_samples=100)

sample_mean = jnp.mean(samples, axis=0)
print(f"Sample mean (target [1.0, 1.0]): {sample_mean}")

# Mean should be within 0.2 of the true value
assert jnp.allclose(sample_mean, 1.0, atol=0.2)
```

The loss should converge to a negative value (since we are maximising
log-likelihood, and the true distribution has low entropy with $\sigma = 0.1$).
For a 2D Gaussian with $\sigma = 0.1$, the expected negative log-likelihood
per sample is approximately $-2 \times (\log(0.1\sqrt{2\pi}) + 0.5) \approx -3.6$.

---

## 7. When to Use Flows vs MDN

The choice between MDN and normalizing flow depends on the complexity of
the posterior and the computational budget.

### Use an MDN when:

- The posterior is roughly **unimodal and elliptical** (most common in
  well-determined microstructure models with good SNR).
- You need **fast training** -- MDNs converge in fewer iterations because
  the Gaussian mixture parameterisation is a strong inductive bias.
- The parameter dimension is **low** ($D \leq 5$) and the number of
  modes is small and known.
- You want **simple posterior summaries** -- mixture weights, means, and
  covariances are directly available without sampling.

### Use a normalizing flow when:

- The posterior has **complex geometry** -- curved degeneracies,
  banana-shaped contours, or sharp boundaries near physical constraints.
- You observe **poor calibration** with MDN (e.g. SBC histograms show
  under- or over-dispersion; see {doc}`training_to_deployment`).
- The parameter space is **higher-dimensional** ($D > 5$) where the
  number of Gaussian components needed grows combinatorially.
- You need **exact density evaluation** at arbitrary points in parameter
  space (flows provide this by construction).

### Quick comparison

| Property | MDN | Normalizing Flow |
|----------|-----|-----------------|
| Posterior family | Gaussian mixture | Arbitrary (learned) |
| Training speed | Fast (fewer params) | Slower (deeper architecture) |
| Expressiveness | Limited by $K$ components | Scales with depth |
| Sampling | Trivial (mixture sampling) | Fast (inverse pass) |
| Density evaluation | Closed-form | Closed-form (change of variables) |
| Multimodality | Explicit ($K$ modes) | Implicit (learned) |
| Implementation | `dmipy_jax.inference.mdn` | `dmipy_jax.inference.flows` |

---

## 8. Integration with the SBI Pipeline

The `FlowNetwork` above is a compact, self-contained implementation that is
useful for understanding how coupling layers and RQS transforms fit together.
In the full SBI4DWI pipeline the flow backend is instead built on
[FlowJAX](https://github.com/danielward27/flowjax) (masked autoregressive
flows with affine or rational-quadratic-spline transformers) and is selected
at training time through the configuration: set
`inference_mode="flow"` in {class}`~dmipy_jax.pipeline.config.SBIPipelineConfig`
and call {func}`~dmipy_jax.pipeline.train.train_sbi`.

The example below trains a deliberately tiny flow on a mono-exponential
(ADC) forward model so that it runs in a few seconds on CPU; `depth` is the
number of flow layers and `knots` the number of spline knots.

```python
import jax.numpy as jnp
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.train import train_sbi

def forward_fn(params, acq):
    # params = [D] in um^2/ms; b-values are SI (s/m^2), hence the 1e-9
    return jnp.exp(-acq.bvalues * params[0] * 1e-9)

acq = JaxAcquisition(
    bvalues=jnp.r_[jnp.zeros(4), jnp.full(28, 1000e6)],
    gradient_directions=jnp.zeros((32, 3)),
)
names, ranges = ["D"], {"D": (0.1, 3.0)}
model_simulator = ModelSimulator(forward_fn, names, ranges, acq,
                                 noise_type="rician", snr=30.0)

config = SBIPipelineConfig(
    model_name="ADC",
    parameter_names=names,
    parameter_ranges=ranges,
    acquisition={"bvalues": acq.bvalues.tolist()},
    inference_mode="flow",    # "mdn", "flow" or "score"
    flow_type="spline",       # "affine" or "spline"
    knots=4,                  # spline knots
    depth=2,                  # number of flow layers
    hidden_dim=32,
    lr_schedule="constant",
    batch_size=128,
    n_steps=100,              # tutorial-sized; use thousands in practice
)

trained_model, losses = train_sbi(config, model_simulator, print_every=50)
```

The resulting `_NormalisedFlow` object wraps the FlowJAX flow together with
the parameter normalisation used during training, so that samples come back
in physical units. It exposes `.log_prob(theta, condition=x)` and
`.sample(key, (n,), condition=x)`:

```python
# FlowJAX requires new-style typed keys (jax.random.key, not PRNGKey)
_, signals = model_simulator.sample_and_simulate(jax.random.key(1), 1)
samples = trained_model.sample(jax.random.key(2), (200,), condition=signals[0])
assert samples.shape == (200, 1)
print(f"posterior mean D = {float(samples.mean()):.2f} um^2/ms")
```

Checkpointing and deployment work identically for both backends. A
checkpoint is a pair of files, `<path>.eqx` (weights) and
`<path>.config.json` (the config), and `SBIPredictor` takes the model and
config together:

```python
import os, tempfile
from dmipy_jax.pipeline.checkpoint import save_checkpoint, load_checkpoint
from dmipy_jax.pipeline.deploy import SBIPredictor

ckpt = os.path.join(tempfile.mkdtemp(), "adc_flow")
save_checkpoint(trained_model, config, ckpt)
loaded, loaded_config = load_checkpoint(ckpt)

predictor = SBIPredictor(loaded, loaded_config)
# predictor.predict_volume("dwi.nii.gz", "dwi.bval", "dwi.bvec",
#                          mask_path="mask.nii.gz", output_dir="out/")
```

See {doc}`training_to_deployment` for a complete run of `predict_volume` on
a NIfTI volume.

---

## References

- Durkan, C. et al. "Neural Spline Flows." *Advances in Neural Information
  Processing Systems* 32, 2019.
- Papamakarios, G. et al. "Normalizing Flows for Probabilistic Modeling and
  Inference." *Journal of Machine Learning Research* 22(57), 2021.
- Rezende, D. J. & Mohamed, S. "Variational Inference with Normalizing
  Flows." *ICML*, 2015.

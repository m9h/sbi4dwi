# Uncertainty Quantification: From Calibration to Clinical Confidence

A trained neural posterior estimator can produce parameter maps for a
whole brain in seconds -- but **would you trust those predictions in a
clinical report?** Without rigorous uncertainty quantification (UQ), the
answer must be no. This tutorial walks through the five complementary UQ
tools in SBI4DWI, building from foundational calibration checks to
production-ready confidence intervals and anomaly detection.

```{contents}
:depth: 3
:local:
```

## Prerequisites

- Familiarity with the `ModelSimulator` and `train_sbi()` pipeline -- see
  {doc}`training_to_deployment`
- Python 3.12+ with `uv sync` from the repository root

```bash
uv sync
```

Every code block on this page is executed as a test on CPU, so the model we
train below is deliberately tiny and the diagnostic sample counts are small.
Production values are noted alongside.

---

## 1. Introduction: Why UQ Matters for Clinical Neuroimaging

Diffusion MRI microstructure parameters -- fractional anisotropy, mean
diffusivity, axon diameter, neurite density -- inform surgical planning,
track disease progression, and guide treatment decisions. A prediction
without a calibrated uncertainty estimate is clinically dangerous: it may
look precise while being systematically wrong.

### The UQ Ladder

SBI4DWI provides five levels of uncertainty quantification, each answering
a different question:

| Level | Module | Question answered |
|-------|--------|-------------------|
| 1. Simulation-Based Calibration | `pipeline.sbc` | Is my posterior well-calibrated *in principle*? |
| 2. Posterior Predictive Checks | `pipeline.ppc` | Can my posterior reproduce observed data? |
| 3. Conformal Prediction | `pipeline.conformal` | What are *distribution-free* prediction intervals with guaranteed coverage? |
| 4. Out-of-Distribution Detection | `pipeline.ood` | Is this voxel within the domain my model was trained on? |
| 5. Ensemble Inference | `pipeline.ensemble` | How much uncertainty comes from the data vs. from model training? |

Each level builds on the previous. A well-calibrated posterior (Level 1)
that reproduces observed signals (Level 2) can be wrapped in conformal
intervals for finite-sample guarantees (Level 3), screened for anomalous
inputs (Level 4), and decomposed into aleatoric and epistemic components
via ensembles (Level 5).

### Setup: a small trained model

The model-based tools below all need a trained posterior estimator and the
simulator it was trained on. We train a Ball-and-Stick MDN exactly as in
{doc}`training_to_deployment`, but with a tiny budget:

```python
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.train import train_sbi

WORK = tempfile.mkdtemp(prefix="sbi4dwi_uq_")

# 6 b=0 + 30 directions at b=1000 + 30 at b=2000 (SI units: s/m^2)
bvals = np.concatenate([np.zeros(6), np.full(30, 1000e6), np.full(30, 2000e6)])
rng = np.random.default_rng(42)
bvecs = rng.standard_normal((len(bvals), 3))
bvecs[:6] = 0.0
bvecs /= np.maximum(np.linalg.norm(bvecs, axis=1, keepdims=True), 1e-8)
acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

parameter_names = ["f", "mu_theta", "mu_phi", "d_par", "d_iso"]
parameter_ranges = {
    "f": (0.05, 0.95), "mu_theta": (0.0, float(jnp.pi)),
    "mu_phi": (0.0, float(2 * jnp.pi)),
    "d_par": (0.5e-9, 3.0e-9), "d_iso": (1.0e-9, 3.5e-9),
}

def forward_fn(params, acquisition):
    f, th, ph, d_par, d_iso = params
    n = jnp.array([jnp.sin(th) * jnp.cos(ph), jnp.sin(th) * jnp.sin(ph), jnp.cos(th)])
    gn = acquisition.gradient_directions @ n
    return f * jnp.exp(-acquisition.bvalues * d_par * gn**2) + \
           (1 - f) * jnp.exp(-acquisition.bvalues * d_iso)

simulator = ModelSimulator(forward_fn, parameter_names, parameter_ranges,
                           acq, noise_type="rician", snr=30.0)

config = SBIPipelineConfig(
    model_name="BallStick",
    parameter_names=parameter_names,
    parameter_ranges=parameter_ranges,
    acquisition={"bvalues": acq.bvalues.tolist(),
                 "gradient_directions": acq.gradient_directions.tolist()},
    inference_mode="mdn", n_components=4, hidden_dim=64, depth=2,
    lr_schedule="constant", batch_size=128,
    n_steps=200,                 # production: 5_000 - 10_000
    seed=0,
)
trained_model, losses = train_sbi(config, simulator, print_every=0)

# A batch of simulated "voxels" that we will treat as observations
theta_dim = len(parameter_names)
n_voxels = 1000
theta_obs, signals = simulator.sample_and_simulate(jax.random.key(1), n_voxels)
batch_size = signals.shape[0]
print(signals.shape)   # (1000, 66)
```

---

## 2. Simulation-Based Calibration (SBC)

### 2.1 The Idea

Simulation-based calibration (Talts et al., 2018) answers a fundamental
question: if the prior and likelihood are correct, does the posterior
actually have the right coverage?

The key insight is a self-consistency property of Bayesian inference. If we:

1. Draw $\boldsymbol{\theta}^* \sim p(\boldsymbol{\theta})$ from the prior
2. Simulate data $\mathbf{x}^* \sim p(\mathbf{x} \mid \boldsymbol{\theta}^*)$
3. Draw posterior samples $\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_L \sim p(\boldsymbol{\theta} \mid \mathbf{x}^*)$

then the **rank** of $\boldsymbol{\theta}^*$ among the posterior samples
should be uniformly distributed on $\{0, 1, \dots, L\}$.

### 2.2 Rank Statistics

The rank for parameter dimension $d$ is simply the count of posterior
samples less than the true value:

$$
r_d = \sum_{l=1}^{L} \mathbb{1}\!\left[\theta_{l,d} < \theta^*_d\right]
$$

Under perfect calibration, $r_d \sim \text{Uniform}\{0, 1, \dots, L\}$.
Deviations reveal specific failure modes:

- **U-shaped histogram** (mass at 0 and $L$): overconfident posterior
- **Inverse-U shape** (mass in the middle): underconfident posterior
- **Skewed histogram**: biased posterior mean

### 2.3 Using `SBCResult`

The `SBCResult` dataclass holds rank statistics and exposes diagnostics
directly.

```python
from dmipy_jax.pipeline.sbc import SBCResult, compute_ranks

# After running SBC (n_sbc repetitions, L posterior samples each)
n_sbc, n_params, L = 1000, 3, 200
rng = np.random.default_rng(42)

# For a well-calibrated model, ranks are uniform
ranks = rng.integers(0, L + 1, size=(n_sbc, n_params))

result = SBCResult(
    ranks=ranks,
    parameter_names=["FA", "MD", "OD"],
    n_posterior_samples=L,
)

# Check shapes
assert result.ranks.shape == (n_sbc, n_params)
assert len(result.parameter_names) == n_params
```

### 2.4 The KS Test for Uniformity

The Kolmogorov-Smirnov test checks whether the empirical rank distribution
departs from Uniform(0, 1). A p-value above 0.05 means we cannot reject
uniformity -- i.e., the posterior passes calibration.

```python
ks_results = result.ks_test()
for name in result.parameter_names:
    stat = ks_results[name]["statistic"]
    pval = ks_results[name]["pvalue"]
    status = "PASS" if pval > 0.05 else "FAIL"
    print(f"  {name}: KS={stat:.4f}, p={pval:.4f} [{status}]")
```

A miscalibrated posterior will fail:

```python
# Overconfident posterior: ranks pile up at 0 and L
choices = np.array([0, L])
bad_ranks = rng.choice(choices, size=(n_sbc, 1))
bad_result = SBCResult(
    ranks=bad_ranks,
    parameter_names=["bad_param"],
    n_posterior_samples=L,
)
ks = bad_result.ks_test()
assert ks["bad_param"]["pvalue"] < 0.05  # Fails uniformity
```

### 2.5 Coverage at Nominal Levels

`coverage_at(level)` computes the fraction of SBC repetitions where the
true parameter falls inside the central credible interval at a given level.
Under perfect calibration, `coverage_at(0.9)` should return approximately
0.9.

```python
# Uniform ranks: coverage should match the nominal level
for level in [0.5, 0.8, 0.9, 0.95]:
    cov = result.coverage_at(level)
    for d, name in enumerate(result.parameter_names):
        print(f"  {name} coverage@{level:.0%}: {cov[d]:.3f}")
        assert abs(cov[d] - level) < 0.05
```

Edge cases provide useful sanity checks:

```python
# All ranks at the centre -> coverage is 1.0
centred_ranks = np.full((100, 2), L // 2)
centred_result = SBCResult(
    ranks=centred_ranks,
    parameter_names=["a", "b"],
    n_posterior_samples=L,
)
cov = centred_result.coverage_at(0.9)
np.testing.assert_allclose(cov, 1.0)

# All ranks at the edge -> coverage is 0.0
edge_ranks = np.zeros((100, 1), dtype=int)
edge_result = SBCResult(
    ranks=edge_ranks,
    parameter_names=["edge"],
    n_posterior_samples=L,
)
cov = edge_result.coverage_at(0.9)
assert cov[0] == 0.0
```

### 2.6 Computing Ranks from Posterior Samples

The `compute_ranks` utility counts how many posterior samples fall below
the true value in each dimension:

```python
theta_true = jnp.array([0.5, 0.3])
posterior = jnp.array([
    [0.1, 0.1],
    [0.4, 0.5],
    [0.6, 0.2],
    [0.8, 0.4],
])

ranks = compute_ranks(theta_true, posterior)
# dim 0: samples < 0.5 are [0.1, 0.4] -> rank 2
# dim 1: samples < 0.3 are [0.1, 0.2] -> rank 2
np.testing.assert_array_equal(np.asarray(ranks), [2, 2])
```

### 2.7 Running SBC on a Trained Model

For a complete end-to-end SBC run, use `SBCDiagnostic`. It loops over
repetitions in Python, so we use a small count here (production: 1000):

```python
import matplotlib
matplotlib.use("Agg")
from dmipy_jax.pipeline.sbc import SBCDiagnostic

diagnostic = SBCDiagnostic(
    model=trained_model,        # _NormalisedMDN or _NormalisedFlow
    simulator=simulator,        # same ModelSimulator used for training
    n_posterior_samples=100,    # L posterior draws per repetition
)

key = jax.random.PRNGKey(0)
result = diagnostic.run(key, n_sbc_samples=40)
assert result.ranks.shape == (40, theta_dim)

# Print summary table with coverage and KS statistics
result.print_summary()

# Visualise rank histograms
result.plot_rank_histograms(save_path=os.path.join(WORK, "sbc_ranks.png"))
```

With only 200 training steps the summary will show failed KS tests for most
parameters -- the posterior is not yet calibrated, and SBC tells you so.

---

## 3. Posterior Predictive Checks (PPC)

### 3.1 The Idea

While SBC checks calibration in parameter space, posterior predictive
checks verify consistency in **data space**. The question is: if I draw
parameters from the posterior and re-simulate signals, do those simulated
signals look like the real observation?

For each test observation $\mathbf{x}^*$:

1. Draw $\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_L \sim p(\boldsymbol{\theta} \mid \mathbf{x}^*)$
2. Re-simulate: $\hat{\mathbf{x}}_l = S(\boldsymbol{\theta}_l, \mathbf{q})$ for $l = 1, \dots, L$
3. Compare $\hat{\mathbf{x}}$ with $\mathbf{x}^*$

### 3.2 Diagnostic Metrics

`PPCResult` computes three complementary metrics:

**Root Mean Square Error (RMSE)**

$$
\text{RMSE} = \sqrt{\frac{1}{M} \sum_{m=1}^{M} \left(x^*_m - \bar{x}_m\right)^2}
$$

where $\bar{x}_m = \frac{1}{L}\sum_l \hat{x}_{l,m}$ is the posterior predictive mean.

**Coverage**: The fraction of measurements $m$ where $x^*_m$ falls within
the 5th--95th percentile interval of $\{\hat{x}_{1,m}, \dots, \hat{x}_{L,m}\}$.
Should be approximately 0.90 for a well-specified model.

**Reduced chi-squared**:

$$
\chi^2_\text{red} = \frac{1}{M} \sum_{m=1}^{M} \frac{(x^*_m - \bar{x}_m)^2}{\text{Var}[\hat{x}_m]}
$$

A value near 1.0 indicates the model's predictive variance matches the
actual spread of residuals. Values much less than 1.0 suggest overfitting
(overestimated variance); values much greater than 1.0 indicate
underestimated uncertainty.

### 3.3 Using `PPCResult`

```python
from dmipy_jax.pipeline.ppc import PPCResult

# Construct from pre-computed diagnostics
n_checks = 200
n_meas = 108  # number of diffusion measurements

rng = np.random.default_rng(42)
result = PPCResult(
    rmse_per_obs=rng.uniform(0.01, 0.05, size=(n_checks,)),
    coverage_per_measurement=rng.uniform(0.85, 0.95, size=(n_meas,)),
    chi_squared=rng.uniform(90, 120, size=(n_checks,)),
    n_measurements=n_meas,
)

# Summary statistics
print(f"Mean RMSE:           {result.mean_rmse:.4f}")
print(f"Median RMSE:         {result.median_rmse:.4f}")
print(f"Mean coverage (90%): {result.mean_coverage:.1%}")
print(f"Reduced chi-squared: {result.reduced_chi_squared:.2f}")

# Print formatted summary
result.print_summary()
```

### 3.4 Interpreting PPC Results

A perfect model (zero-noise reconstruction) yields:

```python
result_perfect = PPCResult(
    rmse_per_obs=np.zeros(50),
    coverage_per_measurement=np.ones(20),
    chi_squared=np.zeros(50),
    n_measurements=20,
)
assert result_perfect.mean_rmse == 0.0
assert result_perfect.mean_coverage == 1.0
assert result_perfect.reduced_chi_squared == 0.0
```

The ideal reduced chi-squared is 1.0, which occurs when the residual
variance exactly matches the posterior predictive variance:

```python
n_meas = 50
chi2_ideal = np.full(300, float(n_meas))
result_ideal = PPCResult(
    rmse_per_obs=np.zeros(300),
    coverage_per_measurement=np.ones(n_meas) * 0.9,
    chi_squared=chi2_ideal,
    n_measurements=n_meas,
)
assert abs(result_ideal.reduced_chi_squared - 1.0) < 1e-10
```

### 3.5 Running PPC on a Trained Model

```python
from dmipy_jax.pipeline.ppc import PPCDiagnostic

diagnostic = PPCDiagnostic(
    model=trained_model,
    simulator=simulator,
    n_posterior_samples=100,
)

key = jax.random.PRNGKey(1)
result = diagnostic.run(key, n_checks=20)     # production: 500
assert result.rmse_per_obs.shape == (20,)

result.print_summary()
```

Because an MDN posterior is unbounded, an under-trained network can place a
few posterior draws at negative diffusivities whose re-simulated signals
explode; this shows up as a huge *mean* RMSE next to a sane *median*.
Compare both, and re-run after longer training.

---

## 4. Conformal Prediction

### 4.1 The Idea

SBC and PPC rely on the prior and likelihood being correct. **Conformal
prediction** provides distribution-free, finite-sample valid coverage
guarantees that hold regardless of whether the model is well-specified.

The only assumption is *exchangeability* of calibration and test data --
no parametric assumptions about the posterior. This makes conformal
prediction the ideal complement to Bayesian UQ for clinical deployment
where model misspecification is the norm.

### 4.2 Split Conformal Inference

Given a held-out calibration set $\{(\boldsymbol{\theta}_i, \hat{\boldsymbol{\theta}}_i)\}_{i=1}^{n}$
of true parameters and model predictions:

1. Compute nonconformity scores $s_i$
2. Set $\hat{q} = \text{Quantile}\left(\{s_i\}, \; \frac{\lceil (1-\alpha)(n+1)\rceil}{n}\right)$
3. For a new prediction $\hat{\boldsymbol{\theta}}$, form the interval
   $C(\hat{\boldsymbol{\theta}}) = [\hat{\boldsymbol{\theta}} - \hat{q}, \; \hat{\boldsymbol{\theta}} + \hat{q}]$

**Coverage guarantee**: $P(\boldsymbol{\theta}_\text{new} \in C(\hat{\boldsymbol{\theta}}_\text{new})) \geq 1 - \alpha$

### 4.3 Two Methods

**Absolute method**: symmetric intervals around the posterior mean.
Nonconformity score: $s_i = |\theta_i - \hat{\theta}_i|$ (per parameter dimension).

**Conformalized Quantile Regression (CQR)**: uses predicted quantile
bounds for heteroscedastic (adaptive-width) intervals. Nonconformity score:
$s_i = \max(\hat{q}^\text{lo}_i - \theta_i, \; \theta_i - \hat{q}^\text{hi}_i)$.

CQR produces tighter intervals where the model is confident and wider
intervals where it is uncertain -- a natural fit for neuroimaging where
some brain regions are more challenging than others.

### 4.4 The Functional API: `calibrate_from_predictions()`

The model-free entry point works directly with pre-computed predictions --
no model object required. This is the easiest way to get started.

**Absolute conformal intervals:**

```python
from dmipy_jax.pipeline.conformal import (
    calibrate_from_predictions,
    predict_intervals_from_predictions,
)

# Synthetic calibration data: predictions = theta + noise
rng = np.random.RandomState(42)
n_cal, n_test, dim = 1000, 500, 2
noise_scale = 0.1

theta_cal = rng.uniform(0, 1, (n_cal, dim))
preds_cal = theta_cal + rng.randn(n_cal, dim) * noise_scale

theta_test = rng.uniform(0, 1, (n_test, dim))
preds_test = theta_test + rng.randn(n_test, dim) * noise_scale

# Calibrate at alpha=0.1 (target 90% coverage)
alpha = 0.1
result = calibrate_from_predictions(
    theta_cal, preds_cal, alpha=alpha,
    parameter_names=["FA", "MD"],
)

assert result.method == "absolute"
assert result.n_calibration == 1000
assert result.empirical_coverage >= 1 - alpha - 0.01
```

**Form prediction intervals on new data:**

```python
intervals = predict_intervals_from_predictions(preds_test, result)

# Verify shapes
assert intervals["mean"].shape == (n_test, dim)
assert intervals["lower"].shape == (n_test, dim)
assert intervals["upper"].shape == (n_test, dim)
assert intervals["width"].shape == (n_test, dim)

# Sanity: lower <= upper, width >= 0
assert np.all(intervals["lower"] <= intervals["upper"])
assert np.all(intervals["width"] >= 0)

# Verify coverage guarantee
covered = (
    (theta_test >= intervals["lower"])
    & (theta_test <= intervals["upper"])
)
marginal_coverage = np.min(np.mean(covered, axis=0))
print(f"Marginal coverage: {marginal_coverage:.1%}")
assert marginal_coverage >= (1 - alpha) - 0.05
```

### 4.5 CQR: Adaptive Intervals

When predicted quantile bounds are available (from the MDN mixture or flow
posterior), CQR produces intervals that adapt to local uncertainty:

```python
# Simulate predicted quantile intervals
interval_width = 0.15
pred_lower_cal = preds_cal - interval_width
pred_upper_cal = preds_cal + interval_width

pred_lower_test = preds_test - interval_width
pred_upper_test = preds_test + interval_width

result_cqr = calibrate_from_predictions(
    theta_cal, preds_cal, alpha=0.1,
    parameter_names=["FA", "MD"],
    pred_lower=pred_lower_cal,
    pred_upper=pred_upper_cal,
)
assert result_cqr.method == "cqr"

intervals_cqr = predict_intervals_from_predictions(
    preds_test, result_cqr,
    pred_lower=pred_lower_test,
    pred_upper=pred_upper_test,
)

covered_cqr = (
    (theta_test >= intervals_cqr["lower"])
    & (theta_test <= intervals_cqr["upper"])
)
marginal_cqr = np.min(np.mean(covered_cqr, axis=0))
print(f"CQR marginal coverage: {marginal_cqr:.1%}")
assert marginal_cqr >= (1 - alpha) - 0.05
```

```{important}
CQR requires quantile bounds at *both* calibration and prediction time.
Calling `predict_intervals_from_predictions` on a CQR result without
`pred_lower` and `pred_upper` raises a `ValueError`.
```

### 4.6 Interval Monotonicity

Stricter coverage targets (lower $\alpha$) produce wider intervals. This
is a useful sanity check:

```python
result_90 = calibrate_from_predictions(theta_cal, preds_cal, alpha=0.1)
result_99 = calibrate_from_predictions(theta_cal, preds_cal, alpha=0.01)

intervals_90 = predict_intervals_from_predictions(preds_test, result_90)
intervals_99 = predict_intervals_from_predictions(preds_test, result_99)

mean_width_90 = np.mean(intervals_90["width"])
mean_width_99 = np.mean(intervals_99["width"])
assert mean_width_99 > mean_width_90
print(f"90% width: {mean_width_90:.4f}, 99% width: {mean_width_99:.4f}")
```

### 4.7 Perfect Predictor

A zero-error predictor should yield near-zero quantiles (the conformal
correction is trivial when predictions are exact):

```python
theta_perfect = rng.uniform(0, 1, (500, 2))
result_perfect = calibrate_from_predictions(
    theta_perfect, theta_perfect, alpha=0.1
)
assert np.all(result_perfect.quantiles < 1e-10)
```

### 4.8 Inspecting Results

```python
result.print_summary()
```

This prints a formatted table showing the method, alpha, target and
empirical coverage, number of calibration samples, and per-parameter
$\hat{q}$ values.

---

## 5. Out-of-Distribution Detection

### 5.1 The Problem

A model trained on simulated data from a Ball-Stick model knows nothing
about tumour tissue, motion artefacts, or susceptibility distortions.
Predictions on such voxels will be confidently wrong. OOD detection flags
voxels where the input signal departs from the training distribution, so
clinicians know which predictions to trust.

### 5.2 Three OOD Scores

`OODDetector` computes three complementary scores:

**Reconstruction error** (RMSE): re-simulate the signal from the posterior
mean and measure how well it matches the observation. High RMSE means the
model cannot explain the observed signal.

$$
\text{RMSE} = \sqrt{\frac{1}{M}\sum_{m=1}^{M}(x_m - \hat{x}_m)^2}
$$

**Predictive entropy**: for MDN models, the entropy of the mixing weights
$H = -\sum_k \pi_k \log \pi_k$. High entropy means the model is uncertain
about which mixture component explains the data.

**Mahalanobis distance**: measures how far a signal is from the reference
distribution in a covariance-normalised sense:

$$
d_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$

where $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ are estimated from
in-distribution reference signals during the `fit()` step.

### 5.3 The Fit-Score Pattern

`OODDetector` follows a scikit-learn-style fit/score pattern:

```python
from dmipy_jax.pipeline.ood import OODDetector

# Create detector from a trained model and its simulator
detector = OODDetector(model=trained_model, simulator=simulator)

# Fit: generate reference distribution statistics (production: 5000)
key = jax.random.PRNGKey(42)
detector.fit(key, n_reference=500)
```

The `fit()` method:
1. Generates `n_reference` in-distribution (theta, signal) pairs from
   the simulator
2. Computes reference signal mean and covariance (for Mahalanobis)
3. Computes reference OOD scores (for percentile thresholds)

### 5.4 Scoring New Signals

```python
# Score new observations
scores = detector.score(signals)

expected_keys = {
    "reconstruction_error",
    "predictive_entropy",
    "mahalanobis_distance",
    "is_ood",
}
assert set(scores.keys()) == expected_keys
assert scores["reconstruction_error"].shape == (n_voxels,)
assert scores["predictive_entropy"].shape == (n_voxels,)
assert scores["mahalanobis_distance"].shape == (n_voxels,)
```

Calling `score()` before `fit()` raises a `RuntimeError`:

```python
# Unfitted detector raises
detector_new = OODDetector(model=trained_model, simulator=simulator)
try:
    detector_new.score(signals)
except RuntimeError as e:
    print(f"Expected error: {e}")  # "OODDetector has not been fitted."
```

### 5.5 In-Distribution vs. OOD

In-distribution signals should have low reconstruction error and
Mahalanobis distance, while random noise should score much higher:

```python
key = jax.random.PRNGKey(300)
k1, k2 = jax.random.split(key)

# In-distribution signals
_, signals_id = simulator.sample_and_simulate(k1, 200)
scores_id = detector.score(signals_id)

# Out-of-distribution: random uniform noise
signals_ood = jax.random.uniform(
    k2, (200, simulator.signal_dim), minval=0.0, maxval=5.0
)
scores_ood = detector.score(signals_ood)

# OOD signals should score higher on all metrics
mean_recon_id = np.mean(scores_id["reconstruction_error"])
mean_recon_ood = np.mean(scores_ood["reconstruction_error"])
print(f"Recon RMSE -- in-dist: {mean_recon_id:.4f}, OOD: {mean_recon_ood:.4f}")
assert mean_recon_ood > mean_recon_id

mean_mahal_id = np.mean(scores_id["mahalanobis_distance"])
mean_mahal_ood = np.mean(scores_ood["mahalanobis_distance"])
print(f"Mahalanobis -- in-dist: {mean_mahal_id:.2f}, OOD: {mean_mahal_ood:.2f}")
assert mean_mahal_ood > mean_mahal_id
```

### 5.6 Flagging a Volume

For whole-brain inference, `flag_volume()` applies percentile-based
thresholds from the reference distribution and returns a structured
`OODResult`:

```python
from dmipy_jax.pipeline.ood import OODResult

result = detector.flag_volume(signals, threshold_percentile=95.0)

assert isinstance(result, OODResult)
assert result.is_ood.shape == (n_voxels,)
assert result.is_ood.dtype == bool

# Summary
result.print_summary()
print(f"OOD fraction: {result.ood_fraction:.1%}")
```

With a brain mask, only active voxels are scored:

```python
mask = np.zeros(n_voxels, dtype=bool)
mask[:500] = True  # only score first 500 voxels

result_masked = detector.flag_volume(signals, mask=mask)
assert result_masked.is_ood.shape == (n_voxels,)
# Masked-out voxels have zero scores
assert np.all(result_masked.reconstruction_error[500:] == 0.0)
assert np.all(result_masked.mahalanobis_distance[500:] == 0.0)
```

### 5.7 `OODResult` Properties

```python
# OOD fraction
is_ood = np.array([True, False, True, False, True])
result = OODResult(
    reconstruction_error=np.zeros(5),
    predictive_entropy=np.zeros(5),
    mahalanobis_distance=np.zeros(5),
    is_ood=is_ood,
    threshold_reconstruction=1.0,
    threshold_entropy=1.0,
    threshold_mahalanobis=1.0,
)
assert abs(result.ood_fraction - 0.6) < 1e-10
```

---

## 6. Ensemble Inference

### 6.1 The Idea

Deep ensembles (Lakshminarayanan et al., 2017) train multiple models with
different random initialisations. The key benefits:

- **Better calibrated uncertainty** via disagreement between members
- **More robust point estimates** via averaging (reduces variance)
- **Aleatoric vs. epistemic decomposition**: within-model uncertainty
  reflects noise in the data (aleatoric), while between-model disagreement
  reflects uncertainty due to limited training (epistemic)

### 6.2 Uncertainty Decomposition

Given $M$ ensemble members, each producing a mean $\hat{\boldsymbol{\theta}}^{(m)}$
and per-voxel standard deviation $\sigma_\text{aleat}^{(m)}$:

**Ensemble mean**:
$$
\bar{\boldsymbol{\theta}} = \frac{1}{M}\sum_{m=1}^{M}\hat{\boldsymbol{\theta}}^{(m)}
$$

**Aleatoric uncertainty** (mean of per-model variances):
$$
\sigma_\text{aleat} = \frac{1}{M}\sum_{m=1}^{M}\sigma_\text{aleat}^{(m)}
$$

**Epistemic uncertainty** (disagreement between models):
$$
\sigma_\text{epist} = \text{Std}\!\left[\hat{\boldsymbol{\theta}}^{(1)}, \dots, \hat{\boldsymbol{\theta}}^{(M)}\right]
$$

**Total uncertainty** (combined):
$$
\sigma_\text{total} = \sqrt{\sigma_\text{aleat}^2 + \sigma_\text{epist}^2}
$$

### 6.3 `EnsemblePredictor`

```python
from dmipy_jax.pipeline.ensemble import (
    EnsemblePredictor,
    train_ensemble,
    save_ensemble,
    load_ensemble,
)

# Train the members (production: n_members=5 with a full training budget)
n_members = 3
members = train_ensemble(config, simulator, n_members=n_members, print_every=0)
ens = EnsemblePredictor.from_training(members, config)
models = ens.models
assert len(models) == n_members
```

### 6.4 Basic Prediction

`predict_mean` returns the ensemble mean and inter-model standard
deviation (epistemic uncertainty):

```python
mean, std = ens.predict_mean(signals)

assert mean.shape == (batch_size, theta_dim)
assert std.shape == (batch_size, theta_dim)
```

With a single member, the epistemic std is zero:

```python
ens_single = EnsemblePredictor([models[0]], config)
mean_single, std_single = ens_single.predict_mean(signals)
assert jnp.allclose(std_single, 0.0, atol=1e-6)
```

### 6.5 Full Uncertainty Decomposition

`predict_with_uncertainty` returns the complete decomposition:

```python
result = ens.predict_with_uncertainty(signals)

assert set(result.keys()) == {"mean", "aleatoric", "epistemic", "total"}

# Verify decomposition: total = sqrt(aleatoric^2 + epistemic^2)
expected_total = jnp.sqrt(
    result["aleatoric"] ** 2 + result["epistemic"] ** 2
)
assert jnp.allclose(result["total"], expected_total, atol=1e-5)

# All uncertainties are non-negative
for key in ("aleatoric", "epistemic", "total"):
    assert jnp.all(result[key] >= 0)
```

### 6.6 Variance Reduction

A fundamental property of ensembling: the ensemble mean is closer to the
truth (on average) than any individual member. Members trained with
different random seeds will disagree, and averaging reduces this variance:

```python
# Individual member predictions
member_preds = []
for model in ens.models:
    m, _ = ens._member_stats(model, signals)
    member_preds.append(m)

stacked = jnp.stack(member_preds, axis=0)  # (M, batch, theta_dim)
member_spread = jnp.std(stacked, axis=0)    # (batch, theta_dim)

# Members should disagree (different random seeds)
assert jnp.mean(member_spread) > 1e-3

# Ensemble mean is the average of the members
ens_mean = jnp.mean(stacked, axis=0)
assert jnp.allclose(ens_mean, mean, atol=1e-5)
```

### 6.7 Save and Load

Ensembles serialise as individual member checkpoints:

```python
# Save
ens_path = os.path.join(WORK, "checkpoints", "my_ensemble")
save_ensemble(members, config, ens_path)
# Creates: my_ensemble_member_0.eqx, my_ensemble_member_1.eqx, ...
assert os.path.exists(ens_path + "_member_0.eqx")

# Load
ens_loaded = load_ensemble(
    ens_path, n_members=n_members,
    key=jax.random.PRNGKey(0),
)
assert len(ens_loaded.models) == n_members
assert ens_loaded.config.model_name == config.model_name

# Verify predictions match after round-trip
loaded_mean, loaded_std = ens_loaded.predict_mean(signals)
assert jnp.allclose(mean, loaded_mean, atol=1e-4)
```

### 6.8 Volume-Level Ensemble Inference

`predict_volume` mirrors `SBIPredictor.predict_volume` but produces
additional uncertainty NIfTI maps (not executed here; see
{doc}`training_to_deployment` Section 6 for building the input files):

```text
results = ens.predict_volume(
    dwi_path="sub-01/dwi/sub-01_dwi.nii.gz",
    bval_path="sub-01/dwi/sub-01_dwi.bval",
    bvec_path="sub-01/dwi/sub-01_dwi.bvec",
    mask_path="sub-01/dwi/sub-01_mask.nii.gz",
    output_dir="output/ensemble/",
    batch_size=4096,
)

# Produces: f.nii.gz, f_aleatoric.nii.gz, f_epistemic.nii.gz, f_total.nii.gz,
#           d_par.nii.gz, d_par_aleatoric.nii.gz, ...
```

---

## 7. Putting It All Together: A Recommended UQ Workflow

For production deployment of SBI models on clinical diffusion MRI data,
we recommend the following workflow. Each step builds confidence before
the model's predictions reach a clinical audience. The code reuses the
ensemble trained in Section 6.

### Step 1: Train an Ensemble

Train 5 models with different random seeds. This provides both better
calibration and the aleatoric/epistemic decomposition.

```python
# members = train_ensemble(config, simulator, n_members=5)   # production
ens = EnsemblePredictor.from_training(members, config)
save_ensemble(members, config, os.path.join(WORK, "production", "ensemble"))
```

### Step 2: Run SBC on Each Member

Verify that each member's posterior is well-calibrated. If any member
fails the KS test, retrain or investigate the simulation pipeline.

```python
for i, model in enumerate(ens.models):
    diagnostic = SBCDiagnostic(model, simulator, n_posterior_samples=100)
    result = diagnostic.run(jax.random.PRNGKey(i), n_sbc_samples=20)   # production: 1000
    result.print_summary()
    result.plot_rank_histograms(save_path=os.path.join(WORK, f"sbc_member_{i}.png"))
```

### Step 3: Run PPC

Verify that posterior predictive distributions are consistent with
simulated observations.

```python
for i, model in enumerate(ens.models):
    diagnostic = PPCDiagnostic(model, simulator, n_posterior_samples=100)
    result = diagnostic.run(jax.random.PRNGKey(100 + i), n_checks=10)   # production: 500
    result.print_summary()
```

Target: mean RMSE < 0.05, mean coverage near 90%, reduced chi-squared
near 1.0.

### Step 4: Calibrate Conformal Intervals

Use a held-out calibration set (not used in training) to compute
distribution-free prediction intervals.

```python
# Generate calibration data (production: 2000+)
key_cal = jax.random.PRNGKey(999)
theta_cal, signals_cal = simulator.sample_and_simulate(key_cal, 500)

# Get ensemble predictions
mean_cal, _ = ens.predict_mean(jnp.array(signals_cal))

# Calibrate
conformal_result = calibrate_from_predictions(
    np.asarray(theta_cal),
    np.asarray(mean_cal),
    alpha=0.1,
    parameter_names=config.parameter_names,
)
conformal_result.print_summary()
assert conformal_result.empirical_coverage >= 0.89
```

### Step 5: Fit the OOD Detector

```python
detector = OODDetector(model=ens.models[0], simulator=simulator)
detector.fit(jax.random.PRNGKey(42), n_reference=500)   # production: 5000
```

### Step 6: Deploy on Real Data

Not executed here (needs scanner files):

```text
# Run ensemble inference
results = ens.predict_volume(
    dwi_path, bval_path, bvec_path,
    mask_path=mask_path,
    output_dir="output/",
)

# Flag OOD voxels
import nibabel as nib
dwi = nib.load(dwi_path)
mask = nib.load(mask_path).get_fdata().astype(bool)
signals_flat = dwi.get_fdata()[mask]
# (b0-normalise signals_flat exactly as the ModelSimulator does)

ood_result = detector.flag_volume(
    jnp.array(signals_flat),
    threshold_percentile=95.0,
)
ood_result.print_summary()

# Apply conformal intervals to ensemble predictions
mean_pred, _ = ens.predict_mean(jnp.array(signals_flat))
intervals = predict_intervals_from_predictions(
    np.asarray(mean_pred), conformal_result
)
```

On the simulated "volume" from the setup section the same three calls look
like this:

```python
ood_result = detector.flag_volume(signals, threshold_percentile=95.0)
mean_pred, _ = ens.predict_mean(signals)
intervals = predict_intervals_from_predictions(np.asarray(mean_pred), conformal_result)

print(f"OOD fraction on simulated data: {ood_result.ood_fraction:.1%}")
covered = (np.asarray(theta_obs) >= intervals["lower"]) & (np.asarray(theta_obs) <= intervals["upper"])
print(f"Conformal coverage per parameter: {np.round(covered.mean(0), 3)}")
assert intervals["lower"].shape == (n_voxels, theta_dim)
```

### Step 7: Report with Confidence

The final output for each parameter includes:

| Output | Source | Interpretation |
|--------|--------|----------------|
| Point estimate | `ens.predict_mean()` | Ensemble-averaged parameter value |
| Aleatoric uncertainty | `ens.predict_with_uncertainty()` | Irreducible noise in the data |
| Epistemic uncertainty | `ens.predict_with_uncertainty()` | Model uncertainty (reducible with more training data) |
| Conformal interval | `predict_intervals_from_predictions()` | Distribution-free coverage guarantee |
| OOD flag | `detector.flag_volume()` | Trustworthiness of this voxel's prediction |

**Clinical interpretation**: if a voxel is flagged OOD, the prediction
should be treated with caution regardless of its confidence interval. If
the conformal interval is wide, the parameter is poorly constrained by the
data. If the epistemic uncertainty is high but aleatoric is low, more
training data or a better model architecture could improve the estimate.

---

## References

- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
  Validating Bayesian inference algorithms with simulation-based calibration.
  *arXiv:1804.06788*.
- Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized Quantile
  Regression. *NeurIPS*.
- Angelopoulos, A. N. & Bates, S. (2023). Conformal Prediction: A Gentle
  Introduction. *Foundations and Trends in Machine Learning*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and
  Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
- Vovk, V., Gammerman, G., & Shafer, G. (2005). *Algorithmic Learning in a
  Random World*. Springer.

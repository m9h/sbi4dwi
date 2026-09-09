# CLAUDE.md — Agent Guide for SBI4DWI

This file orients new Claude Code agents working on SBI4DWI (formerly
dmipy-jax). Read this first before touching any code.

## What is this project?

**SBI4DWI** (Simulation-Based Inference for Diffusion-Weighted Imaging) is a
JAX-accelerated platform for diffusion MRI microstructure estimation. It
combines differentiable biophysical signal models, multi-fidelity physics
simulation, and neural posterior estimation to recover tissue properties
(axon diameter, cell density, fibre orientation) from diffusion-weighted MRI
data. The codebase spans:

- **Analytical forward models** (Ball, Stick, Zeppelin, Sphere, NODDI, SANDI)
- **Differentiable physics simulations** (FEM, Monte Carlo, SDE walkers)
- **Simulation-Based Inference (SBI)** pipeline (MDN + Normalizing Flows)
- **External oracle integration** (ReMiDi, MCMRSimulator.jl, DIPY)
- **Uncertainty quantification** (SBC, PPC, conformal prediction, OOD, ensembles)
- **Clinical deployment** (NIfTI volume inference, comparison framework)

The principle: **build where differentiable, wrap where not**. JAX models
handle gradient-based optimization. External simulators produce training
data at the HDF5 boundary.

## Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Arrays | JAX | GPU acceleration, autodiff |
| Modules | Equinox (`eqx.Module`) | Pytree-compatible neural nets & models |
| Optimisation | Optimistix / Optax | Deterministic fitting / stochastic training |
| ODEs/SDEs | Diffrax | Differentiable solvers for Bloch/diffusion |
| MCMC | BlackJAX | Bayesian inference (NUTS) |
| Flows | FlowJAX | Normalizing flow posteriors |
| IO | nibabel, h5py, DIPY | NIfTI, HDF5, gradient tables |
| Package management | **uv only** | No pip/conda/poetry — see below |

## Critical conventions

### uv only
All Python operations go through `uv`. Never use `pip`, `conda`, or bare
`python`. Use:
```bash
uv sync              # install deps
uv run python x.py   # run scripts
uv run pytest        # run tests
```

### Testing
```bash
uv run pytest tests/ --noconftest     # skip sybil-based conftest
uv run pytest tests/test_oracle.py -v --noconftest  # single file
```
The root `conftest.py` requires `sybil` (for markdown doctests). Use
`--noconftest` when running test files directly. The configured test paths
in pyproject.toml are `dmipy_jax/tests/` and `docs/tutorials/`.

### Equinox patterns
All differentiable objects (models, simulators) are `eqx.Module` subclasses.
They are immutable pytrees — use `eqx.tree_at` for updates, not `__setattr__`.
Plain Python classes (not eqx.Module) are used for pipeline orchestration
(`ModelSimulator`, `ComparisonRunner`, `LibraryGenerator`).

### Units
- b-values: **SI (s/m²)** internally. DIPY and FSL use s/mm² — convert with `/ 1e6`.
- Diffusivity: m²/s (e.g. 2.0e-9 for free water).
- Lengths: metres (e.g. 5e-6 for a 5 μm radius).
- The `JaxAcquisition` class stores b-values in SI.

### b0 normalisation
Training data is b0-normalised in `ModelSimulator.sample_and_simulate()`.
The same normalisation happens at inference time in `SBIPredictor.predict_volume()`.
These **must match** or the network sees different distributions at train vs deploy.

## Directory map

```
dmipy_jax/
├── acquisition.py           # JaxAcquisition (b-values, gradient dirs, delta/Delta)
├── signal_models/           # Analytical forward models (Ball, Stick, Sphere, etc.)
├── core/
│   ├── modeling_framework.py  # Model composition (compose_models)
│   ├── solvers.py             # Optimistix-based voxelwise fitting
│   └── surrogate.py           # Polynomial Chaos Expansion
├── simulation/
│   ├── mesh_sim.py            # FEM MatrixFormalismSimulator (spectral ROM)
│   ├── monte_carlo.py         # Ground-truth MC with SDF geometry
│   ├── differentiable_walker.py  # Differentiable confined Brownian walker
│   ├── simulator.py           # End-to-end SDE simulator (Diffrax)
│   ├── oracle.py              # [PLANNED — not yet written] OracleSimulator protocol (ABC)
│   └── oracles/               # [PLANNED — not yet written] registry + implementations
│       ├── __init__.py         #   get_oracle("dipy"|"remidi"|"mcmr")
│       ├── dipy_sim.py         #   DIPYMultiTensorOracle
│       ├── remidi.py           #   ReMiDiOracle (Python API + Docker)
│       └── mcmr.py             #   MCMROracle (Julia subprocess)
├── pipeline/
│   ├── simulator.py           # ModelSimulator (forward_fn + prior + noise)
│   ├── train.py               # train_sbi() → _NormalisedMDN / _NormalisedFlow
│   ├── config.py              # SBIPipelineConfig
│   ├── checkpoint.py          # save/load checkpoint (.eqx + .config.json)
│   ├── deploy.py              # SBIPredictor for NIfTI volume inference
│   ├── oracle_adapter.py      # [PLANNED — not yet written] OracleModelSimulator
│   ├── multi_fidelity.py      # [PLANNED — not yet written] train_multi_fidelity_sbi()
│   ├── comparison.py          # ComparisonRunner + SimulationComparisonRunner
│   ├── ensemble.py            # Multi-model ensemble inference
│   ├── conformal.py           # Conformal prediction intervals
│   ├── ood.py                 # Out-of-distribution detection
│   ├── ppc.py                 # Posterior predictive checks
│   ├── sbc.py                 # Simulation-based calibration
│   └── metrics.py             # Derived metrics (FA, MD, AD, RD)
├── library/
│   ├── storage.py             # SimulationLibrary (HDF5 + multi-contrast)
│   ├── generator.py           # LibraryGenerator (batch vmap generation)
│   ├── hybrid_generator.py    # HybridLibraryGenerator (multi-fidelity mixing)
│   ├── matcher.py             # DictionaryMatcher (cosine similarity)
│   └── derived_metrics.py     # tensor_to_fa_md, eigenvalues_to_ad_rd
├── inference/
│   ├── mdn.py                 # MixtureDensityNetwork
│   ├── flows.py               # Normalizing flow posterior
│   ├── mcmc.py                # BlackJAX NUTS sampler
│   ├── score_posterior.py     # Score-based diffusion posterior (NEW)
│   └── trainer.py             # Generic training utilities
├── fitting/                   # Classical parameter fitting
├── io/                        # Data loaders (BIDS, HCP, mesh, multi-TE; no SWC loader yet)
├── inverse/                   # AMICO-style linear inversions
└── viz/                       # Surface mapping, visualisation
```

## Key data flow

### SBI training pipeline
```
ModelSimulator                         train_sbi()
  ├── forward_fn(params, acq) → signal    │
  ├── prior_sampler(key, n) → params      │
  └── add_noise(key, signal) → noisy      │
        │                                  │
        ▼                                  ▼
  sample_and_simulate(key, n)         MDN or Flow
        │  (b0-normalise)                  │
        ▼                                  ▼
  (theta, noisy_signals)          _NormalisedMDN / _NormalisedFlow
                                           │
                                           ▼
                                    save_checkpoint()
                                           │
                                           ▼
                                    SBIPredictor.predict_volume()
```

### Oracle / multi-fidelity path
```
OracleSimulator.generate_library()
        │
        ▼
  SimulationLibrary (HDF5)
        │
        ▼
  OracleModelSimulator (adapter)
        │  (k-NN interpolation or neural emulator)
        │
        ▼
  train_sbi()   ←── or HybridLibraryGenerator mixes
                     analytical (70%) + oracle (30%)
```

### SimulationLibrary HDF5 schema
```
/params              (N, P) float32     — tissue parameters
/signals             (N, M) float32     — primary (diffusion) signal
/contrast_signals/   (group, optional)  — multi-contrast data
    /t1              (N, M) float32
    /t2              (N, M) float32
/source              (N,) bytes         — provenance labels
attrs:
    parameter_names  list of str
    contrast_channels list of str
    (arbitrary metadata)
```

## FEM simulator (mesh_sim.py)

`MatrixFormalismSimulator` uses spectral ROM on triangular surface meshes:
1. Constructs FEM stiffness (S) and mass (M) matrices from mesh geometry
2. Generalised eigendecomposition: S v = λ M v (via Cholesky transform)
3. Projects position matrices into eigenspace: V^T M_x V
4. Simulates PGSE via 3 matrix exponentials: pulse1 → gap → pulse2

Key features:
- `__call__(G_amp, delta, Delta, gradient_direction=...)` — arbitrary gradient direction
- `simulate_acquisition(acq)` — vmaps over all measurements in a JaxAcquisition
- `construct_fem_matrices_sparse()` — BCOO sparse for meshes >2K vertices

## Oracle system

> **Status (2026-09-08):** the oracle protocol below is the *intended*
> design. None of `simulation/oracle.py`, `simulation/oracles/`,
> `pipeline/oracle_adapter.py`, `pipeline/multi_fidelity.py` exist yet.
> The only substrate-oracle code in the tree is
> `dmipy_jax/validation/caterpillar.py` (`CATERPillarOracle`). See
> `docs/decisions/006-octopus-differentiable-substrate-landscape.md` §7.

The oracle protocol separates **simulation fidelity** from **JAX requirements**:

- `OracleSimulator` (ABC): `check_available()`, `generate_batch(params, acq)`, `generate_library(n, ...)`
- Implementations: `DIPYMultiTensorOracle`, `ReMiDiOracle`, `MCMROracle`
- Registry: `get_oracle("dipy")` lazy-imports the right class
- Adapter: `OracleModelSimulator` wraps any `SimulationLibrary` into the `ModelSimulator` interface

The `OracleModelSimulator.simulate()` method uses k-NN inverse-distance
interpolation (not single nearest-neighbour) and supports:
- Out-of-support detection (warns when queries are far from library coverage)
- Optional neural emulator (`fit_emulator()` trains a small MLP for smooth interpolation)

## Comparison framework

Two runners:
1. `ComparisonRunner` — method-level comparison (DIPY DTI vs SBI vs dictionary matching)
2. `SimulationComparisonRunner` — simulator-level comparison (analytical vs FEM vs MC vs oracle)

Both produce structured result objects with RMSE, correlation, SSIM, and
per-b-value error analysis.

## Recent development history

The project has been built incrementally over ~2 months:

1. **Core JAX port** — Equinox models, JaxAcquisition, Optimistix fitting
2. **SBI pipeline** — ModelSimulator → train_sbi → checkpoint → deploy
3. **Differentiable simulation** — FEM mesh_sim, Monte Carlo, SDE walkers
4. **UQ suite** — SBC, PPC, conformal prediction, OOD detection, ensembles
5. **Comparison runner** — Multi-method benchmarking framework
6. **Multi-fidelity oracles** — Oracle protocol, DIPY/ReMiDi/MCMR integration,
   hybrid library generation, multi-contrast storage
7. **Score-based posterior** — Score matching for diffusion posterior estimation

## What NOT to do

- Don't use `pip install` — use `uv add`
- Don't mutate `eqx.Module` fields — use `eqx.tree_at`
- Don't hardcode b-value units — always check SI vs FSL
- Don't skip b0 normalisation — train and deploy must match
- Don't make external simulators JAX dependencies — use the oracle boundary
- Don't run `pytest` without `--noconftest` unless `sybil` is installed


# SBI4DWI: Simulation-Based Inference for Diffusion-Weighted Imaging

[![CI](https://github.com/m9h/sbi4dwi/actions/workflows/ci.yml/badge.svg)](https://github.com/m9h/sbi4dwi/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-9cf)](https://github.com/google/jax)

**SBI4DWI** is a JAX-accelerated platform for diffusion MRI microstructure
estimation using simulation-based inference. It combines differentiable
biophysical signal models, multi-fidelity physics simulation, and neural
posterior estimation to recover tissue microstructure from diffusion-weighted
images.

The project originated as a JAX port of the
[dmipy](https://github.com/AthenaEPI/dmipy) signal model library but has
grown into a full research platform spanning forward simulation, amortized
inference, uncertainty quantification, and clinical deployment.

## Architecture

```
                  ┌─────────────────────────────────┐
                  │        Signal Models             │
                  │  Ball, Stick, Zeppelin, Sphere,  │
                  │  NODDI, SANDI, IVIM, EPG, QMT    │
                  └───────────────┬─────────────────┘
                                  │ forward_fn(params, acq) → signal
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
   ┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
   │  Analytical   │     │  Differentiable  │     │    External     │
   │  (closed-form)│     │  Simulation      │     │    Oracles      │
   │               │     │  FEM, MC, SDE    │     │  DIPY, ReMiDi,  │
   │               │     │  (Diffrax)       │     │  MCMRSimulator   │
   └──────┬───────┘     └────────┬────────┘     └───────┬────────┘
          │                      │                      │
          ▼                      ▼                      ▼
   ┌──────────────────────────────────────────────────────────┐
   │              ModelSimulator / SimulationLibrary           │
   │        prior → forward → noise → (theta, signal) pairs   │
   │                    HDF5 multi-contrast storage            │
   └────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
   ┌──────────────────────────────────────────────────────────┐
   │                  Neural Posterior Estimation              │
   │     MDN  ·  Normalizing Flows  ·  Score-Based Diffusion  │
   │             MCMC (NUTS)  ·  Amortized Inference           │
   └────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
   ┌──────────────────────────────────────────────────────────┐
   │              Uncertainty Quantification                   │
   │  SBC  ·  PPC  ·  Conformal Prediction  ·  OOD Detection  │
   │              Ensembles  ·  Calibration                    │
   └────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
   ┌──────────────────────────────────────────────────────────┐
   │                  Clinical Deployment                      │
   │       SBIPredictor  ·  NIfTI volume inference             │
   │       ComparisonRunner  ·  Derived metrics (FA, MD)       │
   └──────────────────────────────────────────────────────────┘
```

## Results

Validated on synthetic data and real **WAND** Siemens Connectom (300 mT/m) acquisitions.

### Normalizing Flow NPE — Ball+2Stick orientation estimation

Neural spline flow posterior trained on simulated multi-shell dMRI.
Achieves near-target orientation accuracy matching
Manzano-Patron et al. (2025).

| Config | Fiber 1 median | d_stick r | f1 r | Steps |
|:-------|:---------------|:----------|:-----|:------|
| Baseline affine flow | 10.7 deg | 0.95 | 0.85 | 30k |
| + spline + noise fix + label-switching | 6.4 deg | 0.978 | 0.902 | 30k |
| + 200k steps | 3.2 deg | 0.986 | 0.935 | 200k |
| + 300k steps | **2.8 deg** | **0.987** | **0.943** | 300k |

Key: spline transformer, train/test noise matching (Rician + b0-norm),
label-switching fix in prior (f1 >= f2), variable SNR augmentation (10-50).

### Score-Based Posterior — denoising score matching + DDPM

MLP with FiLM conditioning, trained via denoising score matching.
Spherical coordinate parameterization eliminates unit-sphere constraint.

| Config | Fiber 1 median | Notes |
|:-------|:---------------|:------|
| MLP + SDE sampler | 29 deg | SDE diverges |
| MLP + DDPM sampler | 15.5 deg | DDPM critical |
| + v-prediction, 1024-wide, 100k | 14.9 deg | Scale up |
| + spherical coords | **12.8 deg** | Best score-based |

### Transcranial Focused Ultrasound — SCI Head Model (A100, April 2026)

Differentiable skull-corrected acoustic simulation via j-Wave on the
SCI Institute head model (208x256x256, 1mm, 8 tissue types). Full 7-experiment
suite validated on Modal A100. [Proposal for Openwater Health](docs/openlifu/).

| Experiment | Result |
|-----------|--------|
| Skull attenuation (400kHz) | **93%** through cortical bone |
| 32-element array optimization | **1.4x** focal improvement (20 iters, 1.8s/iter) |
| Multi-target (cortex→thalamus) | 53-55% depth-dependent attenuation |
| Frequency (180kHz vs 1MHz) | **180kHz best penetration** (32% vs 55% loss) |
| Sensitivity (dp/dc) | Skull region **232x** more sensitive than water |
| Grid convergence | 3 resolutions verified (0.4/0.2/0.1mm) |
| Helmholtz vs time-domain | **r=0.82** correlation, good solver agreement |
| **3D volumetric (52x64x64)** | **94.3%** attenuation, 13s total on A100 |
| **3D 16-element optimization** | **4.7x** focal improvement, 2.6s/iter |

Covers the full device range from 4-element entry-level to 256-element
Oxford-UCL MRI-TFUS helmet (Martin, Stagg, Treeby — Nature Comms 2025).
Vision: multimodal MRI-TFUS-ARFI simulation connecting acoustic fields
to MRI-visible tissue displacement via radiation force elastodynamics.

Integration with [OpenLIFU](https://github.com/OpenwaterHealth/openlifu-python)
via [`HeterogeneousSkullSegmentation`](https://github.com/m9h/openlifu-python/tree/feature/heterogeneous-skull-segmentation).

Run all experiments: `modal run scripts/modal_experiments.py` |
3D validation: `modal run scripts/modal_3d_validation.py`

### Experiment tracking

All results tracked via [Trackio](https://huggingface.co/docs/trackio)
and logged to HuggingFace Hub. Full leaderboard in the companion Julia
package [DMI.jl](https://github.com/m9h/dmijl).

---

## Core Capabilities

### Differentiable Signal Models

Analytical biophysical models implemented as `eqx.Module` pytrees, fully
compatible with `jax.grad`, `jax.vmap`, and `jax.jit`:

| Model | Description |
|-------|-------------|
| Ball, Stick, Zeppelin, Tensor | Standard Gaussian compartments |
| Sphere (GPD, Callaghan, Stejskal-Tanner) | Restricted diffusion in spheres |
| Cylinder (Soderman, Callaghan) | Restricted diffusion in cylinders |
| Plane (Callaghan, Stejskal-Tanner) | Restricted diffusion between planes |
| NODDI / Multi-TE NODDI | Neurite orientation dispersion |
| SANDI | Soma and neurite density |
| IVIM | Intravoxel incoherent motion |
| Free-Water DTI | Free water elimination |
| mcDESPOT | Multi-component relaxometry |
| EPG | Extended phase graphs |
| QMT | Quantitative magnetisation transfer |
| Stimulated-Echo Karger | Exchange-weighted imaging |
| Neural CSD | Neural constrained spherical deconvolution |

Models compose via `compose_models()` with volume fractions, orientation
distributions (Watson, Bingham), and tortuosity constraints.

### Differentiable Physics Simulation

| Simulator | Method | Differentiable |
|-----------|--------|:-:|
| `MatrixFormalismSimulator` | FEM spectral ROM on surface meshes | Yes |
| `MonteCarloSimulator` | Random walk with SDF geometry | No (ground truth) |
| `DifferentiableWalker` | Confined Brownian motion | Yes |
| `SDESimulator` | Diffrax-based SDE integration | Yes |
| `jwave_adapter` | j-Wave pseudospectral acoustic simulation | Yes |
| `tus_optimizer` | Gradient-based TUS delay optimization | Yes |
| `radiation_force` | F=2αI/c + spectral displacement solver (Kuhl CANN) | Yes |
| `fwi_us` | Full waveform inversion for ultrasound (acoustic EIT) | Yes |
| `mr_arfi` | MR-ARFI sequence design + phase prediction (Butts Pauly) | Yes |
| `multimodal_tus` | End-to-end acoustic → force → displacement → MRI phase | Yes |

The FEM simulator constructs stiffness/mass matrices from triangular meshes,
solves a generalised eigendecomposition, and simulates PGSE sequences via
matrix exponentials. Supports arbitrary gradient directions and full
`JaxAcquisition` protocols.

### External Oracle Integration

Non-differentiable simulators wrapped behind a common protocol for
multi-fidelity training data generation:

| Oracle | Backend | Interface |
|--------|---------|-----------|
| `DIPYMultiTensorOracle` | DIPY | Python API |
| `ReMiDiOracle` | ReMiDi | Docker / Python API |
| `MCMROracle` | MCMRSimulator.jl | Julia subprocess |

Oracles produce `SimulationLibrary` (HDF5) datasets. The
`OracleModelSimulator` adapter bridges any library into the SBI training
pipeline via k-NN inverse-distance interpolation or an optional neural
emulator.

### SBI Training Pipeline

```python
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.pipeline.deploy import SBIPredictor

# 1. Define forward model + prior
sim = ModelSimulator(forward_fn, prior_sampler, noise_model)

# 2. Train neural posterior
posterior = train_sbi(sim, n_simulations=100_000, method="flow")

# 3. Deploy on NIfTI volumes
predictor = SBIPredictor.from_checkpoint("model.eqx")
results = predictor.predict_volume(dwi_img, bvals, bvecs, mask)
```

Supported posterior estimators:

- **Mixture Density Networks** (MDN) with Gaussian mixtures
- **Normalizing Flows** via FlowJAX (spline coupling, neural spline)
- **Score-Based Diffusion** with E(3)-equivariant orientation heads
- **MCMC** via BlackJAX NUTS for full Bayesian posteriors
- **Amortized Variational Inference**

### Multi-Fidelity Training

Mix analytical and oracle-generated data to balance speed and accuracy:

```python
from dmipy_jax.library.hybrid_generator import HybridLibraryGenerator

hybrid = HybridLibraryGenerator(
    analytical_sim,     # fast, differentiable (70%)
    oracle_library,     # slow, high-fidelity  (30%)
)
```

The `train_multi_fidelity_sbi()` function validates against oracle ground
truth and reports fidelity-stratified metrics.

### Uncertainty Quantification

| Method | Module | Purpose |
|--------|--------|---------|
| Simulation-Based Calibration | `pipeline/sbc.py` | Posterior coverage diagnostics |
| Posterior Predictive Checks | `pipeline/ppc.py` | Model adequacy |
| Conformal Prediction | `pipeline/conformal.py` | Distribution-free intervals |
| OOD Detection | `pipeline/ood.py` | Flag out-of-support inputs |
| Ensembles | `pipeline/ensemble.py` | Multi-model uncertainty |

### Comparison Framework

Benchmark methods and simulators against each other:

- `ComparisonRunner` — compare DIPY DTI vs SBI vs dictionary matching
- `SimulationComparisonRunner` — compare analytical vs FEM vs Monte Carlo vs oracle

Both produce structured results with RMSE, correlation, SSIM, and
per-b-value error breakdowns.

### Additional Modules

| Module | Description |
|--------|-------------|
| `biophysics/` | Axon conduction delays, neural dynamics (VBJ integration), conductivity mapping |
| `design/` | Optimal experimental design via Fisher information and expected information gain |
| `bayesian/` | Variational inference (NumPyro), Bayesian model discovery |
| `nn/` | E(3)-equivariant score networks, constitutive relation networks |
| `pulseq/` | Bloch equation simulation from PyPulseq sequences |
| `core/surrogate.py` | Polynomial Chaos Expansion for fast surrogate models |
| `core/pinns.py` | Physics-informed neural networks for diffusion PDEs |
| `core/tensor_train.py` | Tensor-train decomposition (ttax) |
| `fitting/` | Neural exchange fitting, algebraic initialisation, AMICO inversion |
| `io/` | BIDS, HCP, IXI, BigMac, WAND, multi-TE, mesh, SWC loaders |
| `viz/` | Surface mapping and visualisation |
| `cli/` | BIDS reporting tool (`dmipy-report`) |

## Companion Project

**[DMI.jl](https://github.com/m9h/dmijl)** — Julia implementation using
the SciML stack (Lux.jl, DifferentialEquations.jl). Features:

- AxCaliber PINN recovering **axon radius R = 3.15 um** from real WAND
  Connectom data via Van Gelderen restricted diffusion
- Neural diffusion tensor field fitting (**MD = 0.74 um^2/ms, FA = 0.42** on CHARMED)
- Native SDE/ODE samplers via DifferentialEquations.jl
- Cross-validated against Microstructure.jl (Ting Gong, MGH/Martinos) at machine precision

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/m9h/sbi4dwi.git
cd sbi4dwi
uv sync
```

For GPU support, ensure CUDA 13+ drivers are installed. JAX will
automatically detect the GPU.

### Optional: Oracle simulators

```bash
# ReMiDi (Docker)
docker build -f docker/Dockerfile.remidi -t remidi .

# MCMRSimulator.jl (Julia)
docker build -f docker/Dockerfile.mcmr -t mcmr .
```

## Usage

### Fit a multi-compartment model to a voxel

```python
from dmipy_jax.core.acquisition import JaxAcquisition
from dmipy_jax.signal_models.cylinder_models import C1Stick
from dmipy_jax.signal_models.gaussian_models import G1Ball
from dmipy_jax.core.modeling_framework import compose_models
from dmipy_jax.core.solvers import fit_voxel

model = compose_models([C1Stick(), G1Ball()])
params = fit_voxel(model, acquisition, signal)
```

### Train an SBI posterior and deploy on a NIfTI volume

```python
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.pipeline.deploy import SBIPredictor

sim = ModelSimulator(forward_fn, prior_sampler, noise_model)
posterior = train_sbi(sim, n_simulations=200_000, method="flow")
predictor = SBIPredictor(posterior, acquisition)
results = predictor.predict_volume(dwi_nifti, mask=brain_mask)
```

### Run a FEM simulation on a mesh

```python
from dmipy_jax.simulation.mesh_sim import MatrixFormalismSimulator

fem = MatrixFormalismSimulator.from_mesh(vertices, triangles, D=2e-9)
signal = fem.simulate_acquisition(acquisition)
```

## Testing

```bash
# Full suite (dmipy_jax/tests + tests); add JAX_PLATFORMS=cpu to stay off the GPU
uv run pytest

# Single file
uv run pytest tests/test_oracle.py -v

# Markdown tutorials as doctests (needs sybil)
uv run pytest docs/tutorials
```

## Tech Stack

| Layer | Library | Role |
|-------|---------|------|
| Arrays & autodiff | JAX | GPU acceleration, automatic differentiation |
| Neural modules | Equinox | Pytree-compatible `eqx.Module` models |
| Optimisation | Optimistix / Optax | Deterministic fitting / stochastic training |
| ODE/SDE solvers | Diffrax | Differentiable Bloch and diffusion simulation |
| MCMC | BlackJAX | Bayesian inference (NUTS) |
| Normalizing flows | FlowJAX | Neural posterior estimation |
| Equivariance | e3nn-jax | E(3)-equivariant score networks |
| Tensor decomposition | ttax | Tensor-train approximation |
| MRI I/O | nibabel, DIPY | NIfTI images, gradient tables |
| Data storage | h5py | HDF5 simulation libraries |
| Package management | uv | Dependency resolution and virtual environments |

## Project Structure

```
dmipy_jax/
├── signal_models/      Analytical forward models (Ball, Stick, Sphere, ...)
├── core/               Model composition, solvers, acquisition, PINNs, surrogates
├── simulation/         FEM, Monte Carlo, SDE walkers, oracle protocol
│   └── oracles/        DIPY, ReMiDi, MCMRSimulator.jl wrappers
├── pipeline/           SBI training, checkpointing, deployment, UQ, comparison
├── inference/          MDN, flows, score posterior, MCMC, amortized
├── library/            HDF5 storage, hybrid generation, dictionary matching
├── models/             Pre-composed models (NODDI, SANDI, mcDESPOT, EPG, ...)
├── biophysics/         Neural dynamics, conduction delays, conductivity
├── fitting/            Neural exchange fitting, algebraic initialisation
├── bayesian/           Variational inference, model discovery
├── nn/                 Equivariant networks, constitutive relations
├── design/             Optimal experimental design
├── io/                 BIDS, HCP, mesh, SWC, multi-TE loaders
├── pulseq/             Bloch simulation from pulse sequences
├── viz/                Surface mapping, visualisation
├── tests/              Module-level test suite
└── examples/           Worked examples by domain
```

## Relationship to dmipy

SBI4DWI builds on the signal model foundations of
[dmipy](https://github.com/AthenaEPI/dmipy) (Fick, Wassermann & Deriche,
2019). The analytical compartment models (Ball, Stick, Zeppelin, Sphere,
Cylinder) are JAX reimplementations of dmipy's NumPy originals using Equinox
modules. Everything else — the SBI pipeline, differentiable simulation,
neural posteriors, oracle integration, UQ framework, and clinical deployment
tools — is new.

If you use the signal models, please cite the original dmipy paper:

> Rutger Fick, Demian Wassermann and Rachid Deriche, "The Dmipy Toolbox:
> Diffusion MRI Multi-Compartment Modeling and Microstructure Recovery Made
> Easy", *Frontiers in Neuroinformatics* 13 (2019): 64.

The SBI pipeline and orientation estimation targets follow:

> Jose P. Manzano-Patron, Michael Deistler, Cornelius Schroder, Theodore
> Kypraios, Pedro J. Goncalves, Jakob H. Macke and Stamatios N.
> Sotiropoulos, "Uncertainty mapping and probabilistic tractography using
> Simulation-Based Inference in diffusion MRI", *Medical Image Analysis*
> 103 (2025): 103580. DOI:
> [10.1016/j.media.2025.103580](https://doi.org/10.1016/j.media.2025.103580)

## License

MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2017 Rutger Fick & Demian Wassermann (original dmipy signal models)
Copyright (c) 2024-2026 Morgan Hough (SBI4DWI platform)

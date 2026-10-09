# SBI4DWI

Diffusion-MRI microstructure in JAX: differentiable compartment models
(ported from dmipy and tested against its closed forms), the PRISM-JAX
multi-fixel refinement with a calibrated fixel posterior that plugs into DIPY,
simulation-based inference for amortised posteriors, and Fisher-information
acquisition design. The import name is `dmipy_jax`.

Start with the README quickstart, then the tutorials. The validated results
and how they were obtained are in the decision records, especially doc 008.

```{toctree}
:maxdepth: 1
:caption: Getting started

tutorials/first_steps
tutorials/model_composition
tutorials/simulation_basics
tutorials/complex_synthetic_data
```

```{toctree}
:maxdepth: 1
:caption: Simulation-based inference

tutorials/sbi_dti
tutorials/training_to_deployment
tutorials/sbi_noddi
tutorials/normalizing_flows
tutorials/uncertainty_quantification
```

```{toctree}
:maxdepth: 1
:caption: Validation results and decisions

decisions/README
decisions/008-positioning-vs-force-prism-sbi
decisions/009-graves-orbit-resolve-design
decisions/010-package-review-and-cleanup-plan
decisions/006-octopus-differentiable-substrate-landscape
```

```{toctree}
:maxdepth: 1
:caption: Project

artifacts
external_projects
bibliography
```

```{toctree}
:maxdepth: 2
:caption: API reference

reference/dmipy_jax
```

## Conventions in one paragraph

Everything is SI: b-values in s/m² (1000 s/mm² = 1e9), diffusivities in m²/s,
lengths in metres. Models are immutable Equinox pytrees, updated with
`eqx.tree_at`. Training data and deployment are b0-normalised in the same
place (`ModelSimulator` and `SBIPredictor`), never by hand in between.
External simulators (CATERPillar, MCMRSimulator.jl) stay behind the HDF5
oracle boundary; nothing non-differentiable is a JAX dependency.

- {ref}`genindex`
- {ref}`modindex`

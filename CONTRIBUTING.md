# Contributing to SBI4DWI

SBI4DWI (Simulation-Based Inference for Diffusion-Weighted Imaging) is a
JAX-accelerated platform for diffusion MRI microstructure estimation. It
combines differentiable biophysical signal models, multi-fidelity physics
simulation, and neural posterior estimation to recover tissue properties from
diffusion-weighted MRI data.

We welcome contributions across the full stack: signal models, simulation
methods, inference pipelines, uncertainty quantification, and documentation.
This guide covers the conventions and workflow we follow.

---

## 1. Getting Started

### Prerequisites

- **Python 3.12+** (see `requires-python` in `pyproject.toml`)
- **uv** for all package management. Do not use `pip`, `conda`, or `poetry`.
  Install uv: <https://docs.astral.sh/uv/getting-started/installation/>
- **Git**

### Clone and install

```bash
git clone https://github.com/<org>/sbi4dwi.git
cd sbi4dwi
uv sync                          # install all deps
uv sync --extra test --extra doc # explicit extras if needed
```

`uv sync` resolves everything from `pyproject.toml` and `uv.lock`. The project
is automatically installed in editable mode.

### Verify the installation

```bash
uv run pytest                         # run tests
uv run python -c "import dmipy_jax"   # import smoke test
```

The root `conftest.py` enables sybil markdown doctests only when sybil is
installed and only for paths you pass explicitly (`uv run pytest docs/tutorials`).

---

## 2. Development Workflow

### Branching

Create a branch from `main` using one of these prefixes:

| Prefix       | Purpose                                      |
|--------------|----------------------------------------------|
| `feature/`   | New models, inference methods, or pipelines  |
| `fix/`       | Bug fixes                                    |
| `docs/`      | Documentation-only changes                   |
| `refactor/`  | Internal restructuring, no new API           |

Example: `feature/score-posterior`, `fix/b0-normalisation-mismatch`.

### Commits

Write concise commit messages. Follow the prefix conventions visible in the
project history:

```
feat: commit all experimental modules + clean gitignore for public release
fix: code review remediation — numerical safety, dead code, eqx.Module consistency
docs: replace Nottingham shorthand with Manzano-Patron et al. (2025) citation
chore: clean root level — move 3.6GB data to /data, organize scripts
```

- Use prefixes: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `results:`
- Keep the first line under 72 characters
- Add detail in the commit body when the change is complex

### Pull requests

- One logical change per PR.
- Descriptive title. Use the body to explain *why*, not just *what*.
- Link related issues with `Closes #N` or `Relates to #N`.
- Ensure `pytest` passes before requesting review.
- Request review from at least one maintainer.

---

## 3. Code Style

### Docstrings

Use **Google-style docstrings** (parsed by Napoleon/sphinx-autodoc-typehints):

```python
def compose_models(models: list[eqx.Module],
                   fractions: Float[Array, "K-1"]) -> Float[Array, "M"]:
    """Compose multiple compartment models into a multi-compartment signal.

    Computes the volume-fraction-weighted sum of compartment signals.
    Fractions are passed through a softmax to ensure they sum to 1.

    Parameters:
        models: List of K compartment signal models.
        fractions: K-1 unconstrained fraction parameters (softmax applied).

    Returns:
        Combined signal vector of length M (number of measurements).

    Raises:
        ValueError: If len(models) < 2.

    See: Fick et al. (2019), "The Dmipy Toolbox", Frontiers in Neuroinformatics.
    """
```

### Type annotations

- Use `jaxtyping` for array shape annotations:
  ```python
  from jaxtyping import Float, Array
  def simulate_signal(D: Float[Array, ""],
                      acq: JaxAcquisition) -> Float[Array, "M"]:
  ```
- Annotate all public function signatures.
- Shape dimensions should use meaningful names: `M` for measurements, `N` for
  samples, `P` for parameters, `K` for compartments.

### Module conventions

- **Module-level docstrings** are required for all new files.
- **Equinox modules** (`eqx.Module`) for all differentiable, JIT-compatible
  objects (signal models, simulators, neural networks). They are immutable
  pytrees -- use `eqx.tree_at` for updates, never `__setattr__`.
- **Plain Python classes** for pipeline orchestration (`ModelSimulator`,
  `ComparisonRunner`, `LibraryGenerator`).
- Follow existing patterns in `dmipy_jax/` -- look at neighbouring modules
  for conventions.

### Units

This project has strict unit conventions. Violations cause silent numerical
errors:

| Quantity     | Unit    | Example               |
|--------------|---------|-----------------------|
| b-values     | s/m^2   | 1e9 (= 1000 s/mm^2)  |
| Diffusivity  | m^2/s   | 2.0e-9 (free water)   |
| Lengths      | metres  | 5e-6 (5 um radius)    |

DIPY and FSL use s/mm^2 for b-values -- convert with `/ 1e6` at the boundary.
The `JaxAcquisition` class always stores b-values in SI.

---

## 4. Testing

### Running tests

```bash
uv run pytest                                  # full suite
uv run pytest tests/test_comparison.py -v      # single file
uv run pytest -k "test_mdn"                    # by name
JAX_PLATFORMS=cpu uv run pytest                # CPU only (CI does this)
```

Coverage is opt-in: `uv run pytest --cov=dmipy_jax --cov-report=term-missing`
(configured in `pyproject.toml`).

### Test requirements

- **All new features need tests.** No exceptions.
- **All bug fixes need a regression test** that fails without the fix.
- Use `pytest` fixtures for shared setup (acquisitions, PRNGKeys, test data).
- Place test files in `tests/` with the naming pattern `test_<module>.py`.

### JAX-specific testing

JAX code needs targeted tests beyond standard unit tests:

- **JIT compilation**: verify that functions work under `jax.jit` (catches
  side effects and Python-mode-only logic).
- **vmap compatibility**: test that batched signal simulation produces the same
  results as manual loops over voxels.
- **Gradient correctness**: use `jax.grad` on fitting objectives and check
  against finite differences (`jax.test_util.check_grads`).
- **b0 normalisation consistency**: ensure training and deployment pipelines
  apply identical normalisation.
- **Numerical stability**: test with `jax.config.update("jax_enable_x64", True)`
  for diffusivity values near machine epsilon.
- **Determinism**: fix `jax.random.PRNGKey` seeds for reproducible tests.

### Oracle tests

Tests that call external simulators (DIPY, ReMiDi, MCMRSimulator.jl) should:
- Be marked with `@pytest.mark.slow` or skip if the backend is unavailable.
- Use small library sizes for CI speed.
- Validate against the `OracleSimulator` protocol ABC.

---

## 5. Documentation

### Where docs live

```
docs/
├── conf.py                # Sphinx configuration
├── index.md               # Landing page
├── tutorials/             # MyST markdown tutorials
├── reference/             # API reference (auto-generated)
├── theory/                # Mathematical background
├── references.bib         # BibTeX bibliography
└── Makefile               # Build target
```

### Writing docs

- **Tutorials** go in `docs/tutorials/` as MyST markdown (`.md`) files.
- **Theory pages** go in `docs/theory/` for mathematical derivations.
- **API reference** is auto-generated via `sphinx-autodoc-typehints`. Write
  good docstrings and the API docs follow.
- **Cross-references** use MyST syntax: `` {func}`dmipy_jax.signal_models.ball.Ball` ``.
- **Citations**: add entries to `docs/references.bib` and cite with
  `` {cite}`AuthorYear` ``.

### Build locally

```bash
cd docs && make html
open _build/html/index.html
```

---

## 6. AI-Assisted Development

We actively use AI coding agents (Claude Code, Copilot, etc.) and have
found them valuable for accelerating development. This section defines our
conventions for responsible AI-assisted contribution.

### CLAUDE.md conventions

This project maintains a `CLAUDE.md` at the repository root that orients
AI agents. Keep it up to date as the project evolves. It should contain:

- **Project summary** -- what the project does, in 2-3 sentences.
- **Tech stack table** -- key libraries and why they are used.
- **Critical conventions** -- things an agent must not violate (e.g., "uv only",
  "Equinox immutability", unit conventions, b0 normalisation rules).
- **Directory map** -- one-level layout of the source tree with brief
  descriptions.
- **Key data flows** -- diagrams of the SBI training pipeline, oracle path, etc.
- **What NOT to do** -- explicit anti-patterns.

`CLAUDE.md` is not documentation for users -- it is a machine-readable project
briefing for AI agents.

### Writing agent-friendly code

AI agents navigate code through docstrings, type hints, and naming. Help them
help you:

- Write **complete Google-style docstrings** on all public functions and classes.
- Use **jaxtyping shape annotations** so agents understand array semantics
  (e.g., `Float[Array, "N M"]` for N samples of M measurements).
- Use **descriptive names** -- `simulate_pgse_signal` not `sim`, `n_measurements`
  not `nm`.
- Keep **module-level docstrings** that explain what the file is for and how it
  fits into the pipeline.
- Avoid deeply nested logic; prefer small, composable functions.
- Document unit conventions in docstrings when a function handles physical
  quantities.

### Reviewing AI-generated code

All AI-generated code must pass human review. Pay special attention to:

- **Hallucinated imports or APIs** -- agents may invent function names or use
  deprecated interfaces. Verify every import against the actual codebase and
  dependency versions.
- **Unnecessary abstractions** -- agents sometimes over-engineer with extra
  classes or patterns. Prefer simplicity.
- **Security issues** -- watch for hardcoded paths, leaked credentials, or
  unsafe deserialization (especially around HDF5 and NIfTI I/O).
- **Test quality** -- agents write tests that pass but may not test meaningful
  behavior. Check that assertions are substantive, not tautological.
- **Scientific correctness** -- this is the most critical failure mode. Agents
  do not understand diffusion physics, microstructure geometry, or MRI signal
  formation. Verify formulas, units, and numerical ranges against published
  references. Check that b-value units are SI throughout.
- **b0 normalisation** -- training and deployment must apply identical
  normalisation. Agents may introduce inconsistencies here.

### Commit attribution

When an AI agent contributed substantially to a commit, add a co-author trailer:

```
feat: add score-based diffusion posterior estimation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

This is informational, not a transfer of responsibility. The human committer
is accountable for the code.

### What agents do well

- Generating boilerplate (test scaffolding, docstrings, `__init__.py` exports)
- Refactoring and renaming across files
- Writing documentation and tutorial drafts
- Translating mathematical formulas into JAX implementations
- Implementing new signal models from published equations
- Code review checklists and static analysis
- Wiring up pipeline components (checkpoint save/load, CLI arguments)

### What needs human oversight

- **Architecture decisions** -- module boundaries, API design, dependency choices
- **Performance-critical JAX transforms** -- custom `jit` partitioning, `vmap`
  axis management for batched voxel inference, `pmap` sharding strategies
- **Scientific correctness** -- signal model equations, biophysical parameter
  ranges, diffusion tensor mathematics, SBI posterior calibration
- **Security-sensitive code** -- authentication, file I/O paths, Docker
  integration (ReMiDi), subprocess calls (MCMRSimulator.jl)
- **Unit consistency** -- b-values (SI vs s/mm^2), diffusivities, length scales
- **Numerical stability** -- signal models near b=0 or extreme diffusivities,
  log-likelihood computations, normalizing flow training

### PR labels

Tag pull requests that were substantially AI-assisted with the `ai-assisted`
label. This helps with auditing and lets reviewers know to apply extra scrutiny
to the areas listed above.

---

## 7. Scientific Standards

SBI4DWI implements published methods from the diffusion MRI microstructure
literature. We hold contributions to a high scientific bar.

### Citations

- Reference the original method paper in docstrings and module-level docs.
  Use the format: `See: Author et al. (Year), "Title", Journal. DOI: ...`
- Add BibTeX entries to `docs/references.bib`.
- If implementing a variant or extension, cite both the original and the
  modification.
- Key references for this project:
  - Fick et al. (2019), "The Dmipy Toolbox", Frontiers in Neuroinformatics
  - Manzano-Patron et al. (2025) -- microstructure estimation methods
  - Panagiotaki et al. (2012), "Compartment models of the diffusion MR signal"

### Mathematical notation

Include mathematical notation in docstrings where it aids understanding:

```python
def ball_signal(D: Float[Array, ""],
                bvals: Float[Array, "M"]) -> Float[Array, "M"]:
    """Compute the isotropic Ball compartment signal.

    Implements the mono-exponential decay:

        S(b) = exp(-b * D)

    where b is the b-value in s/m^2 and D is the isotropic diffusivity
    in m^2/s.

    Parameters:
        D: Isotropic diffusivity (m^2/s). Typical range: [0.5e-9, 3.0e-9].
        bvals: b-values in SI units (s/m^2).

    Returns:
        Normalised signal attenuation for each b-value.

    See: Behrens et al. (2003), "Characterization and propagation of
         uncertainty in diffusion-weighted MR imaging", MRM.
    """
```

### Validation

- Validate new signal models against reference implementations (dmipy, DIPY,
  AMICO) and/or Monte Carlo ground truth.
- Run simulation-based calibration (SBC) when adding new inference methods.
- Report numerical agreement (correlation, RMSE, max absolute error) in PR
  descriptions.
- Include validation scripts in `validation/` or `examples/`.
- For multi-fidelity work, validate the oracle path against analytical models
  on simple geometries before moving to complex microstructures.

---

## Tech Stack Reference

| Layer           | Tool                  | Purpose                                    |
|-----------------|-----------------------|--------------------------------------------|
| Arrays          | JAX                   | GPU acceleration, autodiff                 |
| Modules         | Equinox               | Pytree-compatible models and neural nets   |
| Optimisation    | Optimistix / Optax    | Deterministic fitting / stochastic training|
| ODEs/SDEs       | Diffrax               | Differentiable Bloch/diffusion solvers     |
| MCMC            | BlackJAX              | Bayesian inference (NUTS)                  |
| Flows           | FlowJAX              | Normalizing flow posteriors                |
| IO              | nibabel, h5py, DIPY   | NIfTI, HDF5, gradient tables               |
| Package mgmt    | **uv only**           | No pip/conda/poetry                        |

---

Thank you for contributing to SBI4DWI. If you have questions, open an issue or
start a discussion on the repository.

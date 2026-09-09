# External Projects Reference

Projects that were previously vendored or referenced in this repository.
Removed during repo cleanup (2026-03-25) to reduce tracked size.

## dmipy (Bayesian fork)

- **Source**: https://github.com/AthenaEPI/dmipy
- **License**: MIT (Rutger Fick & Demian Wassermann, 2017); separate
  LICENSE-BAYESIAN for Bayesian fitting additions
- **Version vendored**: 1.0.5 (with Bayesian extensions)
- **Was located at**: `benchmarks/external/dmipy_bayesian/`
- **Purpose**: Legacy benchmarking baseline. Contained the original dmipy
  package plus Bayesian fitting notebooks (`fit_bayes.py`,
  `bayesian-fitting-HCP-example.ipynb`) used to compare against dmipy-jax
  SBI posteriors. Included pre-built eggs (`dmipy-1.0.5-py2.7.egg`,
  `dmipy-1.0.5-py3.9.egg`), Camino simulation data, and example notebooks.
- **Why removed**: 308 tracked files (~113 MB), including 54 MB of egg
  distributions. The upstream repo is publicly available and the comparison
  is reproducible by installing `dmipy==1.0.5` from PyPI or the GitHub repo.

## CATERPillar

- **Source**: https://github.com/jazz031195/CATERPillar
  (Nguyen-Duc J et al., CHUV/EPFL — bioRxiv 10.1101/2025.06.20.660694,
  PubMed 41576825). Not `RafaelNH/CATERPillar` — that GitHub user is Rafael
  Neto Henriques (DIPY), a name collision with author Jonathan Rafael-Patiño.
- **Was located at**: `vendor/CATERPillar/` (this repo — placeholder only).
- **Live checkout**: `/home/mhough/dev/dmipy/vendor/CATERPillar`, cloned and
  compiled. `dmipy_jax/validation/caterpillar.py` (`CATERPillarOracle`)
  defaults to that binary path.
- **Purpose**: C++ tool for generating realistic axon + glial numerical
  substrates (**Computational Axonal Threading Engine for Realistic
  Proliferation**). Grows axons from overlapping spheres with controllable
  density, tortuosity and beading. Output: `*_spheres.csv`
  `(x, y, z, radius, type, id)`, consumed by
  `dmipy_jax/simulation/sphere_sdf.py::MultiSphereSDF`.
- **Why removed from this repo**: the `vendor/` placeholder here was never
  populated; the sibling checkout above is the working copy.
- **Related**: OCTOPUS (same lab, same primitives, adds neurons/glia with
  branching/tapering/undulation/spines) — see
  `docs/decisions/006-octopus-differentiable-substrate-landscape.md`.

## ReMiDi

- **Source**: https://github.com/BioMedAI-UCSC/ReMiDi
  (Khole PP, …, Li J-R, Ianus A, Marinescu R — arXiv:2502.01988, ISMRM 2025).
- **Was located at**: `ReMiDi/` (root)
- **Purpose**: **Re**construction of **Mi**crostructure using a
  **Di**fferentiable diffusion MRI simulator. A PyTorch re-implementation of
  SpinDoctor's FEM matrix formalism (Bloch–Torrey), made differentiable so a
  signal-matching loss can be backpropagated into 3D mesh vertices. It is
  **not** a Monte Carlo random-walk simulator.
- **Integration status**: no wrapper exists in this repo.
  `dmipy_jax/simulation/oracles/remidi.py` and `docker/Dockerfile.remidi`
  were planned (see `docs/prompts/remidi_manager.md`) but never written.
  The same solver family is implemented natively in JAX at
  `dmipy_jax/simulation/mesh_sim.py::MatrixFormalismSimulator`, so the
  honest comparison is solver-vs-solver on identical meshes rather than an
  oracle wrap.
- **Successor**: Spinverse (arXiv:2603.04638, Mar 2026) — differentiable
  Bloch–Torrey on tetrahedral grids with learnable per-face permeability.
- **Why removed**: Directory was empty (placeholder only, never populated).

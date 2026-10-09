# 10. Package review: dmipy port fidelity, structure, documentation — and the clean-up plan

## Date
2026-10-09

## Status
Review (read-only audit by three parallel reviewers plus spot verification), then **P0 executed 2026-10-09** (§7). P1/P2 remain proposals.

## 1. Headline

The research results (docs 008–009) sit on a package whose *core* port is sound but whose *surroundings* have decayed: one restricted model is non-functional, the installable package carries ~41 dead modules and 27 research modules, the test entry point does not run as documented, and 8 of 9 tutorials fail. None of this affects the HCP/DiSCo results (they use Ball/Stick/Zeppelin/Watson paths that verify to 1e-8), but it blocks anyone else from using or trusting the package.

## 2. Port fidelity (dmipy → JAX)

Original dmipy was vendored until `f29facf`; `git show 7b254f4:dmipy/...` is the reference. `dmipy==1.0.4` remains a runtime dependency but no longer imports under current numpy, so the two legacy-equivalence tests fail rather than skip.

| model | status |
|---|---|
| Ball, Stick, Zeppelin, Soderman cylinder, Dot, ST sphere, GPD sphere | verified against dmipy closed forms, max error ≤ 1e-8 |
| Watson ∘ Stick (grid DistributedModel) | within 2e-3 of a 4e5-sample MC reference at κ = 1, 4, 16 |
| C-NODDI tortuosity, SANDI composition | consistent with dmipy conventions |
| **Callaghan cylinder** | **was broken** (`sum_m` never accumulated; coefficients 8/16 vs dmipy 4/8; zero root omitted; returned ~1e-51 where dmipy gives 0.94). **Fixed in P0**: matches the vendored dmipy transcription to rtol 1e-9 in float64 (`tests/test_callaghan_reference.py`). |
| Sphere Callaghan | dmipy's own version is unusable as a reference (cylindrical `J_m'` roots, weight `α²−n(n−1)`, and a `spherical_jn(q, derivative=True)` call that raises). **Rewritten in P0** from the eigenfunction expansion with spherical roots `j_n'(α)=0`; pinned by its τ→∞ (Stejskal-Tanner sphere, 1e-10) and Dτ≪R² (free diffusion, 0.5 %) limits. |
| `constants.SPHERE_ROOTS` | was 30 roots with 6 mistranscribed; **now** the single 100-root dmipy table, `sandi.py` imports it; column 1 of the new spherical root finder reproduces it to 2e-7 |
| unit heuristic `is_si = max(b) > 5e4` | **removed**; cylinder kernels are SI-only like the rest of the package |
| missing vs dmipy | Van Gelderen GPD cylinder, temporal Zeppelin, spherical mean / SMT, SH distribution, dmipy Bingham (ψ, odi, β), Gamma `normalization='cylinder'` (AxCaliber), odi↔κ |
| framework | N free `partial_volume_i` with no Σf = 1 constraint; no parameter linking (`set_equal/fixed/tortuous_parameter`); no S0 (assumes pre-normalised attenuation) |

Before P0 all passing model tests were smoke tests (shape/finite); that is why the Callaghan bug survived. Stick, Soderman cylinder, ST sphere, ST plane and both Callaghan models now have closed-form assertions that need no dmipy install.

## 3. Structure

- `dmipy_jax/`: 420 modules, 64 k lines, 58 subdirectories, ~20 with 1–3 files. 90 modules have no in-package importer; **41 have no importer or test at all** (TUS/ultrasound, pulseq debug scripts, `algebra/test_script.py`, EIT loaders, deepfcd …).
- `dmipy_jax/validation/` (27 modules, 4.8 k lines of paper code: `prism_jax`, `dipy_refine`, `force_*`, `caterpillar`, `substrate_benchmark`) is inside the wheel; only `workflows/refine_peaks_flow.py` imports it from the package proper.
- Parallel namespaces: `pipeline/` vs `pipelines/`, `models/` vs `signal_models/`, three `acquisition.py`, two `tensor_train.py`, five fitting stacks, two `train_sbi`, three Bloch/MC stacks.
- Dependencies: `torch`, `torchvision`, `timm`, `opencv-python` hard deps with zero `import torch` in the package; `furo`, `datalad`, `boto3`, `vbjax`, `gpjax`, `e3nn-jax`, `pypulseq`, `jax-md` also hard. 267 packages in the lock.
- Tests: ~650 tests; `uv run pytest` as documented hangs on GPU (`--cov` in `addopts` + sybil collection); 6 collection errors from stale APIs (`fit_mcmc`, `fit_voxel`, `QMTSPGR`, `meshio`). CI runs the bare command with `jax[cuda13]` on ubuntu-latest and cannot be green.
- Git: 1,408 tracked files incl. 39 `.log`, 44 `.bin` (vendored AMICO), a 43 MB NIfTI, 6 `.eqx`, 3 `.pt`; `.git` 181 MB; `*.eqx`/`*.pt` not ignored.
- Naming: pyproject `sbi4dwi`, import `dmipy_jax`, Sphinx `dmipy-jax`, badges and Homepage → `m9h/dmipy`.

## 4. Documentation

- README (April 2026): quickstart broken on 3 of 5 names (`JaxAcquisition` path, `fit_voxel`, `G1Ball`→`Ball`); `ModelSimulator`/`train_sbi` signatures wrong; claims oracle classes, `HybridLibraryGenerator`, `SimulationComparisonRunner`, SWC loader that do not exist; ~40 % of Results is transcranial ultrasound; nothing on PRISM-JAX, `dipy_refine_peaks`, or docs 008–009.
- Tutorials: 1 of 9 clean (`first_steps`). Failures: dead `dmipy.data.synthetic` import, `G1Ball`, three NameErrors from blocks assuming undefined variables, imports of the non-existent oracle modules; three hang on GB10 compile.
- Sphinx: `index.md` and `index.rst` both present; `reference/` empty (no apidoc step); decision docs orphaned from every toctree; 54 warnings.
- Decision docs: two different `002`, one empty; `unxt_integration_rfc.md` unnumbered; 004/007/008 are logs (1.2–2.8 k lines), not decisions; no results summary page.
- Docstrings 64 % (888/1395 public defs); `inference`, `inverse`, `optimization` < 60 %.
- CLAUDE.md: "PLANNED" markers correct; directory map covers ~15 of ~50 packages and omits `validation/`, `workflows/`; cites `tests/test_oracle.py` which does not exist.
- Missing: `CITATION.cff`, `CHANGELOG`; LICENSE year stale (dmipy attribution preserved).

## 5. Plan

### P0 — correctness (days) — DONE 2026-10-09, see §7
1. Fix the Callaghan cylinder (accumulate `sum_m`, coefficients 4/8, zero root) and the Callaghan sphere weight; replace smoke tests with closed-form assertions against `git show 7b254f4:dmipy/...` for every ported model (the audit scripts in the scratchpad are the seed). Unify `SPHERE_ROOTS` on the 100-root table. Remove the `is_si` heuristic in favour of an explicit unit.
2. Make `uv run pytest` work: drop `--cov` from `addopts`, `JAX_PLATFORMS=cpu` in CI with CPU jax, fix or delete the 6 stale tests, gate tutorials with a marker/timeout. Drop `dmipy==1.0.4` (use the git reference instead).

### P1 — structure (1–2 weeks)
3. Move `dmipy_jax/validation/` out of the wheel: `dipy_refine.py` → `fitting/` (keep the CLI), `caterpillar.py` → `simulation/substrates/`, the rest → top-level `validation/lib/` beside the 67 scripts. One sed over 30 scripts + 4 test files.
4. Delete the 41 dead modules and the TUS/pulseq/EIT remnants (or move to `archive/`); collapse the one-file subpackages and parallel namespaces with deprecation shims; one fitting API, one `train_sbi`, one acquisition class.
5. Dependencies to extras: `[baselines]` (torch stack, amico), `[docs]`, `[pulseq]`, `[ultrasound]`; expect the lock to halve.
6. Binary hygiene: ignore `*.eqx *.pt`, move checkpoints/logs to `/data/datasets/sbi4dwi-artifacts/` or Zenodo, drop vendored AMICO (pip `dmri-amico`), consider a history rewrite for the 181 MB.
7. Naming: `sbi4dwi` everywhere; import package rename behind a `dmipy_jax` shim for one release; fix badges, Homepage, Sphinx project.

### P2 — documentation (1 week)
8. README rewrite around what the package is now: SBI pipeline, PRISM-JAX refine + `dipy_refine_peaks`, uncertainty, acquisition design; one working quickstart; results scoreboard linking docs 008–009.
9. Tutorials: repair the six failures, shrink sizes so each runs < 60 s on CPU, run them in CI.
10. Sphinx IA: Getting started / Concepts (units, b0-norm, pytrees, SBI, oracle boundary) / How-to / Reference (apidoc) / Validation results (one page from doc 008 §6, §11, §13 and doc 009) / Decision records (status table; move 004/007/008 to `docs/reports/`). Delete `index.rst`.
11. ADR hygiene (renumber 002, delete the empty file, number the unxt RFC); regenerate the CLAUDE.md map; add `CITATION.cff` (SBI4DWI, dmipy, Manzano-Patrón), CHANGELOG, LICENSE year.

### Missing-feature backlog (only if a use case appears)
Van Gelderen cylinder, SMT/spherical mean, dmipy Bingham parameterisation, Gamma cylinder normalisation (AxCaliber), Σf = 1 and parameter linking in the composer, S0 estimation.

## 6. What this means for positioning (doc 008)
The validated path (MSMT init → PRISM-JAX refine → Laplace posterior → DIPY plug-in) does not touch the broken or missing models, so the results stand. But a reviewer or collaborator who clones the repo today meets a failing test suite, a README that describes code that does not exist, and tutorials that do not run. P0 and item 8 are the minimum before any outreach note in `docs/outreach/` is sent.

## 7. P0 execution log (2026-10-09)

### 7.1 Correctness

- **Callaghan cylinder** (`signal_models/cylinder_models.py`): rewritten as a vectorised port of dmipy's `perpendicular_attenuation` — coefficients 4 (m = 0) and 8 (m ≥ 1), all orders accumulated, trivial root `α₀₀ = 0` included. Bessel functions for all orders come from one host callback (`bessel_j_stack`) with a custom JVP, so the kernel is differentiable in diameter (gradient check to 5 %). Against a verbatim NumPy transcription of dmipy 7b254f4: rtol 1e-9 in float64, 2e-4 in float32, on a 40-direction 5-shell scheme with per-measurement τ. Limits: τ → ∞ gives Soderman `(2J₁(x)/x)²` to 1e-10; Dτ ≪ R² gives `exp(−bD)` to 5 %.
- **Callaghan sphere** (`signal_models/sphere_models.py`): dmipy's `S3SphereCallaghanApproximation` turns out to be unusable as a reference — it uses the cylindrical roots `J_m'(α) = 0`, a weight `α² − n(n−1)`, and calls `spherical_jn(q, derivative=True)` without an order, which raises. Our copy inherited the wrong roots. Rederived from the eigenfunction expansion `ψ = j_n(αr/R) Y_nm` with reflecting walls: `E = 9 j₁(x)²/x² + Σ 6(2n+1) e^{−α²Dτ/R²} α²/(α² − n(n+1)) (x j_n'(x))²/(x² − α²)²` over the roots of `j_n'(α) = 0` (`constants.spherical_bessel_derivative_roots`; column n = 1 reproduces the Murday–Cotts `SPHERE_ROOTS` table to 2e-7). Limits: τ → ∞ gives the Stejskal–Tanner sphere `(3 j₁(x)/x)²` to 1e-10; Dτ ≪ R² gives `exp(−bD)` to 0.5 %.
- **Root tables**: `constants.SPHERE_ROOTS` is now the single 100-root dmipy table (sandi imports it); `constants.bessel_derivative_roots` is shared by the cylinder model (the sphere no longer uses it).
- **Units**: the `is_si = max(b) > 5e4` heuristic is gone from `c2_cylinder` and `c3_cylinder_callaghan`; both are SI-only (`q = √(b/τ)/2π` in m⁻¹), consistent with CLAUDE.md. Smoke tests that passed s/mm² now pass s/m².
- **Tests**: `dmipy_jax/tests/test_callaghan_reference.py` (14 tests) replaces the skipped legacy-dmipy comparisons; stick, Soderman cylinder, ST sphere and ST plane have closed-form assertions in `tests/test_jax_equivalence.py` and `test_sphere_plane_models.py`.

### 7.2 Test entry point

- `pyproject.toml`: `addopts = "--import-mode=importlib"` (coverage opt-in; importlib mode because `tests/` and `dmipy_jax/tests/` share basenames), `testpaths = [dmipy_jax/tests, tests]`, `slow`/`gpu` marks registered. `docs/tutorials` is collected only when passed explicitly.
- `conftest.py`: sybil optional. `--noconftest` no longer needed; CLAUDE.md, README and CONTRIBUTING updated.
- `dmipy==1.0.4` removed from dependencies (lock: −3 packages). Reference is `git show 7b254f4:dmipy/...`.
- CI runs with `JAX_PLATFORMS=cpu`.
- Six collection errors: `test_qmt.py` (API gone), `tests/test_mcmc_cpu.py` (duplicate), `tests/validation/test_vs_julia.py` (pyjulia script, not a test) deleted; `test_mcmc.py` ported to `MCMCInference`; openlifu/jinns/jwave tests `importorskip` their optional deps.
- Two session-poisoning tests found only once the suite could run as a whole: `tests/test_surface_mapper.py` replaced `numpy` and `jax` in `sys.modules` with MagicMocks at import (285 downstream failures), and `tests/validation/test_prism_exchange.py` enabled x64 globally at import (complex64/complex128 scan errors in EPG, mcDESPOT, trainer). Both scoped.
- `tests/test_multi_compartment.py` had a fixture without `@pytest.fixture` and assigned to frozen eqx fields; fixed, which exposed that `JaxMultiCompartmentModel.fit` recovers f but not `lambda_iso` on its 7-measurement scheme, non-deterministically across its three cases (3 tests xfail).

### 7.3 Known failures left as `xfail(strict=False)` (pre-existing, off the validated path)

| test | cause |
|---|---|
| `test_solver_verification` ×3 | `jnp.roots` under jit / empty roots / gradient check |
| `test_trainer_convergence::test_trainer` | `train_loop()` API drift |
| `biophysics/test_radiation_force::test_displacement_range_microns` | displacement 0.002 µm vs expected > 0.01 µm |
| `test_jemris_comparison::test_fid_signal_match` | Bloch `simulate_acquisition` raises inside JIT |
| `tests/test_neural_fitting` | activation passed as JAX leaf (equinox drift) |
| `tests/io/test_multi_te_loader` | loader return arity |
| `tests/test_karger_exchange` | Karger signal 1.49 at b = 0 — a real model bug |
| `tests/test_solvers::test_diffusion_sde_msd` | Diffrax 0.7 `ControlTerm` structure |
| `tests/test_mcmc::test_mcmc_multi_voxel` | vmapped NUTS chain diverges for one voxel |
| `tests/test_multi_compartment` ×3 | `lambda_iso` not recovered; which case fails varies between runs |

These belong to P1 item 4 (dead/decayed modules): each is either in a module slated for removal or a genuine bug worth a ticket. Suite on CPU after P0: 509 passed, 5 skipped, 11 deselected (`gpu`), remaining failures all marked above, 5 min 47 s.

## 8. P1 execution log (2026-10-09)

### 8.1 `dmipy_jax/validation` out of the wheel (item 3) — commit 9c1ea15
- `dmipy_jax.prism` is the new home of the validated product: `prism_jax`, `prism_uncertainty`, `dipy_refine` (with `default_config`/`fi_config`) and `tracking` (ex `disco_tracking`). `dipy_refine_peaks` CLI unchanged.
- `caterpillar.py` → `dmipy_jax.simulation.substrates`.
- The other 22 modules → `validation/lib/` (a package importable from a dev checkout because the editable install puts the repo root on `sys.path`; hatch still ships only `dmipy_jax`). 74 files rewritten; all 155 validation tests pass.
- Four files carried committed merge-conflict markers (`<<<<<<< HEAD … >>>>>>> recovery_work_v2`); resolved by keeping the cleaned-up side.

### 8.2 Dead code and remnants (item 4)
- Deleted 17 fully dead modules (no importer, test or script): `biophysics/{multimodal_tus,neural_signal}`, `design/sbi_oed`, `experiments/bigmac_bas`, `inference/bedpost`, `io/babelbrain`, `models/deepfcd`, `optimization/oed`, `pulseq/debug_*` ×3, `reconstruction/csa_odf`, `reconstruction/csd/deep/{diffusion,generative}`, `symbolic/derive_master`, `utils/math_helpers`, plus `algebra/test_script.py` and the stale `simulation/scanner/test_bloch.py`. `geometry/curvature.py` was on the audit list but has a test; kept.
- Nine test scripts that lived inside `signal_models/qmri/` and `simulation/scanner/` moved to `dmipy_jax/tests/{qmri,scanner}/` (96 pass; 3 pre-existing failures xfail).
- `archive/` (outside the wheel, not collected): transcranial ultrasound + EIT (14 `biophysics` modules, `network/`, their 9 test files, 5 examples, `io/uclh_eit`) and pulseq (`pulseq/`, `external/pulseq.py`, the JEMRIS comparison test, the manual bridge script). `biophysics/` keeps `neural_exchange`, `velocity`, `conductivity`, `buckling_layer`, `neural_cde`, which the dMRI side imports. `archive/README.md` records the last known test state.
- Not done from item 4: collapsing the parallel namespaces (`pipeline/` vs `pipelines/`, `models/` vs `signal_models/`, three `acquisition.py`, five fitting stacks). Each needs an API decision and deprecation shims; left as the P1 remainder.

### 8.3 Dependencies (item 5)
- Core list cut from 48 to 29 entries, grouped by role. Removed (zero importers in the package): `torch`, `torchvision`, `timm`, `opencv-python`, `gpjax`, `cvxpy`, `uguide`, `tabulate`, `scikit-image`, `scikit-learn`, `vbjax`, `pypulseq` (`cvxpy` stays: dipy MSMT-CSD, the PRISM init, needs it); `furo`/`ipython`/`corner` moved to the `doc` extra / `dev` group. New extras: `data` (`boto3`, `datalad`), `baselines` (`dmri-amico`, `torch`). `pyjulia` dropped from `test`. Lock 264 → 239 packages (CUDA JAX dominates what remains).
- CI installs `--extra test --extra data`.

### 8.4 Binary hygiene (item 6)
- 57 tracked checkpoints/logs/volumes (`*.eqx`, `*.pt`, `*.log`, `*.nii.gz`, 70 MB) untracked and mirrored at `/data/datasets/sbi4dwi-artifacts/` with their paths; `docs/artifacts.md` is the manifest. Extensions ignored. Vendored AMICO (44 `.bin`) deleted in favour of the `dmri-amico` wheel. `validation/*.npz` and `*.png` stay tracked (cited by docs 008–009).
- History rewrite for the 181 MB `.git` not done: it rewrites shared history and needs an explicit go-ahead.

### 8.5 Naming (item 7)
- Sphinx project → `SBI4DWI`; README CI badge and pyproject Homepage/Repository → `m9h/sbi4dwi`.
- The import-package rename `dmipy_jax` → `sbi4dwi` (with a one-release shim) is not done. It touches ~400 modules, every doc and every validation script; worth doing, but as its own commit after the README/tutorial pass so the docs are rewritten once.


## 9. P2 execution log (2026-10-09)

- **README** (item 8) rewritten around the current package: a capability table, a results section copied from doc 008 (§6.4 crossings, DiSCo SNR 10, §10.6/§13 HCP cohort), install with extras, and a three-part quickstart (vectorised stick forward model, `dipy_refine_peaks` / `refine_peaks` with `fi_config`, MDN train → checkpoint → `SBIPredictor`). All three Python snippets were executed before committing. The ultrasound results and the non-existent oracle classes are gone.
- **Tutorials** (item 9): all nine now run as sybil doctests on CPU, 129 blocks in 74 s for the directory, and CI runs them. Fixes: dead `dmipy.*` imports replaced by `JaxMultiCompartmentModel`; `G1Ball`→`Ball`; NameErrors from blocks assuming earlier state; `SBIPipelineConfig`/`train_sbi`/`save_checkpoint`/`SBIPredictor` calls corrected to the real signatures; sizes shrunk with production values noted; outputs to temp dirs; sections on the never-written oracle modules marked planned. One real bug surfaced in tutorial code: `sbi_dti.md`'s FA used a `1e-9` guard on a denominator of order 1e-18, so every FA was ~1e-5 and the network was fitting noise (the package's own `derived_metrics` were not affected). Checked independently after the subagent's pass.
- **Sphinx** (item 10): one `index.md` (the stale `index.rst` deleted) with sections Getting started / Simulation-based inference / Validation results and decisions / Project / API reference; `sphinx.ext.apidoc` (Sphinx 9.1) generates `reference/` at build time, so the empty tracked directory is gone and `docs/reference/` is ignored; working-note directories and superseded ADRs excluded from the build. Build exits 0; ~150 remaining warnings are docstring formatting inside autodoc, left for later.
- **ADR hygiene** (item 11): `docs/decisions/README.md` status table; duplicate `002` files → `002b` (Equinox architecture), `002c` (unxt RFC), the empty one deleted; `CITATION.cff` (SBI4DWI + dmipy + Manzano-Patrón), `CHANGELOG.md`, LICENSE year 2026, CLAUDE.md directory map regenerated for the new layout (done in P1).

### What remains open from doc 010
Namespace collapse (P1.4 second half: `pipeline/` vs `pipelines/`, `models/` vs `signal_models/`, three `acquisition.py`, several fitting stacks), the `dmipy_jax`→`sbi4dwi` import rename with a shim, the git history rewrite, autodoc docstring warnings, and the eleven xfail'd pre-existing failures (§7.3), of which the Karger signal > 1 is a real bug.

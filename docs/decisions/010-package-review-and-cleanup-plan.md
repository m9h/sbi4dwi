# 10. Package review: dmipy port fidelity, structure, documentation — and the clean-up plan

## Date
2026-10-09

## Status
Review (read-only audit by three parallel reviewers plus spot verification). Proposes work; nothing here is done yet.

## 1. Headline

The research results (docs 008–009) sit on a package whose *core* port is sound but whose *surroundings* have decayed: one restricted model is non-functional, the installable package carries ~41 dead modules and 27 research modules, the test entry point does not run as documented, and 8 of 9 tutorials fail. None of this affects the HCP/DiSCo results (they use Ball/Stick/Zeppelin/Watson paths that verify to 1e-8), but it blocks anyone else from using or trusting the package.

## 2. Port fidelity (dmipy → JAX)

Original dmipy was vendored until `f29facf`; `git show 7b254f4:dmipy/...` is the reference. `dmipy==1.0.4` remains a runtime dependency but no longer imports under current numpy, so the two legacy-equivalence tests fail rather than skip.

| model | status |
|---|---|
| Ball, Stick, Zeppelin, Soderman cylinder, Dot, ST sphere, GPD sphere | verified against dmipy closed forms, max error ≤ 1e-8 |
| Watson ∘ Stick (grid DistributedModel) | within 2e-3 of a 4e5-sample MC reference at κ = 1, 4, 16 |
| C-NODDI tortuosity, SANDI composition | consistent with dmipy conventions |
| **Callaghan cylinder** | **broken**: `signal_models/cylinder_models.py:387-414` computes `sum_m` per order and never adds it to `res`; m=0/m>0 coefficients 8/16 vs dmipy 4/8; zero root omitted. Returns ~1e-51 where dmipy gives 0.94. Verified by reading the loop. Test only checks shape. |
| Sphere Callaghan | weight `α²−n(n+1)` vs dmipy `α²−n(n−1)`; untested |
| `constants.SPHERE_ROOTS` | 30 roots, last 6 mistranscribed (residual 5e-5); `sandi.py` carries a correct 100-root table — two copies |
| unit heuristic `is_si = max(b) > 5e4` | silently rescales low-b SI schemes (`cylinder_models.py:94, 310`) |
| missing vs dmipy | Van Gelderen GPD cylinder, temporal Zeppelin, spherical mean / SMT, SH distribution, dmipy Bingham (ψ, odi, β), Gamma `normalization='cylinder'` (AxCaliber), odi↔κ |
| framework | N free `partial_volume_i` with no Σf = 1 constraint; no parameter linking (`set_equal/fixed/tortuous_parameter`); no S0 (assumes pre-normalised attenuation) |

All passing model tests are smoke tests (shape/finite); that is why the Callaghan bug survived.

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

### P0 — correctness (days)
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

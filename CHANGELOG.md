# Changelog

## Unreleased (2026-10-09) — package review follow-up (doc 010)

### Fixed
- Callaghan cylinder (`signal_models/cylinder_models.py`) returned ~1e-51 instead of the
  restricted signal: orders were never accumulated, coefficients were doubled and the
  zero root was missing. Now a faithful port of dmipy 7b254f4 (rtol 1e-9 in float64).
- Callaghan sphere rederived with the roots of the spherical Bessel derivative; pinned by
  its Stejskal–Tanner and free-diffusion limits (dmipy's version was unrunnable).
- `constants.SPHERE_ROOTS` is the single 100-root table (six mistranscribed roots gone).
- `uv run pytest` works from a clean checkout; six stale test modules fixed or removed;
  two tests that poisoned the session (MagicMock `numpy`/`jax`, global x64) scoped.
- Four files with committed merge-conflict markers resolved.

### Changed
- Cylinder kernels are SI-only; the `max(b) > 5e4` unit heuristic is gone.
- `dmipy_jax.validation` split: PRISM-JAX → `dmipy_jax.prism`, CATERPillar →
  `dmipy_jax.simulation.substrates`, paper code → `validation/lib` (not shipped).
- Dependencies: core list 48 → 29; new extras `data` (boto3, datalad) and
  `baselines` (dmri-amico, torch); torch/timm/opencv/gpjax/uguide/vbjax/pypulseq removed.
- Checkpoints, logs and sample volumes untracked (mirrored in
  `/data/datasets/sbi4dwi-artifacts`, see `docs/artifacts.md`); vendored AMICO deleted.
- Transcranial-ultrasound, EIT and pulseq code moved to `archive/`.
- README rewritten around what the package is now; Sphinx project renamed SBI4DWI.

### Removed
- `dmipy==1.0.4` dependency (reference is `git show 7b254f4:dmipy/...`).
- 17 dead modules (no importer, test or script).

## 0.1.0 — earlier history
See `git log` and `docs/decisions/`.

# archive/

Code that was developed inside `dmipy_jax` but is not part of what SBI4DWI is
(diffusion-MRI microstructure estimation, PRISM-JAX refinement, SBI, acquisition
design). Kept in the repository for reference, outside the wheel, not imported,
not tested in CI. Moved here 2026-10-09 (doc 010, P1 item 4).

| directory | what | last known state |
|---|---|---|
| `tus/` | transcranial ultrasound: acoustic property maps, j-Wave adapter, radiation force / MR-ARFI, openlifu bridge, TUS optimiser; EIT conductivity PINN; UCLH EIT loader | tests passed on CPU except `test_radiation_force::test_displacement_range_microns` (0.002 um vs > 0.01 um expected) and the j-Wave tests (need `jwave`) |
| `pulseq/` | Pulseq `.seq` interpreter and Bloch bridge, with its debug scripts | `verify_pulseq_bridge.py` was a manual script, never a collected test |

`dmipy_jax/biophysics/` keeps the modules the dMRI side still imports:
`neural_exchange`, `velocity`, `conductivity`, `buckling_layer`, `neural_cde`.

To revive something: `git mv` it back under `dmipy_jax/` and restore its imports
(`from archive.tus.biophysics import ...` is not supported on purpose).

# To: the FORCE authors / DIPY maintainers — downstream notes with a reproducible package

**Status:** drafted 2026-09-15, not sent. Long form: doc 005 (private) and the published page (claude.ai artifact "FORCE Field Notes"). Data: `/data/datasets/sbi4dwi_benchmark_v1` (built by `validation/build_benchmark_package.py`).

Dear Atharva, Rafael, Alonso, Eleftherios,

I have spent the last few months using FORCE as the reference method in a diffusion-microstructure project (sbi4dwi, a JAX/Equinox stack with a differentiable refinement stage). FORCE is the most noise-robust method in our comparison and reproduces its paper claims on its design data; the notes below are the places where a careful downstream user still trips, each with the number that shows it and a small suggestion. Everything is reproducible from a package I can send (npz exports + a numpy/dipy-only runner).

**1. The library prior is part of the method, and the tutorial does not say so.** On DiSCo the default in-vivo prior gives NDI r = 0.68 (single shell) / 0.92 (three shells) against ground truth with a +0.39 offset; the paper's narrow bands give 0.92 / 0.98. A note after `model.generate(...)` in `reconst_force.py` pointing at §3.2, and a DiSCo example, would have saved me an afternoon. Exposing the diffusivity ranges on the fitted model would help audits.

**2. Orientation error is matching-limited, not library-limited** (dipy master after #4130). For two-fibre crossings at 15–90°, the 500K library contains an entry within 2.8° of the truth for every case, but the cosine search returns entries 7.5° away on average; noise-free signals give 8.0°, a 2M library 7.4°, and signals from FORCE's *own* generator (clean two-fibre WM voxels) 8–10°. So neither noise nor coverage nor model mismatch is the cause: the nearest signal is not the nearest orientation, because partial-volume and microstructure directions dominate the similarity. A few gradient steps on the same forward model from the matched entry would remove most of the 5–7°; the entry is an excellent initialisation. (We do exactly this downstream: FORCE peaks in → 0.79 → 0.91 connectome correlation on the DiSCo protocol.)

**3. `use_posterior=True` does not touch peaks.** Labels, hence `force_peaks` and the ODF, come from the single argmax entry; every direction was identical with the flag on and off across all runs. Either posterior-weighted peaks or a docstring line.

**4. Three small defects on master (2026-09-14).**
- `wm_threshold=1.0` silently zeroes all ODFs (`sims["odfs"]`), so `force_peaks` returns no peaks while `num_fibers > 0`. A warning at generation time would catch it.
- Simulation cache lookup crashes on range-valued `diffusivity_config`: `_diffusivity_matches` calls `np.isclose(stored, current)` on a tuple → `ValueError: The truth value of an array ...`. `use_cache=False` works around it.
- `generate_force_simulations` forks with JAX imported → deadlock warning; a `spawn` context or docstring note.

**5. Composition.** Dirichlet(2,1,1) with three populations yields 70 % three-fibre entries, median WM 0.50, and 0.5 % clean two-fibre WM crossings. Right for in-vivo mixed voxels; a one-line histogram under `verbose=True` and a knob for the fraction prior would let users build orientation-oriented libraries deliberately.

**What works as claimed:** Stanford HARDI maps and FORCE-vs-DTI r = 0.985; reproducibility after #4130; 100 % recall at 15–25° with `two_fiber_min_angle=0`; ND on single bundles is the least biased intra-axonal fraction of any method we ran on Monte Carlo substrates (0.49 vs 0.50) — matching does not suffer the partial-volume attribution that free-compartment optimisers do.

I would be glad to send the package and the runner, and happy to turn any of the above into PRs if useful.

Best regards,
Morgan Hough

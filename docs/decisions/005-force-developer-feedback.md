# FORCE developer feedback (private, not for upstream)

**Date:** 2026-05-10
**Author:** Morgan Hough (sbi4dwi)
**Context:** Replicating FORCE paper §3.2 (DiSCo phantom) and §3.3 (Stanford HARDI) on dipy 1.12.1, then comparing dmipy-JAX dictionary matching against FORCE upstream.
**Audience:** Atharva Shah, Rafael Henriques, Alonso Ramirez-Manzanares, Eleftherios Garyfallidis — the FORCE paper authors / DIPY maintainers, when we have a chance to talk to them.

The findings below are *positive about FORCE's methodology* — most surface places where the upstream packaging or tutorial could be improved so that new users don't fall into the same library-prior pitfall I did on this project.

## 1. The thing the paper says clearly but the tutorial hides

**Paper §3.2 (DiSCo phantom validation), verbatim:**

> Minor adjustments were introduced to align the forward model with the
> characteristics of this numerical phantom as it departs from the regime
> of usual biological tissue diffusion parameters. To ensure consistency
> with the DiSCo simulation model, which represents diffusion using
> stick-like compartments, **diffusivities were sampled from narrow
> uniform bands: D_∥ from Uniform(0.54, 0.66) × 10⁻³ mm²/s and D_⊥ from
> Uniform(0.32, 0.38) × 10⁻³ mm²/s. The isotropic compartment was
> disabled** to match the stick-like DiSCo model.

This is the key methodological footnote. It says: **the library prior is part of the method, and must be retuned per-dataset whenever the data depart from the in-vivo regime.**

But this insight is essentially absent from `dipy/doc/examples/reconst_force.py`, where:

- `model.generate(num_simulations=500000, num_cpus=-1, verbose=True, use_cache=False)` is called with no `diffusivity_config`.
- There is no explanation of when to retune (vs trust defaults).
- The example uses Stanford HARDI (in-vivo, where defaults work) — so a new user runs it, sees plausible output, assumes defaults are universal, and applies them to (e.g.) a phantom or ex-vivo dataset where they aren't.

## 2. Empirical demonstration of why this matters

DiSCo phantom subject 1 highRes, single-shell b=1900, SNR=30, 15,267 masked voxels:

| Method | FORCE NDI vs GT Intra Volume Fraction | FORCE FA vs DTI FA |
|---|:-:|:-:|
| Default library (`wm_d_par_range=(0.002, 0.003)`) | **r = 0.679** | r = 0.827 |
| Paper-protocol retune (`wm_d_par_range=(0.00054, 0.00066)`) | **r = 0.918** | **r = 0.990** |

**24 Pearson-r-point improvement from a ~10-line `diffusivity_config` change.**

More importantly: the **default-library run's NDI mean was 0.75** while ground truth's was ~0.3. That is a **uniform 0.4-magnitude over-estimation across the entire brain mask**. The spatial pattern still tracked GT (r=0.679 isn't zero), so visual inspection alone of the FORCE NDI map would *not* surface the problem. A new user could publish or clinically apply biased numbers in good faith.

## 3. Suggested upstream improvements

Listed in rough order of effort × value.

### (a) Add a paragraph to `dipy/doc/examples/reconst_force.py`

Right after `model.generate(...)`, something like:

```rst
.. note::
   The default diffusivity priors (``D_∥ ~ Uniform(2.0, 3.0)×10⁻³ mm²/s``,
   ``D_⊥ ~ Uniform(0.3, 1.5)×10⁻³ mm²/s``) are calibrated for in-vivo
   human brain at typical clinical resolution. If your data depart from
   this regime — e.g. ex-vivo / fixed tissue, numerical phantoms,
   high-resolution preclinical scanners — **retune via
   `diffusivity_config`**. See FORCE paper §3.2 for an example
   (DiSCo phantom, ``D_∥ ~ Uniform(0.54, 0.66)×10⁻³``).
```

This is 8 lines of docs and would have saved me a full afternoon of debugging in this project.

### (b) Add a tutorial example or section that retunes for DiSCo

DiSCo is already in dipy via `fetch_disco1_dataset()`. A second tutorial example titled "FORCE on the DiSCo phantom" that walks through:

1. Fetching DiSCo
2. Sub-selecting single-shell b≈2000
3. Generating a DiSCo-tuned library with the exact priors from paper §3.2
4. Fitting + showing the NDI ground-truth correlation

…would make the retuning step concrete and copy-pasteable. The current paper has the recipe; a tutorial would surface it.

### (c) Diagnostic: emit a warning when input data appear out-of-distribution

A simple sanity check at the top of `FORCEModel.fit`:

```python
# Heuristic: if observed mean MD outside library's library range,
# warn the user about library/data mismatch.
```

Could compute a rough DTI MD on the input and compare to the library's
expected MD range. If they're a factor of 2+ apart, emit a clear warning
pointing at `diffusivity_config`. Not strictly necessary but would
catch the failure mode early.

### (d) `FORCEModel.__init__` could store the diffusivity ranges from the simulations dict so users can inspect what they're using

Right now you have to either remember what you passed to `generate_force_simulations` or inspect `sims["wm_d_par_range"]`-like keys (if even saved). Surfacing this on the `FORCEModel` instance (`model.diffusivity_config` or `model.summary()` would help post-hoc audits and tutorial reproducibility.

## 4. Other observations from the reproduction work

These are smaller, but in the spirit of "issues a careful user trips over":

### 4a. JAX-fork incompatibility with `num_cpus > 1`

`generate_force_simulations` uses `multiprocessing` with the default `fork` start method on Linux. If JAX has been imported before this call, the fork copies JAX's background threads into the children → deadlock risk. The runtime emits this warning:

```
RuntimeWarning: os.fork() was called. os.fork() is incompatible with
multithreaded code, and JAX is multithreaded, so this will likely lead
to a deadlock.
```

In our work it silently worked at `num_cpus=20`, but the warning is a real footgun. Two cheap fixes for upstream:

- Document this clearly in the `generate_force_simulations` docstring with a recommendation: "If JAX is loaded in the calling process, use `num_cpus=1` or `multiprocessing.set_start_method('spawn')` first."
- Or, if feasible, internally use `multiprocessing.get_context('spawn').Pool(...)` so the worker processes don't inherit the parent's threading state.

### 4b. Library-stored signals use 0–100 scale, not unit-S0

`sims["signals"]` has signal range ~5–100 per voxel, not 0–1. Initially I tried to round-trip a stored signal through `model.fit` as a wiring sanity check and got `FORCEFit.label.sum() == 0`. The reason is that `model.fit` expects unit-S0-normalised input (which it then internally renormalises via cosine similarity). Stored library signals are at the raw amplitude scale.

This is **fine** as long as users always pass real DWI data (S0-normalised by acquisition), but it surprised me as a developer. Worth a comment in the docstring of `save_force_simulations` / `load_force_simulations` explaining the storage convention.

### 4c. Default sphere is 362 vertices; paper uses 724

`generate_force_simulations` doesn't expose a `sphere` argument — the
internal default is `default_sphere` (362 vertices). Paper §2.2.2 (line 182):

> Orientations were sampled uniformly over a unit sphere using a
> **724-vertex electrostatic grid**, providing an angular resolution
> of approximately 4.1° between nearest vertices.

The 362-vertex `default_sphere` has nearest-vertex error ~4.1° too (similar density). But applications requiring sub-4° angular resolution would benefit from being able to pass `sphere=Symmetric724`. Probably a one-line addition to the `generate_force_simulations` signature.

### 4d. `wm_threshold=1.0` silently suppresses ODF generation in the library

Found while reproducing paper §3.2's DiSCo connectivity-matrix protocol.
The paper explicitly says "**The isotropic compartment was disabled** to
match the stick-like DiSCo model", which I implemented as
`wm_threshold=1.0` (the parameter's documented purpose). The resulting
500K-entry library has:

- All scalar metrics (`fa`, `md`, `nd`, `dispersion`, `wm_fraction`, …)
  populated correctly — matcher works, NDI recovery r = 0.918.
- **`sims["odfs"]` is *all zeros* — 0 of 500,000 entries have any
  nonzero ODF value.**

`force_peaks(fit)` builds its `PeaksAndMetrics.peak_dirs` from the
posterior-weighted average of library ODFs, so on a tuned-library fit
it returns zero peaks per voxel — even when `fit.num_fibers > 0` says
the matcher found multi-fibre configurations. Tractography on those
peaks produces zero streamlines, which breaks any downstream
connectivity-matrix or tractography pipeline.

Compare same fixture with the default library: 249,898 / 500,000 ODF
rows nonzero, tractography works.

This is silent and dangerous: the matcher reports `num_fibers > 0`,
the user reasonably assumes peaks exist, and the empty `peak_dirs`
only surfaces when the downstream pipeline crashes (zero streamlines).
Suggestions for upstream:

- Emit a `RuntimeWarning` if `generate_force_simulations` would produce
  zero nonzero-ODF rows.
- Or: ensure ODFs are populated regardless of `wm_threshold`, since
  ODFs are needed for `force_peaks` regardless of the WM-vs-GM/CSF
  fraction policy.
- Or at minimum: document that `wm_threshold=1.0` will suppress ODFs
  and recommend using something like `wm_threshold=0.95` instead, or
  setting `compute_odfs=True` explicitly if such a flag exists.

Pinned in our tests as `test_tuned_library_suppresses_odfs_upstream_bug`
— it will turn red the day upstream fixes the issue.

### 4e. The 70%-3-fibre library composition is undocumented

`Dirichlet(2,1,1)` over (WM, GM, FW) fractions, with WM containing up to 3 fibre populations, results in:

| n_fibres | Library fraction |
|:-:|:-:|
| 1 | 10.0% |
| 2 | 20.0% |
| 3 | **69.9%** |

This means the library is dominated by 3-fibre configurations. For users analysing data with predominantly 1–2 fibre populations (most of the brain), only ~30% of the library is "useful". This is **fine** — FORCE was designed to be general — but it's not documented anywhere I can find.

A `verbose=True` printout from `generate_force_simulations` reporting the achieved n_fibre breakdown (and the total parameter coverage of the library) would help users decide whether their library is dense enough for their geometry of interest. A small histogram of `sims["num_fibers"]` is all it would take.

## 5. Net assessment of FORCE

After substantial back-and-forth in this project's docs/decisions/004, my honest assessment of FORCE upstream:

- **The method works.** §14 (Stanford HARDI maps) and §15 (FORCE vs DTI r=0.985) reproduce the paper's positive claims on its design data.
- **Library-prior alignment is critical.** §16.3 (tuned r=0.918) vs §16.2 (default r=0.679) demonstrates this at a 24-r-point swing.
- **The retuning step is documented in the paper but undersold in the tutorial.** That's the user-experience gap worth surfacing.
- **There are several small footguns** (JAX fork, signal scale, sphere quantisation, fibre-fraction prior) — all easy to document or fix.

If I had read the paper before benchmarking, the §13 / §16.2 wasted compute would not have happened. The fact that I (a careful user with the paper open) still tripped on this is the strongest evidence I have for the "document the retune step" suggestion above.

## 6. What we have running on top of FORCE in sbi4dwi

For context, when discussing with the FORCE authors: the sbi4dwi project has built a dmipy-JAX dictionary matcher (`dmipy_jax/library/matcher.py`) inspired by FORCE but with:

- **Native JAX implementation** (cosine similarity on GPU, no FAISS dependency).
- **Composable forward models** — any combination of dmipy compartments (stick, zeppelin, restricted cylinders, sphere/SANDI, Bingham, Watson, IVIM…), not just FORCE's fixed stick + zeppelin + Bingham + ball.
- **Hybrid initialisation** — dictionary match → Optimistix LM refinement for continuous parameter estimates.
- **End-to-end differentiability** — gradient-based acquisition optimisation via Fisher information / EIG.

The natural collaboration story is: dmipy-JAX adds gradient-based extensions and differentiable physics to the FORCE paradigm; FORCE provides the canonical biophysics + reference implementation. There is no competition story — these are complementary in a way the paper's §6 "future work" section gestures at.

## 7. Update 2026-09-14: what limits FORCE's orientation accuracy (dipy master 1.13.0.dev, seeded)

All of the below is on dipy master after PR #4130 (`seed=2298` wired,
`two_fiber_min_angle=0` allowed), with the default in-vivo priors, on
three shells ≤ 3000 with 64 directions each. Scripts:
`validation/diagnose_force_limits.py` (numpy + dipy only), full log
`validation/external/force_diag.log`; the method is in doc 008 §7.3.

### 7.1 The measurement

Because `force_peaks` reads the vertex labels of the *single* best
cosine match, the angular error of a FORCE peak decomposes exactly into

- **sphere quantisation** — nearest `default_sphere` vertex to the truth,
- **library coverage** — the best entry in the library, judged in
  parameter space (min over entries of the best-match angular error), and
- **matching** — the entry the signal search actually returns.

| dataset | quantisation | coverage floor | FORCE, SNR 30 | FORCE, noise-free |
|---|---|---|---|---|
| two-fibre crossings 15–90°, stick + zeppelin generator, 500K library | 2.76° | **2.79°** | 7.54° / 98.7 % | 7.96° / 97.0 % |
| same, 2M library | 2.76° | 2.79° | 7.37° / 98.7 % | 6.60° / 100 % |
| **FORCE's own generator**, held-out clean two-fibre WM voxels (WM ≥ 0.6, each fibre ≥ 0.15, ODI ≤ 0.2, n = 526), 500K | — | — | 10.44° / 89 % | **10.11° / 90 %** |
| same, 2M | — | — | 8.32° / 94 % | 8.19° / 94 % |
| Monte Carlo axon substrates (CATERPillar), crossings 30–90°, 500K | 3.05° | 2.94° | 9.7–16.6° | — |

### 7.2 What it says

1. **The library is not the bottleneck.** For every configuration tested
   there is an entry within ~2.8° of the truth; the search returns one
   5–14° away. Quadrupling the library moves the benchmark by 0.2° and
   the in-model number by 2°, and is non-monotone on the substrates
   (straight 60°: 13.1 → 6.4°; straight 90°: 9.7 → 11.4° with recall
   100 → 80 %). Nearest-neighbour matching has no smoothness to exploit,
   so a denser library mostly changes *which* wrong entry wins.
2. **Noise is not the bottleneck** — noise-free and SNR 30 differ by
   < 0.6° everywhere.
3. **Model mismatch is not the main cause either** — on FORCE's own
   forward model, clean signals, the error is 8–10°. The cosine
   similarity over 193 measurements is dominated by the partial-volume
   and microstructure directions of the library (WM/GM/CSF fractions,
   D∥, D⊥, ODI): the nearest *signal* is not the nearest *orientation*.
4. **`use_posterior=True` does not touch orientations.** It averages the
   scalar parameters over the `n_neighbors` softmax weights but the
   `labels` (hence peaks and the ODF) still come from the single argmax
   entry. Every direction in every run above was identical with the flag
   on and off. Users reading "posterior" will assume the peaks are
   posterior-weighted too.
5. **Prior composition.** Dirichlet(2,1,1) with three fibre populations
   gives 70 % three-fibre entries, median WM fraction 0.50, and only
   **0.5 %** of the library is a clean two-fibre WM crossing (≥ 0.6 WM,
   each fibre ≥ 0.15, ODI ≤ 0.2). That is the right prior for mixed
   in-vivo voxels and a poor one for any orientation benchmark; it also
   means an "in-model held-out" test that filters to clean crossings has
   only ~0.5 % of the simulations to draw from.

### 7.3 Suggestions (all small)

- **A continuous refinement step after the match.** The matched entry
  is an excellent initialisation (it is within 3° in parameter space
  ~always); a few Gauss–Newton or Rprop steps on the same forward model
  from that start would remove most of the 5–10° matching loss without
  changing the library, the prior or the runtime much. This is exactly
  the gap our differentiable fit closes (2–5° on the same substrates)
  and it is not specific to our model.
- **Posterior-weighted peaks when `use_posterior=True`**, or a doc note
  that peaks/ODF are argmax-only. A weighted ODF over the k neighbours
  (which the ODF machinery can already represent) would be the natural
  thing.
- **Expose the fibre-count / WM-fraction composition** of a generated
  library (a one-line histogram in `verbose=True`) and a
  `diffusivity_config`-style knob for the Dirichlet / `wm_threshold`
  prior, so users can build orientation-oriented libraries deliberately.
- **`fraction_array` is (N, 3) with zeros for absent fibres** — fine,
  but worth documenting, since per-fibre fractions are what a fairness
  filter needs.

### 7.4 What is *not* a problem

- Reproducibility is fixed (same seed → same library, same labels).
- The seed-to-seed scatter of doc 004 §11 is gone.
- Recall at shallow crossings (15–25°) is 100 % with `two_fiber_min_angle=0`.
- ND on single bundles is the least-biased intra-axonal fraction of any
  method we ran (0.49 vs geometry 0.50) — dictionary matching does not
  suffer the partial-volume attribution that free-compartment
  optimisers do (doc 008 §7.2).

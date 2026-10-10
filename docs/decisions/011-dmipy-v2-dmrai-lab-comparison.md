# 11. SBI4DWI vs the dmipy 2.x stack (dmrai-lab), October 2026

Status: comparison, written 2026-10-09 from read-only inspection of shallow
clones of `dmrai-lab/{dmipy,dmipy-fit,dmipy-sim,dmipy-design,dmipy-tract,disco-space}`
(latest commits 2026-07-27 to 2026-10-09). Nothing from those repos was
executed. Numbers quoted for their side are the ones their repos state about
themselves; ours are from doc 008/009.

## 0. Why this document exists

dmipy, the 2019 toolbox our compartment models were ported from, was revived
in July 2026 by its original author, Rutger Fick, as a three-package stack
under github.com/dmrai-lab (`pip install dmipy` 2.2.0, docs at dmipy.org).
The AthenaEPI repo now carries a banner redirecting there. dmipy-fit has a
JAX GPU path, which is the overlap that prompted this review. The companion
paper is Fick, arXiv:2607.09401 (surface relaxivity biases f_intra).

Summary: **the two projects overlap on the compartment-model layer and
nowhere else that matters.** Theirs is a forward-physics and fitting stack
with no uncertainty layer, no inference beyond point-estimate L-BFGS-B, no
fixel refinement, no acquisition-information design and no in-vivo
benchmarks. Ours is weak exactly where theirs is strong: substrate breadth,
relaxation physics, sequence realism, cross-simulator parity. The licences
(AGPL-3.0 or commercial vs our MIT) mean code cannot move from theirs to ours;
the natural relationship is oracle-at-the-boundary plus a scientific
collaboration on the f_i bias question.

## 1. The two stacks at a glance

| | SBI4DWI (`dmipy_jax`) | dmipy 2.x (dmrai-lab) |
|---|---|---|
| packages | one (`dmipy_jax`) + `validation/` | dmipy (umbrella 2.2.0), dmipy-fit 2.3.0, dmipy-sim 2.2.0, dmipy-design 2.0.1, dmipy-tract 0.1.0.dev0, disco-space |
| Python LOC (pkg + tests) | 72.7k, 109 test files, 539 tests passing | fit 42k/98 files/633 tests; sim 103k/247 files; design 3.2k; tract 2.7k; space 7.9k |
| licence | MIT | AGPL-3.0-only OR commercial (fit, sim, umbrella); MIT (tract) |
| contributions | PRs welcome | **no pull requests accepted**; issues only |
| authorship | one author + agent sessions | one author (Fick) + agent sessions (each repo has a CLAUDE.md) |
| numerical core | JAX + Equinox pytrees, Optimistix/Optax, BlackJAX, FlowJAX | numpy classes with a hand-written JAX dispatch table; jaxopt L-BFGS-B/OSQP (jaxopt is deprecated upstream); torch for CSD/DTI |
| units | SI | SI (one stale "10^9 m²/s" docstring) |
| in-vivo / phantom benchmarks | DiSCo vs FORCE/MSMT, 42-subject HCP retest, uncertainty validation (doc 008) | none for fitting; DiSCo r ≈ 0.93 for tracking on their own re-simulated phantom |

## 2. Layer by layer

### 2.1 Compartment models (direct overlap)

Both descend from the same 2019 code and the shared classes agree.

- **Callaghan cylinder.** Identical series and root table (`jnp_zeros`, α₀₀ = 0). Ours: `pure_callback` Bessel stack with a recurrence JVP, float32-safe. Theirs: `jax.scipy.special.bessel_jn`, requires float64 (their note: C3/C4 sums overflow in float32). Ours is tested to 1e-9 against the 2019 code (`test_callaghan_reference.py`); theirs has RTAP, free-diffusion and Söderman-limit tests only.
- **Callaghan sphere.** Both replaced the broken 2019 version (cylinder roots used for a sphere) with the same corrected series: roots of j_n', weight 6(2n+1) α²/(α² − n(n+1)), ground state 9 j₁(x)²/x². Independently derived on both sides; theirs is numpy-only (no JAX path), ours is JAX with a differentiable root-free evaluation. Their defaults are (25 roots, 16 orders), ours (20, 50). Nothing to report to him here; it confirms our derivation.
- **Inventory they have and we lack:** matrix-method pores C5/S5/P5 (Callaghan 1997 eigenmodes for canonical pores, from dmipy-sim), Monte-Carlo-replay compartments C6/S6/P6, planes P4/P5, capped cylinders, `G3TemporalZeppelin`, `X0GeneralizedKarger` (N-pool exchange with T1/T2 under SE/STE), NEXI, and the **occupancy-gated factors** (`TransverseRelaxation`, `LongitudinalRelaxation`, `SurfaceRelaxivity` tied to pore S/V) that implement arXiv:2607.09401. 37 named reference models (noddi, sandi, verdict, axcaliber, mte_* multi-TE variants, …).
- **Inventory we have and they lack:** nothing substantive at this layer. Our SANDI/NODDI/Watson/Bingham/Gamma coverage is a subset of theirs.
- **Composition.** Ours are `eqx.Module` pytrees: `jit`/`vmap`/`grad` for free, Hessians available (we use them for Laplace). Theirs are mutable classes; the JAX path is a separate factory per model and only `value_and_grad` is ever taken.

### 2.2 Fitting and inference (no overlap beyond point estimates)

| capability | ours | theirs |
|---|---|---|
| brute init → gradient refine | `GlobalBruteInitializer`, `VoxelFitter` (Optimistix) | `brute2fine` (scipy), `jax` (vmap'd jaxopt LBFGSB, whole slice on GPU), `mix` |
| Rician / fittable σ / nc-χ NLL | Rician (PRISM) | Rician, fittable σ, spherical-mean Rician, nc-χ, all in log-space i0e |
| CSD | via DIPY (MSMT-CSD used in PRISM init) | own Tournier07 and MSMT solvers in JAX and torch, Dhollander responses |
| MCMC | BlackJAX NUTS (`MCMCInference`) | none |
| Laplace / Hessian posteriors | PRISM fixel posterior | none |
| amortised posteriors (MDN, flows), SBC, PPC, conformal, OOD, ensembles | yes | none |
| multi-fixel refinement with spatial priors | PRISM-JAX | none (only `two_fascicle_noddi`, `BundleModel`) |

Grep for posterior / Laplace / Hessian / Fisher / bootstrap / SBI across all
six repos: zero hits outside Laplace–Beltrami CSD regularisation. **The
uncertainty layer is entirely ours.**

### 2.3 Simulation (complementary, theirs far broader)

- **dmipy-sim** is a pure-JAX Monte Carlo engine (vmap over walkers, scan over time), 1.2e9 walker-steps/s on an L40S. Substrates: analytic pores, RSA-packed cylinders/spheres/myelinated cylinders (three pools, two walls), curved/undulating fibres, triangle meshes to 1e6 faces, sphere-union CSG (native CATERPillar geometry), label volumes from EM/micro-CT. Physics: Powles permeability, Brownstein–Tarr surface relaxivity, per-pool T1/T2, MT bound pool, susceptibility fields, vector Bloch with real RF, EPG, PGSTE, FEXI, concomitant and nonlinearity terms, Pulseq I/O, scanner hardware catalogue.
- **Replay packs** (`.rpk`, safetensors, HF `SubstrateCommons`): walk once, then replay any sequence, T2/T1/ρ, pose, B0. DiSCo pack = 150M walkers, 190 GB, replayed for 192 measurements in 0.3 s. Permeability is *not* a replay knob (one walk per κ).
- **Differentiability:** theirs is differentiable in G(t) only (for sequence design). Ours (`DifferentiableWalker`, FEM `MatrixFormalismSimulator` on meshes) differentiates through geometry and D. Their "matrix method" is Callaghan-1997 eigenmodes for canonical pores in numpy, not a mesh eigenbasis.
- **Parity:** theirs against MC/DC released signals, Disimpy/MISST fixtures, Talabi/Ling wet-lab T2 spectra. Ours against MCMRSimulator.jl (live Julia bridge, doc 007) and ReMiDi. Neither cross-checks the other's engine.
- **Oracle boundary:** ours (HDF5 `SimulationLibrary`, `OracleSimulator` protocol in doc 006 §7) is exactly the shape needed to consume `.rpk` replays as training data without importing AGPL code.

### 2.4 Acquisition design (different problems)

- **dmipy-design** ports NOW (Sjölund 2015): maximise b under slew, amplitude, Maxwell, M1/M2, b-tensor shape, PNS and heating constraints via SLSQP; min-TE search; HS + GRAPE refocusing pulses; Pulseq export. Its README states Fisher/CRLB design is out of scope. Its one information-like objective is two-pack discriminability max |E_A − E_B|.
- **Ours** (doc 009, `validation/design_*.py`) is Gaussian Fisher / Bayesian CRLB over shells, directions and timings. A NOW-deliverable waveform could be the inner loop of our outer Fisher search; nothing overlaps.

### 2.5 Tractography and DiSCo (same phantom, incomparable numbers)

- **dmipy-tract** tracks SH FODs (tournier07 basis) probabilistically or deterministically on JAX with dipy parity; 1.6 s vs 1016 s for dipy on DiSCo. Inputs are FODs only: no peaks, fixels or posteriors. Connectivity is raw endpoint counts.
- **disco-space** is a Hugging Face Gradio app that re-simulates DiSCo from the replay pack for any protocol, fits single-tissue CSD, tracks, and reports Pearson on raw upper-triangle counts (0.927 noiseless, 0.904–0.924 with noise; SNR defined at M0, ≈ SNR 11 at b0). **All 95 absent pairs come out connected** (Dice ≈ 0.35), i.e. the point-estimate regime of doc 008 §1.
- This is a fourth protocol (re-simulated DWI, 364 measurements up to b = 13183 s/mm², ST-CSD, raw-count Pearson) and reinforces doc 008 §1.3.3: DiSCo r is not comparable across papers. Our posterior-pruned r 0.887 / Dice 0.73 at SNR 10 and ICC 0.64 on HCP are the numbers that have no counterpart on their side.

## 3. What each side could take from the other

**For SBI4DWI (no code transfer, AGPL):**
1. Consume `.rpk` replay packs via subprocess/HDF5 as a third oracle beside CATERPillar and MCMRSimulator. Replay over T2, ρ, sequence and pose is close to free; κ is not.
2. Cross-check `mesh_sim` FEM against their `math/matrix_method.py` canonical-pore oracle and their MISST fixtures.
3. Add occupancy-gated T2 / surface-relaxivity factors to the PRISM forward model so f_i can be fitted TE-resolved; this is the test of arXiv:2607.09401 that neither side has run on data. We have the multi-TE loader and the Fisher toolkit to design the TE sampling.
4. Treat his `S3SphereCallaghanApproximation` resonance handling (L'Hôpital at x = α) as a check on our masking of near-resonant terms.

**For dmrai-lab (what we can credibly offer, via issue since PRs are refused):**
1. The uncertainty layer as a downstream consumer: PRISM-JAX takes any FOD/peaks and returns a Laplace fixel posterior; dmipy-tract has no uncertainty-aware input today.
2. HCP scan–rescan and connectome-ICC validation methodology (doc 008 §10–13), which their stack lacks entirely.
3. Our Callaghan cylinder reference test (1e-9 vs the 2019 code) as a regression fixture; our float32-safe Bessel path versus their float64 requirement.
4. Fisher/CRLB protocol design as the outer loop their NOW optimiser does not have.

## 4. Positioning

The honest one-liner: *dmipy 2.x is the forward model and fitter; SBI4DWI is
the posterior.* They have not claimed uncertainty, SBI, fixel refinement or
in-vivo validation, and the jaxopt dependency plus class-based architecture
make it unlikely those would be bolted on quickly. Our shipped claim should
stay what doc 008 supports: calibrated fixel posteriors that improve
connectome reliability and rank edge reliability, with an f_i map on par with
NODDI, now with the explicit caveat that both estimators share the
surface-relaxivity bias Fick derives.

## 5. Decision (2026-10-09)

No collaboration outreach to dmrai-lab. The project is AGPL-or-commercial and
does not accept pull requests, and the preference is to work with fully
open-source, PR-based projects. dmipy 2.x is treated as a reference
implementation and a possible oracle at the HDF5 boundary only. Collaboration
effort goes to DIPY (BSD, PR-based; FORCE/PRISM authors), where PRISM-JAX
already plugs in as a `PeaksAndMetrics` refinement stage; see
`docs/outreach/force-dipy-notes.md`.

Contact for the record: Rutger Fick, fick.rutger@gmail.com (GitHub profile).

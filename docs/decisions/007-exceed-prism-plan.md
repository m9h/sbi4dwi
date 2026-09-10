# 7. Plan: Exceed PRISM on Fixel Recovery and Connectivity

## Date

2026-09-08

## Status

Active plan. Replaces doc 004 §24.7 ("the remaining 0.04 gap to paper")
as the roadmap for the DiSCo / fixel line of work. Builds on doc 006.

## Summary

PRISM (Abouagour, Shah, Garyfallidis — arXiv:2604.00250) is the FORCE
authors' move from dictionary matching to differentiable patch-wise
analysis-by-synthesis. It is the method dmipy-JAX was architected to be,
so it is the right thing to measure against — and its own limitations
section, read alongside doc 004's findings, gives a specific list of
places where it can be beaten.

This document records (1) the comparison infrastructure built today,
(2) the levers, ranked by expected gain and grounded in evidence we
already have, (3) what "exceed" means numerically, and (4) the phased
work with tests.

**The single most important framing decision:** absolute Pearson r on
DiSCo connectivity is *not comparable across papers*. PRISM reports
r=0.934 vs MSMT-CSD r=0.920; our MRtrix SD_STREAM reference on the same
phantom is r=0.13 (doc 004 §18) and our best dmipy-JAX is 0.851 (§24).
The connectivity normalisation is unstated in both FORCE and PRISM. The
comparable quantity is therefore **margin over MSMT-CSD on one's own
tracker** — PRISM's is +1.6 pp at SNR=50, K=5 — plus PRISM's synthetic
angular-error benchmark, which *is* fully specified.

---

## 1. What PRISM actually is (from the full text)

Forward model, per voxel:

```
S = S0 · ( f_csf e^{−b·3.0} + f_gm e^{−b·0.9}
         + Σ_k f_wm,k [ f_i·stick_k + (1−f_i)·zeppelin_k ]
         + f_res e^{−b·0.2} )                        (×1e-3 mm²/s)
```

- **Fixed** D∥ = 1.7, D⊥ = 0.4. f_i shared across fibres, init 0.5.
- Per-voxel learnables: S0 (softplus), K+3 fractions (softmax), K unit
  directions, f_i (sigmoid). K = 2 synthetic, 3 HCP, **5 DiSCo**.
- Priors (all λ fixed globally): Huber–Laplacian on fractions vs
  6/26-neighbour mean (λ=0.01, δ=0.05); direction repulsion (0.01);
  L1 sparsity on minor fibres (τ=0.15, λ=0.02); directional continuity
  (0.005); orphan-WM suppression + fibre ordering (0.01).
- Loss: MSE, or Rician NLL with one learned global σ. Also learns
  intensity-domain calibration (gain/bias-field) parameters.
- Optimiser: Rprop. Whole-volume via 30-slice slabs, 5-slice overlap.
  50–300 iterations. No dispersion. Point estimates only.

Reported results:

| Benchmark | PRISM-MSE | PRISM-NLL | MSMT-CSD | ODF-FP |
|---|---|---|---|---|
| Synthetic, SNR=30, angular error | 3.5° | **2.3°** | 6.8° | 11.6° |
| Synthetic, recall | 95% | **99%** | 83% | 86% |
| DiSCo1, SNR=50, K=5, r (best angle) | — | **0.934** | 0.920 | — |
| HCP 741k voxels, RTX A6000 | | ~12 min | | |

Ablation on DiSCo (from r=0.934): remove restricted compartment
**−9.4 pp**; remove topology priors −1.9 pp; remove Huber–Laplacian −0.7 pp.

Stated limitations: local minima from random init ("warm-start from a
dictionary-based method" proposed); no Bingham dispersion; geometric
preprocessing assumed. **Code: "will be released upon acceptance."**

---

## 2. Infrastructure built (2026-09-08)

| Path | What |
|---|---|
| `dmipy_jax/validation/prism_jax.py` | Faithful JAX PRISM: forward model, all four priors, MSE + Rician-NLL(learned σ), `optax.rprop`, whole-mask joint fit in one `lax.fori_loop`. Plus the PRISM-plus hooks: `learn_diffusivities` (banded sigmoid), `init_dirs/init_fracs` warm start. |
| `dmipy_jax/validation/msmt_baseline.py` | dipy MSMT-CSD with PRISM's oracle-response protocol (WM from top-FA voxels; GM/CSF isotropic 0.9/3.0). |
| `dmipy_jax/validation/disco_tracking.py` | The §21.2 eudx tracker + `peaks_to_pam`, factored out so every method uses the identical pipeline. |
| `validation/validate_prism_disco_connectivity.py` | Driver: `msmt`, `prism-mse`, `prism-nll`, `plus-nll`, `plus-warm`; reports r/CCC/Dice and **Δ vs MSMT** per SNR. |
| `tests/validation/test_prism_jax.py` | 16 tests: stick collapse, simplex/unit-norm contracts, PRISM's default constants, each prior's zero/positive cases, 60°/90° crossing recovery in PRISM's own best-match metric, NLL recovers σ≈1/SNR, sparsity suppresses a spurious second fibre, warm start lifts recall to 1.0, PAM adapter. All green. |

Two things the tests already taught us:

1. **Loss scale matters.** PRISM's ℒ_MSE is Σ over measurements, mean
   over voxels. Using a mean over measurements too shrinks the data term
   ~M× and the sparsity prior then deletes real fibres (recall 0.12 on a
   60° crossing). Fixed; pinned by the crossing tests.
2. **Random init loses fibres at 90°** — recall 0.95, same number PRISM
   reports for its MSE mode. Warm start from a 15°-perturbed estimate
   gives recall 1.0 and 0.9° error. This is direct evidence for lever
   §3.2 before touching DiSCo.

Fidelity caveats of the re-implementation (PRISM code is unreleased):
the sparsity prior form Σ_k min(f_k, τ) is our reading of "L1 on minor
fibres below τ"; orphan-WM suppression, fibre-ordering and the
calibration (gain/bias-field) terms are not implemented; no slab
stitching (DiSCo is 40³, fits whole).

---

## 3. Levers to exceed PRISM — ranked

Each lever names the evidence, the mechanism, and the test that decides it.

### 3.1 Learnable diffusivities (the library-prior problem, again)

**Evidence.** PRISM hard-codes D∥=1.7, D⊥=0.4 (in-vivo values). Doc 004
§13/§16 established that DiSCo needs D∥≈0.6, D⊥≈0.35 and that using
in-vivo priors produced a uniform 0.4 NDI bias in FORCE. PRISM's paper
does not say it retuned for DiSCo. Its ablation shows removing the
restricted compartment (D=0.2, isotropic) costs **−9.4 pp** — far more
than any topology prior. The most economical explanation: the
slow-decaying restricted pool is absorbing the diffusivity mismatch
between the fixed 1.7/0.4 zeppelin and the phantom's actual ~0.6/0.35.

**Mechanism.** `PrismConfig(learn_diffusivities=True)` — one global
(D∥, D⊥) pair in a band, sigmoid-parameterised. Later: per-tissue-class.

**Decisive test.** On DiSCo, with learned D: (a) r ≥ faithful PRISM;
(b) the restricted-compartment ablation costs far less than 9.4 pp;
(c) recovered D∥ lands near 0.6. If (b) holds, the restricted
compartment in PRISM is a fudge for wrong diffusivities, which is a
publishable finding on its own.

### 3.2 Warm start from the dictionary matcher (PRISM's own suggestion)

**Evidence.** PRISM: random init → local minima; proposes warm-starting
from a dictionary method. We have exactly that: the §22 500K-entry
Bingham stick+zeppelin `DictionaryMatcher` at r=0.851. Test file already
shows warm start moving recall 0.95 → 1.0.

**Mechanism.** `plus-warm` in the driver: match → (θ,φ,f) per voxel →
`init_dirs`, `init_fracs`. Cost: one library generation (~minutes on GB10).

**Decisive test.** DiSCo, all SNRs: `plus-warm` ≥ `plus-nll` ≥
`prism-nll`, and recall at 15–25° synthetic crossings ≥ 99%.

### 3.3 Bingham dispersion per fibre

**Evidence.** PRISM lists it as future work. `BinghamNODDI` (120-point
grid) already exists and is what the §22 library uses; the
`_bingham_dispersed_zeppelin` helper in `dmipy_disco_dict.py` gives the
extra-axonal half. DiSCo strands are nearly straight, so the gain on
DiSCo may be small; on HCP it is where MSMT-CSD's smooth FODs beat
point-estimate sticks.

**Mechanism.** Add `odi` per fibre (sigmoid to (0.01, 0.5)); replace
stick/zeppelin kernels with Bingham-averaged ones. Cost on DiSCo:
15k × 5 × 120 × 364 ≈ 3.3 GFLOP per forward — trivial.

**Decisive test.** Synthetic benchmark with dispersed ground truth
(κ ∈ {16, 32, 64}): angular error and f_i bias vs undispersed PRISM.

### 3.4 Uncertainty — the axis PRISM cannot occupy

**Evidence.** PRISM is a point estimate. The repo has MDN/flow posteriors
(`inference/`), SBC (`pipeline/sbc.py`), conformal intervals, and the
Manzano-Patron SBI-tractography reference (doc 004 §2) as the pattern.

**Mechanism.** Two-stage: (i) amortised posterior over (dirs, fracs, f_i)
from the §22 simulator → *both* a warm start (§3.2) and per-fixel
uncertainty; (ii) MAP refinement with PRISM-JAX priors; (iii) feed
posterior direction samples to probabilistic tracking. Calibration by
SBC on the synthetic benchmark.

**Decisive test.** Calibrated 90% intervals on fibre angle (SBC rank
histograms flat); connectivity from posterior-sampled tracking with
credible intervals per edge. No number PRISM reports competes with this.

### 3.5 Model-misspecification robustness (ties to doc 006)

**Evidence.** PRISM's synthetic benchmark is generated by *its own
forward model*, so 2.3° is an in-model number. DiSCo strands are
cylinders — still close to the model. Nobody has evaluated any of these
methods on substrates where the forward model is wrong.

**Mechanism.** CATERPillar (today) and OCTOPUS (on release) substrates →
our MC walker → signals with beading/undulation/glia → all methods.
This is doc 006 Phase 2 pointed at a concrete question.

**Decisive test.** Rank order of methods by angular error and f_i bias
under misspecification; hypothesis: learned-D + dispersion degrade
gracefully, fixed-D PRISM does not.

### 3.6 Smaller levers

- **Noise model.** PRISM: one global σ. Options: per-slab σ, noncentral-χ
  for multi-coil in vivo. Expected gain small on DiSCo.
- **Acquisition design.** §24 showed multi-shell was the dominant lever
  in the whole FORCE study. With a differentiable forward model, optimise
  shells/directions for fixel identifiability (doc 004 §6.6). This is
  where "exceed" becomes "need less data".
- **Speed.** PRISM: 12 min for 741k voxels on an A6000. Our whole-volume
  `fori_loop` on GB10 — measure; not a research claim but worth a row.

---

## 4. What "exceed" means

| Benchmark | PRISM's number | Target | Status |
|---|---|---|---|
| Synthetic, SNR=30, mean angular error | 2.3° (NLL) | **< 2.0°** | met on our protocol (1.52°, §6.2) — protocol caveat |
| Synthetic, recall (all angles) | 99% | **≥ 99%**, and ≥ 95% at 15–20° | met on our protocol (100%, §6.2) |
| DiSCo, K=5: margin over MSMT-CSD on same tracker | +1.6 pp | **> +1.6 pp** | **met**: +6.7 / +9.9 / +12.8 pp at SNR 50/30/10 (§6.5–6.6) |
| DiSCo: beat faithful PRISM-JAX | — | at **every** SNR ∈ {10, 30, 50} | **met** (§6.6); also beats §24 dictionary at SNR 10/30 |
| Restricted-compartment ablation cost with learned D | −9.4 pp | **< −3 pp** (§3.1 hypothesis) | pending |
| Calibrated fixel uncertainty | n/a | SBC-flat, per-edge CIs | pending |
| Misspecified substrates (CATERPillar/OCTOPUS) | n/a | graceful degradation vs fixed-D | pending |

First DiSCo run (SNR=50, K=5, 300 it): `validation/prism_disco_connectivity_results_snr50_first.npz` — results to be recorded in §6 when complete.

---

## 5. Phased work

### Phase A — fair comparison (this week)

- [x] PRISM-JAX + priors + both losses, tested
- [x] MSMT-CSD oracle-response baseline on the shared tracker
- [x] Driver with Δ-vs-MSMT
- [ ] Run SNR {10, 30, 50}, K=5, all five methods; record §6
- [ ] Implement PRISM's synthetic benchmark generator exactly
      (3 shells × 64 dirs, N=193; 15°–90° in 5° steps; 3,400 voxels;
      SNR 10/30/50) as `dmipy_jax/validation/prism_synthetic.py` + tests
- [ ] Time the whole-volume fit vs PRISM's 12 min / 741k

### Phase B — levers 3.1 + 3.2 (next)

- [ ] `plus-nll` and `plus-warm` on DiSCo, all SNRs
- [ ] Restricted-compartment ablation, fixed-D vs learned-D (decides §3.1)
- [ ] Synthetic benchmark: PRISM vs plus-warm at 15–25° crossings

### Phase C — dispersion (3.3)

- [ ] Bingham-averaged stick+zeppelin kernels in `prism_jax.forward`
      (reuse `BinghamNODDI`, `_bingham_dispersed_zeppelin`)
- [ ] Dispersed synthetic ground truth; f_i bias comparison

### Phase D — uncertainty (3.4)

- [ ] Amortised posterior from §22 simulator (MDN first), SBC
- [ ] Posterior-sample tracking with per-edge CIs on DiSCo

### Phase E — misspecification (3.5, = doc 006 Phase 2)

- [ ] CATERPillar substrate → MC → benchmark all methods
- [ ] OCTOPUS on release

### Dropped from doc 004 §24.7

- 1M-entry library and Watson-vs-Bingham as *primary* levers. Both are
  refinements of dictionary matching; the dictionary's new job is warm
  start (§3.2), where 500K is already enough.

---

## 6. Results log

### 6.1 First DiSCo run — SNR=50, K=5, 300 Rprop iterations, max_angle=45°, peak_frac_min=0.05 (2026-09-08)

`validation/prism_disco_connectivity_results_snr50_first.npz`

| method | r | CCC | Dice | Δ vs MSMT | streamlines | fit time | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| MSMT-CSD (oracle responses) | 0.776 | 0.769 | **0.565** | — | 37,869 | 169 s | dipy mcsd, cvxpy per voxel |
| PRISM-JAX MSE (faithful, D∥=1.7 D⊥=0.4) | undefined | — | 0.000 | — | 2,427 | 7 s | f_i→0.82; ~2 peaks/voxel; tracking collapses |
| PRISM-JAX NLL (faithful) | undefined | — | 0.000 | — | 1,796 | 7 s | same failure |
| **PRISM-plus NLL (learned D)** | **0.830** | 0.829 | 0.376 | **+5.4 pp** | 118,181 | 8 s | recovered D∥ = **0.648e-9**, D⊥ = 0.42e-9; 4.8 peaks/voxel |
| (doc 004 §24, dictionary, for reference) | 0.851 | 0.849 | 0.420 | +7.5 pp | — | — | different method, same tracker |

Three findings, in order of importance:

1. **Faithful PRISM with its published fixed diffusivities does not
   work on DiSCo.** The in-vivo D∥=1.7 cannot fit a ~0.6 phantom; f_i
   saturates toward the stick to compensate, direction estimates
   degrade, and the tracker produces so few streamlines that r is
   undefined (zero-variance matrix). This is §3.1's hypothesis
   confirmed in the strongest form. PRISM's paper must have retuned for
   DiSCo (as FORCE §3.2 did) or relied on the restricted compartment +
   calibration terms to absorb the mismatch — either way it is
   unstated. `prism-tuned-nll` (D fixed at 0.6/0.35) is added to the
   driver as the fair "PRISM as its authors would run it" row.
2. **Learning the diffusivities recovers D∥ = 0.648e-9 — inside the
   FORCE-paper DiSCo band (0.54–0.66) — with no prior knowledge**, and
   lands at +5.4 pp over MSMT-CSD on the identical tracker. PRISM's
   reported margin is +1.4 pp. First run, no warm start, no tuning.
3. **MSMT-CSD through dipy peaks + eudx gives r = 0.776 on this
   phantom.** Doc 004 §18 found MRtrix SD_STREAM at r = 0.13 on the same
   data and called the gap "structural to the benchmark". It was
   tracker-specific: the same CSD family with the §21 eudx pipeline is a
   strong, specific baseline (Dice 0.565, the best specificity of any
   method to date). §18.4's conclusion should be amended.

Open issue from this run: PRISM-plus keeps ~4.8 of 5 fibres above the
0.05 cut and emits 3× MSMT's streamlines — high sensitivity, poor
specificity (Dice 0.376). The sparsity prior at λ=0.02 is not selecting
on DiSCo's actual crossings. Next: sweep `peak_frac_min` ∈ {0.05, 0.10,
0.15} and `max_angle` ∈ {25, 30, 45} (PRISM sweeps 15–30° and reports
the best), then warm start.

### 6.2 PRISM synthetic crossing benchmark — SNR=30, 3,400 voxels, 300 iterations (2026-09-08)

`validation/prism_synthetic_results.npz`. Protocol per §1; our fixed
choices (0.5/0.5 fractions, f_i=0.5, no iso compartments in GT) are
stated in `prism_synthetic.py`. **In-model benchmark** — signals come
from the same forward model every PRISM variant fits.

| method | overall error | recall | 15° | 20° | 25° | 30° | fit time |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSMT-CSD (first pass, FA-percentile response) | 10.45° | 85.3% | 7.6° | 10.2° | 12.7° | 15.1° / 50% | 62 s |
| PRISM-JAX MSE (re-impl.) | 1.82° | 100% | 5.9° | 4.1° | 3.1° | 2.4° | 5 s |
| PRISM-JAX NLL (re-impl.) | 1.68° | 100% | 4.4° | 3.4° | 2.9° | 2.2° | 4 s |
| **PRISM-plus warm start** (DictionaryMatcher, 300K PRISM-model library; run 2026-09-09) | **1.59°** | 100% | **3.2°** | 3.1° | 2.8° | 2.3° | 3 s |
| *(earlier stand-in: GT ⊕ 15° noise)* | *1.52°* | *100%* | *2.6°* | *2.9°* | *2.8°* | *2.3°* | *1 s* |
| *PRISM paper, MSE* | *3.5°* | *95%* | | | | | |
| *PRISM paper, NLL* | *2.3°* | *99%* | *3.1–5.8° at ≤30°* | | | | |
| *PRISM paper, MSMT-CSD* | *6.8°* | *83%* | | | | | |

Reading:

- The `< 2.0°` target in §4 is met by all three PRISM-JAX variants on
  *our* protocol. Since our re-implementation beats the paper's own
  numbers for the same method, the paper's benchmark is almost
  certainly harder than ours (varying fractions / f_i / iso signal not
  described in the extract we have). Do **not** claim "1.5° vs 2.3°";
  claim the *within-protocol* ordering: a real dictionary warm start
  (the same mechanism as on DiSCo) cuts the 15° error from 4.4° to 3.2°
  (NLL) and is the only variant ≤ 3.2° at every crossing angle. The
  GT-perturbed stand-in used on the first day is kept for the record
  and is no longer cited.
- Recall is 100% everywhere for PRISM-JAX, including 15°. The paper's
  95%/99% suggests their harder setting or their orphan/ordering priors
  interacting with detection; ours has no such loss.
- Our MSMT baseline: 10.2° / 85.7% recall after switching the WM
  response to the GT single-fibre voxels (the paper's "oracle
  response"; the FA-percentile mask made no difference). Recall matches
  the paper's 83%; error is worse (10.2° vs 6.8°). Diagnosis at 45°:
  dipy's `MultiShellDeconvModel` returns a single merged lobe for
  every voxel regardless of sphere / peak thresholds / SH order 8–10,
  while single-shell CSD on b=3000 resolves 87% of them — the
  multi-shell fit is dominated by the unresolvable b=1000 shell. Volume
  fractions are sane (WM 1.07, iso 0), so it is not tissue leakage.
  Treat our MSMT row as "dipy MSMT-CSD" and do not tune it further; the
  DiSCo MSMT row (r=0.776) is the one that matters for the margin.
- Timing: whole-set joint fit is 4–5 s on GB10 for 3,400 voxels × 193
  measurements; the warm-started fit converges in 1 s.

### 6.3 DiSCo sweep — SNR=50, K=5: `peak_frac_min` × `max_angle` (2026-09-08)

`validation/prism_disco_connectivity_results_sweep_pf*_ma*.npz`. Same
fit each time (fit is 4–8 s); only the peak cut and tracker angle vary.
MSMT-CSD reference on this tracker: r = 0.776 (max_angle 45°).

| method | D∥ / D⊥ (×1e-9) | pf=0.05, 25° | 30° | 45° | pf=0.10, 25° | 30° | **45°** | pf=0.15, 45° |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PRISM-JAX NLL, tuned (D fixed to DiSCo regime) | 0.60 / 0.35 | 0.592 | 0.722 | 0.841 | 0.597 | 0.722 | **0.842** | 0.773 |
| PRISM-plus NLL (D learned) | 0.648 / 0.424 | 0.583 | 0.693 | 0.830 | 0.589 | 0.699 | **0.832** | 0.758 |

Peaks kept per voxel at pf 0.05 / 0.10 / 0.15: tuned 4.8 / 3.4 / 1.8;
plus 4.8 / 3.7 / 2.2.

Reading:

- **Best: tuned PRISM-JAX r = 0.842 (+6.6 pp over MSMT), PRISM-plus
  0.832 (+5.6 pp)** — both ~4× PRISM's reported margin, and within
  0.01–0.02 of the §24 dictionary result (0.851) that took 500K library
  entries. Both exceed PRISM's own margin on our tracker; neither yet
  beats §24 outright.
- **Knowing D beats learning D by ~1 pp.** The learned D⊥ (0.42) sits
  above the phantom's tortuosity value (~0.35); D∥ (0.648) is inside
  the band. The gap is the cost of *not knowing* the regime — small, and
  the learned variant needs no prior knowledge, which is the whole
  point on real data. A D⊥ tortuosity coupling (D⊥ = D∥(1−f_i)), as in
  the §22 simulator, may remove it.
- **max_angle 45° ≫ 30° ≫ 25° on this tracker** (0.84 / 0.72 / 0.59).
  PRISM sweeps 15–30° and reports the best; with eudx + DiSCo's ROI
  geometry the tight angles truncate streamlines. This is another reason
  absolute r is not comparable across papers.
- **pf = 0.10 is the sweet spot**: drops ~1.4 spurious fibres per voxel
  for +0.002 r; pf = 0.15 starts deleting real fibres (−0.07). The
  sparsity prior at PRISM's λ = 0.02 is not doing the selection on
  DiSCo — the hard cut is. Worth revisiting λ_sparse, τ once warm start
  is in.

Updated §4 status: *DiSCo margin over MSMT > +1.6 pp* — **met** (+6.6 /
+5.6 pp). *Beat faithful PRISM-JAX at every SNR* — met at SNR=50 in the
strong sense (faithful fixed-D PRISM is undefined); SNR 10/30 pending.

### 6.4 Warm start from the §22 dictionary — DiSCo SNR=50, K=5, pf=0.05, 45° (2026-09-08)

`validation/prism_disco_connectivity_results_warm_snr50.npz`

| method | r | Δ vs MSMT | D∥ / D⊥ learned | fit time |
|---|---:|---:|---|---:|
| PRISM-plus NLL, random init (§6.1) | 0.830 | +5.4 pp | 0.648 / 0.424 | 8 s |
| **PRISM-plus NLL, warm start from 500K §22 library** | **0.840** | **+6.4 pp** | 0.695 / 0.478 | 33 s (incl. match) |
| PRISM-JAX NLL, tuned D (§6.3, same pf/angle) | 0.841 | +6.5 pp | 0.60 / 0.35 fixed | 4 s |

Warm start recovers the ~1 pp that random init lost to tuned-D, without
knowing D. Both learned diffusivities drift *upward* with warm start
(D⊥ 0.42 → 0.48), so the remaining lever on DiSCo is the D⊥ treatment
(tortuosity coupling), not initialisation. Combined with §6.3's pf=0.10
(+0.002) the expected best is ≈ 0.842–0.845 — level with tuned-D and
~0.01 below the §24 dictionary (0.851), which had Bingham dispersion
and a tortuosity-coupled zeppelin — the two things PRISM-JAX lacks.

### 6.5 Tortuosity-coupled D⊥ — DiSCo SNR=50, K=5, pf=0.10, 45° (2026-09-08)

`validation/prism_disco_connectivity_results_tort_snr50.npz`. D⊥ = D∥(1−f_i)
per voxel (Szafer–Stanisz, as in the §22 simulator); D∥ still learned.

| method | r | Δ vs MSMT | D∥ / mean D⊥ | f_i mean | peaks/voxel |
|---|---:|---:|---|---:|---:|
| PRISM-plus NLL, learned D (§6.3, same pf/angle) | 0.832 | +5.6 pp | 0.648 / 0.424 | 0.25 | 3.7 |
| PRISM-plus + tortuosity, random init | 0.832 | +5.6 pp | 0.629 / 0.470 | 0.25 | 3.8 |
| PRISM-plus + warm start (§6.4, pf 0.05) | 0.840 | +6.4 pp | 0.695 / 0.478 | 0.25 | 4.6 |
| **PRISM-plus + tortuosity + warm start** | **0.843** | **+6.7 pp** | 0.636 / 0.483 | 0.24 | 3.6 |
| PRISM-JAX NLL, tuned D (oracle, §6.3) | 0.842 | +6.6 pp | 0.60 / 0.35 fixed | 0.24 | 3.4 |
| doc 004 §24 dictionary (Bingham + tortuosity) | 0.851 | +7.5 pp | — | v_ic 0.34 | — |

**PRISM-plus with everything learned now equals the oracle-tuned PRISM
(0.843 vs 0.842)** and sits 0.008 below the §24 dictionary. Tortuosity
alone is worth nothing at random init and +0.003 with warm start — the
D⊥ hypothesis in §6.4 was wrong in the specific: what matters is that
f_i is estimated at ~0.25 (a physically sensible DiSCo intra fraction)
and the tracker sees fewer, cleaner peaks (3.6/voxel). The remaining
0.008 to §24 is within the seed-to-seed noise of a 300-iteration Rprop
fit and a single tracking seed; establishing it as real needs ≥3 seeds.
Bingham dispersion (§3.3) is the only model-form difference left.

### 6.6 DiSCo SNR 10 and 30 — K=5, pf=0.10, 45° (2026-09-08)

`validation/prism_disco_connectivity_results_snr10_30.npz`

| SNR | MSMT-CSD | PRISM-JAX tuned D (oracle) | PRISM-plus learned D | **PRISM-plus warm** | §24 dictionary |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.721 | 0.845 (+12.3 pp) | 0.833 (+11.2) | **0.850 (+12.8 pp)** | 0.772 |
| 30 | 0.758 | 0.849 (+9.1 pp) | 0.845 (+8.7) | **0.857 (+9.9 pp)** | 0.811 |
| 50 | 0.776 | 0.842 (+6.6 pp) | 0.832 (+5.6) | 0.843 (+6.7 pp) *(§6.5, +tort)* | 0.851 |

- **PRISM-plus (warm) beats the §24 dictionary outright at SNR 10 and
  30** — by 0.078 and 0.046 — and ties it at SNR 50. Every
  differentiable variant, including learned-D random-init, beats §24 at
  SNR ≤ 30. The dictionary's advantage at high SNR was resolution; at
  low SNR the joint spatial fit's priors are worth far more than
  library density. This settles doc 004 §24.7: the remaining "gap to
  paper" was never a library-design problem.
- Margin over MSMT grows as SNR falls (+6.7 → +12.8 pp), i.e. the
  spatial priors are doing exactly what PRISM claims for them, and
  more strongly than PRISM's +1.4–1.9 pp.
- **Learned D∥ drifts out of band at SNR 10** (1.04e-9 vs the phantom's
  ~0.6): the fit is absorbing noise into diffusivity. It still wins on
  connectivity, but the recovered microstructure is wrong. Fix options:
  tighten `d_par_range`, or a weak prior on D. This is the concrete
  hazard of lever §3.1 and should be tested before any in-vivo claim.
- Dice is flat at 0.35–0.38 for all PRISM variants vs 0.47–0.55 for
  MSMT: the differentiable fits still emit more false-positive edges.
  Specificity is the open weakness, not sensitivity.

Updated §4 status: *beat faithful PRISM-JAX at every SNR* — **met**
(faithful fixed-D is undefined; oracle-tuned PRISM is beaten at 10/30
and tied at 50). *Margin > +1.6 pp* — **met at all SNRs**.

### 6.7 Seeds and the D∥ prior (2026-09-09)

**Seeds.** `plus-tort-warm` at SNR=50 with seeds {0,1,2}: r = 0.8437 all
three, sd 0. With a warm start the pipeline is deterministic — the
library key is fixed, eudx tracking is deterministic given the seed
mask, and DiSCo ships one noise realisation per SNR. The 0.008 between
0.843 and §24's 0.851 is therefore *not* seed noise in this pipeline;
the sources of variance that remain are the library key (warm start)
and random direction init (non-warm variants). Treat 0.008 as real but
small until a library-seed sweep says otherwise.

**D∥ prior** (`lam_diffusivity_prior=1.0`, sd = 0.3 log units around
D∥ = 0.6e-9), warm + tortuosity, pf 0.10, 45°:

| SNR | no prior (§6.5/6.6) | with prior | D∥ no prior → with |
|---:|---:|---:|---|
| 10 | 0.850 | 0.837 | 1.03e-9 → **0.73e-9** |
| 30 | 0.857 | 0.848 | 0.72e-9 → 0.65e-9 |
| 50 | 0.843 | 0.837 | 0.64e-9 → 0.64e-9 |

The prior does its job — D∥ is pulled back inside the phantom's band at
SNR 10 — at a cost of ~1 pp connectivity at every SNR. So the
out-of-band D∥ at low SNR was *helping* tractography: a longer stick
sharpens the angular contrast the tracker uses, at the expense of the
microstructure being wrong. Two honest deployments, then: connectivity-
optimal (no prior, D free) and microstructure-honest (prior). Neither
is "the" PRISM-plus; report both. Still +11.6 / +9.0 / +6.1 pp over
MSMT with the prior on.

### 6.8 Dispersion — synthetic benchmark with Watson-dispersed ground truth, SNR=30 (2026-09-09)

`validation/prism_synthetic_results_odi{0.1,0.2,0.3}.npz`. Same 3,400-voxel
protocol as §6.2, but every fibre is Watson-dispersed at the stated ODI
(generated by the dispersed forward model). This is the §3.3 decisive
test: PRISM has no dispersion; PRISM-plus learns an ODI per fibre.

| GT ODI | method | overall err | recall | err @15° | err @30° | single-fibre err | f_i (GT 0.5) | ODI recovered |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 0.1 | PRISM-JAX NLL (no dispersion) | 4.80° | 100% | 10.2° | 7.0° | 10.5° | **0.35** ± 0.05 | — |
| 0.1 | PRISM-plus warm, no dispersion | 4.78° | 100% | 10.3° | 7.1° | 9.9° | 0.35 | — |
| 0.1 | **PRISM-plus warm + dispersion** | **3.30°** | 100% | **4.4°** | **4.0°** | **3.8°** | **0.53** ± 0.04 | 0.087 |
| 0.2 | PRISM-JAX NLL | 9.10° | 97.7% | 12.9° | 10.8° | 12.8° | **0.27** ± 0.10 | — |
| 0.2 | **PRISM-plus warm + dispersion** | **6.77°** | 98.2% | **7.5°** | **5.8°** | **7.2°** | **0.49** ± 0.05 | 0.135 |
| 0.3 | PRISM-JAX NLL | 13.29° | 85.1% | 14.1° | 12.7° | 13.7° | **0.25** ± 0.12 | — |
| 0.3 | **PRISM-plus warm + dispersion** | **10.72°** | 89.4% | **9.9°** | **8.5°** | **9.3°** | **0.47** ± 0.06 | 0.170 |

Three things, in order:

1. **Undispersed PRISM absorbs dispersion into f_i.** With GT f_i = 0.5,
   PRISM-JAX recovers 0.35 / 0.27 / 0.25 as ODI rises — a 30–50 %
   under-estimate of the intra-axonal fraction, with a per-voxel spread
   (±0.10–0.12) that makes it look like biology. The dispersed variant
   returns 0.53 / 0.49 / 0.47. This is the model-form analogue of §6.1's
   fixed-diffusivity finding: a parameter PRISM does not model is paid
   for by a parameter it reports.
2. **Direction accuracy at narrow crossings halves** with dispersion
   modelled (15°: 10.2° → 4.4° at ODI 0.1; 12.9° → 7.5° at 0.2). Wide
   crossings (45–60°) are unchanged — dispersion mostly blurs the
   *narrow* case, exactly where PRISM claims its advantage.
3. **ODI itself is under-recovered** (0.087 / 0.135 / 0.170 for 0.1 /
   0.2 / 0.3): three shells at b ≤ 3000 do not separate dispersion from
   f_i fully, and 300 iterations from ODI-init 0.1 may not be converged.
   Directions and f_i benefit anyway. Do not report the ODI map as
   quantitative without the DiSCo-style high-b shell.

Cost: the dispersed fit is ~30 s vs 1–7 s for 3,400 voxels (13-term
Legendre scan through Rprop's 300 iterations); the GPU compile
pathology that made this minutes-to-never is fixed (`b47bb8f`).

### 6.9 Dispersion on DiSCo — K=5, pf=0.10, 45°, warm + tortuosity + learned D (2026-09-09)

`validation/prism_disco_connectivity_results_disp.npz`

| SNR | PRISM-plus warm+tort (§6.5/6.6) | **+ Watson dispersion** | Δ | ODI recovered | peaks/voxel | Dice | fit time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.850 | **0.855** | +0.005 | 0.020 | 3.7 | 0.350 | 349 s |
| 30 | 0.857 | 0.839 | −0.018 | 0.025 | 3.1 | 0.391 | 349 s |
| 50 | 0.843 | **0.850** | +0.007 | 0.030 | 2.9 | **0.420** | 408 s |

- **On DiSCo, dispersion is a wash for connectivity** (+0.5 / −1.8 / +0.7
  pp) — as §3.3 predicted: the strands are nearly straight and the fit
  says so (ODI 0.02–0.03, at the bottom of the allowed range). The
  SNR 30 loss is a single deterministic run; treat as noise until a
  library-seed sweep says otherwise.
- **It does buy specificity**: 2.9 peaks/voxel and Dice 0.42 at SNR 50
  vs 3.6 / 0.38 undispersed — the best Dice of any differentiable
  variant, closing a third of the gap to MSMT's 0.565. A model that can
  explain angular blur as dispersion no longer needs a spurious extra
  fibre to do it.
- **SNR 50 now equals the §24 dictionary (0.850 vs 0.851)** while also
  beating it at SNR 10/30; the last model-form difference between
  PRISM-plus and §24 is gone, and with it the last item on doc 004
  §24.7.
- Cost: ~6 min per fit vs 30 s. The 13-term Legendre scan over
  (N,K,M) with 300 Rprop iterations is memory-bound (~66 GB on GB10).
  Halving `n_legendre` to 7 is safe for ODI ≥ 0.05 and would roughly
  halve this; on DiSCo specifically, dispersion should simply be
  switched off.

### 6.10 Where this leaves the §4 targets

| Target | Status |
|---|---|
| Synthetic error < 2.0° / recall ≥ 99% | met on our protocol (1.52° / 100%); protocol caveat stands |
| DiSCo margin over MSMT > +1.6 pp | met: +6.7 / +9.9 / +12.8 pp (SNR 50/30/10) |
| Beat faithful PRISM-JAX at every SNR | met; fixed-D PRISM is undefined on DiSCo |
| Restricted-compartment ablation with learned D | **done** — learned-D model is better *without* it (+1.4 pp); fixed-D loses 1.9 pp vs PRISM's 9.4 (§6.11) |
| Calibrated fixel uncertainty | **done** — 90 % cones cover 87–88 % at SNR ≥ 30 (§6.12); posterior-mean connectome r = 0.859, FP edges 4.7× more uncertain (§6.13) |
| Misspecified substrates | **done** (§6.14) — dispersion-aware PRISM-plus best on single bundles; fixed-D PRISM matches/beats learned-D at crossings; all methods over-estimate f_i under hindrance |

Honest summary of what "exceeding PRISM" currently rests on: (i) fixing
a design flaw PRISM shares with FORCE (fixed in-vivo diffusivities),
(ii) implementing the warm start PRISM proposed but did not build, and
(iii) modelling dispersion, which PRISM lists as future work and which
we show it otherwise absorbs into f_i. None of these is a new
estimator; all are things the JAX forward-model stack made a one-day
job each. The uncertainty axis (§3.4) is where the claim would become
qualitative.

### 6.11 Restricted-compartment ablation — DiSCo SNR=50, K=5, pf=0.10, 45° (2026-09-09)

`validation/prism_disco_connectivity_results_ablation_snr50.npz`. PRISM
reports **−9.4 pp** when its restricted isotropic pool (D=0.2) is removed.

| method | with restricted | without | Δ | D∥ learned |
|---|---:|---:|---:|---|
| PRISM-JAX NLL, tuned D (oracle 0.6/0.35) | 0.847 | 0.829 | **−1.9 pp** | fixed |
| PRISM-plus (learned D, tortuosity, warm) | 0.837 | **0.851** | **+1.4 pp** | 0.636 → 0.584 |

- With diffusivities *learned*, the restricted pool is not just
  unnecessary — removing it helps (+1.4 pp) and D∥ moves further into
  the phantom's band (0.58). With diffusivities *fixed*, removing it
  costs 1.9 pp. Both directions support §3.1's reading: on a phantom
  whose diffusivity the model does not know, the slow isotropic pool
  is a diffusivity fudge. Our −1.9 pp (fixed-but-correct D) vs PRISM's
  −9.4 pp is consistent with PRISM having run DiSCo with diffusivities
  further from the truth than our tuned 0.6/0.35.
- **Cross-process variance caveat.** `plus-tort-warm` scored 0.8437 in
  §6.5 (three seeds, one process) and 0.8367 here (new process, same
  flags). The pipeline is deterministic *within* a process but the
  500K library generation / XLA autotuning is not bit-reproducible
  across processes: ±0.7 pp is the run-to-run floor. Every DiSCo Δ
  under ~1 pp in this document is inside that floor, including this
  ablation's +1.4 pp being "only" marginally outside it. The
  fixed-D −1.9 pp and the SNR-10/30 margins over §24 (+4.6 / +7.8 pp)
  are not.

**PRISM-plus without the restricted pool, learned D, tortuosity, warm
start: r = 0.851 at SNR 50 — level with the §24 dictionary (0.851)
without dispersion, at 20 s per fit.** New default for the `plus-*`
rows going forward.

### 6.12 Fixel uncertainty — Laplace posterior, calibration on the synthetic benchmark (2026-09-09)

`dmipy_jax/validation/prism_uncertainty.py`; `validation/prism_uncertainty_calibration.npz`.
PRISM is a point estimate. We take the per-voxel **Gauss–Newton
(Fisher) curvature** JᵀJ/σ² of the Rician NLL at the PRISM-plus MAP over
[tangent coords of each fibre, f_i logit, fraction logits] (globals σ, D
fixed; spatial coupling ignored → block diagonal; weak priors — 57° on
directions, 3 logit-units on fractions — so degenerate co-linear fibre
pairs and unidentified minor fibres cannot leak unbounded variance into
the main fibre). The exact Hessian was tried first and is indefinite at
a 300-iteration MAP: its clipped inverse gave σ_θ of 10³–10⁴°. Output:
a 2×2 tangent covariance per fixel → angular σ_θ, direction samples.

**Calibration**: fraction of (voxel, true fibre) pairs whose truth lies
inside the 90 % credible cone of the best-matching fixel (Mahalanobis
in the tangent plane vs χ²₂). 6,600 fixels per SNR.

| SNR | overall coverage (target 90 %) | 15° | 25° | 30° | 45° | 60° | 90° | single |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 80.5 % | 55 % | 68 % | 75 % | 84 % | 86 % | 86 % | 97 % |
| 30 | **87.3 %** | 71 % | 88 % | 86 % | 92 % | 90 % | 90 % | 98 % |
| 50 | **88.0 %** | 80 % | 89 % | 88 % | 92 % | 90 % | 90 % | 100 % |

Median σ_θ vs actual mean error, SNR 30: 25° → 2.71° vs 2.93°; 45° →
1.30° vs 1.24°; 90° → 0.92° vs 0.93°. **At every crossing ≥ 25° and
SNR ≥ 30 the Laplace σ_θ predicts the realised angular error to within
~0.2°**, and the 90 % cone covers 86–92 %.

Where it fails, honestly: **narrow crossings (≤ 20°) under-cover** —
52–84 % — because two nearly-merged fibres give a bimodal, non-Gaussian
posterior that a quadratic expansion at the MAP cannot represent; SNR 10
compounds this. Single-fibre voxels *over*-cover (97–99 %) because the
fit sometimes splits the fibre into two co-linear copies; their
individual σ_θ (median 5.6° at SNR 50 vs 1.0° actual error) is honest
about that degeneracy even though the voxel's mean direction is fine.
Both are diagnosable from σ_θ itself.

This is the first of the §3.4 deliverables: a per-fixel angular
uncertainty that is calibrated where PRISM's point estimate is most
confident and flags where it isn't. Cost: one vmapped Hessian, 1–12 s
for 3,400–15,000 voxels.

### 6.13 Posterior-sampled tractography on DiSCo — SNR 50, K=5, 20 samples (2026-09-09)

`validation/validate_prism_posterior_disco.py`; `validation/prism_posterior_disco_snr50.npz`.
Best PRISM-plus (learned D, tortuosity, no restricted pool, warm start)
→ Gauss–Newton Laplace posterior per fixel (median σ_θ = 4.9°, 90th
percentile 17°) → 20 direction-field samples → 21 connectomes on the
§21.2 tracker.

| quantity | value |
|---|---:|
| r(MAP connectome) | 0.851 |
| **r(posterior-mean connectome)** | **0.859** |
| r over the 20 samples | 0.852 ± 0.009 [0.839, 0.868] |
| median 90 % CI relative width, GT-positive edges | 0.39 |
| median 90 % CI relative width, GT-zero edges | 1.61 |
| GT-zero edges whose CI lower bound reaches 0 | 26 % (of 95) |
| coefficient of variation, GT-zero vs GT-positive edges | 0.61 vs 0.13 (Mann–Whitney p = 1.3 × 10⁻¹³) |

- **The posterior-mean connectome is the best DiSCo number in this
  document (0.859)** — above the MAP (0.851) and above the §24
  dictionary (0.851). Averaging over direction uncertainty acts as a
  principled version of the probabilistic-tracking smoothing that
  helped the FORCE paper's own results.
- **The uncertainty separates false-positive edges from true ones.**
  Spurious connections have ~4.7× the relative spread of real ones, and
  a quarter of them have a credible interval that includes zero. This
  is the qualitative capability §3.4 was after: PRISM's Dice deficit
  vs MSMT (§6.6) is a *specificity* problem, and per-edge credible
  intervals are a direct handle on it that no point-estimate method
  can offer. A CI-thresholded connectome is the obvious next
  experiment.
- Cost: fit 5 s (warm) + Laplace 4 s + 21 trackings 17 s. The whole
  posterior connectome takes less time than one MSMT-CSD fit.
- Honest caveats: the Laplace ignores spatial-prior coupling (intervals
  are per-voxel conservative); the 90 % cones under-cover at crossings
  ≤ 20° (§6.12); this is one noise realisation, one tracking seed.

### 6.14 Model misspecification — Monte Carlo on CATERPillar substrates (2026-09-09)

`dmipy_jax/validation/substrate_benchmark.py`, `validation/validate_prism_substrate.py`,
Slurm job 1739 log. Every earlier benchmark was in-model (PRISM's own
stick+zeppelin) or near-model (DiSCo cylinders). Here the signal comes
from Brownian walkers in CATERPillar axon substrates (overlapping-sphere
axons, ICVF ≈ 0.5, c₂ = 0.98, 20 µm box; a second population rotated
into a sheet crossing; intra walkers periodic along y/z, extra periodic
in all axes, elastic reflection with an axon-id check that forbids
tunnelling between axons — the first version leaked and gave
free-diffusion signals, caught by the physics tests). Two geometries:
*straight* and *tortuous + beaded* (beading amplitude 0.3). PRISM's
3-shell scheme, δ = 6 ms, Δ = 12 ms, D = 2.0 µm²/ms in both
compartments, SNR 30, 64 voxels per condition. No method is told any of
this; PRISM-plus variants start from D∥ = 1.7. MSMT gets its WM response
from the single-bundle voxels of the same geometry.

**All 10 conditions (2 geometries × 5 angles)**

| method | mean error | mean recall | mean f_i bias (est − geom) |
|---|---:|---:|---:|
| prism-nll | 5.63° | 99.8 % | +0.150 |
| plus | 5.53° | 98.4 % | +0.086 |
| plus-dprior | 5.63° | 98.2 % | +0.098 |
| plus-disp | 5.20° | 99.1 % | +0.187 |
| msmt | 10.46° | 88.5 % | — |

**Single bundles (angle 0), straight + tortuous**

| method | mean error | mean recall | mean f_i bias (est − geom) |
|---|---:|---:|---:|
| prism-nll | 7.58° | 100.0 % | +0.124 |
| plus | 5.00° | 100.0 % | +0.097 |
| plus-dprior | 5.18° | 100.0 % | +0.106 |
| plus-disp | 2.43° | 100.0 % | +0.159 |
| msmt | 4.74° | 100.0 % | — |

**Crossings (30–90°), straight + tortuous**

| method | mean error | mean recall | mean f_i bias (est − geom) |
|---|---:|---:|---:|
| prism-nll | 5.14° | 99.7 % | +0.156 |
| plus | 5.66° | 98.0 % | +0.083 |
| plus-dprior | 5.75° | 97.8 % | +0.096 |
| plus-disp | 5.89° | 98.8 % | +0.195 |
| msmt | 11.89° | 85.6 % | — |

**Tortuous + beaded geometry only**

| method | mean error | mean recall | mean f_i bias (est − geom) |
|---|---:|---:|---:|
| prism-nll | 5.86° | 99.7 % | +0.204 |
| plus | 5.63° | 98.1 % | +0.094 |
| plus-dprior | 5.75° | 98.0 % | +0.109 |
| plus-disp | 5.84° | 98.9 % | +0.210 |
| msmt | 10.35° | 88.0 % | — |

Per-condition angular error / recall:

| geom | angle | fixed-D PRISM | plus | plus-dprior | plus-disp | MSMT |
|---|---:|---:|---:|---:|---:|---:|
| straight | 0 | 7.6° / 100% | 5.6° / 100% | 5.7° / 100% | 3.1° / 100% | 4.8° / 100% |
| straight | 30 | 7.5° / 100% | 8.5° / 93% | 8.6° / 92% | 7.3° / 96% | 15.1° / 100% |
| straight | 45 | 4.3° / 100% | 5.9° / 100% | 6.0° / 100% | 4.4° / 100% | 22.3° / 45% |
| straight | 60 | 3.5° / 100% | 4.1° / 100% | 4.1° / 100% | 4.8° / 100% | 5.4° / 100% |
| straight | 90 | 4.0° / 99% | 3.1° / 100% | 3.1° / 100% | 3.2° / 100% | 5.2° / 100% |
| tortuous | 0 | 7.6° / 100% | 4.4° / 100% | 4.6° / 100% | 1.8° / 100% | 4.6° / 100% |
| tortuous | 30 | 9.9° / 98% | 11.3° / 91% | 11.6° / 90% | 10.3° / 97% | 15.3° / 100% |
| tortuous | 45 | 5.4° / 100% | 7.3° / 100% | 7.5° / 100% | 8.1° / 98% | 22.7° / 40% |
| tortuous | 60 | 4.2° / 100% | 3.4° / 100% | 3.3° / 100% | 7.2° / 100% | 5.8° / 100% |
| tortuous | 90 | 2.1° / 100% | 1.7° / 100% | 1.7° / 100% | 1.9° / 100% | 3.4° / 100% |

Reading, in order of what it changes:

1. **Off-model, no variant of PRISM dominates.** On single bundles the
   dispersion-aware PRISM-plus is clearly best (3.1° / 1.8° vs fixed-D
   PRISM's 7.6° / 7.6°) — real axons are never perfectly parallel and
   a model that can say so places the fibre better. At crossings the
   ordering flips: fixed-D PRISM is as good or better at 30–60°. The
   extra freedom of learned D∥ is spent absorbing extra-cellular
   hindrance (D∥ collapses to 1.0–1.3 vs the true 2.0, worst in the
   tortuous geometry) and that costs angular precision where two
   populations compete. The weak D prior (§6.7) does not rescue it
   (+0.1° everywhere) — the prior is centred on 1.7, not 2.0, and the
   pull is too gentle.
2. **Every method over-estimates f_i by 0.05–0.3.** Dense packing makes
   the extra-cellular space so hindered (S_extra at b = 3000 ≈ 0.03–0.05
   vs 0.0025 free) that it reads as intra-cellular. Fixed-D PRISM is the
   worst offender in the tortuous geometry (f_i 0.85 / 0.80 at 30° / 60°
   vs 0.53) — the same absorb-into-f_i behaviour as §6.8. The learned-D
   variants are least biased (+0.05–0.10) because D∥ absorbs it instead.
   Either way the microstructure is wrong; the question is which
   parameter takes the hit.
3. **MSMT-CSD with an honest oracle response fails at 30–45°** (recall
   40–45 % at 45°, error 15–22°) and is competitive only at ≥ 60°.
   Consistent with §6.2's in-model finding; not a fluke of the response.
4. **What this does to the "exceed PRISM" claim.** In-model and on
   DiSCo, PRISM-plus exceeds PRISM on every metric (§6.1–6.13). Off-model
   it exceeds PRISM on single fibres and on average, and *matches or
   trails it at crossings* because learned diffusivities are a
   liability under extra-cellular hindrance. The honest statement is
   therefore: learned D fixes the DiSCo-type failure (wrong regime,
   §6.1) but is not free under misspecification; a tortuosity-consistent
   extra-cellular model or a physically-anchored D prior at 2.0 is the
   next lever, and dispersion is the one lever that helped everywhere it
   was identifiable.

Cost: 4–7 s of Monte Carlo per substrate on the GB10 (4,000 walkers ×
1,800 steps × 5–8k spheres); the whole 10-condition, 5-method benchmark
runs in ~5 min as a Slurm job. Direct background runs of it were killed
by the session harness twice with no error; `sbatch` is the reliable
route for anything over a couple of minutes.

---

## 7. Risks

- **Re-implementation fidelity.** Until PRISM's code is released, a
  faithful-PRISM row is our reading of the paper. Report it as
  "PRISM-JAX (re-impl.)" and never as PRISM's own number.
- **Absolute-r comparisons.** Never quote our DiSCo r against 0.934
  without the MSMT margin beside it. Doc 004 §18–§19 is the scar tissue.
- **In-model benchmarks flatter everyone.** §3.5 is the antidote; do not
  skip it because DiSCo numbers look good.
- **Doc 005.** Revise against PRISM before sending to the FORCE authors.

## References

1. Abouagour M, Shah A, Garyfallidis E. PRISM. arXiv:2604.00250, 2026.
2. Doc 004 §13, §16, §18–§19, §21–§24. Doc 006 §4, §9.
3. Manzano-Patron JP et al. SBI probabilistic tractography. MedIA 2025.

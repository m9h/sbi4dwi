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
| Synthetic, SNR=30, mean angular error | 2.3° (NLL) | **< 2.0°** | pending |
| Synthetic, recall (all angles) | 99% | **≥ 99%**, and ≥ 95% at 15–20° | pending |
| DiSCo, SNR=50, K=5: margin over MSMT-CSD on same tracker | +1.6 pp | **> +1.6 pp** | running |
| DiSCo: beat faithful PRISM-JAX | — | at **every** SNR ∈ {10, 30, 50} | running |
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

*(appended as runs complete)*

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

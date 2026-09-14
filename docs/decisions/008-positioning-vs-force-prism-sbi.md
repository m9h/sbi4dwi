# 8. Positioning: sbi4dwi against FORCE, PRISM and the SBI groups on datasets 1–3

## Date

2026-09-14

## Status

Strategy note. Consolidates docs 004 (FORCE), 006 (landscape) and 007
(PRISM campaign) into one answer to: *on the datasets with ground truth,
what has each method shown, what have we shown, and what is genuinely
ours?*

## Summary

Four method families now compete on the same three ground-truth datasets:

| family | representative | inference | uncertainty | spatial prior | forward model |
|---|---|---|---|---|---|
| dictionary matching | **FORCE** (Indiana/DIPY, dipy ≥ 1.12) | kNN in signal space, 500K–1M library | kNN spread ("uncertainty", "ambiguity" maps), uncalibrated | none | stick + zeppelin + Bingham + GM/CSF, library priors |
| differentiable analysis-by-synthesis | **PRISM** (same authors, arXiv 2604.00250, no code) | Rprop, whole-slab | none | Huber–Laplacian + topology | stick + zeppelin + restricted + GM/CSF, **fixed D** |
| amortised SBI | **SBI_dMRI** (Nottingham, MedIA 2025), **SBIDTI** (Alicante, bioRxiv 2025) | neural spline flows, per voxel | full posterior, amortised | none | ball-and-sticks (Nottingham); DTI/DKI/AxCaliber (Alicante) |
| differentiable FEM | ReMiDi / Spinverse (UCSC) | gradient on mesh / permeability | none | mesh | Bloch–Torrey on geometry |
| **sbi4dwi** | this repo | dictionary warm start → Rprop MAP → Laplace posterior (→ NPE optional) | Gauss–Newton Laplace per fixel, calibrated; posterior tractography; CV-pruned connectome | PRISM's priors | PRISM's model with **learned D∥, decoupled D∥,ex, tortuosity, optional Watson dispersion**, no restricted pool |

The distinguishing claim, as the evidence now supports it: **on point
estimates we are level with the best of them; the difference is a
calibrated per-fixel posterior that turns into a measurably better and
far more specific connectome, plus a documented map of where every
method — ours included — fails off-model.**

---

## 1. Dataset 1 — DiSCo (strand-level connectivity ground truth)

### 1.1 What the others report
- FORCE paper: r = 0.894 (SNR 50), 0.868 (SNR 10); CSD 0.847. Protocol
  now public (doc 007 §8): 3 shells b < 3100, eroded ROI seeds, mask-only
  stopping, eudx 60°, lower-triangle Pearson.
- PRISM paper: r = 0.934 (SNR 50, K = 5, best tracking angle) vs MSMT 0.920
  — a different, unstated tracker; absolute r not comparable.
- SBI_dMRI / SBIDTI: not evaluated on DiSCo.

### 1.2 What we have (same tracker for every row)

Under the FORCE authors' protocol (doc 007 §8.3):

| | SNR 50 r / Dice | SNR 10 r / Dice |
|---|---|---|
| FORCE (dipy 1.12.1, their config) | 0.850–0.856 / 0.36 | 0.800–0.811 / 0.35 |
| MSMT-CSD | 0.779 / 0.45 | 0.752 / 0.37 |
| sbi4dwi MAP | 0.863 / 0.35 | 0.797 / 0.35 |
| **sbi4dwi posterior, CV-pruned** | 0.811 / **0.67** | **0.887 / 0.73** |

Under the all-shell protocol with our §21 tracker (doc 007 §6.18):
sbi4dwi posterior CV-pruned r = 0.858 / 0.859 / **0.871** at SNR 50/30/10
with Dice 0.80 / 0.79 / 0.71; MSMT Dice 0.57 / 0.55 / 0.47.

### 1.3 What distinguishes us here
1. **Specificity.** Every point-estimate method — FORCE, PRISM-style MAP,
   ours — sits at Dice ≈ 0.35 (80–95 false-positive edges of 95). The
   posterior-pruned connectome is the only row above 0.6. This is not a
   tuning artefact: the CV < 0.3 rule was fixed at SNR 50 and held at 30
   and 10.
2. **Low-SNR behaviour.** At SNR 10 the posterior mean alone beats every
   MAP (0.878 vs 0.80–0.81); pruning adds the rest. Uncertainty is worth
   most where the point estimate is worst.
3. **Reproduction discipline.** We are the only group to have run FORCE,
   MSMT and a PRISM-class model through one identical tracker on DiSCo,
   and to have shown that the paper-to-paper r differences (0.87 vs 0.93
   vs 0.13) were protocol, not method (doc 004 §18.6, doc 007 §8.2).

### 1.4 Where we do not win
- FORCE under its own protocol matches our MAP at SNR 50 and beats it at
  SNR 10. The +6–13 pp margins of doc 007 §6.6 were over MSMT, a weaker
  baseline; they should not be the headline.
- Dropping b = 13190 (their protocol) costs our learned-D model
  identifiability (D∥ drifts to 0.98); FORCE's library priors are
  indifferent to it.

---

## 2. Dataset 2 — synthetic crossings (exact fixel ground truth)

### 2.1 What the others report
- PRISM (its own protocol, SNR 30): 2.3° / 99 % recall (NLL), 3.5° / 95 %
  (MSE); MSMT 6.8° / 83 %; ODF-FP 11.6° / 86 %.
- FORCE (Fig. 6f,g and `experiments/synthetic_angle_accuracy`): peak
  detection 60–92 % across 10–90° and SNR 10–50, vs CSA/CSD/GQI/ODF-FP;
  needs `two_fiber_min_angle=0` (dipy ≥ 1.13) to represent < 30°.
- FORCE `recovery`: N = 5000 Watson-generated voxels, ND MAE 0.035 /
  0.068 / 0.177 at SNR 50/20/10, ODI MAE ≈ 0.12; vs AMICO-NODDI, DTI, DKI.
- SBI_dMRI: ~2° primary / ~6° secondary orientation error vs MCMC
  reference (not vs ground truth); posterior calibration by SBC.
- SBIDTI: AxCaliber angular error 0.065 rad (3.7°) on minimal schemes.

### 2.2 What we have
- PRISM protocol, in-model (doc 007 §6.2): PRISM-JAX 1.68° / 100 %; with
  dictionary warm start 1.59° / 100 %, 3.2° at 15°.
- Dispersed ground truth (§6.8): undispersed PRISM absorbs ODI into f_i
  (0.35 / 0.27 / 0.25 for true 0.5); dispersion-aware model 0.53 / 0.49
  / 0.47 and halves the 15–30° error.
- Laplace calibration (§6.12): 90 % cones cover 87–88 % at SNR ≥ 30;
  σ_θ predicts realised error to ~0.2° at ≥ 25°; under-covers ≤ 20°.

### 2.3 What distinguishes us here
1. **Calibration numbers exist.** PRISM and FORCE report accuracy only.
   SBI_dMRI reports SBC but against a Ball-and-Sticks posterior with no
   spatial model; ours is on a spatially regularised stick + zeppelin +
   dispersion model, per fixel, with the failure region (≤ 20°) stated.
2. **Parameter-absorption diagnosis.** We are the only group to show
   *which* parameter takes the hit when the model is wrong (ODI → f_i;
   D mismatch → restricted pool; hindrance → D∥). FORCE's `recovery`
   experiment reports bias but does not attribute it.

### 2.4 What we owe
- Our synthetic protocol is easier than PRISM's (fixed fractions, no iso
  signal); the within-protocol ordering is what we claim, not "1.6° vs
  2.3°".
- We have not run FORCE (dipy ≥ 1.13, min-angle 0) or SBI_dMRI on our
  synthetic sets. Both are scriptable; the second has a Docker scaffold
  (`docker/Dockerfile.sbi_dmri`, never run to a result).

---

## 3. Dataset 3 — Monte Carlo substrates (geometry ground truth, off-model)

### 3.1 What the others report
- FORCE `mc_phantom` / `out_of_sample_angle_accuracy`: disimpy walkers in
  packed impermeable cylinders (single fibre; crossings for the angle
  test), Watson-convolved, ICVF 0.40–0.72; vs DTI/DKI/AMICO and vs
  CSA/CSD/GQI/ODF-FP. Numbers not in the preprint text we have.
- CHUV (CATERPillar / NEXI validation / OCTOPUS): substrates as
  *validation of their own models*, not as a method comparison.
- PRISM, SBI_dMRI, SBIDTI: no off-model test.

### 3.2 What we have (doc 007 §6.14–6.15, 6.19)
CATERPillar overlapping-sphere axons (straight and tortuous + beaded),
sheet crossings 30–90°, intra + extra walkers, PRISM's scheme, SNR 30:

| method | mean error (10 conditions) | f_i bias |
|---|---:|---:|
| fixed-D PRISM-JAX | 5.63° | +0.15 |
| learned D | 5.53° | +0.09 |
| learned D + decoupled D∥,ex (**-x**) | best at 30–45° (7.0° vs 7.7 / 8.4) | +0.09 |
| + dispersion | 5.20° overall, best on single bundles (2.4° vs 7.6°) | +0.19 |
| MSMT-CSD | 10.46° | — |

And the negatives: every method over-estimates f_i by 0.05–0.30 in
packed tissue; the dispersion and -x levers do not stack; Laplace-
evidence model selection does not recover best-of-both.

### 3.3 What distinguishes us here
1. **Richer geometry than anyone's off-model test.** FORCE's cylinders
   are straight and impermeable; ours have tortuosity, beading, radius
   distributions, glia-capable substrates from the CHUV generator, and
   the physics tests that caught three leakage bugs are in the repo.
2. **The f_i result is a finding, not a failure to report.** No
   Gaussian-compartment method recovers the intra-cellular fraction from
   hindered, restricted extra-cellular space. That bounds every method
   in the table, including FORCE's NDI and PRISM's f_i, and it is the
   argument for the substrate-trained forward model (doc 006 Phase 3).
3. **Same engine, three paradigms.** Dictionary (matcher), differentiable
   MAP, and Laplace/NPE posterior are one JAX stack; the dictionary
   *warm-starts* the gradient fit — the bridge PRISM proposed and did
   not build.

### 3.4 What we owe
- A head-to-head on *their* substrate (disimpy cylinders) and *ours*
  with FORCE and SBI_dMRI in the table.
- Re-running on un-severed CATERPillar substrates (upstream fix
  2026-09-10) once `caterpillar.py` is ported to the JSON/CLI.

---

## 4. How to state the distinction (and what not to claim)

**Claim.** sbi4dwi is a differentiable, uncertainty-aware fixel and
microstructure estimator: level with FORCE and PRISM-class MAP fits on
point estimates, with a calibrated per-fixel posterior that (i) lifts
low-SNR connectivity above every point estimate, (ii) removes ~85 % of
false-positive connectome edges at fixed recall, and (iii) reports its
own failure regions. Its forward model learns the diffusivity regime
instead of assuming it, which is the failure mode that undoes both FORCE
(library priors, doc 004 §13) and PRISM (fixed D, doc 007 §6.1) on
out-of-regime data.

**Do not claim.** "Beats FORCE/PRISM on DiSCo" (level on their protocol);
"1.6° vs 2.3°" (different protocol); "recovers microstructure off-model"
(f_i is biased for everyone); "SBI" in the amortised sense on DiSCo (the
DiSCo posterior is Laplace; the NPE path exists in `pipeline/train.py`
but has not been run on these benchmarks).

**Versus the SBI groups specifically.** Nottingham's SBI_dMRI is the
closest in spirit (posterior → probabilistic tractography, scan–rescan
r 0.95). Differences: their model is Ball-and-Sticks with no dispersion,
no extra-cellular compartment and no spatial prior; their validation is
against MCMC and scan–rescan, not ground truth; ours is against DiSCo
strands and MC geometry with Dice and coverage. Alicante's SBIDTI is
about minimal acquisitions for DTI/DKI/AxCaliber — orthogonal; its
acquisition-minimisation idea is the one thing we should borrow (doc
004 §6.6). The honest gap on our side: the project is named for SBI and
the headline results use a Laplace approximation; running the flow
posterior (`inference_mode="flow"`) on datasets 1–3 with SBC, and
comparing it to the Laplace intervals, is the experiment that earns the
name.

---

## 5. The three runs that would settle the positioning

1. **SBI_dMRI on datasets 1–3** via the existing Docker scaffold — the
   only SBI competitor with code, never run here to a result.
2. **FORCE on datasets 2–3 with dipy ≥ 1.13** (`two_fiber_min_angle=0`,
   `seed=`), on both disimpy cylinders and CATERPillar.
3. **sbi4dwi amortised flow posterior + SBC on datasets 1–3**, alongside
   the Laplace rows: if calibration and Dice hold, "SBI" is earned; if
   Laplace is as good, that is worth knowing and cheaper.

## References

Docs 004, 006, 007; FORCE repo `experiments/`; Manzano-Patron et al.
MedIA 2025; Eggl & De Santis bioRxiv 2025; Abouagour et al. arXiv
2604.00250; Nguyen-Duc et al. CATERPillar 2026; Brammerloh et al.
OCTOPUS bioRxiv 2026.

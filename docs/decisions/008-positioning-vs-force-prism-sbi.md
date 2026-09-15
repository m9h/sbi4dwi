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


---

## 6. Results of the settling runs (2026-09-14)

### 6.1 Run 2 — FORCE (dipy master 1.13.0.dev, `seed=2298`, `two_fiber_min_angle=0`) on datasets 2 and 3

Slurm 1755, `validation/run_force_external.py`, 500K library, FORCE's
default in-vivo priors (no tuning — the same footing as our PRISM-JAX
rows, which start from D∥ = 1.7). Same best-match metric as every other
row (recall = matched within 20°).

**Dataset 2 — synthetic crossings (same exported sets as ours):**

| | FORCE (dipy master) | sbi4dwi PRISM-JAX / plus-warm (§2.2) | MSMT-CSD |
|---|---|---|---|
| in-model, SNR 30 | 7.54° / 98.7 % | 1.68° / 100 % · 1.59° / 100 % | 10.2° / 86 % |
| in-model, SNR 30, 15° / 20° / 25° | 7.7° / 9.5° / 7.1° | 4.4° / 3.4° / 2.9° | 7.6° / 10.2° / 12.7° |
| dispersed GT (ODI 0.2), SNR 30 | 14.72° / 70.2 % | plain 9.10° / 97.7 % · dispersed 6.77° / 98.2 % | — |
| in-model, SNR 10 | 8.59° / 96.4 % | (not run at SNR 10 in-model; Laplace calibration run gave 6.58° at 15°, ~3° at ≥ 45°) | — |

- With the < 30° library limit lifted, FORCE detects shallow crossings
  (100 % recall at 15–25°) but at 7–10° error — the ~4° angular
  quantisation of a 500K library. The differentiable fit is 3–4× more
  precise at every angle.
- **Dispersion hurts FORCE most**: recall collapses to 17–36 % at 50–65°
  under ODI 0.2, because its library shares one Bingham ODI across
  fibres and the matched entry's fixed directions inherit the blur. The
  dispersion-aware sbi4dwi model keeps 98 % recall at 6.8°.
- SNR 10 barely moves FORCE (8.6° vs 7.5°): dictionary matching is
  noise-robust in the way a nearest-neighbour method is — the library
  spacing, not the noise, sets the floor.

**Dataset 3 — CATERPillar substrates (identical MC signals, job 1754):**

| condition | FORCE err / recall / ND | sbi4dwi PRISM-JAX (fixed D) | sbi4dwi plus-x (learned D, decoupled D∥,ex) | geom f_i |
|---|---|---|---|---:|
| straight 0° | 8.7° / 100 / **0.49** | 6.6° / 100 / 0.62 | 6.9° / 100 / 0.62 | 0.50 |
| straight 30° | 15.3° / 95 / 0.68 | 8.1° / 98 / 0.78 | **8.0° / 97 / 0.61** | 0.49 |
| straight 45° | 14.3° / 98 / 0.63 | **4.8°** / 100 / 0.73 | 4.8° / 100 / 0.60 | 0.49 |
| straight 60° | 13.1° / 81 / 0.59 | **3.0°** / 100 / 0.67 | 3.3° / 100 / 0.64 | 0.49 |
| straight 90° | 9.7° / 100 / 0.62 | 2.4° / 100 / 0.58 | **2.3°** / 100 / 0.61 | 0.50 |
| tortuous 0° | **6.7°** / 100 / 0.67 | 7.1° / 100 / 0.68 | 6.7° / 100 / 0.60 | 0.53 |
| tortuous 30° | 15.1° / 100 / 0.70 | 7.8° / 100 / 0.75 | **6.5°** / 100 / 0.59 | 0.53 |
| tortuous 45° | 16.6° / 77 / 0.64 | 5.1° / 100 / 0.70 | **4.4°** / 100 / 0.64 | 0.52 |
| tortuous 60° | 12.3° / 91 / 0.67 | 3.5° / 100 / 0.65 | **3.1°** / 100 / 0.59 | 0.52 |
| tortuous 90° | **6.7°** / 100 / 0.64 | 2.5° / 100 / 0.59 | 2.8° / 100 / 0.64 | 0.53 |

- **Off-model, the differentiable fit beats FORCE by 2–4× at every
  crossing** (e.g. tortuous 45°: 4.4° vs 16.6°, with FORCE's recall
  dropping to 77 %). On single bundles the two are level (6.7–8.7°) and
  the dispersion-aware sbi4dwi variant (2.4°, §3.2) is far ahead of both.
- **FORCE's ND is *less* biased than any of our f_i estimates on single
  bundles** (0.49 and 0.67 vs geometry 0.50 / 0.53) but drifts to
  0.59–0.70 at crossings; ours sits at 0.58–0.78 throughout. Nobody
  recovers the intra fraction at crossings — the §3.3 finding holds
  with the strongest dictionary method included.
- This is the first run where FORCE, a PRISM-class MAP and our learned-D
  model see byte-identical off-model signals. Head-to-head rows 2 and 3
  of the positioning claim (§4) are now supported by direct evidence,
  not by cross-paper comparison.

### 6.2 Run 1 — SBI_dMRI (Nottingham NPE) on datasets 2 and 3

Slurm 1761 / 1763, `docker/sbi_dmri_run.py` in `nvcr.io/nvidia/pytorch:26.06-py3`.
Their Ball-and-Sticks forward model, priors (hemisphere, nfib = 2,
modelnum = 2), noise policy (Rician, SNR 5–80) and layout, trained as a
neural spline flow on **1M simulations, 60 epochs (14 min on the GB10)**,
one posterior for the PRISM 3-shell scheme (cached and reused). Two
things had to change to run it at all: their `DirectPosterior` rejection
sampler looped for hours on observations where the flow leaks mass
outside the box prior (sampled the density estimator directly and
clipped instead), and their per-voxel sampling loop (~1 s/voxel) was
batched. Two post-processings are reported: *published* (sort fibres by
mean fraction) and *clustered* (pool both fibres' direction samples,
2-means on the sphere — the same fix our flow needed).

| dataset 2 | published | clustered | sbi4dwi PRISM-JAX / Laplace |
|---|---|---|---|
| in-model SNR 30 | 10.3° / 93 % | **5.1° / 98.5 %** | 1.6° / 100 % |
| in-model SNR 30, 15° / 25° / 40° / 90° | 7.0 / 10.9 / 17.3 (77 %) / 5.9° | — | 4.4 / 2.9 / 1.6 / 0.9° |
| dispersed GT (ODI 0.2) | **8.3° / 99 %** | 13.2° / 80 % | dispersed model 6.8° / 98 % |
| in-model SNR 10 | 11.5° / 88 % | **8.9° / 94 %** | Laplace run: 6.6° at 15°, ~3° at ≥ 45° |

| dataset 3 (same MC signals) | published | clustered | FORCE (§6.1) | sbi4dwi plus-x |
|---|---|---|---|---|
| straight 0° | **1.2° / 100** | 6.8° | 8.7° | 6.9° (disp. model 2.4°) |
| straight 30 / 45 / 60 / 90° | 13.7 / 20.1 (50 %) / 26.0 (29 %) / 30.5° (23 %) | 8.6 / 9.4 / 7.2 / 4.6° | 15.3 / 14.3 / 13.1 / 9.7° | **8.0 / 4.8 / 3.3 / 2.3°** |
| tortuous 0° | **2.8° / 100** | 5.6° | 6.7° | 6.7° (disp. 2.3°) |
| tortuous 30 / 45 / 60 / 90° | 13.0 / 19.9 (50 %) / 27.1 (16 %) / 22.3° (51 %) | 11.7 / 13.1 / 10.9 / 6.5° | 15.1 / 16.6 / 12.3 / 6.7° | **6.5 / 4.4 / 3.1 / 2.8°** |
| Σ stick fractions vs geometry f_i | **0.48–0.61 vs 0.49–0.53** | same | ND 0.49–0.70 | f_i 0.58–0.64 |

- **Their single-fibre estimates are the best of any method** (1.2°,
  2.8°), and their stick-fraction sum is the least biased microstructure
  number in the whole comparison (within 0.02–0.1 of geometry) — the
  Ball-and-Sticks model's lack of an extra-cellular compartment turns
  out to be an advantage when the real extra-cellular space is this
  hindered.
- **Crossings are the weakness.** As published, recall collapses to
  16–50 % at 45–90° off-model (fibre identities average); clustered, it
  recovers recall but sits at 7–13° vs 2–5° for the differentiable
  fit. The amortised posterior is 2–4× less precise than a
  spatially-regularised MAP + Laplace on the same signals.
- Fairness note: the clustering post-process is ours, not theirs; where
  it hurts (single bundles, dispersed GT) the published reading is
  reported. Their paper also reports a *classifier* model-selection
  variant (SBI_ClassiFiber) we did not run.

### 6.3 Run 3 — sbi4dwi amortised flow posterior + SBC

Slurm 1758 / 1762, `validation/validate_flow_sbi.py`: spline NPE
(256 × 6, 8 knots) on the PRISM K = 2 model, 30k × 512 = 15.4M
simulations (19 min), raw 193-d signal as condition, SNR 8–60.

| | value |
|---|---|
| SBC, 300 held-out sims, 90 % coverage per parameter | 87.7–91.7 % (KS p > 0.05 on 6/7) |
| synthetic SNR 30, angular error / recall (published-style sort) | 18.5° / 58 % |
| same, mode-clustered | 19.6° / 58 % — error grows with angle (7.6° at 15° → ~25° at 60–90°) |
| single-fibre voxels | 2.5° |
| 90 % cone coverage (clustered) | 99.9 %, σ_θ 17–25° everywhere |
| Laplace posterior on the same data (doc 007 §6.12) | 1.6–1.7° / 100 %, coverage 87–88 % |

**Negative result, stated plainly** (superseded by §7.1: most of this number is the benchmark sitting on the parameterisation seams; tilted, the same flow gives 9.1° / 94.5 %)**.** The flow is calibrated in
parameter space — SBC passes — but its direction posterior is one
broad, honest cloud rather than two resolved modes, so it is far less
precise than both our own Laplace posterior and Nottingham's NSF on the
same task. The difference to SBI_dMRI is implementation, not principle:
they use a hemisphere prior, more epochs over a fixed 1M set, and
report ~6–11° at shallow crossings; we trained on streamed simulations
with φ ∈ (0, 2π) (antipodal ambiguity doubles every mode) and no
embedding net. Fixes are known (unit-vector or dyadic parameterisation
of directions, fraction ordering in the prior, an embedding net, longer
training) and none was attempted tonight. Until they are, **the SBI
claim for sbi4dwi rests on the Laplace posterior, which is the
better-calibrated *and* more precise of the two by a wide margin.**

### 6.4 Consolidated: datasets 2 and 3, every method, identical inputs

| method | synthetic SNR 30 | dispersed GT | substrates, single | substrates, crossings 30–90° | f_i / ND / Σf bias |
|---|---|---|---|---|---|
| FORCE (dipy master, seeded, min-angle 0) | 7.5° / 99 % | 14.7° / 70 % | 6.7–8.7° | 9.7–16.6°, recall 77–100 % | ND: best on single, +0.1–0.2 at crossings |
| SBI_dMRI (best of published / clustered) | 5.1° / 98.5 % | 8.3° / 99 % | **1.2–2.8°** | 4.6–13.1° | **Σf: within 0.02–0.1 everywhere** |
| sbi4dwi flow NPE | 18.5° / 58 % (calibrated) | — | 2.5° | — | — |
| sbi4dwi PRISM-JAX MAP, fixed D | 1.7° / 100 % | 9.1° / 98 % | 6.6–7.1° | 2.4–8.1° | +0.15 |
| sbi4dwi plus-x (learned D, decoupled D∥,ex) | — | — | 6.7–6.9° | **2.3–8.0°** | +0.09 |
| sbi4dwi + dispersion | 1.5° / 100 % | **6.8° / 98 %** | **2.3–2.4°** | 5–10° | +0.19 |
| sbi4dwi Laplace posterior | 1.6° / 100 %, 88 % coverage | — | — | — | — |
| MSMT-CSD | 10.2° / 86 % | — | 4.5–5.5° | 5–23°, recall 40–100 % | — |

What the settling runs settle, relative to §4's claim:

1. **Precision at crossings** — the differentiable, spatially regularised
   fit is 2–4× more precise than FORCE and than the amortised NPE on
   identical off-model signals. This is the strongest single row for
   sbi4dwi and it is now direct evidence.
2. **Microstructure off-model** — we do not win. SBI_dMRI's stick sum
   and FORCE's single-bundle ND are closer to geometry than any of our
   f_i. The §3.3 finding ("nobody recovers f_i") needs the qualifier
   "…with an extra-cellular Gaussian compartment": the model *without*
   one does better here. Worth a dedicated experiment.
3. **Calibrated uncertainty** — ours (Laplace) is the only posterior with
   both calibration and precision; the two amortised flows are
   calibrated (ours) or precise-ish (theirs) but not both. This remains
   the distinguishing capability, and on DiSCo it is what produces the
   best connectome (doc 007 §8.3).
4. **"SBI" as a name** — earned by the Laplace posterior's calibration
   numbers and by the SBC harness now in the repo, *not* by the
   amortised flow, which needs the fixes in §6.3 before it is competitive.

## 7. Where the limits come from (2026-09-14)

§6 settled *who* wins on which row. This section asks *why*, one
diagnostic per limitation, each isolating one mechanism at a time on the
same exported benchmarks. Drivers: `validation/diagnose_force_limits.py`
(dipy-master venv), `validation/diagnose_substrate_compartments.py`,
`validation/diagnose_flow_variants.py`, `validation/diagnose_posterior_width.py`,
plus `docker/sbi_dmri_run.py --hidden/--transforms/--batch`.

### 7.1 Our amortised flow: the benchmark sat on the parameterisation seams

`diagnose_flow_variants.py` retrains the §6.3 flow six ways — same
architecture (spline MAF 256 × 6), budget (20k × 512 streamed unless
stated) and evaluation (mode-clustered posterior, best-match error) — and
`diagnose_flow_rotated.py` re-evaluates every checkpoint on the same
benchmark rotated 45° about y then x. `validation/flow_variants_slurm1768.log`,
`validation/flow_variants_results.npz`, checkpoints `validation/flow_variant_*.eqx`.

| variant | equatorial benchmark (§6.3 geometry) err / recall | antipodal split | **tilted 45°** err / recall | 45° crossing, tilted | 90°, tilted | SBC 90 % cov (7 params) |
|---|---|---|---|---|---|---|
| base (θ ∈ (0,π), φ ∈ (0,2π)) | 23.9° / 41 % | 0.43 | **9.1° / 94.5 %** | 16.4° / 90 % | 7.0° | 87–94 % |
| hemi (θ ≤ π/2) | 23.1° / 39 % | 0.28 | 10.0° / 88 % | 22.2° / 48 % | 4.5° | 90–93 % |
| disk ((x,y) → upper hemisphere) | 23.8° / 39 % | 0.42 | 8.6° / 93 % | 19.4° / 54 % | 3.0° | 88–96 % |
| hemi + embedding MLP 193→64 | 20.9° / 47 % | 0.34 | 10.1° / 93 % | 22.0° / 50 % | 11.4° | 90–94 % |
| hemi, 2× budget (40k steps) | 21.6° / 48 % | 0.35 | **7.8° / 95 %** | **11.6° / 96 %** | 3.9° | 90–96 % |
| hemi, fixed 1M set × 60 epochs, batch 4096 (SBI_dMRI regime) | 19.2° / 56 % | 0.36 | 10.9° / 87 % | 22.5° / 28 % | 5.7° | 89–94 % |

1. **The §6.3 number was an evaluation artefact first.** The synthetic
   benchmark puts fibre 1 at (1, 0, 0) — φ = 0, the wrap seam of the
   θ/φ parameterisation — and both fibres at z = 0, the boundary of every
   hemisphere prior. A posterior mode on a seam is split in two by the
   flow, and `summarise()` then reads one fibre as two. Tilting the same
   configurations off the seams takes the *unchanged* base flow from
   23.9° / 41 % to 9.1° / 94.5 %. The hemisphere variants show the same
   thing through the antipodal-split column (0.28–0.43 on the seam,
   0.00–0.10 tilted). The same seam hurts SBI_dMRI (§7.5).
2. **What remains is a mid-angle resolution limit, shared with the other
   amortised flow.** Tilted, every variant is 1–1.5° on single fibres and
   3–7° at 90°, but 12–22° with 28–54 % recall at 45°: the posterior
   merges two fibres 45° apart into one broad mode (σ_θ 17–19°,
   label-switch fraction ≈ 0). SBI_dMRI's published reading has the same
   hole (45°: 19.7° / 48 %, 60°: 26.8° / 2 %).
3. **Budget helps where parameterisation does not.** Doubling the
   streamed budget is the only change that moves the 45° row
   (22.5° → 11.6°, recall 48 → 96 %); hemisphere, disk, embedding and the
   fixed-set regime are within noise of each other. The amortised
   posterior is capacity/budget-limited at mid angles, not
   parameterisation-limited.
4. Calibration is unaffected throughout (SBC coverage 87–96 % for every
   variant) — the flows are honest about their breadth, which is why
   err ≈ σ_θ in §7.4.

Corrected reading of §6.3: on a seam-free benchmark our streamed flow is
**8–9° / 94 %** (7.8° with 2× budget) against SBI_dMRI clustered 4.8°
on the identical tilted set and the Laplace posterior's 1.6°. The
amortised flow is still the least precise of the three, but by 2×, not
by 10×, and the gap is one more training doubling wide, not a design
flaw.

### 7.2 Our f_i bias on substrates: compartment attribution, not the compartment model

`diagnose_substrate_compartments.py` regenerates the CATERPillar
substrates (straight / tortuous × 0, 45, 90°), keeps the intra- and
extra-cellular MC signals separate, and fits them with the directions
fixed at truth so orientation estimation is out of the picture
(`validation/substrate_compartment_diagnostic{,_noiso}.{json,log}`).

**MC estimator check.** The benchmark's signal is |mean e^{iφ}|, which
has a positive floor ≈ √(π/4N) where the true signal is ≈ 0. Swapping to
Re mean e^{iφ} and raising N from 4,000 to 32,000 changes S_ex(b = 3000)
by 0.006 and every fitted fraction by ≤ 0.002. The benchmark signals are
not the problem.

**Compartment-resolved fits, directions fixed, clean signals** (Δf = fit − geometry):

| substrate | sz-free (f, D∥, D⊥) | sz-tort (D⊥ = (1−f)D∥) | sz-tort-x (decoupled D∥,ex) | sz-fixed (PRISM 1.7/0.4) | ball+stick (SBI_dMRI) | in-oracle (true S_in + zeppelin) | ex-oracle (stick + true S_ex) |
|---|---|---|---|---|---|---|---|
| straight 0° | −0.01 | −0.09 | −0.05 | −0.26 | +0.00 | +0.03 | −0.04 |
| straight 45° | +0.05 | −0.12 | −0.03 | −0.05 | −0.01 | +0.02 | −0.05 |
| straight 90° | +0.01 | −0.06 | −0.02 | −0.20 | +0.01 | +0.01 | −0.04 |
| tortuous 0° | +0.02 | −0.03 | −0.01 | −0.22 | +0.06 | +0.04 | −0.03 |
| tortuous 45° | +0.05 | −0.12 | −0.03 | −0.01 | −0.00 | +0.04 | −0.06 |
| tortuous 90° | +0.05 | −0.01 | +0.02 | −0.15 | +0.05 | +0.03 | −0.01 |

- With directions known, **stick + zeppelin with a free D⊥ recovers f_i
  within ±0.05 everywhere**, and so does ball + stick. The Gaussian
  compartment model is not what biases f_i.
- The extra-cellular space is *less* tortuous than Szafer–Stanisz
  predicts: fitted D⊥,ex ≈ 0.8–1.1 µm²/ms against (1 − f)·D∥ ≈ 0.75, and
  it is mildly non-Gaussian (apparent radial D falls 1.15 → 1.04 from
  b = 1000 to 3000). Tortuosity coupling therefore biases f_i *low* by
  0.01–0.12; PRISM's fixed D⊥ = 0.4 biases it low by 0.15–0.26. The
  half-oracles agree: replacing the extra model by the true S_ex costs
  −0.04, replacing the intra model by the true S_in costs +0.03 — both
  small, opposite in sign.
- Intra-axonal signals fit a stick with an *apparent* D∥ of 1.3–1.5
  (single bundles, ⟨cos²⟩ = 0.93 about the axis) and 0.8–0.9 at the
  sheet crossings: dispersion and finite-length effects show up as
  reduced D∥, not as a non-stick shape. Convolving the stick with the
  true segment fODF does not fit better than the single stick (the
  diffusion-scale fODF is narrower than the segment-scale one).

**So why does the full pipeline report +0.09 – +0.19?** Same signals,
64 Rician SNR-30 voxels, the actual PRISM-JAX plus-x fit with one knob
turned at a time. WM-internal f_i / isotropic (CSF + GM) fraction /
voxel-level intra fraction f_i·f_wm + f_res, geometry in brackets:

| substrate | plus-x | no tortuosity | no spatial / sparsity / D prior | fixed-D PRISM | **no CSF/GM** (`use_isotropic=False`) | no CSF/GM, no tortuosity |
|---|---|---|---|---|---|---|
| straight 0° [0.47] | 0.61 / 0.19 / 0.49 | 0.54 / 0.27 / 0.39 | = plus-x | 0.50 / 0.22 / 0.39 | **0.52** / 0 / 0.52 | 0.53 |
| straight 45° [0.46] | 0.62 / 0.47 / 0.33 | 0.52 / 0.45 / 0.28 | = plus-x | 0.69 / 0.34 / 0.51 | **0.43** / 0 / 0.43 | 0.45 |
| straight 90° [0.47] | 0.86 / 0.56 / 0.38 | 0.86 / 0.52 / 0.42 | = plus-x | 0.96 / 0.52 / 0.46 | **0.46** / 0 / 0.46 | 0.51 |
| tortuous 0° [0.52] | 0.57 / 0.21 / 0.45 | 0.55 / 0.22 / 0.42 | = plus-x | 0.63 / 0.19 / 0.55 | **0.51** / 0 / 0.51 | 0.53 |
| tortuous 45° [0.53] | 0.55 / 0.38 / 0.34 | 0.17 / 0.37 / 0.10 | = plus-x | 0.94 / 0.30 / 0.67 | **0.40** / 0 / 0.40 | 0.17 |
| tortuous 90° [0.52] | 0.62 / 0.41 / 0.36 | 0.60 / 0.42 / 0.35 | = plus-x | 0.61 / 0.27 / 0.50 | **0.47** / 0 / 0.47 | 0.52 |

(the earlier GPU run of the same ablation on a different CATERPillar
realisation gives the same picture: plus-x 0.63–0.73 / 0.25–0.41 /
0.40–0.51 against 0.53–0.54.)

1. **The bias is compartment attribution.** The near-isotropic
   extra-cellular space (D⊥ ≈ 1.0, D∥ ≈ 1.7) is absorbed by the CSF and
   GM balls — 19–56 % of the voxel — so the WM-internal f_i is inflated
   while the voxel-level intra fraction is *under*-estimated by 0.1–0.2.
   The "+0.09 – +0.19" of §6.4 and doc 007 §6.14 is this ratio effect;
   the same fit read as f_i·f_wm is biased the other way.
2. **No prior knob touches it**: spatial, sparsity, repulsion, D-prior
   and loss variants reproduce plus-x to two decimals. It is the model's
   partial-volume freedom, not the regularisation.
3. **Disabling the isotropic compartments fixes it** in five of six
   substrates (within 0.05); the tortuous 45° sheet crossing remains
   −0.13 with D∥ pulled to 1.46 — a genuine crossing/dispersion
   confound, the one row where tortuosity coupling is what keeps the
   fit identifiable (without it f_i collapses to 0.17).
4. **Why the others looked better.** SBI_dMRI's ball + stick has no
   separate isotropic *and* extra-cellular compartment, so nothing can
   leak; FORCE's ND on single bundles is the matched entry's, and it
   drifts at crossings for the same partial-volume reason. The
   "nobody recovers f_i" line of §6 becomes: *any model with both an
   anisotropic extra-cellular and free isotropic compartments will
   mis-attribute a hindered extra-cellular space, unless the isotropic
   compartments are constrained (WM-only voxels) or reported jointly.*

Consequence for the substrate benchmark and for DiSCo: report f_i·f_wm
alongside f_i, and add a `use_isotropic=False` row (now a `PrismConfig`
flag, 31 tests green) wherever the voxel is known to be WM-only.

### 7.3 FORCE: matching-limited, not library-limited

FORCE's peaks are the sphere vertices labelled in the *single* best
cosine match (`FORCEModel.fit`: argmax over `n_neighbors`; the
`use_posterior` flag averages scalar parameters but not the labels, and
it changed no direction in any run). Its angular error therefore
decomposes into sphere quantisation, library coverage (the entry nearest
the truth in parameter space) and the matching loss (the entry the
signal search actually returns). `validation/external/force_diag.log`.

| dataset 2, synthetic SNR 30 | 500K library | 2M library |
|---|---|---|
| sphere quantisation floor (362 vertices) | 2.76° | 2.76° |
| library coverage floor, 2-fibre GT | **2.79°** | 2.79° |
| FORCE, noisy | 7.54° / 98.7 % | 7.37° / 98.7 % |
| FORCE, **noise-free** | 7.96° / 97.0 % | 6.60° / 100 % |
| in-model held-out (FORCE's own generator, clean 2-fibre WM voxels, n = 526), clean | 10.11° / 90 % | 8.19° / 94 % |
| same, SNR 30 | 10.44° / 89 % | 8.32° / 94 % |
| share of library that is a clean 2-fibre WM crossing (WM ≥ 0.6, each fibre ≥ 0.15, ODI ≤ 0.2) | 0.53 % (2,630) | 0.53 % (10,656) |

| dataset 3, substrates (err / recall / ND) | 500K | 2M |
|---|---|---|
| coverage floor, 2-fibre | 2.94° | 2.94° |
| straight 30 / 45 / 60 / 90° | 15.3 / 14.3 / 13.1 (81 %) / 9.7° | 12.5 / 12.5 / 6.4 / **11.4° (80 %)** |
| tortuous 30 / 45 / 60 / 90° | 15.1 / 16.6 (77 %) / 12.3 / 6.7° | 15.1 / 14.2 / 8.2 / **9.9° (78 %)** |

What this says:

1. **The library already contains an entry within 2.8° of every
   crossing tested.** FORCE returns entries 5–14° further away. The gap
   is the matching loss; quantisation and coverage together account for
   < 3°.
2. **Noise is not the cause.** Noise-free signals give the same error
   (8.0° at 500K); SNR 30 vs clean differ by 0.3–0.6° everywhere.
3. **Model mismatch is not the main cause either.** On signals from
   FORCE's *own* generator — clean two-fibre WM voxels, no noise — the
   error is 10.1° (500K) / 8.2° (2M), *worse* than on our synthetic.
   The cosine search over 193-d signals is dominated by the
   partial-volume and microstructure degrees of freedom (WM/GM/CSF
   fractions, D∥, D⊥, ODI): the nearest signal is not the nearest
   orientation.
4. **A 4× library buys 0.2° on the benchmark and 2° in-model**, and is
   erratic off-model (straight 60° improves 13 → 6°, straight 90° worsens
   9.7 → 11.4° with recall dropping to 80 %). Nearest-neighbour matching
   has no smoothness to exploit: a denser library changes *which* wrong
   entry wins.
5. **Prior composition.** 70 % of the library is three-fibre and the
   median WM fraction is 0.50; only 0.5 % of entries are clean two-fibre
   WM crossings. FORCE is optimised for the in-vivo mixed-tissue voxel,
   not the crossing benchmark — which is fair to say in its favour on
   real data and against it on any orientation benchmark.

The differentiable fit does not have this failure mode because it
descends the same likelihood continuously: orientation error is set by
the noise-limited curvature (Laplace σ ≈ 1–3° at SNR 30), not by which
discrete neighbour wins a similarity vote. This is the mechanism behind
the 2–4× precision margin in §6.4, and it cannot be closed by a bigger
library.

### 7.4 Amortised posteriors: breadth or bias?

`diagnose_posterior_width.py` compares, per crossing angle, mean
best-match error with the posterior's own σ_θ (RMS angle of direction
samples about their mode) on the cached SNR-30 samples of §6.2–6.3.
err ≈ σ_θ is an honest but broad posterior; err ≪ σ_θ is over-dispersed;
err ≫ σ_θ is confidently wrong.

| angle | SBI_dMRI published err / σ_θ | SBI_dMRI clustered err / σ_θ | sbi4dwi flow (clustered) err / σ_θ |
|---|---|---|---|
| 15° | 7.0 / 10.6° | 7.6 / 15.6° | 7.6 / 21.0° |
| 30° | 13.1 / 13.0° | 8.0 / 15.3° | 15.0 / 17.6° |
| 45° | 18.3 / 19.5° | 5.3 / 15.1° | 22.5 / 17.5° |
| 60° | 11.1 / 24.9° | 3.0 / 14.1° | 24.6 / 19.6° |
| 90° | 5.9 / 38.3° | 2.4 / 15.7° | 22.4 / 24.7° |
| all | 10.0 / 21.7° (ratio 0.46) | 5.1 / 15.0° (0.34) | 19.1 / 19.9° (**0.96**) |

- **SBI_dMRI as published**: at 15–45° the error tracks σ_θ (a broad,
  honest posterior — the amortisation limit of a 1M-sim NSF); at 60–90°
  σ_θ balloons to 25–38° while the error falls to 5° — the per-fibre
  sample sets contain both fibres (label switching), so the reported
  uncertainty is inflated 5–7× and the sort-by-fraction fixel is a
  mode average. Clustering removes the switching (error 2–3° at ≥ 60°)
  but leaves σ_θ at 15° because the pooled 2-means split still mixes
  tails. Their remaining 7–10° at shallow angles is genuine breadth.
- **Our flow**: err/σ_θ ≈ 1 at *every* angle. It is calibrated (as the
  SBC said) and the posterior is simply not resolved — no bias, no
  label-switching signature, just a flow that never learned the sharp
  conditional. That is what §7.1 tests knob by knob.

### 7.5 SBI_dMRI: training budget vs noise prior vs label switching

`docker/sbi_dmri_run.py` with `--hidden/--transforms/--batch`, Slurm 1769
(`validation/external/sbi_dmri_diag_slurm1769.log`). Three estimators on
the identical exported sets: *published* = 1M sims, 60 epochs, NSF 64 × 8,
SNR 5–80 (§6.2); *big* = 3M sims, 100 epochs, 128 × 10 (84 min on the
GB10); *snr30* = 1M, SNR 25–35. Published-style sort-by-fraction reading
and the clustered reading of each.

| dataset 2, synthetic SNR 30 | published | big | snr30 |
|---|---|---|---|
| sort-by-fraction: overall (15 / 30 / 45 / 60 / 90°) | 10.3° / 93 % (6.9 / 13.1 / 18.3 / 11.1 / 5.9) | 11.1° / 90 % (4.8 / 11.9 / 12.3 / 6.3 / 14.1) | 14.7° / 79 % (6.7 / 12.8 / 16.2 / 20.2 / 11.3) |
| clustered | **5.1° / 98.5 %** (7.6 / 8.0 / 5.3 / 3.0 / 2.4) | 4.6° / 93.5 % (7.3 / 4.8 / 6.5 / 5.3 / 1.3), σ_θ 7.6° | 5.7° / 99 % (8.4 / 7.4 / 3.8 / 5.0 / 3.3) |
| **tilted 45°**, sort-by-fraction | 19.1° / 52 % (45°: 19.7 / 48 %; 60°: 26.8 / 2 %) | 19.1° / 51 % | — |
| tilted 45°, clustered | **4.8° / 97 %** (45°: 2.8 / 99 %; 60°: 7.3 / 82 %) | 20.4° / 48 % (45°: 22.5 / 0 %; 60°: 30.0 / 0 %) | — |

| dataset 3 substrates, crossings 30–90° (err / recall) | published | big | snr30 |
|---|---|---|---|
| sort-by-fraction, straight | 13.7/95 · 20.1/50 · 26.0/29 · 30.5/23 | 11.1/100 · 15.2/81 · 16.5/70 · 19.6/57 | 12.4/100 · 20.3/49 · 24.4/32 · 25.7/36 |
| clustered, straight | 8.6 · 9.4 · 7.1 · 4.6 | **7.7 · 6.3 · 5.2 · 2.8** | 10.6 · 10.9 · 8.8 · 6.3 |
| clustered, tortuous | 11.7 · 13.1 · 10.9 · 6.5 | **7.7 · 9.4 · 5.7 · 3.6** | 12.2 · 13.9 · 11.8 · 8.8 |
| Σ stick fractions vs geometry | within 0.02–0.10 | within 0.03–0.06 | within 0.02–0.07 |

1. **Label switching is the published method's dominant crossing
   failure, at every budget and noise prior.** Sort-by-fraction recall
   collapses to 2–57 % at 45–90° for all three estimators; the
   clustering post-process recovers it every time. This is a reporting
   choice in their pipeline, not a property of the posterior.
2. **Budget buys precision on the substrates, not on the synthetic
   set.** 3× sims + a wider net takes the clustered substrate crossings
   from 5–13° to 3–9° (now within 1–2× of our differentiable fit's
   2–5°) but leaves the synthetic set at 4.6° vs 5.1° — and it
   *fails* on the tilted set even clustered (45–60°: 0 % recall). A
   flow that is sharper (σ_θ 7.6° vs 15°) where it is right and
   confidently wrong elsewhere is the classic over-trained NPE
   signature; their 1M/60-epoch default is the better-calibrated choice.
3. **A narrow noise prior does not help** (snr30 is the worst of the
   three): the amortised posterior's breadth is not coming from noise
   marginalisation.
4. Their microstructure number is robust to all of this (Σf within
   0.1 of geometry in every run) — consistent with §7.2: a model with no
   free isotropic + anisotropic extra-cellular pair cannot mis-attribute.

### 7.6 What the limits are, in one table

| limitation (from §6.4) | mechanism, now measured | fixable by | evidence |
|---|---|---|---|
| FORCE 7–17° at crossings | nearest-neighbour matching in signal space picks entries 5–14° from the truth although the library holds one within 2.8°; unchanged noise-free, 0.2° better with 4× library, 8–10° in-model | not by library size or noise; only by a continuous refinement step after the match (which is what the differentiable fit is) | §7.3 |
| FORCE ND drifts at crossings | same partial-volume freedom as ours: matched entry's WM/GM/CSF split | — | §7.3, §7.2 |
| our f_i +0.1–0.2 on substrates | CSF/GM balls absorb the hindered (D⊥ ≈ 1.0) extra-cellular space; WM-internal f_i inflates, voxel-level intra fraction deflates | `use_isotropic=False` in WM-only voxels (5/6 substrates within 0.05); report f_i·f_wm otherwise | §7.2 |
| our tortuosity coupling | Szafer–Stanisz under-predicts D⊥,ex at Δ = 12 ms (real 0.8–1.1 vs 0.75) → f_i low by 0.01–0.12 with directions known; but it is what keeps the tortuous 45° crossing identifiable | keep, with a wider D⊥ prior | §7.2 |
| our flow 18.5° / 58 % | benchmark on the φ = 0 and z = 0 seams (→ 9° / 94 % tilted); residual mid-angle merging is budget-limited (2× budget: 45° 22 → 12°) | tilt-free evaluation; ≥ 2× training; clustering | §7.1 |
| SBI_dMRI crossings 16–50 % recall | sort-by-fraction under label switching; posterior itself resolves crossings (clustered 4.8–5.1°) | clustering (ours); their classifier variant untested | §7.5, §7.4 |
| SBI_dMRI σ_θ 25–38° at 90° | same label switching pooled into per-fibre samples → 5–7× inflated uncertainty | clustering | §7.4 |
| SBI_dMRI residual 5–9° at crossings | amortisation breadth (err ≈ σ_θ at 15–45°); more budget sharpens on substrates but over-trains on tilted synthetic | — (fundamental to a 1M–3M amortised NSF at this SNR) | §7.5 |
| all amortised flows vs Laplace | Laplace 1.6° / 88 % coverage is per-voxel optimisation + local curvature: no amortisation gap, no seam, no label ambiguity | — | §7.1, doc 007 §6.12 |

The positioning claim of §4 survives with sharper wording: the
differentiable, spatially regularised MAP + Laplace posterior is the
only method here whose orientation error is set by the noise (1–3°)
rather than by a discrete library, an amortisation gap, or a
parameterisation seam; its microstructure bias is a partial-volume
attribution that a one-flag model change removes on WM-only tissue.

## 8. What is left to surpass the other tools (2026-09-14)

Ordered by how much each closes a measured gap, with the number that
would show it is done.

1. **Make the microstructure win real.** §7.2 showed the f_i bias is
   isotropic-compartment attribution. Productise it: a tissue prior on
   the CSF/GM fractions (spatial, from a T1 or FA mask; hard zero in
   WM-only phantoms), voxel-level intra fraction as the reported number,
   a wider D⊥ prior instead of strict tortuosity. Target: DiSCo intra-VF
   r ≥ 0.92 *without* per-dataset retuning (FORCE reaches 0.918 only
   after retuning its library), and substrates within 0.05 in 6/6.
2. **Bring the amortised flow to parity, then hybridise.** Evaluate on
   the tilted benchmark only, 4× budget, keep SBC. Target: ≤ 5° / ≥ 95 %
   clustered (SBI_dMRI's best reading is 4.8°). Then use the flow as the
   initialiser for the Laplace refinement: amortised speed with the
   1.6° / 88 %-coverage posterior nobody else has.
3. **In-vivo evidence, with a metric the others do not report.** Every
   comparison so far is phantom or synthetic. Run HCP-YA test–retest
   (the retest subjects are the point) and Stanford HARDI: scan–rescan
   reproducibility of fixel directions, f_i and the CV-pruned connectome
   against FORCE and MSMT-CSD. Calibrated uncertainty predicts
   reproducibility; that is the demonstration only we can make.
4. **Time dependence in the forward model.** MC shows extra-cellular
   D⊥ falling 1.15 → 1.04 across shells and a chain-tortuosity D∥ of
   1.3–1.5. Add a first-order time-dependent extra-cellular term (or
   kurtosis) and fit multi-Δ data; PRISM, FORCE and SBI_dMRI are all
   Gaussian-compartment and cannot.
5. **Substrate realism.** Port `caterpillar.py` to the CMake/JSON CLI
   (un-severed axons), get MCMRSimulator `main` (sphere+cylinder SWC)
   once FillArrays 1.17 lands, add permeability and OCTOPUS-style glia.
   Two engines agreeing on intra *and* extra signals is the credibility
   bar for the substrate rows.
6. **Ship the benchmark.** Export the tilted synthetic set, the
   DiSCo-protocol data and the substrate signals with the external
   runners (FORCE, SBI_dMRI, MSMT) as a reproducible package; publish
   the seam finding — it affects every hemisphere-prior NPE in the
   literature. Push the repository to origin (it has never been pushed).
7. **Runtime.** Whole-brain plus-x fit and Laplace posterior timed
   against FORCE (seconds) and MSMT-CSD; batched voxel chunks on the
   GB10, NIfTI CLI. Precision that takes an hour per brain will not be
   adopted.
8. **Upstream engagement.** Send the FORCE notes (doc 005 / the
   published page), the MCMRSimulator chain limitation with the two
   reproducers, and the CATERPillar severed-axon confirmation; ask the
   SBI_dMRI group for their classifier variant.

Not worth more time now: per-voxel Laplace-evidence model selection
(negative, doc 007 §6.16), stacking dispersion with decoupled D∥,ex
(negative), bigger FORCE libraries (§7.3).

### 8.1 Item 1 done: DiSCo intra-axonal fraction without retuning (2026-09-14)

`validation/validate_disco_microstructure.py` (Slurm 1772) and
`validation/run_force_disco_ndi.py` (dipy master, `use_cache=False` —
the cache lookup crashes on range-valued `diffusivity_config`, doc 005
§7.3). Same data for everyone: DiSCo 1, the FORCE authors' protocol
(3 shells ≤ 3100), 15,267 mask voxels, Pearson r and mean bias against
`Strand_Intra_Volume_Fraction` (mean 0.178). Ours: K = 3, dictionary
warm start, 300 Rprop iterations, **no dataset-specific settings**
(D∥ initialised at 0.6 for the learned-D rows, the same value we use for
the DiSCo connectome; everything else as on in-vivo data). FORCE: 1M
libraries with its default in-vivo prior, the authors' DiSCo ranges from
`experiments/force_disco.py`, and the paper's narrow bands.

| method | SNR 30: r (bias) | SNR 10: r (bias) | needs per-dataset retune? |
|---|---|---|---|
| **PRISM-JAX plus-x, no isotropic (`use_isotropic=False`)** | **0.993** (+0.07) | **0.975** (+0.16) | no — D∥ learned 0.72 / 0.88 |
| PRISM-JAX plus-x (CSF + GM on), f_i | 0.985 (+0.07) | 0.952 (+0.17) | no — D∥ learned 0.70 / 0.86 |
| PRISM-JAX plus-x, voxel-level f_i·f_wm | 0.991 (+0.06) | 0.973 (+0.15) | no |
| PRISM-JAX + tortuosity scale / no tortuosity | 0.993 / 0.992 | 0.975 / 0.975 | no (bias +0.11–0.18) |
| PRISM as published (fixed D∥ 1.7, D⊥ 0.4), f_i | 0.822 (+0.36) | 0.770 (+0.31) | yes |
| PRISM with DiSCo-tuned fixed D (0.6 / 0.3) | 0.943 (−0.01) | 0.800 (+0.06) | yes |
| FORCE, default in-vivo prior, ND | 0.919 (+0.39) | 0.883 (+0.45) | — |
| FORCE, authors' DiSCo ranges, ND | 0.950 (+0.15) | 0.923 (+0.22) | yes |
| FORCE, paper's narrow bands, ND | 0.979 (+0.15) | 0.957 (+0.20) | yes |
| FORCE, ND·f_wm (best prior) | 0.962 (+0.01) | 0.941 (+0.06) | yes |

1. **The target is met.** r ≥ 0.92 without retuning was the bar; the
   learned-D fit gives 0.985–0.993 at SNR 30 and 0.95–0.975 at SNR 10,
   above FORCE with its *best* hand-tuned prior (0.979 / 0.957) and far
   above FORCE untuned (0.919 / 0.883). The bias (+0.07) is a third of
   FORCE's tuned bias (+0.15) and a fifth of its untuned one (+0.39).
2. **What does it: learned diffusivities.** The same model with PRISM's
   fixed in-vivo D gives 0.82; with the DiSCo-tuned fixed D 0.94; learned,
   0.99. The fit finds D∥ ≈ 0.7 on its own — the number FORCE has to be
   told. This is the exact inverse of doc 005 §1's finding on FORCE, and
   it is the argument for gradient-based refinement in one line.
3. **Disabling the isotropic compartments** (DiSCo has no CSF/GM) adds
   +0.008 and removes the attribution split of §7.2; with them on, the
   voxel-level product f_i·f_wm is the right number to report (0.991).
4. Multi-shell matters for FORCE too: its default prior went from
   r = 0.68 on the single b = 1900 shell (doc 005 §2) to 0.92 on three
   shells. The doc 005 single-shell contrast overstated the retuning
   penalty; the 3-shell one (0.92 → 0.98) is the fair number.
5. Tortuosity scale and free D⊥ do not help on DiSCo (bias grows); keep
   tortuosity as default, wider prior as an option.

## 9. Hybrid posterior: amortised proposal → MAP → Laplace (2026-09-15)

### 9.1 Synthetic, K = 2: initialisation does not matter in-model

`validation/validate_hybrid_posterior.py` (Slurm 1773). Same MAP + Laplace
(fixed-D PRISM K = 2, 300 Rprop iterations) from three starts, 3400
voxels, SNR 30.

| tilted 45° (seam-free) | err / recall | 90 % cone coverage | median σ_θ | time (3400 vox) |
|---|---|---|---|---|
| flow alone (hemi-2x checkpoint) | 7.85° / 94.8 % | — | 13.8° | 29 s |
| dictionary alone (300K) | 16.81° / 66.1 % | — | — | 39 s |
| flow → MAP → Laplace | **1.73° / 100 %** | 85.7 % | 0.89° | + 10 s + 21 s |
| dict → MAP → Laplace | 1.77° / 100 % | 85.1 % | 0.88° | + 1 s + 1 s (warm JIT) |
| no init → MAP → Laplace | 1.65° / 100 % | 86.9 % | 0.91° | + 1 s + 1 s |
| original seam benchmark, the same three | 1.76 / 1.84 / 1.68° | 86.1 / 85.4 / 87.2 % | | |

The flow is the better proposal (2× the dictionary's accuracy, 100 %
recall after refinement either way), but the spatially regularised MAP
reaches the same 1.7° from a random start: on in-model data the
objective has no local minima the priors do not smooth away. The
15° row is the only one where the start shows (4.4–6.3°). So the
amortised stage buys nothing here; where it can matter is K = 3 on real
protocol data, where doc 007 §8.3 needed the dictionary warm start —
that is §9.2. Timing note: the whole 3400-voxel hybrid runs in about a
minute on the GB10 including the 29 s of flow sampling; the sampling,
not the refinement, is the cost.

### 9.2 DiSCo, K = 3, the FORCE authors' protocol: initialisation is a second-order effect there too

`validation/validate_hybrid_disco.py` (Slurm 1775) with a flow trained
on the DiSCo 3-shell scheme in 3.5 min (`train_flow_k.py`, K = 3, D∥ as
a flow parameter, `validation/flow_disco_k3.eqx`). Same plus-x MAP (K = 3,
learned D, 300 iterations) → Laplace → protocol tractography, 20
posterior connectomes, CV < 0.3 pruning.

| SNR 30 | intra-VF r | D∥ | MAP connectome r / Dice | CV-pruned posterior r / Dice | time |
|---|---|---|---|---|---|
| flow proposal alone (amortised) | 0.912 | 0.59 | — | — | 65 s |
| flow → MAP → Laplace | 0.973 | 0.70 | 0.881 / 0.35 | **0.907 / 0.77** | + 3 s + 2 s |
| dict → MAP → Laplace | 0.985 | 0.70 | **0.904** / 0.35 | 0.907 / 0.72 | + 11 s + 3 s |
| none → MAP → Laplace | 0.968 | 0.64 | 0.877 / 0.35 | 0.905 / **0.80** | + 2 s + 1 s |
| doc 007 §8.3 (same protocol, earlier settings) | | | 0.863 | 0.887 / 0.73 | |
| FORCE, same protocol (§8.3) | | | 0.850–0.856 | — | |

| SNR 10 | intra-VF r | MAP r | CV-pruned r / Dice |
|---|---|---|---|
| flow / dict / none → MAP → Laplace | 0.814 / 0.952 / 0.862 | 0.814 / 0.776 / 0.788 | 0.806 / 0.804 / **0.841** (0.64 / 0.59 / 0.69) |
| doc 007 §8.3 | | 0.797 | 0.811 |

1. **The CV-pruned posterior connectome is the robust number** —
   0.905–0.907 at SNR 30 from all three starts, 0.80–0.84 at SNR 10 —
   and it is now clearly above FORCE under FORCE's own protocol
   (0.856). The MAP alone swings 0.78–0.90 with the start; the posterior
   does not.
2. **The amortised proposal is not what the MAP needs.** With spatial
   regularisation the plus-x objective is benign enough that a random
   start reaches the same posterior connectome; the dictionary gives the
   best MAP at SNR 30, no init the best at SNR 10. Item 2 of §8 is
   therefore closed as "hybrid wiring works, adds no accuracy" — the
   flow's value is as a 65-second standalone posterior (intra-VF
   r = 0.91, D∥ regime detected on its own) and as an initialiser where
   there is no spatial prior (single voxels, streaming, ex-vivo slabs).
3. These numbers supersede doc 007 §8.3 as the DiSCo-protocol
   comparison row: PRISM-JAX posterior connectome **0.907 / 0.841**
   (SNR 30 / 10) vs FORCE 0.856 / 0.800.

### 9.3 Item 2 started: exchange-aware kernel and acquisition design

`dmipy_jax/validation/prism_exchange.py` (6 tests): intra stick ⇄ extra
zeppelin with Kärger exchange time τ_ex, (a) closed-form 2×2 matrix
exponential per measurement and (b) a Diffrax ODE over the explicit
trapezoid waveform with b matched to the integrated q(t); the two agree
to 3·10⁻³ and the ODE is differentiable in δ, Δ and the waveform.
`validation/design_exchange_protocol.py` puts Fisher information on top:
θ = (f_i, τ_ex, D∥, D⊥) in log space, single fibre averaged over 5
orientations, 3 shells × 32 directions, Gaussian noise with the T2 = 70 ms
echo-time penalty so longer Δ costs SNR, D-optimal + τ-weighted gradient
ascent over (b_k, Δ_k).

| protocol (96 measurements) | CRLB relative SD of τ_ex, true τ = 10 / 25 / 50 / 100 ms, SNR 30 | same, SNR 50 | f_i rel SD (τ = 25) |
|---|---|---|---|
| DiSCo / PRISM timing (δ 17.7, Δ 35.8, b 1/2/3) | 16 / 18 / 26 / 44 | 9.6 / 11 / 16 / 27 | 3.2 / 1.9 |
| fixed 3 Δ (20 / 45 / 80 ms, b 1/2/3) | 5.5 / 5.4 / 5.9 / 8.4 | 3.3 / 3.3 / 3.6 / 5.0 | 2.7 / 1.6 |
| **optimised** (b 1.4 / 5 / 5, Δ 15 / 15 / 51.5 ms) | **1.1 / 0.83 / 1.0 / 1.6** | 0.69 / 0.50 / 0.62 / 0.98 | **0.12 / 0.07** |

- The single-Δ protocols every method in this comparison uses cannot
  see exchange at all (τ_ex relative SD 16–44). Three spread Δ's help by
  3–5×; the optimised protocol — two short-Δ high-b shells that pin f_i
  and D, one long-Δ shell that carries the exchange — by 15–30×.
- Even then τ_ex is marginal: relative SD 0.5–1 at SNR 50 with 96
  measurements. Exchange needs SNR ≥ 50, ≥ 200 measurements, or a fixed
  D prior; that is a quantitative statement no Gaussian-compartment
  method can make, and the design layer is what makes it.
- Caveats: the optimiser ran to the b = 5 ms/µm² cap, where the stick +
  zeppelin approximation is weakest; single fibre with known
  orientation; Gaussian noise. Next: validate against MCMRSimulator
  permeable cylinders (extra-cellular parity established in doc 007 §8.6)
  — fitted τ_ex scatter vs CRLB for the DiSCo and optimised protocols.

### 8.2 Substrates with the isotropic compartments off (2026-09-15)

`validate_prism_substrate.py --methods plus-x plus-x-noiso plus-x-disp-noiso`
(Slurm 1777; a new CATERPillar realisation, so compare within the table).
err / recall / f_i, geometry f_i in brackets.

| substrate | plus-x (§6.4 model) | plus-x, no CSF/GM | plus-x + dispersion, no CSF/GM |
|---|---|---|---|
| straight 0° [0.52] | 5.3° / 100 / 0.63 | 6.0° / 100 / **0.51** | **4.5°** / 100 / 0.53 |
| straight 30° [0.51] | 12.0° / 79 / 0.66 | 14.9° / 56 / 0.54 | **6.4° / 97** / 0.66 |
| straight 45° [0.51] | 8.4° / 100 / 0.67 | 11.0° / 95 / **0.51** | **4.5°** / 100 / 0.71 |
| straight 60° [0.51] | 7.0° / 100 / 0.64 | 8.1° / 100 / **0.51** | **4.4°** / 100 / 0.72 |
| straight 90° [0.52] | 4.4° / 100 / 0.66 | 4.4° / 100 / **0.51** | 4.4° / 100 / 0.53 |
| tortuous 0° [0.54] | 4.1° / 100 / 0.66 | 4.1° / 100 / **0.57** | **1.7°** / 100 / 0.62 |
| tortuous 30° [0.55] | 7.4° / 98 / 0.62 | 8.7° / 88 / **0.50** | **5.7°** / 97 / 0.76 |
| tortuous 45° [0.53] | 8.0° / 100 / 0.62 | 8.7° / 99 / **0.49** | **4.8°** / 100 / 0.79 |
| tortuous 60° [0.54] | 6.8° / 100 / 0.60 | 7.2° / 100 / **0.50** | **4.4°** / 100 / 0.82 |
| tortuous 90° [0.55] | 2.9° / 100 / 0.65 | 2.9° / 100 / **0.54** | 2.8° / 100 / 0.62 |

- **f_i is fixed by the flag**: within ±0.05 of geometry in 9/10
  substrates (−0.01 to −0.05), against +0.08–0.16 for the §6.4 model.
  This is the substrate half of §8 item 1; DiSCo was §8.1.
- **The cost is orientation at shallow crossings** (30°: recall 79 → 56 %
  straight): the isotropic compartments were absorbing the hindered
  extra-cellular signal that otherwise pulls the second fibre.
- **Dispersion + no-iso is the orientation winner everywhere**
  (1.7–6.4°, recall ≥ 97 %) but inflates f_i to 0.6–0.8: the Watson
  kernel absorbs the extra-cellular anisotropy into the intra
  compartment. Orientation and microstructure are best served by
  different compartment models on the same voxel — the practical recipe
  is the dispersed no-iso fit for fixels and tractography and the
  non-dispersed no-iso fit for f_i, or (better) a joint model with a
  dispersion prior tied to the extra-cellular D⊥, which is now the open
  modelling question rather than an attribution bug.

### 7.1 addendum: 4× budget

The hemi flow at 80k × 512 (Slurm 1776) on the tilted set: **6.59° /
94.8 %**, σ_θ 11.2°, SBC 90–95 %; the seam benchmark stays at 20°. Per
budget doubling: 9.1 → 7.8 → 6.6°. The ≤ 5° target (SBI_dMRI clustered
4.8°) is one more doubling away by extrapolation but the returns are
shrinking; an embedding that respects the antipodal symmetry (dyadic
input features) is the cheaper route than more steps.

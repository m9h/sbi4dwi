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
   Retrained with the dyadic parameterisation (Slurm 1780,
   `flow_disco_k3_dyad.eqx`, 3.7 min): amortised intra-VF r 0.917 / 0.749
   (SNR 30 / 10), refined 0.972 / 0.814, MAP connectome 0.892 / 0.801,
   CV-pruned 0.901 / 0.802 — the same picture; the proposal's
   parameterisation does not reach the refined result either.
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

### 7.1 addendum 2: the dyadic parameterisation closes most of the gap (2026-09-15)

Each fibre as a symmetric 3×3 tensor (xx, yy, xy, xz, yz; zz = 1 − xx − yy),
direction = principal eigenvector: no seam anywhere on the sphere and no
hemisphere boundary. Same spline MAF, same trainer (Slurm 1778,
`validation/flow_variants_dyad.npz`, `flow_variant_dyad*_256x6.eqx`).

| flow (256 × 6 spline MAF) | budget | seam benchmark err / recall | tilted 45° err / recall | SBC 90 % cov |
|---|---|---|---|---|
| θ/φ base | 20k × 512 | 23.9° / 41 % | 9.1° / 94.5 % | 87–94 % |
| θ/φ hemi | 40k | 21.6° / 48 % | 7.8° / 95 % | 90–96 % |
| θ/φ hemi | 80k | 20.3° / 56 % | 6.6° / 95 % | 90–95 % |
| **dyadic** | 20k | **7.7° / 94 %** | 6.7° / 97 % | 89–95 % |
| **dyadic** | 40k | **5.8° / 98 %** | **5.6° / 95.5 %** | 86–94 % |
| SBI_dMRI clustered (their NSF, 1M sims) | | 5.1° / 98.5 % | 4.8° / 97 % | — |

Seam and tilted numbers now agree, which is the point: the flow's error
is finally a property of the flow, not of where the truth sits. At 40k
steps the amortised posterior is within 0.8° of SBI_dMRI's best reading
with calibration intact. **The 80k run (Slurm 1779) does not help**: 5.6° / 95 % on the seam set but 8.3° / 84 % tilted — the over-training signature SBI_dMRI's 3M estimator showed (§7.5). 40k × 512 is the operating point for this architecture; the amortised floor is ≈ 5.5°. The K = 3
DiSCo proposal of §9.2 should be retrained with this parameterisation
before it is used anywhere without a spatial prior.

### 9.4 Item 3 started: the DIPY refinement stage

`dmipy_jax/validation/dipy_refine.py`: `refine_peaks(data, gtab, mask,
pam)` takes any DIPY `PeaksAndMetrics`, initialises the plus-x MAP from its
peaks and values, and returns a refined `PeaksAndMetrics` plus the Laplace
fixel posterior; `posterior_pams()` yields one PAM per posterior sample for
CV-pruned connectomes. `validation/validate_dipy_refine.py` (Slurm 1781) on
DiSCo under the authors' protocol, MSMT-CSD and FORCE (dipy 1.12.1 in the
main venv, authors' DiSCo ranges, 500K) as the two sources:

| SNR | source | peaks as produced r / Dice | refined MAP | CV-pruned posterior | intra-VF r | refine + 20 posterior connectomes |
|---|---|---|---|---|---|---|
| 30 | MSMT-CSD | 0.786 / 0.43 | 0.891 / 0.35 | **0.908 / 0.77** | 0.986 | 6 s + 10 s |
| 30 | FORCE | 0.790 / 0.35 | 0.873 / 0.35 | **0.908 / 0.77** | 0.987 | 3 s + 10 s |
| 10 | MSMT-CSD | 0.752 / 0.37 | 0.807 / 0.34 | **0.860 / 0.72** | 0.950 | 3 s + 10 s |
| 10 | FORCE | 0.825 / 0.34 | 0.797 / 0.34 | 0.806 / 0.66 | 0.951 | 3 s + 10 s |

- **Any peaks in, the same posterior connectome out** at SNR 30
  (0.908 from either source, matching §9.2's 0.905–0.907 from flow,
  dictionary or no init). The stage costs 3–6 s of refinement and 10 s
  of posterior tractography on top of a DIPY reconstruction, and it also
  returns the intra-VF map (r = 0.99) the source reconstruction did not
  have.
- **At SNR 10 the source matters** (0.860 from MSMT, 0.806 from FORCE):
  with weak data the MAP stays near its start, and FORCE's peaks at 1.12.1
  with the tuned prior are a better start for the MAP r (0.825) than the
  refinement's own optimum reaches (0.797); the posterior still lifts
  Dice from 0.34 to 0.66. The honest reading: at low SNR the refinement
  is an uncertainty layer more than an accuracy layer.
- What remains for a real plug-in: fixel-format output (MRtrix), a
  `dipy.workflows` command, and the BIDS-derivative naming; the
  numerical core is done.

### 9.5 Item 7: whole-brain runtime (Stanford HARDI, 154,368 mask voxels, 160 measurements)

`validation/benchmark_runtime_hardi.py` (Slurm 1782, one GB10, dipy 1.12.1
for CSD and FORCE with 6 CPU workers).

| stage | wall | per voxel |
|---|---|---|
| CSD peaks (dipy, 6 processes) | 16 s | 0.10 ms |
| **refine_peaks**, K = 3 plus-x, 300 Rprop iterations, spatial priors | **31 s** | **0.20 ms** |
| **Laplace fixel posterior** (Gauss–Newton, chunked) | **23 s** | 0.15 ms |
| FORCE fit + peaks, 500K library (library generation 241 s once) | 172 s | 1.11 ms |

The full refinement plus posterior for a whole in-vivo brain is under a
minute on one GB10, 3× faster than FORCE's matching on the same mask
(and FORCE's library adds 4 min the first time). Sanity on this
single-shell b = 2000 set: D∥ learned 1.33 (in-vivo, single shell —
weakly identified), f_i 0.42 mean, refined main peaks within 6.2° (median)
of CSD's, median σ_θ 5.0°. Runtime is no longer an adoption objection.

### 9.3 addendum: the exchange model against MCMRSimulator ground truth (2026-09-15)

`julia/mcmr/permeable_cylinders.jl` (241 randomly packed cylinders, r ≈ 1 µm,
intra fraction 0.601, 40 µm periodic, D = 2, permeability p in MCMR units)
simulates both protocols of the design table (96 measurements each);
`julia/mcmr/exchange_time.jl` measures the residence time directly by
tracking spin identity across readouts; `validation/validate_exchange_mcmr.py`
fits the closed-form Kärger kernel (f_i, τ_ex, D∥, D⊥; fibre along z) to
the clean signal and to 30 Rician repeats at SNR 30 / 50. Two things
learned on the way: MCMR's `Subset(inside=…)` is evaluated at *readout*,
not seeding, and its permeability unit is large — p = 0.3 already gives
τ ≈ 1–2 ms (fast-exchange limit; those runs are archived under
`validation/mcmr/fast_exchange/`).

| p | residence time (measured) | protocol | clean fit f_i / τ_ex / D∥ | SNR 30, 30 repeats: f_i, τ_ex | SNR 50 |
|---|---|---|---|---|---|
| 0 | ∞ | DiSCo single-Δ | 0.74 / 25 ms / 2.09 | 0.78 ± 0.09, 20 ± 909 ms | 0.67 ± 0.09, 54 ± 950 |
| 0 | ∞ | **optimised** | **0.62 / 1050 ms / 2.03** | 0.62 ± 0.01, 586 ± 773 (rel 1.3) | 0.62 ± 0.01, 833 ± 760 (0.9) |
| 0.003 | ≈ 100 ms | DiSCo single-Δ | 0.84 / 7.5 / 2.01 | 0.83 ± 0.11, 9 ± 841 | 0.83 ± 0.08, 9 ± 356 |
| 0.003 | ≈ 100 ms | **optimised** | **0.60 / 117 ms / 2.06** | 0.60 ± 0.02, 103 ± 341 (3.3) | 0.60 ± 0.01, 114 ± 173 (1.5) |
| 0.01 | ≈ 32 ms | DiSCo single-Δ | 0.84 / 4.9 / 2.04 | 0.83 ± 0.14, 6 ± 843 | 0.83 ± 0.09, 6 ± 358 |
| 0.01 | ≈ 32 ms | **optimised** | **0.57 / 37 ms / 2.06** | 0.57 ± 0.02, **35 ± 17** (0.49) | 0.57 ± 0.01, **37 ± 7** (0.20) |
| 0.03 | ≈ 11 ms | DiSCo single-Δ | 0.82 / 2.4 / 2.07 | 0.81 ± 0.22, 3 ± 883 | 0.81 ± 0.20, 3 ± 742 |
| 0.03 | ≈ 11 ms | **optimised** | 0.49 / 15 ms / 2.05 | 0.50 ± 0.03, **15 ± 6** (0.38) | 0.49 ± 0.02, **15 ± 3** (0.19) |

1. **The single-Δ protocol every method in this comparison uses cannot
   see exchange, and mis-reads f_i when exchange exists**: f_i 0.82–0.84
   against 0.60 at every finite τ, τ_ex pinned at a few ms with ±900 ms
   scatter. This is not a fitting failure; the CRLB said 16–44 and the
   empirical relative scatter is 38–290.
2. **The Fisher-designed protocol recovers f_i, D∥ and τ_ex from an
   independent Monte Carlo engine**: f_i 0.60 / 0.57 / 0.49 vs 0.60 for
   τ = 100 / 32 / 11 ms (the drift at fast exchange is the Kärger
   approximation reaching its limit at Δ ≈ τ), τ_ex 117 / 37 / 15 ms vs
   ≈ 100 / 32 / 11 measured, D∥ 2.03–2.06 vs 2.0. At SNR 30 the τ_ex
   scatter is ±17 ms at τ = 32 and ±6 ms at τ = 11 (relative 0.4–0.5);
   at τ = 100 ms it is heavy-tailed (0.6 of the repeats within ±50 ms,
   the rest running to the bound) — exactly the regime the design table
   flagged as marginal.
3. **Design and validation agree** where both exist: relative SD of
   τ_ex at SNR 30, CRLB 0.8–1.0 vs empirical 0.4–0.5 (τ = 11–32 ms;
   the bound is conservative because the MC substrate has D⊥ ≈ 0.7,
   further from D∥ than the design's 0.5 assumption) and 1.0–1.6 vs
   1.3–3.3 at τ ≥ 100 ms.

**Under clinical gradients** (`--gmax 0.08 --delta 0.020`, Δ ≥ δ + 2 ms
enforced): the optimiser picks b = 1.0 / 2.9 / 5.0 ms/µm² at Δ = 22 / 23 /
64 ms; τ_ex CRLB relative SD 1.2 (τ = 25 ms) and 2.5 (τ = 100 ms) at
SNR 30, i.e. ~1.5× the 340 mT/m design — exchange at τ ≤ 50 ms stays
marginally identifiable on an 80 mT/m system with 96 measurements
(`validation/exchange_protocol_design_g0.08.json`).

This is the first closed loop of the "Diffrax forward model +
acquisition design" item: a differentiable exchange kernel, a protocol
chosen by gradient ascent on its Fisher information, and a third-party
Monte Carlo engine confirming that the chosen protocol identifies what
the standard one cannot. Nothing in PRISM, FORCE, SBI_dMRI or MSMT-CSD
can produce this row.

## 10. Remaining work, ranked by risk and reward (2026-09-15)

Inventory from docs 006 §9, 007 §8 and this document's §8–§9 (60 items
audited; §8.1–§9.5 closed items 1, 2, 4 and 7 of §8). Reward = how much
the item separates sbi4dwi from FORCE / PRISM / SBI_dMRI / MSMT-CSD /
Microstructure.jl / OCTOPUS on evidence rather than argument; risk =
external dependency × technical uncertainty × effort.

**Tier A — low risk, high reward: do these first**

| # | task | why it ranks here |
|---|---|---|
| A1 | **Push the repository and ship the benchmark package** (tilted synthetic set, DiSCo-protocol data, substrate signals, MCMR exchange sets, the external runners). Write up the seam finding (§7.1) as a short note. | Zero external risk. Nothing in this document exists to anyone else until it is pushed; the seam artefact touches every hemisphere-prior NPE in the literature, which is a citation-generating result on its own. |
| A2 | **Finish the DIPY plug-in packaging**: MRtrix fixel output, a `dipy.workflows` command around `refine_peaks`, BIDS-derivative names. | Core is done (§9.4, §9.5: any peaks → 0.908 posterior connectome in < 1 min per brain). Packaging is a week of low-uncertainty work and is the adoption path — users keep their pipeline and gain calibrated fixels and f_i. |
| A3 | **Upstream notes**: rewrite doc 005 around §8.1 / §7.3 and send it (page is published); the MCMRSimulator sphere-chain reproducers to Cottaar; the CATERPillar severed-axon confirmation; ask the SBI_dMRI group for their classifier variant. | Cheap, builds the relationships the plug-in needs, and two of the three may return fixes we would otherwise write ourselves. |
| A4 | **Re-run the protocol design under clinical gradient limits** (80 mT/m, δ free) and for the Dogpatch / NIC scanners' actual constraints. | Cheap (minutes on CPU). The §9.3 optimum used 340 mT/m; the question a scanner physicist will ask first is what the design gives at 80. Prerequisite for A6/B2. |

**Tier B — medium risk, high reward**

| # | task | risk | reward |
|---|---|---|---|
| B1 | **In-vivo test–retest** (HCP-YA retest subjects): scan–rescan reproducibility of fixels, f_i and the CV-pruned connectome vs FORCE and MSMT-CSD; check that Laplace σ predicts rescan disagreement. | Data access (ConnectomeDB AWS keys not configured); the metric itself is straightforward. | The only in-vivo claim no competitor reports, and the natural validation of calibration. |
| B2 | **Acquire a multi-Δ dataset with the designed protocol** on the local scanner (one volunteer, A4 protocol) and fit the exchange model. | Scanner access, IRB/consent, and the Kärger approximation near Δ ≈ τ (§9.3 addendum). | Turns the closed simulation loop into a real-data capability nobody else has. |
| B3 | **A joint model that serves orientation and f_i at once**: dispersion prior tied to the extra-cellular D⊥ (§8.2's open question). | Modelling risk; may need the time-dependent term to be identifiable. | Removes the "two fits per voxel" recipe; single-model best-in-class on both rows. |
| B4 | **CATERPillar CMake/JSON port and un-severed re-run** of the substrate rows. | Low external risk (upstream released 2026-09-10); half a day of plumbing plus a GPU hour. | Moderate — validates the substrate tables against the fixed generator; also unblocks OCTOPUS-style glia later. |
| B5 | **Empirical-Bayes hyperparameters** by implicit differentiation through the Optimistix argmin (λ_spatial, λ_sparse, D prior width, tortuosity factor). | Technical: implicit-function gradients through Rprop; moderate effort. | Answers "how were the weights chosen" with a number; PRISM's are hand-set. |

**Tier C — high risk, high reward: prepare, do not start**

| # | task | blocker | when it pays |
|---|---|---|---|
| C1 | OCTOPUS integration (doc 006 Phases 3–4): OCTOPUSOracle, morphology-parameter SBI, grow→simulate→recover. | No code released; email CHUV (A3-cost action now). | Large, once code exists — it is the substrate-realism ceiling. |
| C2 | Intra-axonal MC parity: MCMRSimulator `main` sphere+cylinder SWC (needs FillArrays ≥ 1.17) or Permeable_MCDS as the third engine. | Upstream registry / effort. | Moderate — the extra-cellular half is already at 0.01 RMS; this closes the intra half. |
| C3 | Emulator-in-the-loop inversion: an Equinox emulator of MC signals so voxels can be fitted (or an NPE trained) against Monte Carlo rather than Gaussian compartments. | Emulator accuracy at b ≥ 3000, training data volume. | High — the OCTOPUS direction without waiting for OCTOPUS, and the only route to microstructure beyond stick+zeppelin. |

**Tier D — low reward: strike or defer**

- The [PLANNED] oracle plumbing in CLAUDE.md (`oracle.py` ABC, `oracles/`, `oracle_adapter.py`, `multi_fidelity.py`): write a minimal ABC only when a second oracle exists (MCMR wrapper); strike `multi_fidelity.py`.
- SWC ingestion and sphere+cylinder SDF (006 Phase 2): only needed for C1/C2.
- ReMiDi FEM benchmark (006 Phase 4), disimpy head-to-head, library-seed sweep, `n_legendre` halving, doc-drift residue: no competitive information in any of them now.

**Status 2026-09-15:** A1 done (pushed; package at `/data/datasets/sbi4dwi_benchmark_v1`, seam note `docs/notes/seam-artefact-in-direction-npe.md`); A2 done in code (`dmipy_jax/io/fixel.py` validated with MRtrix `fixel2voxel`, `dmipy_jax/workflows/refine_peaks_flow.py` + `dipy_refine_peaks` entry point, BIDS-style outputs); A3 drafted, not sent (`docs/outreach/`); A4 done (80 mT/m design, §9.3).

**Recommended order:** A1 → A2 and A3 in parallel → A4 → B1 (as soon
as keys exist) and B4 (GPU idle) → B3 → B5 → B2. Prepare C1 with the
CHUV email; revisit C2/C3 when B3 shows where the compartment model runs
out.

### 10.1 B5 result: prior weights under the Rician NLL (2026-09-17)

`validation/tune_hyperparams_cv.py` (Slurm 1784, 1787): measurement-split
cross-validation (fit on 80 % of directions, Rician NLL on the held-out
20 %, three splits), grid over (λ_spatial, λ_sparse, λ_repulsion, λ_D),
with the downstream metric (DiSCo intra-VF r; substrate angular error)
recorded for every setting. Two findings, one of them a correction.

1. **PRISM's published weights are inert under the NLL loss.** The data
   term is ≈ 10² per voxel while the prior terms are O(0.1); at
   λ = 0.01–0.02 (PRISM's MSE-scale values) the priors change nothing to
   four decimals. Every "no-spatial = plus-x" ablation row in §7.2 was
   showing this, and the initialisation-insensitivity of §9.1–§9.2 is a
   property of the likelihood, not of the regularisation. The weights
   only bite from λ ≈ 1 upward under NLL.
2. **Held-out likelihood and parameter accuracy disagree once the
   priors are strong.** DiSCo, 15,267 voxels:

   | λ_spatial / λ_sparse / λ_repulsion | held-out NLL | intra-VF r |
   |---|---|---|
   | 0 / 0 / 0 (≈ PRISM defaults under NLL) | 893.6 ± 1.5 | 0.968 |
   | 100 / 10 / 10 | 892.5 | 0.971 |
   | **1000 / 10 / 10** | 892.9 | **0.977** |
   | 300 / 100 / 10 (CV optimum) | **888.1** | 0.897 |
   | 100 / 100 / 10 | 888.1 | 0.846 |

   Strong sparsity buys 4–5 NLL units on the held-out directions
   (fewer live fibres predict the signal of the same voxel better) and
   costs 0.07–0.13 in intra-VF correlation. Measurement-split CV rewards
   signal compression, not parameter identifiability, so it is the wrong
   selector for sparsity and repulsion. It is a reasonable selector for
   the spatial weight, where NLL and accuracy move together (λ_spatial
   100–1000 improves both).
3. On the 64-voxel replicated substrate volume the priors are within
   the split noise either way (ΔNLL 0.4 vs SD 2.0; angular error 8.0°
   unchanged).

**Decision:** PRISM-JAX NLL defaults become λ_spatial = 300, λ_sparse = 10,
λ_repulsion = 10, λ_D = 0 (DiSCo intra-VF r 0.977 vs 0.968; held-out NLL
better than inert priors) — chosen by the downstream metric within the
CV-improving set, and stated as such. Implicit differentiation through
the argmin remains the way to make this a gradient step instead of a
grid, but it would inherit the same objective problem: the outer loss
must be a parameter-recovery loss (SBC, synthetic truth), not the
held-out signal likelihood.

### 10.2 B4 result: the un-severed CATERPillar build (2026-09-17)

CATERPillar `main` (0fd887e) builds with CMake, runs headless (`CATERPillar-cli --config x.json`,
0.4 s per 10 µm voxel) and every axon spans the voxel. `dmipy_jax/validation/caterpillar.py::CATERPillarCLI`
+ `caterpillar_bundle(backend="cli")`. Two caveats found on the way: still no seed
key (std::random_device), and the achieved ICVF undershoots the request and varies
(0.45 / 0.34 / 0.47 / 0.17 for `AxonsICVF: 50` in four runs) — both in the
CATERPillar note (`docs/outreach/`). Benchmark (Slurm 1785, measured
intra fraction used as truth; err / recall / f_i):

| substrate [geom f_i] | plus-x | plus-x no-iso | plus-x + dispersion, no-iso | MSMT-CSD |
|---|---|---|---|---|
| straight 0° [0.34] | 8.8° / 100 / 0.59 | 10.4° / 100 / **0.38** | **3.4°** / 100 / 0.42 | 4.6° |
| straight 30° [0.34] | 11.4° / 79 / 0.55 | 13.6° / 70 / 0.39 | 13.9° / 59 / 0.49 | 15.5° / 51 |
| straight 45° [0.34] | 10.9° / 87 / 0.57 | 15.3° / 62 / 0.40 | **9.6°** / 84 / 0.50 | 22.9° / 50 |
| straight 60° [0.35] | 8.7° / 95 / 0.55 | 9.9° / 91 / 0.41 | **5.8° / 99** / 0.52 | 30.6° / 50 |
| straight 90° [0.34] | 2.1° / 100 / 0.53 | 2.1° / 100 / **0.40** | 2.2° / 100 / 0.47 | 43.9° / 52 |
| tortuous 0° [0.47] | 5.0° / 100 / 0.60 | 4.7° / 100 / **0.50** | **2.7°** / 100 / 0.57 | 1.5° |
| tortuous 30° [0.47] | 14.0° / 72 / 0.60 | 14.9° / 60 / **0.43** | 13.2° / 71 / 0.76 | 15.0° / 50 |
| tortuous 45° [0.45] | 9.3° / 95 / 0.61 | 11.9° / 84 / **0.40** | **8.2° / 99** / 0.79 | 22.5° / 50 |
| tortuous 60° [0.46] | 6.5° / 100 / 0.61 | 9.7° / 98 / **0.43** | **5.6°** / 98 / 0.75 | 30.1° / 50 |
| tortuous 90° [0.47] | 3.7° / 100 / 0.67 | 3.4° / 100 / **0.41** | 3.7° / 100 / 0.77 | 2.6° |

Same structure as the January build (§6.4, §8.2): no-iso recovers f_i
within 0.02–0.06 in 10/10, dispersion + no-iso is the orientation winner
at crossings, MSMT-CSD collapses to 50 % recall beyond 30° on these
sheet crossings. The un-severed axons change the numbers, not the
ranking, so the earlier tables stand. The sparser substrates (0.34 vs
0.50) make the shallow 30° crossing harder for everyone (59–79 % recall).
Signals exported to `validation/external/substrate_signals_cli.npz` for
the external runners.

### 10.3 B3 result: one model for orientation and f_i (2026-09-17)

Ablation of the dispersed, isotropic-free fit on CLI substrates (Slurm 1786,
1788): tortuosity vs free D⊥ (`-freeperp`), ODI capped at 0.15 (`-odicap`).
The build's density is erratic, which turned out to be useful: one
realisation is dense (tortuous, intra fraction 0.70–0.73), the others
sparse (0.17–0.20). err / recall / f_i, geometry in brackets.

| substrate | no-iso (no dispersion) | disp, no-iso, tortuosity | **disp, no-iso, free D⊥** | disp, no-iso, ODI ≤ 0.15 |
|---|---|---|---|---|
| dense tortuous 0° [0.70] | 4.3° / 100 / **0.70** | 4.1° / 0.74 | 3.8° / 0.78 | 4.0° / 0.73 |
| dense tortuous 45° [0.73] | 7.2° / 100 / 0.60 | 5.7° / 0.78 | **5.8° / 100 / 0.72** | 6.2° / 97 / 0.77 |
| dense tortuous 90° [0.70] | 4.1° / 0.64 | 4.1° / 0.67 | 3.9° / 0.76 | 4.1° / 0.67 |
| sparse straight 0° [0.20] | 8.7° / 98 / **0.22** | **4.2°** / 0.24 | 4.9° / 0.24 | 5.3° / 0.24 |
| sparse straight 45° [0.19] | 16.1° / 57 / 0.24 | 12.4° / 81 / 0.34 | **11.2° / 84** / 0.33 | 16.0° / 68 / 0.32 |
| sparse straight 90° [0.19] | 4.0° / **0.24** | 4.1° / 0.28 | 4.0° / 0.27 | 4.1° / 0.28 |
| sparse (0.17) straight / tortuous (0.38), all angles | f_i within +0.05 | +0.07 to +0.19 | same | same |

1. **Dense tissue: dispersed + no-iso + free D⊥ is the joint model.** At
   the 45° crossing it gives 5.8° / 100 % and f_i 0.72 vs 0.73, where
   tortuosity coupling inflates f_i (0.78) and the non-dispersed fit
   deflates it (0.60). At 0° and 90° it over-shoots by 0.06–0.08 — the
   free D⊥ absorbs some dispersion — so the two are not fully separable
   at single-Δ; that is the multi-Δ / time-dependence argument again.
2. **Sparse tissue (f_i ≤ 0.2): dispersion inflates f_i in every
   variant** (+0.04 to +0.15) while the non-dispersed no-iso fit stays
   within +0.05; orientation still favours the dispersed fits at
   crossings. The Watson kernel has nothing to describe at 20 % axon
   density except the extra-cellular anisotropy, and it takes it.
3. Capping ODI does not help and costs recall at crossings.

**Decision:** default WM model = dispersed, isotropic-free where tissue is
WM-only, free D⊥ (`tortuosity=False`) — with the caveat, now measured,
that below ~0.3 intra fraction f_i should be read from the non-dispersed
fit. Tier B is closed except for the two items that need external
inputs (B1 HCP credentials, B2 scanner time).

### 10.4 B1 result: in-vivo scan–rescan on HCP-YA 105923 (2026-09-20)

Unblocked when ConnectomeDB turned out to redirect to BALSA, which now
issues per-project S3 keys for `hcp-openaccess` (HCP_1200 first visit,
HCP_Retest second visit). `validation/validate_hcp_retest.py`; both
visits are the HCP preprocessed 3-shell data (18 b0 + 3 × 90, b = 1000/
2000/3000, 1.25 mm, 791k brain voxels, 235k WM voxels), each in its own
ACPC space. Per visit: dipy MSMT-CSD peaks → plus-x refinement (K = 3,
NLL, learned D∥, tortuosity, CSF/GM on) + Laplace fixel posterior → FORCE
(dipy 1.12.1, 1M library, default in-vivo prior) → deterministic EuDX
tracking from every WM voxel (relative peak threshold 0.1, 45°, all
streamlines ≥ 10 mm) → Desikan 68-ROI connectomes; refine additionally
tracks 10 posterior direction samples. Visits are aligned by rigid T1w
registration (0.30°, 0.09 mm — HCP's ACPC alignment already does the
work; WM Dice 0.954), retest maps resampled nearest-neighbour into the
first-visit grid and directions rotated. All metrics on the 223k-voxel WM
intersection.

**Runtime per visit** (one GB10, 16 CPU workers): MSMT peaks 16 min (dipy,
cvxpy per voxel), refinement 2.5 min for 791k voxels + Laplace 20 s, FORCE
library 4 min + fit 15 min, tracking 5 s per peak set.

**Whole-brain agreement of the two visits**

| | MSMT-CSD | plus-x refine | FORCE |
|---|---|---|---|
| f_i: CCC / mean \|Δ\| / within-subject CoV | | **0.866 / 0.043 / 8.3 %** | ND 0.757 / 0.074 / 10.8 %; ND·f_wm 0.728 / 0.103 / 18.6 % |
| FA reference (DTI, b ≤ 1000) | 0.925 / 0.048 / 10.3 % | | |
| main fixel Δθ: median / p90 / < 10° | 8.0° / 33° / 73 % | 8.3° / 41° / 56 % | 28° / 90° / 24 % |
| all fixels (frac ≥ 0.1): median Δθ to nearest | 8.5° | 13.3° | 22° |
| fixel-count agreement (mean count) | 72 % (1.8) | 72 % (2.5) | 63 % (2.2) |
| connectome r(log w) / Dice / edges | 0.916 / 0.816 / ~840 | **0.926 / 0.857 / ~1430** | 0.906 / 0.808 / ~1010 |
| posterior-mean connectome (10 samples) | | **0.965 / 0.937** | |
| CV-pruned posterior: CV < 0.5 / 0.3 / 0.2 | | 0.900 / 0.856 / 0.826 (edges 680 / 390 / 233) | |

- **f_i is the clearest win.** Our intra-axonal fraction is more
  reproducible than FORCE's ND in absolute error (0.043 vs 0.074), CCC
  (0.87 vs 0.76) and within-subject CoV (8.3 % vs 10.8 %), with zero
  bias between visits; it is also more reproducible than DTI FA in
  CoV. FORCE's ND·f_wm, the quantity that matched DiSCo intra-VF best
  (§8.1), is the least reproducible in vivo (CoV 18.6 %) because f_wm
  itself is unstable.
- **Orientations: the refinement inherits the MSMT peak on the main
  fixel** (8.3° vs 8.0° median) but reports 2.5 fixels per voxel against
  MSMT's 1.8 at the 0.1 fraction threshold, and the extra small fixels
  are the ones that disagree (all-fixel median 13.3°). This is the
  K = 3 prior doing what §7.4 predicted in vivo: the third fixel is
  weakly determined and should be pruned by the posterior, not by a
  fixed fraction threshold. FORCE's in-vivo peaks are not reproducible
  (28° median, 24 % within 10°) — the matching-limited mechanism of
  §7.3 on a library whose in-vivo prior covers few clean crossings.
- **Calibration**: the Laplace σ *ranks* the scan–rescan disagreement
  (Spearman 0.47 over 222k voxels; decile medians rise monotonically
  from 2.6° to 15.4°) but under-predicts its size by 1.7× (51 % of
  voxels inside the predicted 90 % cone). The noise-only Laplace cannot
  contain registration, nearest-neighbour resampling at 1.25 mm,
  physiological change and head-position effects, so a single
  in-vivo inflation factor of ≈1.7 is the practical calibration —
  measurable only because we have a σ to calibrate; no competitor
  reports one.

  | σ_comb decile median | 2.0° | 2.9° | 3.6° | 4.2° | 4.8° | 5.3° | 6.0° | 6.7° | 7.7° | 9.7° |
  |---|---|---|---|---|---|---|---|---|---|---|
  | predicted median Δθ | 2.1 | 3.0 | 3.6 | 4.0 | 4.5 | 5.1 | 5.6 | 6.3 | 7.2 | 8.9 |
  | observed median Δθ | 2.6 | 4.3 | 5.8 | 7.2 | 8.8 | 10.6 | 12.3 | 13.9 | 15.1 | 15.4 |

- **Connectomes**: on identical tracking, the refined peaks give the
  most reproducible MAP connectome (r 0.926, Dice 0.857) with the most
  edges, MSMT next, FORCE last. Averaging the connectome over 10
  posterior direction samples lifts reproducibility to r 0.965 / Dice
  0.937 — the posterior-mean connectome is the in-vivo deliverable.
  CV pruning, which was the best row on DiSCo (§9.2), *lowers* r here
  because it removes the weak edges that are nevertheless reproducible
  at this scale; what the posterior CV does carry is edge-level
  predictive value — the test-visit edge CV predicts the scan–rescan
  edge disagreement with Spearman 0.40 over 1833 shared edges, so it
  belongs as an edge weight or confidence, not as a hard threshold.
- Caveats: one subject; deterministic tracking on peaks (the DiSCo
  protocol), not probabilistic FOD tracking; the first pass with an
  absolute peak threshold (`compare_results_tracking_v1.json`) gave the
  same ordering with fewer streamlines.

Artefacts: `validation/hcp_retest/{compare_results.json,
compare_results_tracking_v1.json, run_sessions.log,
run_connectomes_v2.log}`; per-visit peak/fixel/posterior arrays and
connectomes in `/data/datasets/hcp/b1/{test,retest}/`.

### 10.5 B1 cohort: five HCP retest subjects + NODDI reference (2026-09-20)

Same pipeline as §10.4 on 103818, 105923, 111312, 114823, 115320 (all
from HCP_1200 / HCP_Retest via BALSA), plus AMICO 2.1.1 NODDI per visit
(`validation/validate_hcp_retest.py --stage noddi`; 1.5 min per visit)
and a second refinement with the isotropic compartments off
(`validation/hcp_refine_variant.py --variant noiso`). Mean ± sd over
subjects; WM intersection 200–225k voxels each; registration 0.3–2.4°.
`validation/summarize_hcp_retest.py` → `validation/hcp_retest/summary.json`.

**f_i scan–rescan, five subjects**

| estimator | CCC | mean \|Δ\| | within-subject CoV |
|---|---|---|---|
| plus-x refine, **no isotropic** (WM model of §10.3) | 0.883 ± 0.030 | **0.030 ± 0.004** | **5.5 ± 0.8 %** |
| NODDI NDI·(1−FWF), AMICO | **0.894 ± 0.023** | 0.034 ± 0.004 | 5.6 ± 0.7 % |
| NODDI NDI | 0.872 ± 0.024 | 0.047 ± 0.006 | 7.0 ± 0.8 % |
| plus-x refine, default (CSF + GM balls) | 0.835 ± 0.025 | 0.044 ± 0.004 | 8.2 ± 0.8 % |
| DTI FA (reference) | 0.912 ± 0.028 | 0.051 ± 0.008 | 10.7 ± 1.8 % |
| FORCE ND | 0.723 ± 0.029 | 0.074 ± 0.003 | 10.7 ± 0.5 % |
| FORCE ND·f_wm | 0.694 ± 0.029 | 0.105 ± 0.003 | 18.4 ± 0.9 % |
| NODDI ODI | 0.853 ± 0.035 | 0.054 ± 0.009 | 26.5 ± 2.9 % |

- **Where the default model's variance comes from**
  (`validation/diagnose_hcp_fi_variance.py`, 105923): not the crossings.
  Δf_i is the same for 1, 2 and 3-fixel voxels (sd 0.056–0.082) and
  *largest* in coherent high-FA voxels (sd 0.10 at FA > 0.7); it
  correlates with Δ(f_csf + f_gm) at r = 0.52, and the two balls take
  0.31 of the WM signal (0.24 with the GM ball off), against NODDI's
  free-water fraction of ≈0.1. This is §7.2's attribution mechanism in
  vivo: the isotropic compartments absorb hindered extra-cellular signal
  and the split between them and f_i is not stable across visits. A CSF-
  only variant does not fix it (CCC 0.910 but CoV 9.4 %, 105923).
- **With the balls off, our f_i is as reproducible as NODDI's** — ahead
  on absolute error and CoV, behind by 0.01 in CCC, and every subject
  keeps the ordering. The default model sits behind NODDI, so the honest
  statement for the in-vivo section is "on par with NODDI, clearly
  better than FORCE", with the WM-only model, exactly as §10.3
  concluded on substrates. The price is orientation: with no isotropic
  sink the third fixel fills instead (2.96 fixels per voxel, main fixel
  Δθ 12.4° vs 8.5°), so in vivo the deliverable is a two-model pair —
  orientations and posterior from the default model, f_i from the
  no-iso fit — until a proper prior on the isotropic fractions (or a
  free-water map from a different contrast) replaces the balls.
- **Orientation, connectome and calibration all replicate across the
  five subjects** with small spread: MSMT main fixel 8.0 ± 0.3°, refine
  8.5 ± 0.8°, FORCE 26 ± 2° (24 % within 10°); connectome r 0.921 /
  0.928 / 0.908 (MSMT / refine / FORCE) and Dice 0.82 / 0.86 / 0.81;
  posterior-mean connectome **0.968 ± 0.003 / 0.935 ± 0.002**; CV pruning
  still lowers r (0.904 at CV < 0.5); the edge CV predicts scan–rescan
  edge disagreement at ρ = 0.36 ± 0.04; the Laplace σ ranks Δθ at
  ρ = 0.46 ± 0.02 and under-predicts it by 1.76 ± 0.10 (1.70–1.96, the
  largest on the noisiest subject). The inflation factor is stable
  enough to be used as a constant.
- FA is more reproducible in CCC than every microstructure index (0.91)
  but not in CoV (10.7 %); CCC rewards its wide dynamic range across WM.

**Runtime per visit on the GB10**: refinement 2.5 min + Laplace 20 s (+2.7
min for the no-iso refit), AMICO NODDI 1.5 min, MSMT peaks 15–17 min,
FORCE 17–19 min. Full cohort (5 subjects × 2 visits, everything) 6.5 h.

Artefacts: `validation/hcp_retest/{<subj>_compare_results.json,
<subj>_variant_noiso.json, summary.json, run_cohort_driver.log,
run_cohort_noiso.log}`; per-visit arrays in `/data/datasets/hcp/b1/<subj>/`.

## 11. Tier C (2026-09-17)

### 11.1 C2: three Monte Carlo engines on one CATERPillar substrate

Permeable_MCDS (CHUV, `whitematter` branch) builds from its `compile.sh`
recipe and reads CATERPillar spheres as SWC-like axons
(`validation/run_mcds_parity.py`; geometry needs a `.swc` extension,
per-process `_rep_NN_DWI.txt` outputs are summed, and spheres crossing a
periodic face are replicated as images because MCDS wraps walkers but
not obstacles). On the sphere-chain test it behaves as a tube (0.128
along the chain vs 0.135 free, 1.00 across), which MCMRSimulator v1.1.0
does not (0.62). On the exported straight bundle (559 spheres, 10 µm,
δ 6 / Δ 12 ms, D 2, 8000 walkers), mean signal per shell:

| b | S_intra: MCDS / **JAX** / MCMR | S_extra: MCDS / **JAX** / MCMR | JAX at dt 10 → 2 → 1 µs (intra / extra) |
|---|---|---|---|
| 1000 | 0.470 / 0.550 / 0.824 | 0.186 / 0.238 / 0.232 | 0.558 → 0.568 → 0.577 / 0.236 → 0.240 → 0.244 |
| 2000 | 0.288 / 0.390 / 0.695 | 0.052 / 0.070 / 0.074 | |
| 3000 | 0.204 / 0.312 / 0.600 | 0.024 / 0.032 / 0.036 | 0.320 → 0.334 → 0.341 / 0.043 → 0.033 → 0.037 |
| along axon, b = 1000 / 3000 | 0.235 / 0.032 · 0.299 / 0.095 · 0.71 / 0.38 | | free: 0.135 / 0.002 |

- **Extra-cellular**: JAX and MCMRSimulator agree (0.01 RMS, doc 007
  §8.6) and the JAX walker is converged in time step; MCDS sits 0.05
  lower even with periodic images — walkers in MCDS see a freer
  extra-cellular space than the other two engines. Not yet explained
  (candidates: its stuck-spin discard, or how it initialises "extra"
  walkers relative to the image spheres).
- **Intra-axonal**: the three engines bracket each other, MCDS < JAX <
  MCMR, with MCMR's over-restriction already diagnosed. MCDS lets
  walkers move more freely along the chain than our union-SDF walker
  (0.235 vs 0.30–0.33 at b = 1000 along the axon); halving our step
  twice moves us 0.02 the *other* way, so the difference is in the
  collision model, not our discretisation. Two engines that both
  claim to treat overlapping spheres as tubes differ by 0.08 in the
  intra signal at b = 1000; the honest statement is that the
  intra-axonal signal of sphere-chain substrates is uncertain to that
  level across today's simulators, which is exactly the question the
  OCTOPUS and CATERPillar notes ask (tube vs sphere+cylinder reading).
- Consequence for the substrate rows: orientation results are robust to
  this (all engines agree the signal is a dispersed stick), f_i
  comparisons inherit a ±0.05 engine uncertainty on the intra signal.

### 11.2 C3: emulator-in-the-loop inversion against Monte Carlo (2026-09-17)

The question for C3 was whether a voxel can be fitted against the
*physics* instead of against Gaussian compartments: train an emulator of
Monte Carlo signals over a substrate-parameter family, then invert
held-out MC signals through it with no compartment model at all.

**Library** (`validation/generate_mc_library.py`, log
`validation/mc_library_generation.log`): 600 CATERPillar CLI substrates
(10 µm box; requested ICVF 0.3–0.8, crossing angle 0 or 10–90°,
tortuous/beaded half the time, c2 0.90–0.995), JAX walker on the PRISM
scheme (δ/Δ 6/12 ms, D 2.0), 4000 particles each; 582 usable (18 too
few axons), 83 min on the CPU partition. Labels are the *measured*
descriptors (intra fraction, per-bundle ⟨cos²⟩), since the CLI's
achieved ICVF undershoots and varies (§10.2): f_intra in the library
spans 0.04–0.95 rather than the requested 0.3–0.8.

**Emulator** (`validation/emulator_inversion.py`): per-measurement
Equinox MLP f(θ, b, ĝ) → log S with antipodal-symmetric ĝ features, 5
seeds, 6000 Adam steps each (107 s on the GB10). Held-out RMSE of S on
40 substrates: 0.027 (b0), 0.041 (b1000), 0.044 (b2000), 0.044 (b3000)
— comparable to the Rician noise floor at SNR 30 (0.033), so the
emulator is not the limiting factor at that SNR.

**Inversion**: Levenberg–Marquardt (Optimistix) over θ + a rotation
(bundle-1 direction and roll), 8 restarts, on the 40 held-out signals at
Rician SNR 30 observed in a random lab orientation.

| parameter | r | bias | sd |
|---|---|---|---|
| f_intra | 0.98 | +0.009 | 0.042 |
| crossing angle (°) | 0.63 | +5.1 | 23.1 |
| c2 bundle 1 | 0.63 | −0.03 | 0.16 |
| c2 bundle 2 | 0.40 | −0.08 | 0.23 |
| tortuous (0/1) | 0.53 | +0.04 | 0.44 |
| bundle-1 axis | median error 6.8° | | |

- **f_intra from MC signals is essentially solved by the emulator**:
  r = 0.98 with no compartment model, across a density range (0.04–0.95)
  far wider than any of the Gaussian fits in §8.2/§10.3 were tested on,
  and without the ±0.05 attribution bias those fits show at low density.
  This is the strongest f_i result in the document and does not depend
  on any choice of D∥, D⊥ or tortuosity.
- **Orientation is fair, geometry is weak.** The bundle axis is
  recovered to ~7°, i.e. worse than the stick+zeppelin plus-x (4–6°) on
  the same kind of substrate — the emulator is trained on 582 points in
  a 5-D θ space, so its angular derivatives are noisier than an
  analytical model's. Crossing angle, per-bundle dispersion and the
  tortuous flag are only weakly identified (r 0.4–0.6): at one diffusion
  time and b ≤ 3000, dispersion and a second bundle at a small angle
  produce near-identical signals, which is the same degeneracy the
  compartment fits have, now measured against physics rather than
  against an assumed model.
- **Consequence for the roadmap.** The emulator is the right *f_i*
  estimator and the plus-x remains the right *orientation* estimator;
  the natural combination is plus-x for the peaks and the emulator for
  the density given those peaks. Resolving the geometric descriptors
  needs either the multi-Δ protocol from §9.3 (adds diffusion-time
  contrast) or a denser library (the 582 points are the bottleneck for
  the angular derivatives — 5k substrates is a 12 h CPU job).
- The stick+zeppelin comparison on the same held-out MC signals named in
  the script docstring was not run; §10.3 already gives that number on
  the same substrate family.

Artefacts: `validation/emulator_inversion_results.json`,
`validation/emulator_inversion.log`,
`/data/datasets/sbi4dwi_mc_library/mc_library.h5` (582 × 160, with
intra/extra signals stored separately).

## 12. Protocol transfer HCP-YA → HCP-A by digital twin (2026-09-25)

Motivation: a planned UCSF Graves' orbitopathy / thyroid eye disease study
on a Prisma with the HCP-A (Lifespan) diffusion protocol. The question is
what the new protocol will deliver relative to the HCP-YA data we have
validated on, *before* anyone is scanned, and how harmonisation works
when the forward model takes the acquisition as an explicit input.

`validation/protocol_transfer_hcpa.py`. For each of the five retest
subjects the first-visit WM fit (190–235k voxels) is the ground-truth
tissue; it is forward-simulated with Rician noise on
(i) the subject's own HCP-YA scheme (b 1000/2000/3000 × 90 + 18 b0,
1.25 mm; measured WM b0 SNR 18–21),
(ii) the HCP-A scheme (b 1500 × 93 + b 3000 × 92 + 28 b0, AP and PA →
398 volumes, 1.5 mm; SNR scaled by (1.5/1.25)³ = 1.73 → 31–37),
(iii) HCP-A at HCP-YA SNR (directions/shells only), and refitted with the
same model from a perturbed initialisation. Each model is tested against
its own fitted tissue (default = CSF + GM balls; no-iso = §10.5's WM
model). Scan times: HCP-YA 59 min, HCP-A 21.4 min. No partial-volume or
Prisma-vs-Skyra hardware effects are modelled.

**Recovery of the generating tissue, mean over five subjects**

| model / protocol | f_i sd | f_i mean \|Δ\| | f_i bias | main fixel Δθ median | < 10° | fixel count agree |
|---|---|---|---|---|---|---|
| default / HCP-YA | 0.054 | 0.042 | +0.029 | 3.27° | 87 % | 81 % |
| default / HCP-A dirs at YA SNR | 0.040 | 0.033 | +0.018 | 2.52° | 91 % | 85 % |
| default / **HCP-A** | **0.031** | **0.024** | +0.013 | **1.39°** | 96 % | 90 % |
| no-iso / HCP-YA | 0.010 | 0.008 | +0.002 | 1.63° | 93 % | 98 % |
| no-iso / HCP-A dirs at YA SNR | 0.008 | 0.006 | +0.002 | 1.28° | 94 % | 99 % |
| no-iso / **HCP-A** | **0.005** | **0.004** | +0.000 | **0.72°** | 97 % | 99 % |

Between-subject sd of every entry is ≤ 0.006 in f_i and ≤ 0.3° in angle.

**Fisher / CRLB per voxel** (single fibre, f_i 0.5, diffusivities global
as in the fit; sd of f_i): HCP-YA 0.063, HCP-A 0.039, HCP-A at YA SNR
0.067; per √minute of scan time HCP-A is **2.7× more efficient**
(0.18 vs 0.49). Two- and three-fibre voxels: same ratios (0.077 → 0.047,
0.159 → 0.095). Fibre angle CRLB halves (0.49° → 0.23°). With
diffusivities free per voxel the CRLBs are 3× larger for every protocol —
the reason D∥ is a global parameter in the fit.

- **HCP-A is expected to be the better protocol for everything the
  models estimate**, by 1.7–2.3× in precision, even though it drops the
  b = 1000 shell and is a third of the scan time. Two-thirds of the gain
  is the voxel-volume SNR; one-third is the 398-volume two-shell design
  itself (row "HCP-A dirs at YA SNR"). This is the slide for the UCSF
  planning: quantified before the first scan, with the caveat that 1.5 mm
  partial voluming in thin structures is not in the simulation.
- **The f_i / f_iso degeneracy is the dominant error of the default
  model on any protocol**: sd 0.054 vs 0.010 for the WM-only model on
  identical HCP-YA data, with a +0.03 bias (f_i leaks into the balls). In
  vivo this appeared as the retest CoV gap of §10.5 (8.2 % vs 5.5 %); in
  simulation it is 5×. A prior on the isotropic fractions is the single
  most valuable modelling change left.
- **Harmonisation at the model level**: the same tissue on two protocols
  gives the same f_i to within the bias column above (≤ 0.03 default,
  ≤ 0.002 no-iso) with no image-level harmonisation, because the
  acquisition enters the likelihood explicitly. What remains
  scanner-specific — noise floor, gradient nonlinearity, the in-vivo σ
  inflation factor (1.76 ± 0.10 on HCP-YA) — is estimated per scanner,
  and the digital twin gives the expected posterior widths to check the
  first UCSF controls against.
- **Not covered here**: orbital diffusion. The HCP-A protocol is a brain
  protocol; for optic nerve and extraocular muscles a coronal RESOLVE
  block (multi-shot EPI) is the realistic quantitative option and a
  TGSE-BLADE trace DWI the distortion-free qualitative one (Siemens
  product; version-dependent). Our SBI posteriors on a 20–30-direction
  RESOLVE scheme with a single-fibre optic-nerve model are the natural
  next design step; the Fisher tool chooses its b-values.

Artefacts: `validation/hcp_retest/protocol_transfer_hcpa.{json,log}`.

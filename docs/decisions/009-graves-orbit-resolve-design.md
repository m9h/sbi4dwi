# 9. Orbital RESOLVE diffusion block for Graves orbitopathy — Fisher design

## Date
2026-10-09

## Status
Design calculation (no orbit data acquired). Companion to doc 008 §12/§13.5 (HCP-A whole-brain transfer). For the UCSF psychiatry Prisma / Silkiss TED project.

## Question
The HCP-A brain protocol does not image the orbit (doc 008 §12). A separate non-EPI block is needed: coronal readout-segmented EPI (Siemens RESOLVE), 2 × 2 × 3 mm, ~20 slices, 5 readout segments, TR ≈ 3.6 s. What diffusion scheme, within a fixed time budget, best measures (a) extraocular-muscle (EOM) free-water fraction f_w as an oedema/activity marker and (b) optic-nerve intra-axonal fraction f_i for compressive optic neuropathy?

## Method
`validation/design_orbit_resolve.py` (results `validation/orbit_resolve_design.json`). Gaussian-noise Fisher information / CRLB per tissue, b0 SNR 25 at TE = 0 (2 × 2 × 3 mm, 3 T; assumed), single TE set by b_max via TE = 42 + 22·√(b_max/1000) ms, per-compartment T2 weighting (muscle 35 ms, nerve 70 ms, water 2 s). Scan time = n_vol · 5 segments · TR; a scheme shorter than the budget is repeated (information × budget/time). Tissue models:

- EOM: zeppelin muscle (D∥ 1.9, D⊥ 1.4 µm²/ms) + free water f_w = 0.15; orientation two nuisance angles.
- Optic nerve: stick f_i = 0.55 + tortuous zeppelin + CSF f_csf = 0.15 (sheath partial volume); orientation nuisance.
- Weak Gaussian prior (sd ≈ 30°) on each orientation angle (Bayesian CRLB; keeps 3-direction schemes finite).

Named candidates plus an exhaustive search over 1–3 shells, b ∈ {250 … 2500}, 6/12/20 directions, objective sd(f_w) + sd(f_i).

## Results (8-minute budget)

| scheme | vol | TE ms | min | repeats | EOM sd f_w | EOM sd D⊥ | nerve sd f_i | nerve sd f_csf | nerve angle |
|---|---|---|---|---|---|---|---|---|---|
| clinical b1000 × 3 dir | 4 | 64 | 1.2 | 6.7 | 0.019 | 31 % | 0.209 | 0.027 | 14.5° |
| **b1000 × 6** | 7 | 64 | 2.1 | 3.8 | 0.025 | 30 % | 0.061 | 0.036 | 7.0° |
| b1500 × 6 | 7 | 69 | 2.1 | 3.8 | 0.025 | 45 % | 0.059 | 0.035 | 9.7° |
| b500/1000 × 6 | 13 | 64 | 3.9 | 2.1 | 0.033 | 29 % | 0.072 | 0.048 | 7.2° |
| b1000/1250/1500 × 6 | 20 | 69 | 6.0 | 1.3 | 0.030 | 37 % | 0.062 | 0.041 | 6.5° |
| b800/1500 × 12 | 27 | 69 | 8.1 | — | over budget | | | | |

Search optimum by sd(f_w)+sd(f_i): single shell b1000–1500 with 6 directions, repeated to fill the budget. Best for EOM alone: b2500 × 6 (f_w 0.024, but D⊥ unidentifiable at 147 %). Best for nerve alone: b1500 × 6 (f_i 0.059).

## Reading

1. **The orbit block is SNR-limited, not design-limited.** Every feasible scheme lands at f_w sd 0.025–0.033 and f_i sd 0.06–0.07. Adding shells costs TE (muscle T2 35 ms makes every 5 ms of TE a 13 % signal loss) and costs repeats; the extra contrast does not pay for either. Under this model sd scales as 1/(SNR₀·√budget), so the levers are voxel size, coil, slice count and minutes, not b-values.
2. **Clinical 3-direction DWI is adequate for EOM f_w and useless for the nerve.** With orientation unknown, three directions cannot separate f_i from orientation (sd 0.21); six directions are the minimum. The recommendation is therefore b = 1000 s/mm², 6 directions, 1 b0 per block, repeated ~4× in 8 min (≈ 28 volumes), or b1500 if the nerve is the priority (f_i 0.059 vs 0.061, at a 50 % worse D⊥).
3. **What the precision buys.** Scan–rescan minimal detectable change (95 %) is 2.77·sd: f_w ≈ 0.07 per muscle per visit, f_i ≈ 0.17 per nerve segment. The EOM figure is in range for active-vs-inactive TED if oedema shifts free water by ≥ 0.1 (to be checked against published EOM ADC changes before relying on it; ROI averaging over a muscle belly of ~100 voxels brings per-muscle sd to ~0.003). The nerve figure means per-patient optic-neuropathy staging by f_i is not achievable in an 8-minute RESOLVE block at this SNR; a group-level contrast or a longer, nerve-only block (3 mm isotropic, 15 min: sd ≈ 0.03) is.
4. **Where sbi4dwi adds something here.** (i) The joint muscle + free-water model with T2 weighting fitted directly, rather than an ADC; (ii) the per-voxel Laplace σ, which tells the reader which muscles and which nerve segments carried enough signal in a given patient (doc 008 §13.1: the σ ranks reliability even where it under-predicts by 1.8×); (iii) this Fisher tool, which lets the budget-vs-precision trade be made before the pilot rather than after.

## Limitations (all are assumptions, none measured)
Gaussian not Rician noise (optimistic at b ≥ 1500 for muscle); SNR₀ = 25 is a guess to be replaced by a pilot b0; single-TE model (a two-TE block would separate f_w from T2 and is worth a Fisher pass once the pilot SNR is known); no fat compartment (assumes fat suppression succeeds in the orbit, which RESOLVE's SPAIR usually does); no partial voluming at 2 × 2 × 3 mm on 4–8 mm muscles; muscle T2 35 ms and nerve T2 70 ms are literature-typical, not orbit-measured; EOM orientation prior of 30° is generous (muscle axes are largely known from anatomy, which would tighten D⊥ but not f_w). The digital-twin step of doc 008 §12 cannot be run for the orbit without orbit data; the first pilot scan should be used to calibrate SNR₀ and TE in this script and re-run it.

## Recommendation for the protocol meeting
Whole brain: HCP-A as in doc 008 §12/§13.5 (21 min). Orbit: coronal RESOLVE, 2 × 2 × 3 mm, b = 0/1000 (or 1500), 6 directions, repeated to ~8 min; fat-suppressed; one b0 pilot first to fix SNR₀. Analyse EOM with the free-water zeppelin model and report f_w with its σ per muscle; treat optic-nerve f_i as exploratory unless a nerve-only block is added.

# A benchmark seam artefact in amortised fibre-direction posteriors

**Short note, 2026-09-15.** Evidence: sbi4dwi doc 008 §7.1 (with addenda), §7.5; scripts `validation/diagnose_flow_variants.py`, `validation/diagnose_flow_rotated.py`.

## Claim

Any neural posterior estimator (NPE) whose direction parameters are spherical angles (θ, φ) or a hemisphere-constrained equivalent has *seams*: the φ wrap at 0 / 2π and the hemisphere boundary at z = 0. A posterior mode sitting on a seam is represented by the flow as two half-modes, and any fixel-extraction step (mean, dyadic, clustering) reads it as a broad or doubled distribution. Synthetic crossing benchmarks that place ground-truth fibres in the x–y plane — the natural choice — put every fibre on both seams at once. The resulting angular errors measure the parameterisation, not the posterior.

## Evidence

Spline masked-autoregressive flow, 256 × 6, streamed training, PRISM K = 2 stick+zeppelin model, three-shell scheme, SNR 30, 3,400 voxels at 16 crossing angles:

| flow | budget (steps × 512) | fibres in the z = 0 plane: err / recall | same configurations tilted 45° off the plane |
|---|---|---|---|
| θ ∈ (0, π), φ ∈ (0, 2π) | 20k | 23.9° / 41 % | 9.1° / 94.5 % |
| θ ≤ π/2 (hemisphere) | 40k | 21.6° / 48 % | 7.8° / 95 % |
| unit-disk → hemisphere | 20k | 23.8° / 39 % | 8.6° / 93 % |
| **dyadic** (xx, yy, xy, xz, yz; direction = principal eigenvector) | 20k | **7.7° / 94 %** | 6.7° / 97 % |
| dyadic | 40k | **5.8° / 98 %** | **5.6° / 95.5 %** |

The same flow, unchanged, moves from 24° to 9° when the truth is moved off the seams. Simulation-based calibration is unaffected throughout (87–96 % coverage at nominal 90 %): the flow is honest about a posterior it cannot represent compactly. The "antipodal split" diagnostic — the fraction of a fibre's samples in the hemisphere opposite its mode — is 0.28–0.43 on the seam and 0.00–0.10 tilted for the θ/φ flows.

A published NPE (SBI_dMRI, Ball-and-Sticks, hemisphere prior) shows the same dependence: sort-by-fraction reading 10.3° in-plane vs 19.1° tilted; after clustering its direction samples, 5.1° vs 4.8°.

## Remedy

Parameterise each fibre by the five free entries of a symmetric 3 × 3 tensor and take the principal eigenvector. The prior is then uniform in a box rather than on the sphere, which is acceptable for a proposal or for a posterior that is subsequently refined; it is seam-free by construction, and the seam and tilted numbers agree. Alternatively, evaluate θ/φ flows only on benchmarks whose ground truth has been rotated off both seams, and say so.

## Also worth knowing

More training over-trains: the dyadic flow at 80k steps is 5.6° on the seam set but 8.3° / 84 % tilted; the 3M-simulation SBI_dMRI variant shows the same pattern (0 % recall at 45–60° tilted). Budget should be chosen on a held-out, tilted set.

# To: the SBI_dMRI authors (Nottingham) — comparison results and a request for the classifier variant

**Status:** drafted 2026-09-15, not sent. Context: doc 008 §6.2, §7.4, §7.5.

Hello,

We have been running SBI_dMRI's Ball-and-Sticks NPE alongside our own methods on shared benchmarks (synthetic crossings on a 3-shell scheme, Monte Carlo axon substrates) and wanted to share what we found, since two of the findings are about evaluation rather than about your method, and one is a request.

**Setup.** Your forward model, priors (hemisphere, nfib = 2), noise policy and layout, trained as an NSF with the current `sbi` API on a GB10 GPU: 1M simulations / 60 epochs (14 min), and a 3M / 100-epoch / wider-net variant (84 min). Two practical changes were needed: `DirectPosterior`'s rejection sampler looped for hours on observations where the flow leaks mass outside the box prior, so we sample the density estimator directly and clip; and the per-voxel sampling loop was batched (it was ~1 s per voxel).

**What we found.**

1. *Your single-fibre estimates are the best of any method we ran* (1.2–2.8° on the substrates), and your stick-fraction sum is the least biased microstructure number in the comparison (within 0.02–0.10 of the geometric intra fraction on every substrate) — a model without a separate anisotropic extra-cellular compartment cannot mis-attribute a hindered extra-cellular space, which every stick+zeppelin model in our comparison, ours included, does.
2. *The published sort-by-fraction fixel extraction is what fails at crossings*, not the posterior. Recall collapses to 16–50 % at 45–90° on off-model substrates because the per-fibre sample sets contain both fibres (label switching), and the reported σ_θ is inflated 5–7× for the same reason. Pooling both fibres' direction samples and clustering them on the sphere (2-means, antipodal) restores recall to ~100 % and gives 4.6–5.1° at every budget we tried.
3. *An evaluation artefact that affects both of us.* A benchmark whose fibres lie in the z = 0 plane sits exactly on the boundary of a hemisphere prior; a posterior mode there is split by the flow and read as two. Our own θ/φ flow went from 24° to 9° just by tilting the same configurations 45° off the plane; yours changes too (published reading 10.3° → 19.1°, clustered 5.1° → 4.8°). If your published benchmarks used in-plane fibres, this is worth a check. A dyadic output parameterisation (each fibre as a symmetric rank-1 tensor, direction = principal eigenvector) removes the seam entirely for us.
4. *More training over-trains*: the 3M / wider variant is sharper where it is right (σ_θ 7.6° vs 15°) but confidently wrong on the tilted set (0 % recall at 45–60°). Your 1M / 60-epoch default is the better-calibrated choice.

**The request.** Your paper reports a classifier-based model-selection variant (SBI_ClassiFiber) that we did not run. If the code or a trained model is available, we would like to include it — the fibre-count decision is exactly where the sort-by-fraction reading loses recall, and it may change item 2 above.

We would be glad to share the exported benchmarks (npz) and the runner so the numbers can be checked directly.

Best regards,
Morgan Hough

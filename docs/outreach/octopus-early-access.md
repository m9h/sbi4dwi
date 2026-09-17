# To: Ivan Brammerloh / Ileana Jelescu (CHUV) — OCTOPUS early access and a benchmark offer

**Status:** drafted 2026-09-17, not sent. Context: doc 006 (OCTOPUS landscape and phased plan), doc 007 §8.5–8.6, doc 008 §7–§10.

Dear Ivan, dear Ileana,

I read the OCTOPUS preprint (bioRxiv 2026.09.02.748537) with great interest: the combination of realistic glia, branching and beading in one differentiable-friendly substrate generator is exactly the piece missing from the validation chain we have been building in sbi4dwi, a JAX/Equinox stack for microstructure estimation with calibrated uncertainty.

Where we are, briefly, so you can judge whether early access makes sense:

- We use CATERPillar substrates (January build, and since this week the new CMake/JSON build) with our own JAX Monte Carlo walker and, as an independent check, MCMRSimulator v1.1.0. The two engines agree on the extra-cellular PGSE signal to 0.01 RMS up to b = 3000; on intra-axonal sphere chains MCMRSimulator's overlapping-sphere passage currently over-restricts along the chain (reported to Michiel Cottaar with reproducers), so the intra-axonal cross-check still rests on analytical limits. A third engine that reads CATERPillar/OCTOPUS geometry natively would close that gap; we are building Permeable_MCDS for the same reason.
- On these substrates we have measured *why* each estimator fails: FORCE is matching-limited (library holds an entry within 2.8° of every crossing, the search returns one 7–10° away), Gaussian-compartment fits mis-attribute the hindered extra-cellular space to isotropic compartments, and amortised flows were being judged on a benchmark whose fibres sat on the parameterisation seams. With those fixed, a differentiable stick+zeppelin fit with dispersion and a free extra-cellular D⊥ recovers orientation to 4–6° and f_i within 0.03 on dense crossings — and fails predictably on sparse tissue and at single diffusion times.
- The next step is exactly what OCTOPUS enables: an emulator of Monte Carlo signals over a substrate-parameter family (density, dispersion, beading, glial fraction) so voxels can be fitted, and neural posteriors trained, against the physics rather than against Gaussian compartments. We have the pipeline (library → Equinox emulator → Optimistix inversion → SBC) and would like to run it on OCTOPUS substrates rather than on our own approximations of them.

Two concrete asks:

1. Early access to the OCTOPUS code (or a batch of generated substrates with their configurations) under whatever terms suit you. We would report back with the three-engine comparison on the same geometries and with the estimator results.
2. Two technical questions for the CATERPillar side, which we have also written to the CATERPillar maintainers: is there a way to seed the generator (runs are not reproducible from the config, and achieved ICVF varies 0.17–0.47 for a 0.5 request in the new build), and should overlapping spheres within an axon be read as a union tube or as sphere+cylinder segments? Downstream simulators are diverging on the second point.

Everything above is reproducible from a benchmark package we can share (npz signals, substrate CSVs, runners for FORCE / SBI_dMRI / MCMRSimulator).

With best regards,
Morgan Hough

# To: Michiel Cottaar (MCMRSimulator.jl) — overlapping-sphere chains restrict diffusion along the chain

**Status:** drafted 2026-09-15, not sent. Reproducers: `julia/mcmr/chain_test.jl`, `julia/mcmr/chain_spacing.jl` (Julia 1.13.0, MCMRSimulator v1.1.0 from the FMRIB repository, MRIBuilder 0.4.1).

Hi Michiel,

We are using MCMRSimulator v1.1.0 as an independent check on our own JAX Monte Carlo (sbi4dwi) and wanted to report one thing, and thank you for another.

**The thanks first.** On randomly packed cylinders and on CATERPillar axon substrates, the *extra-cellular* PGSE signals from MCMRSimulator and from our walker agree to 0.010–0.015 RMS on three shells up to b = 3000 s/mm², single bundle and 90° crossing alike. Your permeable-cylinder geometry then let us validate an exchange model end to end: a Fisher-designed three-shell multi-Δ protocol recovers the residence time we measure directly in your simulator (117 / 37 / 15 ms fitted vs ≈ 100 / 32 / 11 ms tracked) where a single-Δ protocol cannot. That is a genuinely useful third engine.

**The report.** `Spheres(...; overlapping=true)` (new in v1.1.0) does not let a chain of overlapping spheres behave as a tube. A straight chain of r = 0.5 µm spheres along z in a 10 µm periodic box, D = 2 µm²/ms, PGSE δ = 6 / Δ = 12 ms, b = 1000 along the chain:

| geometry | S along chain | S perpendicular |
|---|---|---|
| chain, `overlapping=true`, spacing 0.9 / 0.7 / 0.5 / 0.35 / 0.25 µm | 0.68 / 0.59 / 0.62 / 0.65 / 0.65 | 1.00 |
| chain, `overlapping=false` | 1.00 | 1.00 |
| chain, `overlapping=true, permeability=Inf` | 0.08 | 0.14 |
| `Cylinders` r = 0.5 (reference) | 0.11 | 1.00 |
| free diffusion exp(−bD) | 0.135 | — |

So spins are no longer trapped (0.65 vs 1.0), but they are far more restricted along the chain than a cylinder (0.11), and the number does not depend on the spacing — even at 0.25 µm, where each sphere overlaps three neighbours on each side and the necks are 0.48 µm wide, the along-chain signal is the same as at 0.9 µm. That pattern suggests the passage is granted between a spin's *current* sphere and one neighbour but something in the collision handling (surface normal at the neck? the `inside_other_rounds` filter?) still reflects most of the along-chain steps. On a real CATERPillar substrate (559 spheres, 40k-spin readout) the intra-axonal along-axon signal comes out at 0.71 at b = 1000 against 0.30 from our walker and 0.135 for a free stick.

Your CHANGELOG already notes that cylindrical connections between SWC spheres are coming; I mention it because anyone using `read_swc` or CATERPillar output with v1.1.0 today will get intra-axonal signals that look like ~50 % restriction along the axon, and it is not obvious from the outside that this is the overlapping-sphere passage rather than the geometry. If it helps, the two scripts above reproduce everything in under two minutes on 8 threads.

Two smaller notes from the same work:

- `Subset(inside=true)` is evaluated at readout time, not at seeding. That is fine and arguably right, but it surprised us when measuring residence times (the "fraction still inside" was 1.0 at every time). Tracking spin identity across sequences works; a sentence in the `readout` docstring would save others the detour.
- `main` currently requires FillArrays ≥ 1.17, which is not in General yet, so it cannot be installed alongside the registry; v1.1.0 installs cleanly on Julia 1.13.0.

Happy to share the CATERPillar substrate (CSV of spheres) and our reference signals if you want to look at the chain case directly.

Best,
Morgan

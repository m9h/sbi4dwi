# To: CATERPillar maintainers (jazz031195/CATERPillar, CHUV) — confirmation of the severed-axon symptom and two questions

**Status:** drafted 2026-09-15, not sent. Context: doc 007 §6.14–6.19 and §8; doc 008 §7.2, §8.2.

Hello,

We use CATERPillar to generate axon substrates for a Monte Carlo benchmark of fibre-orientation and microstructure estimators (sbi4dwi; our own JAX walker, cross-checked against MCMRSimulator). Your 2026-09-10 commit "Fix severed axons, livelock and unoptimised build; adaptive glial/axon packing" describes exactly what we had been seeing on the January build, so this is a confirmation with numbers, plus two questions.

**What we saw on the pre-fix build** (10 µm box, `axons_without_myelin_icvf` 0.5, `c2` 0.98, `tortuous` 0/1, `beading_variation` 0.3):

- A fraction of axons ended as stubs with `zmax` ≈ 0.2 µm — one or two spheres — rather than crossing the box. Our loader had to drop axons with fewer than 5 spheres before the bundle axis was well defined.
- The type-2 sphere rows were exact duplicates of the type-0 rows and were dropped.
- Realisations are not seed-stable: the same config gives a different substrate on each run (intra fraction 0.47–0.54 for the same target 0.5). We now compare methods within a run only. If a seed argument exists in the new CLI, that would remove a real nuisance for benchmarking.

None of this stopped the substrates from being useful — the extra-cellular signals from two independent engines agree to 0.01 RMS, and the intra-axonal signal behaves as a dispersed stick with apparent D∥ 1.3–1.5 at ⟨cos²⟩ = 0.93 — but the stubs inflate the extra-cellular fraction locally and the duplicates double the sphere count.

**Two questions**

1. In the new JSON/CLI build, is the sphere order within an axon still growth order (we rebuild segment directions from consecutive centres), and does `parent_id` in the new CSV give the connectivity directly?
2. Are overlapping spheres within one axon intended to be read as a connected tube (union of spheres), or as sphere+cylinder segments? MCMRSimulator's overlapping-sphere passage currently over-restricts along such chains (we have reported this separately), so a recommended geometric reading from you would help everyone downstream converge.

We will port our loader to the new CLI and re-run the crossing benchmark on un-severed axons; happy to share the results and the loader.

Best regards,
Morgan Hough

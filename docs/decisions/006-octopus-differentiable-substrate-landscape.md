# 6. OCTOPUS and the 2026 Differentiable-Substrate Landscape

## Date

2026-09-08

## Status

Research survey + incorporation plan. Supersedes nothing in doc 004; extends
its §5–§6 landscape framing to the 2026 forward/inverse substrate literature.

## Summary

Doc 004 surveyed **forward modelling as an inference engine** (FORCE, SBI for
dMRI, SBIDTI, cuDIMOT) and then spent §9–§24 empirically stress-testing FORCE
on the DiSCo phantom. That whole line of work operated on **analytical
compartment geometry** — sticks, zeppelins, spheres — and its recurring
finding was that the *design envelope* (library priors, acquisition shells,
compartment richness) dominates the result.

Four papers published since doc 004's last update move the field off analytical
geometry entirely, in two opposite directions:

| Direction | Paper | Group | What it does |
|---|---|---|---|
| **Forward** | **OCTOPUS** (Sep 2026) | CHUV Lausanne (Jelescu) | Grows realistic numerical brain *cells*; simulates signal with MCDC |
| **Inverse** | **ReMiDi** (Feb 2025) | UCSC (Marinescu) + INRIA (Li) | Differentiable FEM; backprops signal loss into mesh vertices |
| **Inverse** | **Spinverse** (Mar 2026) | UCSC (Marinescu) + INRIA (Li) | Differentiable Bloch–Torrey on tet grids with learnable face permeability |
| **Inverse** | **PRISM** (Mar 2026) | Indiana/DIPY (Shah, Garyfallidis) | Patch-wise differentiable analysis-by-synthesis for fixel recovery |

The through-line: **everyone has converged on differentiable physics, and the
open gap is that the forward and inverse halves have never been run against
each other.** OCTOPUS has no inverse method; Spinverse has no ground-truth
substrate set. This repository already holds working pieces of both halves.

This doc records what each paper is, what the repo already has, what is
mis-documented, and a concrete plan to incorporate OCTOPUS.

---

## 1. OCTOPUS — realistic numerical brain cells (the forward half)

**Paper:** Brammerloh M, de Riedmatten I, Beaubis J, Nguyen-Duc J, Oliveira AR,
Le Boeuf Fló A, Fischi-Gomez E, Rafael Patiño Lopez J, Jelescu IO.
*OCTOPUS: A versatile open-source tool creating realistic numerical brain cells.*
bioRxiv, posted 2026-09-07. DOI: 10.64898/2026.09.02.748537
Europe PMC: PPR1313907. Corresponding author: Malte Brammerloh (CHUV).

**Not to be confused with** the 2022 paper on dMRI tractography of the actual
octopus brain, or the unrelated Octopus TDDFT code. The acronym is never
expanded in the preprint.

### 1.1 What it does

OCTOPUS is a **substrate generator, not an estimator**. It grows digital
replicas of brain cells carrying the morphological features that make
analytical signal expressions impossible:

- **Branching** — hierarchical process development to a target branching order
- **Tapering** — `r(x) = r₀ + (r_start − r₀)·exp(−x / l_tapering)`
- **Undulation** — Ornstein–Uhlenbeck process on the process tangent direction
- **Beading** — lognormal radius variation along process length
- **Protrusions** — dendritic spines as a cylindrical neck + spherical head

Two growth modes:

1. **Free growth** — no predefined structure, random branch orientations; new
   branches preferentially grown at segment ends from which only one branch
   emerges.
2. **Skeleton-based growth** — a histological cell skeleton is used as
   scaffold, preserving branch direction, arborisation and segment lengths.

Both start by placing a soma, adding primary branches, then iteratively growing
secondary processes to the target branching order.

Cell types recreated from histology: pyramidal, GABAergic and glutamatergic
neurons; astrocytes; microglia. Eight cell types reproduced in total.

**Signal simulation is not part of OCTOPUS** — the paper uses "a customized
version of the MCDC simulator" (Monte Carlo Diffusion & Collision, from
Rafael-Patiño's line of work). OCTOPUS emits geometry; MCDC emits signal.

**Python interface:** `OCTOpool`.

### 1.2 Validation strategy — plausibility by signature, not ground truth

This is worth noting because it defines the gap the repo can fill. OCTOPUS is
validated by reproducing *known qualitative dMRI signatures*, not by any
quantitative inverse benchmark:

1. Short-range disorder: diffusivity and kurtosis follow 1/√t scaling in
   undulated and beaded processes
2. High-b power law: signal follows 1/√b, consistent with stick models
3. Histological reproduction: eight cell types via both growth modes
4. Branching reduces effective diffusivity at long diffusion times (~500 ms)
5. Dendritic spines decrease signal decay by ~37% (hindered intracellular
   diffusion)

Finding of note: **different growth strategies are adequate for more isotropic
vs. more anisotropic cells** — i.e. free growth and skeleton growth are not
interchangeable, and which one is right depends on the cell's anisotropy.

### 1.3 Availability — the blocker

> "The OCTOPUS and OCTOpool source code will be openly available upon
> publication."

Checked 2026-09-08: nothing on GitHub under that name (the `octopool` repos on
GH are unrelated — a Go worker pool and a GitHub read relay). **This is a
watch-item, not something that can be wired up today.**

Mitigating factor: **Jasmine Nguyen-Duc is an author on both OCTOPUS and
CATERPillar**, and CATERPillar is already cloned and compiled locally
(`/home/mhough/dev/dmipy/vendor/CATERPillar`, upstream
`github.com/jazz031195/CATERPillar`). Requesting early access is a warm ask,
not a cold one.

---

## 2. ReMiDi — differentiable FEM (the UCSC inverse half)

**Paper:** Khole PP, Petiwala ZK, Magesh SP, Mirafzali E, Gupta U, Li J-R,
Ianus A, Marinescu R. *ReMiDi: Reconstruction of Microstructure Using a
Differentiable Diffusion MRI Simulator.* arXiv:2502.01988 (Feb 2025);
presented at ISMRM 2025 (abstract 3396).
**Code:** https://github.com/BioMedAI-UCSC/ReMiDi

A PyTorch re-implementation of **SpinDoctor's FEM matrix formalism**, made
differentiable so that a signal-matching loss can be backpropagated into the
**3D mesh vertices**, recovering microstructure as an arbitrary mesh.
Jing-Rebecca Li (SpinDoctor's author) is a co-author, which is why this is
SpinDoctor-lineage rather than an independent solver.

**Constraint:** mesh topology must be fixed up front — you start from a mesh of
roughly the right shape and deform it.

**Note on naming:** ReMiDi is **not** a Monte Carlo random-walk simulator, and
it is not "Reference MRI Diffusion". See §7 for the drift this caused in our
docs.

---

## 3. Spinverse — differentiable Bloch–Torrey with learnable permeability

**Paper:** Khole PP, Brenes MM, Petiwala ZK, Mirafzali E, Gupta U, Li J-R,
Ianus A, Marinescu R. *Spinverse: Differentiable Physics for
Permeability-Aware Microstructure Reconstruction from Diffusion MRI.*
arXiv:2603.04638, submitted 2026-03-04. License CC BY-SA 4.0.
No code link in the preprint as of 2026-09-08.

ReMiDi's successor and the more consequential of the two. It inverts dMRI
measurements through a **fully differentiable Bloch–Torrey simulator**,
representing tissue on **tetrahedral grids with learnable per-face
permeability**. Microstructural boundaries therefore **emerge** from
optimisation rather than requiring fixed mesh topology — removing ReMiDi's
central constraint.

Two consequences that matter here:

- **Exchange/permeability enters the forward model.** This is the same axis the
  CHUV lab is chasing with NEXI, approached from the opposite side.
- **Topology-free recovery makes ground-truth validation harder, not easier.**
  If the mesh can become anything, you need externally-specified ground-truth
  geometry to know whether it became the *right* thing. That is exactly what
  OCTOPUS produces.

**Caveat:** §3 is sourced from the arXiv abstract page only. Full text not yet
read; treat the method description as a summary, not a specification.

---

## 4. PRISM — the FORCE successor (directly relevant to doc 004/005)

**Paper:** Abouagour M, Shah A, Garyfallidis E. *PRISM: Differentiable
Analysis-by-Synthesis for Fixel Recovery in Diffusion MRI.* arXiv:2604.00250,
submitted 2026-03-31. cs.CV.

**These are the FORCE authors.** Doc 004 §9–§24 and doc 005 are entirely about
FORCE's behaviour and its library-prior fragility. PRISM is those same authors
moving from dictionary/cosine-similarity matching to fitting an explicit
multi-compartment forward model **end-to-end over spatial patches**:

- Compartments: CSF, grey matter, white-matter fibres, restricted
- Patch-wise rather than voxel-independent — this is what buys tight crossings
- Two optimisation modes: fast MSE, and a Rician noise model that learns image
  quality parameters automatically
- Reported: **2.3° angular error, 99% detection**, 20° crossings resolved;
  ~741k HCP voxels full-brain in **~12 min on a single GPU**

### 4.1 Why this reframes doc 004 and doc 005

Doc 004's central empirical finding was that FORCE's **library prior is the
method** — retuning `diffusivity_config` per the paper's §3.2 protocol moved
DiSCo NDI correlation from r = 0.679 to r = 0.918, and the default library
carried a uniform 0.4-magnitude NDI bias that visual inspection would not
surface. Doc 005 was drafted as private feedback to Shah/Henriques/
Ramirez-Manzanares/Garyfallidis on exactly that.

PRISM's answer to the library-prior problem is to **delete the library** and
optimise the forward model directly — which is what this repository was built
to do (JAX forward models + Optimistix). So:

- **Doc 005 should be read against PRISM before it is sent.** As written it
  critiques a method its authors may already consider superseded. It is still
  worth sending — the tutorial/packaging points in §3 stand for anyone using
  `dipy.reconst.force` — but the framing should acknowledge PRISM.
- **The natural next benchmark** is dmipy-JAX's differentiable forward model
  vs. PRISM on DiSCo, on the same connectivity metric as §21–§24. That is a
  like-for-like comparison; FORCE vs dmipy-JAX never was.
- **§24's unresolved ~0.04 gap** may be less interesting now. The remaining
  candidates in §24.7 (1M-entry library, Watson vs Bingham dispersion) are all
  *library-design* levers, i.e. refinements of the paradigm PRISM abandons.

---

## 5. JEMRIS — what it is and is not

Recorded because it came up and the attribution is easy to get wrong.

**JEMRIS is not a UCSC project.** It is Stöcker's general-purpose Bloch/MRI
sequence simulator (Jülich). The recent development there is *GPU-accelerated
JEMRIS for extensive MRI simulations*, MAGMA 2025 — no overlap with the
Marinescu group.

JEMRIS sits on a **different axis** from everything else in this doc: it
simulates full pulse-sequence physics and k-space acquisition, not
microstructure geometry. It is complementary, not competing.

The repo already has hooks:

- `dmipy_jax/external/jemris.py` — `load_jemris_signal()` (complex k-space from
  ISMRMRD HDF5), `load_jemris_xml()`
- `dmipy_jax/tests/test_jemris_comparison.py` — analytic FID/90°-pulse checks
  run; the actual file-comparison test is `@pytest.mark.skip(reason="Requires
  external JEMRIS run & data file")`
- `dmipy_jax/external/pulseq.py` alongside it

Nothing to do here beyond not confusing it with the FEM line.

---

## 6. The forward/inverse seam — why this repo sits on it

OCTOPUS and Spinverse are **opposite directions on the same object**:

```
   growth rules ──OCTOPUS──> cell geometry ──MCDC──> dMRI signal
                                  ▲                       │
                                  └──── Spinverse ────────┘
                                     (differentiable Bloch–Torrey,
                                      learnable permeability)
```

Neither half validates the other today:

- **OCTOPUS has no inverse method.** Its validation is qualitative signature
  reproduction (§1.2), which cannot answer "would a fitting method recover
  these parameters?"
- **Spinverse has no ground-truth substrate library.** Topology-free mesh
  recovery is only meaningful against externally-specified truth geometry.

Closing that loop — grow a cell with OCTOPUS, simulate, recover with a
differentiable solver, compare recovered geometry against the known input — is
a real and currently unoccupied seam. This repo holds working pieces of both
halves (§7), which is the reason to care about OCTOPUS beyond curiosity.

It is also the **fourth axis of doc 004's recurring "check the design envelope"
lesson**:

| Axis | Where established |
|---|---|
| Library priors | §13, §17.5 |
| Acquisition (single- vs multi-shell) | §22 / §24 |
| Compartment richness | §22, §23 |
| **Geometry (analytical vs realistic)** | **this doc** |

---

## 7. What the repo actually has — verified inventory (2026-09-08)

Checked against the working tree, not against CLAUDE.md.

### 7.1 Present and relevant

| Path | What it gives us |
|---|---|
| `dmipy_jax/validation/caterpillar.py` | `CATERPillarOracle` — writes config, shells out to the binary, parses `*_spheres.csv` → DataFrame `(x, y, z, radius, type, id)` |
| `dmipy_jax/simulation/sphere_sdf.py` | `MultiSphereSDF` — consumes that sphere list as geometry |
| `dmipy_jax/simulation/monte_carlo.py` | Ground-truth MC with SDF geometry |
| `dmipy_jax/simulation/differentiable_walker.py` | Differentiable confined Brownian walker |
| `dmipy_jax/simulation/mesh_sim.py:366` | `MatrixFormalismSimulator` — **FEM matrix formalism in JAX, "ReMiDi / SpinDoctor flavor"** (the docstring says so at `mesh_sim.py:368`) |
| `dmipy_jax/validation/compare_disimpy.py` | CATERPillar substrate → dmipy-jax vs disimpy MC cross-check |
| `dmipy_jax/external/jemris.py`, `pulseq.py` | Sequence-level IO |
| `dmipy_jax/signal_models/sandi.py`, `sphere_models.py` | Soma/neurite compartments to stress-test |
| `/home/mhough/dev/dmipy/vendor/CATERPillar` | Cloned + compiled binary, upstream `jazz031195/CATERPillar` |

The important one: **`mesh_sim.py` already implements ReMiDi's solver family in
JAX.** The honest comparison against ReMiDi/Spinverse is therefore a
`SimulationComparisonRunner` job, not an integration job.

### 7.2 Documented but absent

`CLAUDE.md`'s directory map and `docs/external_projects.md` both describe an
oracle protocol that is not in the tree:

```
MISS  dmipy_jax/simulation/oracles/        (whole directory)
MISS  dmipy_jax/simulation/oracle.py       (OracleSimulator ABC)
MISS  dmipy_jax/pipeline/oracle_adapter.py (OracleModelSimulator)
MISS  dmipy_jax/pipeline/multi_fidelity.py
OK    dmipy_jax/library/, dmipy_jax/inference/, core/surrogate.py, io/hcp.py
```

The only oracle-shaped code present is `validation/caterpillar.py` and
`benchmarks/oracle_wrapper.py` (the latter wraps *pretrained PyTorch SBI
posteriors*, not simulators). **The oracle boundary CLAUDE.md advertises is
aspirational** — which matters directly, because it is the seam both OCTOPUS
and Spinverse would land through. Building it is a prerequisite, not a
detail.

---

## 8. Documentation drift found while checking

All verified against the tree on 2026-09-08. Each is a factual error, not a
style issue.

| File:line | Says | Correct |
|---|---|---|
| `docs/external_projects.md:39` | ReMiDi is a "GPU-accelerated **Monte Carlo random-walk** simulator" | Differentiable **FEM Bloch–Torrey** solver (SpinDoctor matrix formalism, PyTorch). Opposite family. |
| `docs/external_projects.md:36` | Source `remidi.org` / `github.com/jingrebeccali/ReMiDi` | `github.com/BioMedAI-UCSC/ReMiDi` |
| `docs/external_projects.md:42` | Wrapper lives at `dmipy_jax/simulation/oracles/remidi.py` | Path does not exist (§7.2) |
| `docs/prompts/remidi_manager.md:3` | "ReMiDi (**Reference MRI Diffusion**)" | "**Re**construction of **Mi**crostructure using a **Di**fferentiable diffusion MRI simulator" |
| `docs/external_projects.md:27` | CATERPillar = "Computer-Assisted Tissue Engineering for Reproducible Phantoms in Localised fibre arrangements" | "**Computational Axonal Threading Engine for Realistic Proliferation**" (upstream README) |
| `docs/external_projects.md:24` | Source `github.com/RafaelNH/CATERPillar` | `github.com/jazz031195/CATERPillar`. `RafaelNH` is Rafael Neto Henriques (DIPY/FORCE co-author) — a name collision with Jonathan Rafael-Patiño. |
| `docs/external_projects.md:31` | CATERPillar "was empty (placeholder only, never populated)" | True of *this* repo's `vendor/`, but misleading: the tool is cloned and compiled in the sibling checkout at `/home/mhough/dev/dmipy/vendor/CATERPillar`, which is exactly what `validation/caterpillar.py:17` defaults to. The note should point there. |

Fixing these is cheap and blocks nothing, but the ReMiDi mischaracterisation in
particular would send a future agent down the wrong path — it is the difference
between "wrap a black-box MC simulator" and "we already have this solver in
JAX".

---

## 9. Incorporation plan for OCTOPUS

Phased so that everything before Phase 3 is useful **whether or not OCTOPUS
ever releases**.

### Phase 0 — housekeeping (now, ~1h)

- [ ] Fix the seven drift items in §8.
- [ ] Commit the pending MRtrix 3.0.4 re-run as a doc 004 §18 addendum. Result:
      mean Pearson r across SNR {10, 30, 50} moves **0.11789 → 0.11794** —
      the SD_STREAM structural gap is version-stable to 5e-5 and is not an
      MRtrix artifact. Files: `validation/mrtrix_disco_connectivity_results.npz`
      (modified), `.v3.0.4.npz` (untracked).
- [ ] Resolve the two deleted-unstaged FORCE files
      (`force_disco_results_tuned.npz`, `force_disco_tuned.png`) — `git rm`,
      they are superseded by the connectivity-era artifacts.

### Phase 1 — build the oracle boundary for real (prerequisite)

The thing CLAUDE.md already claims exists. Nothing about OCTOPUS is blocked on
OCTOPUS; it is blocked on this.

- [ ] `dmipy_jax/simulation/oracle.py` — `OracleSimulator` ABC:
      `check_available()`, `generate_batch(params, acq)`, `generate_library(n)`
- [ ] `dmipy_jax/simulation/oracles/` — move `validation/caterpillar.py` in as
      the **reference implementation** and make it conform. It already has the
      right shape (config → subprocess → parse → geometry).
- [ ] `dmipy_jax/pipeline/oracle_adapter.py` — `OracleModelSimulator`
      (k-NN interpolation over a `SimulationLibrary`, out-of-support warning)
- [ ] Either implement `pipeline/multi_fidelity.py` or strike it from CLAUDE.md.
      Do not leave it documented-but-absent.

### Phase 2 — a substrate-geometry track that OCTOPUS will slot into

- [ ] **SWC / skeleton ingestion.** OCTOPUS's skeleton-based growth mode
      consumes histological cell skeletons; CLAUDE.md claims an SWC loader in
      `io/` but `grep -i swc` over `dmipy_jax/` returns nothing. Write
      `io/swc.py` (NeuroMorpho SWC → soma + branch tree). Independently useful,
      and it is the format OCTOPUS scaffolds on.
- [ ] **Extend `MultiSphereSDF` to sphere+cylinder unions.** CATERPillar is
      pure overlapping spheres; OCTOPUS adds cylindrical spine necks and
      tapered cylindrical segments. Same primitive family, one more shape.
- [ ] **Re-derive OCTOPUS's five signatures on CATERPillar substrates now.**
      All five (§1.2) are checkable with the existing MC walker: 1/√t
      diffusivity-and-kurtosis scaling, 1/√b high-b power law, branching
      effect at ~500 ms, beading, undulation. CATERPillar already produces
      beading and tortuosity. This gives a **regression harness that OCTOPUS
      substrates can be dropped into unchanged** — and it independently
      validates our MC walker against published signatures.
- [ ] **Gray-matter model stress test.** Use those substrates to test where
      `sandi.py`'s sphere-soma assumption breaks. Doc 004 §9–§24 was entirely
      white-matter/DiSCo; this is the unexplored half.

### Phase 3 — OCTOPUS proper (on release, or on early access)

- [ ] `OCTOPUSOracle` alongside `CATERPillarOracle`. Given the shared authorship
      (Nguyen-Duc, Rafael-Patiño) and shared primitives, expect the
      config → subprocess → sphere/cylinder-list → SDF path to transfer with
      modest changes.
- [ ] Cross-validate **three MC engines** on one OCTOPUS substrate: our
      `monte_carlo.py`, disimpy (harness exists at `compare_disimpy.py`), and
      OCTOPUS's customised MCDC. Same three-way agreement discipline as the
      MRtrix version check.
- [ ] Add OCTOPUS as the fourth leg of `SimulationComparisonRunner`:
      analytical vs FEM (`mesh_sim`) vs our MC vs OCTOPUS-MCDC.
- [ ] **SBI on morphology parameters.** Branching order, tapering length,
      undulation amplitude, beading amplitude, spine density form a prior space
      for `ModelSimulator` that no analytical forward model exposes. This is
      the axon-diameter/cell-density target in the project brief, with
      parameters that are only reachable through simulation.

### Phase 4 — close the forward/inverse loop (the seam in §6)

- [ ] Benchmark `mesh_sim.py`'s JAX matrix formalism against ReMiDi's PyTorch
      implementation on identical meshes. We claim the same solver family;
      verify it.
- [ ] Grow OCTOPUS cell → simulate → recover geometry with our differentiable
      solver → compare recovered vs known geometry. This is the experiment
      neither group can currently run.
- [ ] Only then consider Spinverse-style learnable permeability. Read the full
      preprint first (§3's caveat).

### Outreach (parallel, cheap)

- [ ] Email Brammerloh (CHUV) for OCTOPUS/OCTOpool early access. We already run
      CATERPillar from their lab; mention it.
- [ ] Revisit doc 005 against PRISM before sending (§4.1).

---

## 10. Risks and open questions

- **OCTOPUS may not release soon.** "Upon publication" on a preprint posted six
  days ago could mean months. Phases 0–2 are deliberately independent of this.
- **§3 and §4 are abstract-only.** Spinverse and PRISM method descriptions here
  come from arXiv landing pages, not full text. Read both before building
  anything against them.
- **MCDC is a fourth external simulator.** We already carry DIPY, ReMiDi and
  MCMR at the oracle boundary (nominally — see §7.2). Adding MCDC needs a
  reason beyond completeness; the reason here is that it is the engine OCTOPUS
  was validated with, so reproducing their signatures requires it.
- **Scope discipline.** Doc 004 ran to 24 sections chasing one r-value on one
  phantom. The forward/inverse seam in §6 is a larger surface. Phase 2's
  signature-regression harness is the checkpoint: if our MC walker cannot
  reproduce the five published signatures on CATERPillar substrates, stop and
  fix that before adding any new external tool.
- **PRISM may make parts of doc 004 §21–§24 moot.** Deciding whether to keep
  chasing §24.7's residual ~0.04 is a real call, not a formality.

---

## References

1. Brammerloh M, de Riedmatten I, Beaubis J, Nguyen-Duc J, Oliveira AR,
   Le Boeuf Fló A, Fischi-Gomez E, Rafael Patiño Lopez J, Jelescu IO.
   *OCTOPUS: A versatile open-source tool creating realistic numerical brain
   cells.* bioRxiv 2026. DOI: 10.64898/2026.09.02.748537

2. Khole PP, Petiwala ZK, Magesh SP, Mirafzali E, Gupta U, Li J-R, Ianus A,
   Marinescu R. *ReMiDi: Reconstruction of Microstructure Using a
   Differentiable Diffusion MRI Simulator.* arXiv:2502.01988, 2025.
   ISMRM 2025 abstract 3396. Code: github.com/BioMedAI-UCSC/ReMiDi

3. Khole PP, Brenes MM, Petiwala ZK, Mirafzali E, Gupta U, Li J-R, Ianus A,
   Marinescu R. *Spinverse: Differentiable Physics for Permeability-Aware
   Microstructure Reconstruction from Diffusion MRI.* arXiv:2603.04638, 2026.

4. Abouagour M, Shah A, Garyfallidis E. *PRISM: Differentiable
   Analysis-by-Synthesis for Fixel Recovery in Diffusion MRI.*
   arXiv:2604.00250, 2026.

5. Nguyen-Duc J et al. *CATERPillar: a flexible framework for generating white
   matter numerical substrates with incorporated glial cells.* bioRxiv
   10.1101/2025.06.20.660694; PubMed 41576825.
   Code: github.com/jazz031195/CATERPillar

6. Villarreal-Haro JL, Gardier R, Canales-Rodríguez EJ, Fischi-Gomez E,
   Girard G, Thiran J-P, Rafael-Patiño J. *CACTUS: a computational framework
   for generating realistic white matter microstructure substrates.*
   Front Neuroinform 17:1208073, 2023.

7. Shah AJ et al. *FORCE: FORward modeling for Complex microstructure
   Estimation.* Research Square preprint, 2025. DOI: 10.21203/rs.3.rs-8151109/v1
   (see doc 004 §9–§24, doc 005)

8. GPU-accelerated JEMRIS for extensive MRI simulations. MAGMA, 2025.
   DOI: 10.1007/s10334-025-01281-z (unrelated to the UCSC line; see §5)

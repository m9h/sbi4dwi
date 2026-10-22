# Decision records and result logs

Numbered files are architecture decision records (ADRs) or, from 004 on,
research logs that read like lab notebooks. The table says which is which and
what state each is in. Numbers are stable: other docs, commit messages and the
memory of previous sessions cite them as "doc 008 §13".

| doc | type | status (2026-10-09) | one line |
|---|---|---|---|
| [001](001-optimistix-over-optax.md) | ADR | accepted | Optimistix for deterministic voxel fitting, Optax for stochastic training |
| [002](002-adoption-of-kidger-stack.md) | ADR | accepted | Equinox / Diffrax / Lineax / Optimistix as the numerical stack |
| [002b](002b-equinox-architecture.md) | ADR | accepted | models and simulators are immutable `eqx.Module` pytrees |
| [002c](002c-unxt-integration-rfc.md) | RFC | open, not adopted | unit-safe arrays with `unxt`; convention today is SI everywhere |
| [003](003-simulation-stack.md) | ADR | accepted | FEM matrix formalism, Monte Carlo and SDE walkers; external simulators at the HDF5 oracle boundary |
| [004](004-landscape-forward-modeling-sbi.md) | survey | historical | forward-modelling and SBI landscape that set the roadmap |
| [005](005-force-developer-feedback.md) | log | historical | notes from the FORCE/DIPY exchange (seed fix, protocol) |
| [006](006-octopus-differentiable-substrate-landscape.md) | survey + log | live | differentiable substrate generators (OCTOPUS unreleased, CATERPillar, ConCeG, MCMRSimulator) |
| [007](007-exceed-prism-plan.md) | log | superseded by 008 | the plan and experiments to beat PRISM/FORCE on DiSCo |
| [008](008-positioning-vs-force-prism-sbi.md) | **results log** | live, primary | PRISM-JAX vs FORCE / SBI_dMRI / MSMT / NODDI: DiSCo, synthetic, substrates, 42-subject HCP retest, protocol transfer, uncertainty validation, CHUV realism ladder + glial compartment (§6, §10.6, §12, §13, §14) |
| [009](009-graves-orbit-resolve-design.md) | design calc | done, no data yet | Fisher design of an orbital RESOLVE block for Graves orbitopathy |
| [010](010-package-review-and-cleanup-plan.md) | review + plan | P0–P2 executed; deferred items in §9 | port fidelity, structure, documentation audit and the clean-up log (§7–§9) |
| [011](011-dmipy-v2-dmrai-lab-comparison.md) | comparison | done | SBI4DWI vs the revived dmipy 2.x stack (dmrai-lab): overlap only at the compartment layer; uncertainty/SBI/PRISM/validation are ours, substrates/relaxation/sequences are theirs; AGPL |

Conventions: one file per topic; append dated sections rather than rewriting
history; numbers in the tables are copied from the JSON the scripts in
`validation/` write, never retyped from memory.

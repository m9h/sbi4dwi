#!/usr/bin/env python3
"""
Assemble the sbi4dwi benchmark package (doc 008 §10 A1): every exported
dataset and external runner needed to reproduce docs 007–008, in one
directory with a README and checksums.

  python validation/build_benchmark_package.py --out /data/datasets/sbi4dwi_benchmark_v1
"""
import argparse, hashlib, shutil, json, subprocess
from pathlib import Path

FILES = {
    "synthetic (dataset 2)": ["validation/external/synthetic_snr30.npz", "validation/external/synthetic_snr10.npz",
                              "validation/external/synthetic_snr30_odi0.2.npz", "validation/external/synthetic_snr10_odi0.2.npz",
                              "validation/external/synthetic_clean.npz", "validation/external/synthetic_snr30_tilt45.npz"],
    "substrates (dataset 3)": ["validation/external/substrate_signals.npz", "validation/mcmr/straight_0_spheres.csv",
                               "validation/mcmr/straight_90_spheres.csv", "validation/mcmr/straight_0_ref.csv",
                               "validation/mcmr/straight_90_ref.csv", "validation/mcmr/meta.json"],
    "exchange (MCMR permeable cylinders)": [f"validation/mcmr/perm_{p}_{q}.csv" for p in ("0p0", "0p003", "0p01", "0p03") for q in ("disco", "optimised")]
                                            + [f"validation/mcmr/perm_{p}_meta.txt" for p in ("0p0", "0p003", "0p01", "0p03")],
    "runners": ["validation/run_force_external.py", "validation/diagnose_force_limits.py", "validation/run_force_disco_ndi.py",
                "docker/sbi_dmri_run.py", "docker/Dockerfile.sbi_dmri", "validation/validate_exchange_mcmr.py",
                "validation/design_exchange_protocol.py", "julia/mcmr/substrate_pgse.jl", "julia/mcmr/permeable_cylinders.jl",
                "julia/mcmr/exchange_time.jl", "julia/mcmr/chain_test.jl", "julia/mcmr/Project.toml", "julia/mcmr/Manifest.toml"],
    "results": ["validation/external/force_diag_synthetic.json", "validation/external/force_diag_substrates.json",
                "validation/external/force_disco_ndi.json", "validation/disco_microstructure_results.json",
                "validation/exchange_mcmr_results.json", "validation/exchange_protocol_design.json",
                "validation/dipy_refine_results.json", "validation/runtime_hardi.json"]
                + sorted(str(p) for p in Path("validation/external").glob("sbi_dmri_*.json")),
    "docs": ["docs/decisions/007-exceed-prism-plan.md", "docs/decisions/008-positioning-vs-force-prism-sbi.md",
             "docs/notes/seam-artefact-in-direction-npe.md"],
}

README = """# sbi4dwi benchmark package v1 (2026-09-15)

Datasets and runners behind sbi4dwi docs 007 and 008 (see docs/). DiSCo itself is
fetched with `dipy.data.fetch_disco1_dataset()`; the FORCE-authors' protocol loader is
`validation/validate_disco_force_protocol.py::load_protocol_data` in the repository.

- synthetic/: PRISM K=2 stick+zeppelin crossings, 3 shells × 64 dirs (b = 1000/2000/3000 s/mm²),
  17 conditions × 200 voxels; `_tilt45` is the seam-free version (fibres rotated 45° about y then x).
  Keys: data (17,10,20,193), mask, bvals (s/mm²), bvecs, gt_dirs (N,2,3), angle, gt_fracs, fintra, gt_odi.
- substrates/: CATERPillar axon substrates, MC signals from the JAX walker (δ 6 / Δ 12 ms, D 2 µm²/ms)
  on the same scheme; sphere CSVs (µm) and MCMRSimulator reference signals.
- exchange/: MCMRSimulator permeable-cylinder signals (241 cylinders, f_intra 0.601) for the DiSCo
  single-Δ and the Fisher-optimised protocols; residence times ≈ ∞ / 100 / 32 / 11 ms.
- runners/: FORCE (dipy ≥ 1.13 master), SBI_dMRI (Docker), MCMRSimulator (Julia 1.13) and the
  exchange-model fit and design scripts.
- results/: JSON outputs quoted in the docs.

Units: bvals in s/mm² in the npz files, SI (s/m²) inside sbi4dwi.
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {}; missing = []
    for group, files in FILES.items():
        d = out / group.split(" ")[0]; d.mkdir(exist_ok=True)
        for f in files:
            p = Path(f)
            if not p.exists():
                missing.append(f); continue
            shutil.copy2(p, d / p.name)
            manifest[f"{d.name}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / "README.md").write_text(README + "\n## Provenance\n\ngit commit: " + subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout)
    json.dump(manifest, open(out / "SHA256SUMS.json", "w"), indent=1)
    print(f"{len(manifest)} files → {out}; missing: {missing}")


if __name__ == "__main__":
    main()

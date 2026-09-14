#!/usr/bin/env python3
"""Export CATERPillar substrates (spheres CSV in µm) and our JAX MC reference
signals (S_intra, S_extra on the PRISM scheme, δ = 6 ms, Δ = 12 ms, D = 2 µm²/ms)
for the MCMRSimulator.jl parity run (doc 006 Phase 3 / doc 007 §8.5)."""
import json, numpy as np, pandas as pd
from dmipy_jax.validation import substrate_benchmark as sb
from dmipy_jax.validation.prism_synthetic import prism_scheme
bvals, bvecs = prism_scheme()
df = sb.caterpillar_bundle(icvf=0.5, box_um=10.0)
out = {}
for ang in (0, 90):
    sub = sb.make_crossing_substrate(df, ang, 10.0)
    S, S_in, S_ex = sb.simulate_substrate_signal(sub, bvals, bvecs, n_particles=8000)
    tag = f"straight_{ang}"
    pd.DataFrame({"x": sub.centers_m[:, 0] * 1e6, "y": sub.centers_m[:, 1] * 1e6, "z": sub.centers_m[:, 2] * 1e6,
                  "r": sub.radii_m * 1e6, "axon": sub.axon_ids}).to_csv(f"validation/mcmr/{tag}_spheres.csv", index=False)
    np.savetxt(f"validation/mcmr/{tag}_ref.csv", np.column_stack([bvals / 1e6, bvecs, S, S_in, S_ex]), delimiter=",",
               header="b_s_mm2,gx,gy,gz,S,S_intra,S_extra", comments="")
    out[tag] = {"box_um": 10.0, "f_intra": float(sub.f_intra), "axes": sub.axes.tolist(), "n_spheres": int(len(sub.radii_m)),
                "delta_ms": 6.0, "Delta_ms": 12.0, "D_um2_ms": 2.0}
    print(tag, out[tag], "S_in(b3000)", S_in[-64:].mean().round(3), "S_ex(b3000)", S_ex[-64:].mean().round(3), flush=True)
json.dump(out, open("validation/mcmr/meta.json", "w"), indent=1)

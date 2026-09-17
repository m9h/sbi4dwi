#!/usr/bin/env python3
"""
Tier C3 (doc 008 §10): a Monte Carlo substrate library for emulator-in-the-loop
inversion. Samples a substrate-parameter family, generates the CATERPillar
substrate (CLI build), simulates the PGSE signal with the JAX walker on the
PRISM scheme in a canonical frame (bundle 1 along z, crossing in the y–z
sheet), and stores (params, measured descriptors, signal) in an HDF5
SimulationLibrary.

Parameters sampled (uniform):   requested ICVF 0.3–0.8, crossing angle
0–90° (0 = single bundle, p = 0.25), tortuous {0,1}, beading 0–0.3
(tortuous only), c2 0.90–0.995.
Stored descriptors (measured): intra fraction, per-bundle ⟨cos²⟩ about the
axis, number of axons, mean radius — the labels the emulator is trained
on, since the request ≠ realisation for this generator.
"""
import argparse, time, json
import numpy as np, jax, jax.numpy as jnp
from pathlib import Path
from dmipy_jax.validation import substrate_benchmark as sb
from dmipy_jax.validation.prism_synthetic import prism_scheme


def segment_c2(sub):
    C, I = sub.centers_m, sub.axon_ids; out = []
    for a in sub.axes:
        num = den = 0.0
        for k in np.unique(I):
            c = C[I == k]
            if len(c) < 2: continue
            d = np.diff(c, axis=0); n = np.linalg.norm(d, axis=1); ok = (n > 0) & (n < 2e-6)
            num += float((((d[ok] / n[ok, None]) @ a) ** 2 * n[ok]).sum()); den += float(n[ok].sum())
        out.append(num / max(den, 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-particles", type=int, default=4000); ap.add_argument("--box-um", type=float, default=10.0)
    ap.add_argument("--out", default="/data/datasets/sbi4dwi_mc_library/mc_library.h5")
    a = ap.parse_args()
    import h5py
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    bvals, bvecs = prism_scheme(); rng = np.random.default_rng(a.seed)
    names = ["icvf_req", "angle_deg", "tortuous", "beading", "c2_req"]
    desc_names = ["f_intra", "c2_meas_1", "c2_meas_2", "n_axons", "mean_radius_um", "n_spheres"]
    P, D, S, S_in, S_ex = [], [], [], [], []
    t_start = time.time()
    for i in range(a.n):
        icvf = rng.uniform(0.3, 0.8); ang = 0.0 if rng.random() < 0.25 else rng.uniform(10, 90)
        tort = int(rng.random() < 0.5); bead = rng.uniform(0.0, 0.3) if tort else 0.0; c2 = rng.uniform(0.90, 0.995)
        try:
            df = sb.caterpillar_bundle(icvf=icvf, box_um=a.box_um, tortuous=tort, beading=bead, c2=c2, backend="cli")
            if df["id"].nunique() < 3: raise RuntimeError("too few axons")
            sub = sb.make_crossing_substrate(df, ang, a.box_um)
            Sig, Si, Se = sb.simulate_substrate_signal(sub, bvals, bvecs, n_particles=a.n_particles, seed=i)
        except Exception as e:
            print(f"  [{i}] skipped: {e}", flush=True); continue
        c2m = segment_c2(sub); c2m = c2m + [np.nan] * (2 - len(c2m))
        P.append([icvf, ang, tort, bead, c2]); D.append([sub.f_intra, c2m[0], c2m[1], df["id"].nunique(), 1e6 * df["radius"].mean(), len(sub.radii_m)])
        S.append(Sig); S_in.append(Si); S_ex.append(Se)
        if i % 20 == 0:
            print(f"  [{i}/{a.n}] icvf req {icvf:.2f} → {sub.f_intra:.2f}, angle {ang:.0f}, tort {tort}, S(b3000) {Sig[-64:].mean():.3f}  ({time.time()-t_start:.0f}s)", flush=True)
            with h5py.File(out, "w") as f:
                f.create_dataset("params", data=np.array(P, np.float32)); f.create_dataset("descriptors", data=np.array(D, np.float32))
                f.create_dataset("signals", data=np.array(S, np.float32)); f.create_dataset("signals_intra", data=np.array(S_in, np.float32))
                f.create_dataset("signals_extra", data=np.array(S_ex, np.float32)); f.create_dataset("bvals", data=bvals); f.create_dataset("bvecs", data=bvecs)
                f.attrs["parameter_names"] = json.dumps(names); f.attrs["descriptor_names"] = json.dumps(desc_names)
                f.attrs["frame"] = "bundle 1 along z; bundle 2 = Rx(angle)·z; box periodic"; f.attrs["delta_Delta_ms"] = "6, 12"; f.attrs["D_um2_ms"] = 2.0
    with h5py.File(out, "w") as f:
        f.create_dataset("params", data=np.array(P, np.float32)); f.create_dataset("descriptors", data=np.array(D, np.float32))
        f.create_dataset("signals", data=np.array(S, np.float32)); f.create_dataset("signals_intra", data=np.array(S_in, np.float32))
        f.create_dataset("signals_extra", data=np.array(S_ex, np.float32)); f.create_dataset("bvals", data=bvals); f.create_dataset("bvecs", data=bvecs)
        f.attrs["parameter_names"] = json.dumps(names); f.attrs["descriptor_names"] = json.dumps(desc_names)
        f.attrs["frame"] = "bundle 1 along z; bundle 2 = Rx(angle)·z; box periodic"; f.attrs["delta_Delta_ms"] = "6, 12"; f.attrs["D_um2_ms"] = 2.0
    print(f"wrote {len(S)} substrates to {out} in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()

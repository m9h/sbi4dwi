#!/usr/bin/env python3
"""
Tier C2: Permeable_MCDS (CHUV, `whitematter` branch) as the third MC engine —
intra- and extra-cellular PGSE signals on the exported CATERPillar substrates
(`validation/mcmr/<tag>_spheres.csv`, µm) on the PRISM scheme, vs the JAX
walker reference (`<tag>_ref.csv`) and MCMRSimulator (`<tag>_mcmr.csv`).

  python validation/run_mcds_parity.py --tags straight_0 straight_90 --walkers 8000 --threads 16
"""
import argparse, subprocess, time, json
from pathlib import Path
import numpy as np, pandas as pd

BIN = "/home/mhough/dev/_external/Permeable_MCDS/MC-DC_Simulator_built"
GAMMA = 2.6751525e8


def write_geometry(df, path, box_um=None, images=True):
    """SWC-like MCDS geometry. With `images`, spheres that cross a periodic face are replicated
    at ±L so walkers near the faces see the same obstacles as in the JAX/MCMR periodic setups
    (MCDS wraps walkers, not obstacles)."""
    import itertools
    rows = ["id_ax id_sph id_branch Type X Y Z Rin Rout P"]; next_id = int(df["axon"].max()) + 1
    for aid, g in df.groupby("axon", sort=True):
        for j, r in enumerate(g.itertuples()):
            rows.append(f"{aid} {j} 0 axon {r.x:.6f} {r.y:.6f} {r.z:.6f} {r.r:.6f} {r.r:.6f} 0")
    if images and box_um is not None:
        L = box_um
        for shift in itertools.product((-L, 0.0, L), repeat=3):
            if shift == (0.0, 0.0, 0.0): continue
            for aid, g in df.groupby("axon", sort=True):
                x = g["x"].values + shift[0]; y = g["y"].values + shift[1]; z = g["z"].values + shift[2]; r = g["r"].values
                keep = (x > -r) & (x < L + r) & (y > -r) & (y < L + r) & (z > -r) & (z < L + r)
                if keep.sum() == 0: continue
                for j, k in enumerate(np.where(keep)[0]):
                    rows.append(f"{next_id} {j} 0 axon {x[k]:.6f} {y[k]:.6f} {z[k]:.6f} {r[k]:.6f} {r[k]:.6f} 0")
                next_id += 1
    Path(path).write_text("\n".join(rows) + "\n")


def write_scheme(bvals_mm2, bvecs, path, delta=6e-3, Delta=12e-3, TE=20e-3):
    lines = ["VERSION: STEJSKALTANNER"]
    for b, g in zip(bvals_mm2, bvecs):
        G = np.sqrt(b * 1e6 / (GAMMA ** 2 * delta ** 2 * (Delta - delta / 3.0))) if b > 0 else 0.0
        gx, gy, gz = (g if np.linalg.norm(g) > 0 else (1.0, 0.0, 0.0))
        lines.append(f"{gx:.6f} {gy:.6f} {gz:.6f} {G:.6f} {Delta:.6f} {delta:.6f} {TE:.6f}")
    Path(path).write_text("\n".join(lines) + "\n")


def run(geom, scheme, out_prefix, box_um, init, walkers, threads, D=2.0e-9, steps=20000, seed=0):
    L = box_um * 1e-3
    conf = f"""N {walkers}
T {steps}
duration 0.020
diffusivity_intra {D}
diffusivity_extra {D}
scheme_file {scheme}
exp_prefix {out_prefix}
scale_from_stu 1
write_txt 1
write_bin 0
write_traj_file 0
num_process {threads}
ini_walkers_pos {init}
<obstacle>
<axons_list>
{geom}
permeability global 0.0
</axons_list>
</obstacle>
<voxel>
0 0 0
{L} {L} {L}
</voxel>
<END>
"""
    cpath = f"{out_prefix}_{init}.conf"; Path(cpath).write_text(conf)
    t0 = time.time(); r = subprocess.run([BIN, cpath], capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(r.stdout[-1500:] + r.stderr[-500:])
    import glob
    reals = sorted(glob.glob(f"{out_prefix}*_DWI.txt")); imags = sorted(glob.glob(f"{out_prefix}*_DWI_img.txt"))
    if not reals: raise FileNotFoundError(f"no DWI output for {out_prefix}: {r.stdout[-800:]}")
    re_ = sum(np.loadtxt(f) for f in reals); im = sum(np.loadtxt(f) for f in imags)     # per-process sums over walkers
    S = np.sqrt(re_ ** 2 + im ** 2); return S / S[0], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["straight_0", "straight_90"]); ap.add_argument("--walkers", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--out-dir", default="validation/mcmr/mcds")
    a = ap.parse_args()
    out = Path(a.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True); meta = json.load(open("validation/mcmr/meta.json"))
    summary = {}
    for tag in a.tags:
        df = pd.read_csv(f"validation/mcmr/{tag}_spheres.csv"); ref = pd.read_csv(f"validation/mcmr/{tag}_ref.csv")
        geom = str(out / f"{tag}_geometry.swc"); scheme = str(out / f"{tag}_scheme.scheme")
        write_geometry(df, geom, box_um=meta[tag]["box_um"]); write_scheme(ref.b_s_mm2.values, ref[["gx", "gy", "gz"]].values, scheme)
        res = {}
        for init in ("intra", "extra"):
            S, dt = run(geom, scheme, str(out / f"{tag}_{init}"), meta[tag]["box_um"], init, a.walkers, a.threads)
            res[init] = S; print(f"{tag} {init}: {dt:.0f}s", flush=True)
        ref["S_intra_mcds"] = res["intra"]; ref["S_extra_mcds"] = res["extra"]; ref.to_csv(out / f"{tag}_mcds.csv", index=False)
        mc = Path(f"validation/mcmr/{tag}_mcmr.csv"); mcmr = pd.read_csv(mc) if mc.exists() else None
        summary[tag] = {}
        for bv in (1000, 2000, 3000):
            m = np.isclose(ref.b_s_mm2, bv, atol=50)
            row = {"S_in": (float(ref.S_intra_mcds[m].mean()), float(ref.S_intra[m].mean())), "S_ex": (float(ref.S_extra_mcds[m].mean()), float(ref.S_extra[m].mean())),
                   "rms_in": float(np.sqrt(np.mean((ref.S_intra_mcds[m] - ref.S_intra[m]) ** 2))), "rms_ex": float(np.sqrt(np.mean((ref.S_extra_mcds[m] - ref.S_extra[m]) ** 2)))}
            if mcmr is not None:
                mm = np.isclose(mcmr.b, bv, atol=50); row["S_in_mcmr"] = float(mcmr.S_intra_mcmr[mm].mean()); row["S_ex_mcmr"] = float(mcmr.S_extra_mcmr[mm].mean())
            summary[tag][str(bv)] = row
            print(f"  b={bv}: S_intra MCDS {row['S_in'][0]:.3f} / JAX {row['S_in'][1]:.3f}" + (f" / MCMR {row['S_in_mcmr']:.3f}" if mcmr is not None else "")
                  + f"  RMS {row['rms_in']:.3f} | S_extra MCDS {row['S_ex'][0]:.3f} / JAX {row['S_ex'][1]:.3f}" + (f" / MCMR {row['S_ex_mcmr']:.3f}" if mcmr is not None else "") + f"  RMS {row['rms_ex']:.3f}", flush=True)
        ax = np.array(meta[tag]["axes"][0]); c = np.abs(ref[["gx", "gy", "gz"]].values @ ax)
        for bv in (1000, 3000):
            m = np.isclose(ref.b_s_mm2, bv, atol=50) & (c > 0.9)
            print(f"  along axon b={bv}: MCDS {ref.S_intra_mcds[m].mean():.3f}  JAX {ref.S_intra[m].mean():.3f}  free {np.exp(-bv/1000*2):.3f}", flush=True)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

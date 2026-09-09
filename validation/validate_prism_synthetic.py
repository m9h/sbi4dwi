#!/usr/bin/env python3
"""
PRISM synthetic crossing benchmark: PRISM-JAX (mse/nll), PRISM-plus
(warm start), MSMT-CSD. Paper numbers @ SNR=30: PRISM-MSE 3.5°/95%,
PRISM-NLL 2.3°/99%, MSMT-CSD 6.8°/83%, ODF-FP 11.6°/86%.
"""
import argparse, time
from dataclasses import replace
from pathlib import Path
import numpy as np

from dmipy_jax.validation.prism_jax import PrismConfig, fit_prism
from dmipy_jax.validation.prism_synthetic import (
    ANGLES, make_benchmark, msmt_peaks_on_benchmark, score_by_angle,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=float, nargs="+", default=[30])
    ap.add_argument("--methods", nargs="+", default=["msmt", "prism-mse", "prism-nll", "plus-warm"])
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--warm-noise-deg", type=float, default=15.0,
                    help="stand-in for a dictionary warm start: GT perturbed by this angle")
    args = ap.parse_args()

    rows = {}
    for snr in args.snrs:
        b = make_benchmark(snr=snr)
        for meth in args.methods:
            t0 = time.time()
            if meth == "msmt":
                dirs, fracs = msmt_peaks_on_benchmark(b)
            else:
                cfg = PrismConfig(n_fibres=2, n_iter=args.n_iter,
                                  loss="mse" if meth == "prism-mse" else "nll")
                init = None
                if meth == "plus-warm":
                    rng = np.random.default_rng(2)
                    sd = np.tan(np.radians(args.warm_noise_deg))
                    init = b["gt_dirs"] + rng.normal(0, sd, b["gt_dirs"].shape)
                    init[b["angle"] == 0, 1] = rng.normal(size=3)
                fit = fit_prism(b["data"], b["mask"], b["bvals"], b["bvecs"], cfg, init_dirs=init)
                dirs, fracs = fit.dirs, fit.wm_fracs
            sc = score_by_angle(dirs, fracs, b)
            rows[(meth, snr)] = sc
            e, r = sc["overall"]
            print(f"{meth:10s} SNR={snr:>4.0f}  overall err={e:5.2f}°  recall={100*r:5.1f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print("\nper-angle mean error (°) / recall (%):")
    print("angle  " + "".join(f"{m:>18s}" for (m, s) in rows))
    for a in list(ANGLES) + [0]:
        line = f"{a:>5d}  "
        for key in rows:
            e, r = rows[key][a]
            line += f"{e:8.2f} /{100*r:6.1f}"
        print(line)
    out = Path("validation/prism_synthetic_results.npz")
    np.savez(out, **{f"{m}_snr{int(s)}_{a}": np.array(rows[(m, s)][a])
                     for (m, s) in rows for a in list(ANGLES) + [0, "overall"]})
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

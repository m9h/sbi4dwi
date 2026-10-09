#!/usr/bin/env python3
"""
2-fibre crossing-angle benchmark across an SNR sweep — four tools:

  - dmipy-JAX dictionary (200K-entry 2-stick library)
  - dipy upstream FORCEModel (500K library, n_neighbors=50, force_peaks)
  - DIPY CSD peaks
  - DIPY GQI peaks

Mirrors validate_force_replication_v2.py's 17-angle sweep (10-90 deg) at
each of SNR ∈ {10, 20, 30, 50}. Drops the hybrid/LM dmipy-JAX variants
since this experiment isolates the noise-robustness question, not the
hybrid-LM characterisation. The SNR=30 column should reproduce v2's
dict / dipy_force / csd / gqi numbers within sampling noise.

Cost: ~2-3 hours on GB10. Use ``--smoke`` for a 1-SNR x 2-angle x 10-trial
end-to-end check (~30 s).
"""

import argparse
import os
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dipy.data import default_sphere
from dipy.reconst.force import FORCEModel, load_force_simulations

from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.matcher import DictionaryMatcher
from dmipy_jax.library.storage import SimulationLibrary
from validation.lib.force_baselines import (
    csd_peaks_from_signal,
    dipy_force_peaks_from_signal,
    gqi_peaks_from_signal,
)
from validation.lib.force_helpers import (
    best_two_peaks,
    check_both_detected,
    params_to_orientations,
)
from validation.lib.two_fiber import (
    acq_to_gtab_si,
    add_rician_noise,
    build_two_stick_simulator,
    make_multishell_acquisition,
    two_stick_signal,
)


DIPY_FORCE_CACHE = Path(os.environ.get(
    "DIPY_FORCE_CACHE",
    str(Path.home() / ".cache" / "dipy_force" / "force_v2_500k.npz"),
))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--lib-size", type=int, default=200_000)
    ap.add_argument("--snrs", type=float, nargs="+",
                    default=[10.0, 20.0, 30.0, 50.0])
    args = ap.parse_args()

    print("=" * 78)
    print("FORCE SNR sweep: 4 tools × angle × SNR")
    print("=" * 78)

    acq = make_multishell_acquisition()
    sim = build_two_stick_simulator(acq)

    print(f"\nGenerating dmipy-JAX 2-stick library ({args.lib_size:,} entries)...")
    gen = LibraryGenerator(sim, chunk_size=50_000)
    lib_params, lib_signals = gen.generate(args.lib_size, key=jax.random.PRNGKey(1))
    jax.block_until_ready(lib_signals)
    lib = SimulationLibrary(
        params=lib_params, signals=lib_signals,
        parameter_names=sim.parameter_names,
    )
    matcher = DictionaryMatcher(lib, k_best=10)
    print(f"Library: {lib.n_entries:,} entries, signal dim {lib.signal_dim}")

    if not DIPY_FORCE_CACHE.exists():
        raise SystemExit(
            f"DIPY FORCE library cache not found at {DIPY_FORCE_CACHE}. "
            "Run validate_force_replication_v2.py first to build it."
        )
    dipy_sims = load_force_simulations(str(DIPY_FORCE_CACHE))
    dipy_force = FORCEModel(gtab=acq_to_gtab_si(acq), simulations=dipy_sims, n_neighbors=50)
    gtab = acq_to_gtab_si(acq)
    response = (np.array([1.7e-3, 3e-4, 3e-4]), 1.0)
    sphere = default_sphere
    print("DIPY FORCEModel + CSD/GQI baselines ready.")

    if args.smoke:
        snrs = [args.snrs[len(args.snrs) // 2]]
        crossing_angles = [20, 60]
        n_trials = 10
    else:
        snrs = list(args.snrs)
        crossing_angles = list(np.arange(10, 95, 5))
        n_trials = args.n_trials

    method_names = ["dict", "dipy_force", "csd", "gqi"]
    results = {
        m: np.zeros((len(snrs), len(crossing_angles))) for m in method_names
    }

    angle_threshold = 15.0
    f1_true = 0.45
    f_iso_true = 0.05
    key = jax.random.PRNGKey(42)

    for s_idx, snr in enumerate(snrs):
        print(f"\n--- SNR = {snr:g} ---")
        for a_idx, angle_deg in enumerate(crossing_angles):
            rad = np.radians(angle_deg)
            mu1_true = np.array([0.0, 0.0, 1.0])
            mu2_true = np.array([np.sin(rad), 0.0, np.cos(rad)])
            counts = {m: 0 for m in method_names}

            for _ in range(n_trials):
                key, k_noise = jax.random.split(key, 2)
                signal = two_stick_signal(
                    acq, jnp.asarray(mu1_true), jnp.asarray(mu2_true),
                    f1=f1_true, f_iso=f_iso_true,
                )
                noisy = add_rician_noise(signal, snr=float(snr), key=k_noise)
                noisy_np = np.asarray(noisy)

                # dmipy-JAX dictionary
                d_params, _ = matcher.match_single(noisy)
                mu1_d, mu2_d = params_to_orientations(np.asarray(d_params))
                if check_both_detected(
                    mu1_true, mu2_true, mu1_d, mu2_d, angle_threshold,
                ):
                    counts["dict"] += 1

                # dipy upstream FORCE
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pks = dipy_force_peaks_from_signal(noisy_np, dipy_force)
                    pair = best_two_peaks(pks, mu1_true, mu2_true)
                    if pair is not None and check_both_detected(
                        mu1_true, mu2_true, pair[0], pair[1], angle_threshold,
                    ):
                        counts["dipy_force"] += 1
                except Exception:
                    pass

                # CSD
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pks = csd_peaks_from_signal(
                            noisy_np, gtab, response, sphere,
                            relative_peak_threshold=0.5,
                            min_separation_angle=10.5,
                        )
                    pair = best_two_peaks(pks, mu1_true, mu2_true)
                    if pair is not None and check_both_detected(
                        mu1_true, mu2_true, pair[0], pair[1], angle_threshold,
                    ):
                        counts["csd"] += 1
                except Exception:
                    pass

                # GQI
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pks = gqi_peaks_from_signal(
                            noisy_np, gtab, sphere,
                            relative_peak_threshold=0.5,
                            min_separation_angle=10.5,
                        )
                    pair = best_two_peaks(pks, mu1_true, mu2_true)
                    if pair is not None and check_both_detected(
                        mu1_true, mu2_true, pair[0], pair[1], angle_threshold,
                    ):
                        counts["gqi"] += 1
                except Exception:
                    pass

            for m in method_names:
                results[m][s_idx, a_idx] = counts[m] / n_trials

            line = f"  {angle_deg:3d}°:" + "".join(
                f"  {m}={results[m][s_idx, a_idx]:.0%}" for m in method_names
            )
            print(line)

    print("\n" + "=" * 78)
    print("Detection rate by SNR × crossing angle")
    print("=" * 78)
    for m in method_names:
        print(f"\n{m}:")
        header = "       " + "  ".join(f"{a:>5d}°" for a in crossing_angles)
        print(header)
        for s_idx, snr in enumerate(snrs):
            row = f"  SNR={int(snr):3d}: " + "  ".join(
                f"{results[m][s_idx, a_idx] * 100:>5.1f}%"
                for a_idx in range(len(crossing_angles))
            )
            print(row)

    if not args.smoke:
        out = Path("validation/force_snr_sweep_results.npz")
        np.savez(
            out,
            snrs=np.asarray(snrs),
            angles=np.asarray(crossing_angles),
            **{m: results[m] for m in method_names},
        )
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Companion to validate_force_replication_v2.py: 3-fiber crossing benchmark.

dipy's FORCE library is ~70% 3-fiber configurations (Dirichlet(2,1,1)
prior over fibre fractions); 2-fiber benchmarks understate its capability.
This script tests whether FORCE *excels* in the 3-fiber regime its library
was designed for, and how a freshly-built dmipy-JAX 3-stick library
compares.

Setup
-----
- Synthetic: 3 sticks coplanar in +x/+z, equal fractions (0.317 each),
  isotropic FW = 0.05, d_par = 1.7e-9, d_iso = 3e-9.
- Geometry sweep: vary the *minimum inter-fibre angle*. The three sticks
  are placed at theta = -alpha, 0, +alpha relative to z (alpha controls
  spread). At alpha=15 deg the three sticks span -15 to +15 (min sep 15).
  At alpha=60 deg they span -60 to +60 (min sep 60).
- Acquisition: same 90-direction 2-shell as v2.
- SNR 30, Rician noise, 200 trials per geometry.
- Detection criterion: all 3 peaks within 15 deg of their assigned truth.

Methods
-------
- dmipy-JAX 3-stick dictionary (fresh 200K-entry library)
- DIPY upstream FORCEModel (reuses 500K cached library from v2)
- DIPY FORCE-internal (label-based, bypasses force_peaks)
- DIPY CSD peaks
- DIPY GQI peaks

Cost: ~30-40 sec library build + ~2-2.5 h sweep on GB10. Use
``--smoke`` to do a single 10-trial check at one alpha.
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
    dipy_force_label_directions_from_signal,
    dipy_force_peaks_from_signal,
    gqi_peaks_from_signal,
)
from validation.lib.force_helpers import (
    best_three_peaks,
    check_all_three_detected,
    params3_to_orientations,
)
from validation.lib.three_fiber import (
    acq_to_gtab_si,
    build_three_stick_simulator,
    make_multishell_acquisition,
    three_stick_signal,
)


DIPY_FORCE_CACHE = Path(os.environ.get(
    "DIPY_FORCE_CACHE",
    str(Path.home() / ".cache" / "dipy_force" / "force_v2_500k.npz"),
))


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Run a single alpha=30 deg, 10-trial smoke test")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--lib-size", type=int, default=200_000,
                    help="dmipy-JAX 3-stick library size")
    args = ap.parse_args()

    print("=" * 78)
    print("FORCE 3-fiber benchmark: dipy FORCE vs dmipy-JAX 3-stick dict + ODF baselines")
    print("=" * 78)

    acq = make_multishell_acquisition()
    sim = build_three_stick_simulator(acq)

    print(f"\nGenerating dmipy-JAX 3-stick library ({args.lib_size:,} entries)...")
    gen = LibraryGenerator(sim, chunk_size=50_000)
    lib_params, lib_signals = gen.generate(args.lib_size, key=jax.random.PRNGKey(1))
    jax.block_until_ready(lib_signals)
    lib = SimulationLibrary(
        params=lib_params, signals=lib_signals,
        parameter_names=sim.parameter_names,
    )
    matcher = DictionaryMatcher(lib, k_best=10)
    print(f"Library: {lib.n_entries:,} entries, signal dim {lib.signal_dim}")

    gtab = acq_to_gtab_si(acq)
    response = (np.array([1.7e-3, 3e-4, 3e-4]), 1.0)
    sphere = default_sphere

    if not DIPY_FORCE_CACHE.exists():
        raise SystemExit(
            f"DIPY FORCE library cache not found at {DIPY_FORCE_CACHE}. "
            "Run validate_force_replication_v2.py first to build it."
        )
    dipy_sims = load_force_simulations(str(DIPY_FORCE_CACHE))
    dipy_force = FORCEModel(gtab, simulations=dipy_sims, n_neighbors=50)
    print("DIPY FORCEModel ready (500K sims, n_neighbors=50).")

    # Geometry: alpha controls fiber spread. Three sticks at theta = -alpha, 0, +alpha.
    if args.smoke:
        alphas = [30]
        n_trials = 10
    else:
        alphas = list(range(15, 65, 5))   # 15..60 deg fiber spread
        n_trials = args.n_trials

    method_names = ["dict3", "dipy_force", "dipy_force_internal", "csd", "gqi"]
    results = {m: [] for m in method_names}
    results["alpha"] = []
    snr = 30.0
    angle_threshold = 15.0
    f1_true = 0.317
    f2_true = 0.317
    f_iso_true = 0.05

    key = jax.random.PRNGKey(42)

    for alpha in alphas:
        rad = np.radians(alpha)
        mu1_true = np.array([-np.sin(rad), 0.0, np.cos(rad)])
        mu2_true = np.array([0.0, 0.0, 1.0])
        mu3_true = np.array([np.sin(rad), 0.0, np.cos(rad)])
        counts = {m: 0 for m in method_names}

        for trial in range(n_trials):
            key, k_noise = jax.random.split(key, 2)
            signal = three_stick_signal(
                acq,
                jnp.asarray(mu1_true), jnp.asarray(mu2_true), jnp.asarray(mu3_true),
                f1_true, f2_true, f_iso_true,
            )
            sigma = 1.0 / snr
            k1, k2 = jax.random.split(k_noise)
            n1 = jax.random.normal(k1, signal.shape) * sigma
            n2 = jax.random.normal(k2, signal.shape) * sigma
            noisy = jnp.sqrt((signal + n1) ** 2 + n2 ** 2)
            noisy_np = np.asarray(noisy)

            # dmipy-JAX 3-stick dictionary
            dict3_params, _ = matcher.match_single(noisy)
            mu1_d, mu2_d, mu3_d = params3_to_orientations(np.asarray(dict3_params))
            if check_all_three_detected(
                mu1_true, mu2_true, mu3_true, mu1_d, mu2_d, mu3_d, angle_threshold,
            ):
                counts["dict3"] += 1

            # dipy upstream FORCE
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pks = dipy_force_peaks_from_signal(noisy_np, dipy_force)
                triple = best_three_peaks(pks, mu1_true, mu2_true, mu3_true)
                if triple is not None and check_all_three_detected(
                    mu1_true, mu2_true, mu3_true,
                    triple[0], triple[1], triple[2], angle_threshold,
                ):
                    counts["dipy_force"] += 1
            except Exception:
                pass

            # dipy FORCE-internal (read FORCEFit.label directly)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pks = dipy_force_label_directions_from_signal(
                        noisy_np, dipy_force, sphere)
                triple = best_three_peaks(pks, mu1_true, mu2_true, mu3_true)
                if triple is not None and check_all_three_detected(
                    mu1_true, mu2_true, mu3_true,
                    triple[0], triple[1], triple[2], angle_threshold,
                ):
                    counts["dipy_force_internal"] += 1
            except Exception:
                pass

            # CSD peaks
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pks = csd_peaks_from_signal(
                        noisy_np, gtab, response, sphere,
                        relative_peak_threshold=0.3, min_separation_angle=10.0)
                triple = best_three_peaks(pks, mu1_true, mu2_true, mu3_true)
                if triple is not None and check_all_three_detected(
                    mu1_true, mu2_true, mu3_true,
                    triple[0], triple[1], triple[2], angle_threshold,
                ):
                    counts["csd"] += 1
            except Exception:
                pass

            # GQI peaks
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pks = gqi_peaks_from_signal(
                        noisy_np, gtab, sphere,
                        relative_peak_threshold=0.3, min_separation_angle=10.0)
                triple = best_three_peaks(pks, mu1_true, mu2_true, mu3_true)
                if triple is not None and check_all_three_detected(
                    mu1_true, mu2_true, mu3_true,
                    triple[0], triple[1], triple[2], angle_threshold,
                ):
                    counts["gqi"] += 1
            except Exception:
                pass

        results["alpha"].append(alpha)
        for m in method_names:
            results[m].append(counts[m] / n_trials)

        line = f"  alpha={alpha:3d} deg:" + "".join(
            f"  {m}={results[m][-1]:.0%}" for m in method_names
        )
        print(line)

    print("\n" + "=" * 78)
    print("Summary: all-3-fibres detection rate by fiber spread (alpha = half-spread)")
    print("=" * 78)
    header = f"{'alpha':>5s}  " + "  ".join(f"{m:>16s}" for m in method_names)
    print(header)
    for i, a in enumerate(results["alpha"]):
        row = f"{a:>5d}  " + "  ".join(
            f"{results[m][i]:>15.1%}" for m in method_names
        )
        print(row)

    out = Path("validation/force_3fiber_results.npz")
    if not args.smoke:
        np.savez(
            out,
            alpha=np.asarray(results["alpha"]),
            **{m: np.asarray(results[m]) for m in method_names},
        )
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FORCE-paper-matched re-run of the 2-fibre crossing benchmark (§13 of doc 004).

Differences from validate_force_replication_v2.py / validate_force_snr_sweep.py
that bring the test inside FORCE's published design envelope:

  - **Acquisition**: 150 dirs × b=2000 single-shell (Stanford HARDI), instead
    of 90 dirs × {b=1000, b=2000} 2-shell.
  - **Synthetics with dispersion**: each fibre is Bingham-dispersed (axisym)
    with ODI ∈ Uniform(0.01, 0.30), matching the range FORCE Table 1 uses
    to generate its library.
  - **Tolerance**: 20° detection threshold (paper) instead of 15°.
  - **Min peak separation**: 10° (paper) instead of 10.5°.
  - **dmipy-JAX dict library** is also dispersion-aware: 6-param simulator
    [d_par, θ1, θ2, ODI, f1, f_iso] with ODI sampled in the same band as
    FORCE's, so neither library has a structural advantage.

Methods compared: dict, dipy_force, csd, gqi.

Cost: ~2-3 hours on GB10 (dipy FORCE library regeneration takes ~14 min
single-CPU; sweep itself ~1.5h). Use ``--smoke`` for a 1-SNR × 2-angle
× 10-trial check (~2-5 min plus library build).
"""

import argparse
import os
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dipy.data import default_sphere
from dipy.reconst.force import (
    FORCEModel,
    generate_force_simulations,
    load_force_simulations,
    save_force_simulations,
)

from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.matcher import DictionaryMatcher
from dmipy_jax.library.storage import SimulationLibrary
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.signal_models.bingham import BinghamNODDI
from validation.lib.force_baselines import (
    csd_peaks_from_signal,
    dipy_force_peaks_from_signal,
    gqi_peaks_from_signal,
)
from validation.lib.force_helpers import best_two_peaks, check_both_detected
from validation.lib.force_matched import (
    dispersed_two_stick_signal,
    make_stanford_hardi_acquisition,
    odi_to_kappa,
)
from validation.lib.three_fiber import acq_to_gtab_si


DIPY_FORCE_CACHE = Path(os.environ.get(
    "DIPY_FORCE_MATCHED_CACHE",
    str(Path.home() / ".cache" / "dipy_force" / "force_matched_500k.npz"),
))


# --------------------------------------------------------------------------- #
# Dispersion-aware dmipy-JAX 2-stick simulator (matches FORCE library prior)
# --------------------------------------------------------------------------- #

def build_dispersed_two_stick_simulator(acq):
    """6-param simulator: ``[d_par, theta1, theta2, odi, f1, f_iso]``."""
    bingham = BinghamNODDI(grid_points=120)

    def forward_fn(params, acq):
        d_par = params[0]
        t1, t2 = params[1], params[2]
        odi = params[3]
        f1, f_iso = params[4], params[5]
        f2 = 1.0 - f1 - f_iso
        # ODI -> kappa (axisymmetric: kappa1 = kappa2)
        kappa = 1.0 / jnp.tan(jnp.pi * odi / 2.0)
        mu1 = jnp.array([jnp.sin(t1), 0.0, jnp.cos(t1)])
        mu2 = jnp.array([jnp.sin(t2), 0.0, jnp.cos(t2)])
        s1 = bingham(acq.bvalues, acq.gradient_directions,
                     mu=mu1, kappa1=kappa, kappa2=kappa, lambda_par=d_par)
        s2 = bingham(acq.bvalues, acq.gradient_directions,
                     mu=mu2, kappa1=kappa, kappa2=kappa, lambda_par=d_par)
        s_iso = jnp.exp(-acq.bvalues * 3.0e-9)
        return f1 * s1 + f2 * s2 + f_iso * s_iso

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_par", "theta1", "theta2", "odi", "f1", "f_iso"],
        parameter_ranges={
            "d_par": (1.0e-9, 2.5e-9),
            "theta1": (0.0, float(jnp.pi)),
            "theta2": (0.0, float(jnp.pi)),
            "odi": (0.01, 0.30),
            "f1": (0.1, 0.8),
            "f_iso": (0.0, 0.2),
        },
        acquisition=acq,
    )


def get_or_make_dipy_force_library(gtab, n_sims=500_000, num_cpus=1):
    DIPY_FORCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if DIPY_FORCE_CACHE.exists():
        print(f"Loading cached DIPY FORCE library from {DIPY_FORCE_CACHE}")
        return load_force_simulations(str(DIPY_FORCE_CACHE))
    print(f"Generating DIPY FORCE library (n={n_sims:,}, num_cpus={num_cpus}); "
          f"~14 min single-CPU...")
    sims = generate_force_simulations(
        gtab, num_simulations=n_sims, num_cpus=num_cpus, verbose=True,
    )
    save_force_simulations(sims, str(DIPY_FORCE_CACHE))
    print(f"Saved to {DIPY_FORCE_CACHE}")
    return sims


def add_rician_noise(signal, snr, key):
    """Inline copy from two_fiber.add_rician_noise (avoids cross-import)."""
    sigma = 1.0 / snr
    k1, k2 = jax.random.split(key)
    n1 = jax.random.normal(k1, signal.shape) * sigma
    n2 = jax.random.normal(k2, signal.shape) * sigma
    return jnp.sqrt((signal + n1) ** 2 + n2 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--lib-size", type=int, default=100_000,
                    help="dmipy-JAX dispersed library size")
    ap.add_argument("--snrs", type=float, nargs="+", default=[10.0, 20.0, 50.0])
    args = ap.parse_args()

    print("=" * 78)
    print("FORCE-paper-matched re-run: 4 tools × 3 SNR × 17 angles")
    print("Stanford HARDI acquisition, Bingham-dispersed sticks (ODI∈[0.01,0.30])")
    print("=" * 78)

    acq = make_stanford_hardi_acquisition()
    bvals_si = np.asarray(acq.bvalues)
    print(f"Acquisition: {bvals_si.shape[0]} dirs, "
          f"{(bvals_si == 0).sum()} b=0, {(np.abs(bvals_si - 2e9) < 5e7).sum()} b=2000")

    sim = build_dispersed_two_stick_simulator(acq)
    print(f"\nGenerating dispersed dmipy-JAX library "
          f"({args.lib_size:,} entries, Bingham-integrated)...")
    gen = LibraryGenerator(sim, chunk_size=10_000)
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

    dipy_sims = get_or_make_dipy_force_library(gtab, n_sims=500_000, num_cpus=1)
    dipy_force = FORCEModel(gtab=gtab, simulations=dipy_sims, n_neighbors=50)
    print("DIPY FORCEModel ready.")

    if args.smoke:
        snrs = [args.snrs[len(args.snrs) // 2]]
        crossing_angles = [20, 60]
        n_trials = 10
    else:
        snrs = list(args.snrs)
        crossing_angles = list(np.arange(10, 95, 5))
        n_trials = args.n_trials

    method_names = ["dict", "dipy_force", "csd", "gqi"]
    results = {m: np.zeros((len(snrs), len(crossing_angles))) for m in method_names}
    angle_threshold = 20.0  # FORCE paper convention
    min_separation = 10.0
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
                key, k_odi, k_noise = jax.random.split(key, 3)
                # Sample dispersion in the FORCE library's range
                odi_true = float(jax.random.uniform(
                    k_odi, shape=(), minval=0.01, maxval=0.30,
                ))
                kappa = odi_to_kappa(odi_true)

                signal = dispersed_two_stick_signal(
                    acq, jnp.asarray(mu1_true), jnp.asarray(mu2_true),
                    kappa1=kappa, kappa2=kappa, f1=f1_true, f_iso=f_iso_true,
                )
                noisy = add_rician_noise(signal, snr=float(snr), key=k_noise)
                noisy_np = np.asarray(noisy)

                # dmipy-JAX dict (dispersion-aware library)
                d_params, _ = matcher.match_single(noisy)
                p = np.asarray(d_params)
                t1, t2 = float(p[1]), float(p[2])
                mu1_d = np.array([np.sin(t1), 0.0, np.cos(t1)])
                mu2_d = np.array([np.sin(t2), 0.0, np.cos(t2)])
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
                            min_separation_angle=min_separation,
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
                            min_separation_angle=min_separation,
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
    print("Detection rate (20° tolerance) by SNR × crossing angle")
    print("=" * 78)
    for m in method_names:
        print(f"\n{m}:")
        header = "       " + "  ".join(f"{a:>5d}°" for a in crossing_angles)
        print(header)
        for s_idx, snr in enumerate(snrs):
            row = f"  S={int(snr):3d}: " + "  ".join(
                f"{results[m][s_idx, a_idx] * 100:>5.1f}%"
                for a_idx in range(len(crossing_angles))
            )
            print(row)

    if not args.smoke:
        out = Path("validation/force_matched_results.npz")
        np.savez(
            out,
            snrs=np.asarray(snrs),
            angles=np.asarray(crossing_angles),
            **{m: results[m] for m in method_names},
        )
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

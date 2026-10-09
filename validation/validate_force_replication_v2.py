#!/usr/bin/env python3
"""
FORCE replication v2: extends v1 with five additional methods

  - Hybrid (MSE-guarded): only accept LM if MSE improves on dict
  - Hybrid (maxiter=15): tighter LM polish
  - DIPY FORCE: dipy.reconst.force.FORCEModel (the official upstream)
  - DIPY CSD: peaks from constrained spherical deconvolution
  - DIPY GQI: peaks from generalized q-sampling imaging

Same crossing-angle sweep (10-90 deg, 200 trials/angle, SNR 30) so results
are directly comparable to validate_force_replication.py.

Requires dipy>=1.12.1 (which introduced dipy.reconst.force).

Usage::

    python validation/validate_force_replication_v2.py
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import warnings

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.storage import SimulationLibrary
from dmipy_jax.library.matcher import DictionaryMatcher
from dmipy_jax.fitting.optimization import VoxelFitter

# DIPY imports
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.force import (
    FORCEModel, force_peaks,
    generate_force_simulations, save_force_simulations, load_force_simulations,
)
from dipy.direction import peaks_from_model
from dipy.data import default_sphere

DIPY_FORCE_CACHE = Path(os.environ.get(
    "DIPY_FORCE_CACHE",
    str(Path.home() / ".cache" / "dipy_force" / "force_v2_500k.npz"),
))


# --------------------------------------------------------------------------- #
# Acquisition + forward model — copied from v1 for direct comparability
# --------------------------------------------------------------------------- #

def make_multishell_acquisition():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    def rand_vecs(k, n):
        z = jax.random.normal(k, (n, 3))
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    v0 = jnp.array([[1.0, 0.0, 0.0]] * 2)
    v1 = rand_vecs(k1, 32)
    v2 = rand_vecs(k2, 56)
    bvals = jnp.concatenate([jnp.zeros(2), jnp.full(32, 1e9), jnp.full(56, 2e9)])
    bvecs = jnp.concatenate([v0, v1, v2], axis=0)
    return JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)


def two_stick_signal(acq, mu1, mu2, f1, d_par=1.7e-9, d_iso=3.0e-9, f_iso=0.05):
    f2 = 1.0 - f1 - f_iso
    cos1 = acq.gradient_directions @ mu1
    cos2 = acq.gradient_directions @ mu2
    s1 = jnp.exp(-acq.bvalues * d_par * cos1 ** 2)
    s2 = jnp.exp(-acq.bvalues * d_par * cos2 ** 2)
    s_iso = jnp.exp(-acq.bvalues * d_iso)
    return f1 * s1 + f2 * s2 + f_iso * s_iso


def build_library_simulator(acq):
    def forward_fn(params, acq):
        d_par = params[0]
        theta1 = params[1]
        theta2 = params[2]
        f1 = params[3]
        f_iso = params[4]
        f2 = 1.0 - f1 - f_iso
        mu1 = jnp.array([jnp.sin(theta1), 0.0, jnp.cos(theta1)])
        mu2 = jnp.array([jnp.sin(theta2), 0.0, jnp.cos(theta2)])
        cos1 = acq.gradient_directions @ mu1
        cos2 = acq.gradient_directions @ mu2
        s1 = jnp.exp(-acq.bvalues * d_par * cos1 ** 2)
        s2 = jnp.exp(-acq.bvalues * d_par * cos2 ** 2)
        s_iso = jnp.exp(-acq.bvalues * 3.0e-9)
        return f1 * s1 + f2 * s2 + f_iso * s_iso

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_par", "theta1", "theta2", "f1", "f_iso"],
        parameter_ranges={
            "d_par": (1.0e-9, 2.5e-9),
            "theta1": (0.0, float(jnp.pi)),
            "theta2": (0.0, float(jnp.pi)),
            "f1": (0.1, 0.8),
            "f_iso": (0.0, 0.2),
        },
        acquisition=acq,
    )


# Geometry helpers + acquisition adapter live in validation.lib.force_helpers
from validation.lib.force_helpers import (
    angular_error,
    params_to_orientations,
    check_both_detected,
    best_two_peaks,
    acq_to_gtab,
)


def csd_peaks(noisy_signal_np, gtab, response, sphere, ang_threshold):
    """Return up to two peak directions from CSD on a single-voxel signal."""
    data4d = noisy_signal_np[None, None, None, :]
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=8)
    peaks = peaks_from_model(
        model=csd_model,
        data=data4d,
        sphere=sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=max(10.0, ang_threshold * 0.7),
        return_odf=False,
        normalize_peaks=True,
        npeaks=3,
    )
    return [peaks.peak_dirs[0, 0, 0, k] for k in range(peaks.peak_dirs.shape[-2])]


def gqi_peaks(noisy_signal_np, gtab, sphere, ang_threshold):
    data4d = noisy_signal_np[None, None, None, :]
    gqi_model = GeneralizedQSamplingModel(gtab, sampling_length=1.2)
    peaks = peaks_from_model(
        model=gqi_model,
        data=data4d,
        sphere=sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=max(10.0, ang_threshold * 0.7),
        return_odf=False,
        normalize_peaks=True,
        npeaks=3,
    )
    return [peaks.peak_dirs[0, 0, 0, k] for k in range(peaks.peak_dirs.shape[-2])]


def get_or_make_dipy_force_simulations(gtab, n_sims=100_000, num_cpus=1):
    """Cache dipy.reconst.force simulations to ~/.cache/dipy_force/.

    Generated single-process to avoid os.fork() warnings under JAX.
    """
    DIPY_FORCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if DIPY_FORCE_CACHE.exists():
        print(f"Loading cached DIPY FORCE simulations from {DIPY_FORCE_CACHE}")
        return load_force_simulations(str(DIPY_FORCE_CACHE))
    print(f"Generating DIPY FORCE simulations (n={n_sims:,}, num_cpus={num_cpus})...")
    sims = generate_force_simulations(
        gtab, num_simulations=n_sims, num_cpus=num_cpus, verbose=True,
    )
    save_force_simulations(sims, str(DIPY_FORCE_CACHE))
    return sims


def dipy_force_peaks(noisy_signal_np, force_model, ang_threshold):
    """dipy's standard FORCE pipeline: matcher -> force_peaks (SH on default_sphere)."""
    data4d = noisy_signal_np[None, None, None, :].astype(np.float32)
    fit = force_model.fit(data4d)
    peaks = force_peaks(fit)
    return [peaks.peak_dirs[0, 0, 0, k] for k in range(peaks.peak_dirs.shape[-2])]


def dipy_force_internal_peaks(noisy_signal_np, force_model, sphere):
    """Bypass force_peaks: read FORCEFit.label directly off the matched sphere.

    FORCEFit.label is shape (n_sphere_vertices,) with nonzero entries at
    fiber-direction indices the matcher selected. Returns those vertex
    directions, ranked by FORCEFit.fracs.
    """
    data4d = noisy_signal_np[None, None, None, :].astype(np.float32)
    mvfit = force_model.fit(data4d)
    fit = mvfit[0, 0, 0]
    label = np.asarray(fit.label)
    nz_idx = np.nonzero(label)[0]
    if nz_idx.size == 0:
        return []
    dirs = sphere.vertices[nz_idx]
    # If matcher is reporting >2 active labels, sort by FORCEFit.fracs so the
    # most-weighted directions come first.
    fracs = np.asarray(fit.fracs)
    n_take = min(len(nz_idx), len(fracs))
    if n_take and fracs[:n_take].sum() > 0:
        order = np.argsort(-fracs[:n_take])
        dirs = dirs[order[:n_take]]
    return [dirs[i] for i in range(len(dirs))]


# --------------------------------------------------------------------------- #
# LM fitter
# --------------------------------------------------------------------------- #

def build_lm_fitter(forward_fn, maxiter=50, tol=1e-6):
    param_ranges = [
        (1.0e-9, 2.5e-9), (0.0, float(jnp.pi)), (0.0, float(jnp.pi)),
        (0.1, 0.8), (0.0, 0.2),
    ]
    scales = jnp.array([1e-9, 1.0, 1.0, 1.0, 1.0])
    return VoxelFitter(forward_fn, param_ranges, scales=scales,
                       solver_settings={"maxiter": maxiter, "tol": tol})


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #

def main():
    print("=" * 78)
    print("FORCE Replication v2: dict / hybrid (3 variants) / pure LM / CSD / GQI")
    print("=" * 78)

    acq = make_multishell_acquisition()
    sim = build_library_simulator(acq)
    forward_fn = sim.forward_fn

    print("\nGenerating library (200K entries)...")
    gen = LibraryGenerator(sim, chunk_size=50_000)
    lib_params, lib_signals = gen.generate(200_000, key=jax.random.PRNGKey(1))
    jax.block_until_ready(lib_signals)
    lib = SimulationLibrary(
        params=lib_params, signals=lib_signals,
        parameter_names=sim.parameter_names,
    )
    matcher = DictionaryMatcher(lib, k_best=10)
    print(f"Library: {lib.n_entries:,} entries, signal dim {lib.signal_dim}")

    fitter50 = build_lm_fitter(forward_fn, maxiter=50)
    fitter15 = build_lm_fitter(forward_fn, maxiter=15)

    @jax.jit
    def lm50(data, init):
        return fitter50.fit(data, acq, init)

    @jax.jit
    def lm15(data, init):
        return fitter15.fit(data, acq, init)

    @jax.jit
    def predict(params):
        return forward_fn(params, acq)

    @jax.jit
    def mse(signal, params):
        return jnp.mean((signal - forward_fn(params, acq)) ** 2)

    # DIPY setup (single response, fixed)
    gtab = acq_to_gtab(acq)
    # Synthetic single-fiber response: typical WM tensor eigenvalues, S0=1.0
    response = (np.array([1.7e-3, 3e-4, 3e-4]), 1.0)
    sphere = default_sphere

    # DIPY upstream FORCE — paper-default settings (500K library, n_neighbors=50)
    dipy_sims = get_or_make_dipy_force_simulations(gtab, n_sims=500_000, num_cpus=1)
    dipy_force = FORCEModel(gtab, simulations=dipy_sims, n_neighbors=50)
    print("DIPY FORCEModel ready (500K sims, n_neighbors=50).")

    crossing_angles = np.arange(10, 95, 5)
    n_trials = 200
    snr = 30.0
    angle_threshold = 15.0

    method_names = [
        "dict", "hybrid50", "hybrid15", "hybrid_guard", "lm",
        "dipy_force", "dipy_force_internal", "csd", "gqi",
    ]
    results = {m: [] for m in method_names}
    results["angle"] = []

    key = jax.random.PRNGKey(42)

    for angle_deg in crossing_angles:
        angle_rad = np.radians(angle_deg)
        counts = {m: 0 for m in method_names}

        for trial in range(n_trials):
            key, k_noise, k_init = jax.random.split(key, 3)

            mu1_true = np.array([0.0, 0.0, 1.0])
            mu2_true = np.array([np.sin(angle_rad), 0.0, np.cos(angle_rad)])
            f1 = 0.45

            signal = two_stick_signal(
                acq, jnp.array(mu1_true), jnp.array(mu2_true), f1,
            )
            sigma = 1.0 / snr
            k1, k2 = jax.random.split(k_noise)
            n1 = jax.random.normal(k1, signal.shape) * sigma
            n2 = jax.random.normal(k2, signal.shape) * sigma
            noisy = jnp.sqrt((signal + n1) ** 2 + n2 ** 2)
            noisy_np = np.asarray(noisy)

            # Dict
            dict_params, _ = matcher.match_single(noisy)
            mu1_d, mu2_d = params_to_orientations(np.asarray(dict_params))
            if check_both_detected(mu1_true, mu2_true, mu1_d, mu2_d, angle_threshold):
                counts["dict"] += 1

            # Hybrid maxiter=50 (the v1 default)
            try:
                h50_params, _ = lm50(noisy, dict_params)
                mu1, mu2 = params_to_orientations(np.asarray(h50_params))
                if check_both_detected(mu1_true, mu2_true, mu1, mu2, angle_threshold):
                    counts["hybrid50"] += 1
            except Exception:
                pass

            # Hybrid maxiter=15 (tighter polish)
            try:
                h15_params, _ = lm15(noisy, dict_params)
                mu1, mu2 = params_to_orientations(np.asarray(h15_params))
                if check_both_detected(mu1_true, mu2_true, mu1, mu2, angle_threshold):
                    counts["hybrid15"] += 1
            except Exception:
                h15_params = dict_params

            # Hybrid MSE-guarded: take h15 only if data fidelity strictly improves
            try:
                mse_dict = float(mse(noisy, dict_params))
                mse_h15 = float(mse(noisy, h15_params))
                guarded = h15_params if mse_h15 < mse_dict else dict_params
                mu1, mu2 = params_to_orientations(np.asarray(guarded))
                if check_both_detected(mu1_true, mu2_true, mu1, mu2, angle_threshold):
                    counts["hybrid_guard"] += 1
            except Exception:
                pass

            # Pure LM (random init)
            try:
                rand_init = sim.prior_sampler(k_init, 1)[0]
                lm_params, _ = lm50(noisy, rand_init)
                mu1, mu2 = params_to_orientations(np.asarray(lm_params))
                if check_both_detected(mu1_true, mu2_true, mu1, mu2, angle_threshold):
                    counts["lm"] += 1
            except Exception:
                pass

            # DIPY upstream FORCE -> force_peaks (SH-on-sphere postprocessing)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dpf = dipy_force_peaks(noisy_np, dipy_force, angle_threshold)
                pair = best_two_peaks(dpf, mu1_true, mu2_true)
                if pair is not None and check_both_detected(
                    mu1_true, mu2_true, pair[0], pair[1], angle_threshold
                ):
                    counts["dipy_force"] += 1
            except Exception:
                pass

            # DIPY FORCE-internal: read FORCEFit.label directly off the sphere
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dpfi = dipy_force_internal_peaks(noisy_np, dipy_force, sphere)
                pair = best_two_peaks(dpfi, mu1_true, mu2_true)
                if pair is not None and check_both_detected(
                    mu1_true, mu2_true, pair[0], pair[1], angle_threshold
                ):
                    counts["dipy_force_internal"] += 1
            except Exception:
                pass

            # CSD
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cs = csd_peaks(noisy_np, gtab, response, sphere, angle_threshold)
                pair = best_two_peaks(cs, mu1_true, mu2_true)
                if pair is not None and check_both_detected(
                    mu1_true, mu2_true, pair[0], pair[1], angle_threshold
                ):
                    counts["csd"] += 1
            except Exception:
                pass

            # GQI
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gs = gqi_peaks(noisy_np, gtab, sphere, angle_threshold)
                pair = best_two_peaks(gs, mu1_true, mu2_true)
                if pair is not None and check_both_detected(
                    mu1_true, mu2_true, pair[0], pair[1], angle_threshold
                ):
                    counts["gqi"] += 1
            except Exception:
                pass

        results["angle"].append(angle_deg)
        for m in method_names:
            results[m].append(counts[m] / n_trials)

        line = f"  {angle_deg:3d} deg:"
        for m in method_names:
            line += f"  {m}={results[m][-1]:.0%}"
        print(line)

    # ----------------------------------------------------------------- #
    # Summary
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("Summary: Both-fiber detection rate by crossing angle")
    print("=" * 78)
    header = f"{'Angle':>5s}  " + "  ".join(f"{m:>12s}" for m in method_names)
    print(header)
    for i, a in enumerate(results["angle"]):
        row = f"{a:>5d}  " + "  ".join(f"{results[m][i]:>11.1%}" for m in method_names)
        print(row)

    npz_path = "validation/force_replication_v2_results.npz"
    np.savez(
        npz_path,
        angles=np.asarray(results["angle"]),
        **{m: np.asarray(results[m]) for m in method_names},
    )
    print(f"\nResults saved to {npz_path}")


if __name__ == "__main__":
    main()

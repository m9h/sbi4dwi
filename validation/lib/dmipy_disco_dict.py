"""
dmipy-JAX dictionary matcher on DiSCo — scalar microstructure recovery.

Counterpart to `validation/lib/force_disco.py` (which fits dipy's
upstream `dipy.reconst.force.FORCEModel`) — this module uses sbi4dwi's
own `dmipy_jax.library.matcher.DictionaryMatcher` for the same task.

Library design (matches FORCE paper §3.2's tuned DiSCo protocol):
  - 2-stick + Bingham dispersion + isotropic (small ceiling)
  - Diffusivity priors: D_∥ ∈ Uniform(0.54, 0.66) × 10⁻³ mm²/s
  - ODI ∈ Uniform(0.01, 0.30)
  - f_iso ∈ Uniform(0.0, 0.05) (paper "disabled" isotropic compartment)

dmipy-JAX NDI prediction = matched `f1 + f2 = 1 - f_iso`. This is a
proxy for "intra-cellular volume fraction" since our 2-stick model has
no extra-axonal compartment; for DiSCo's pure-cylinder phantom this is
the most defensible analogue to FORCE's `fit.nd`.
"""

from __future__ import annotations

from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.signal_models.bingham import (
    BinghamNODDI, get_rotation_matrix_from_z_to_vector,
)
from dmipy_jax.signal_models import g2_zeppelin


def build_disco_tuned_two_stick_simulator(acq: JaxAcquisition) -> ModelSimulator:
    """6-param dispersion-aware 2-stick simulator with DiSCo-aligned priors.

    Parameter layout: ``[d_par, theta1, theta2, odi, f1, f_iso]``.
    """
    bingham = BinghamNODDI(grid_points=120)

    def forward_fn(params, acq):
        d_par = params[0]
        t1, t2 = params[1], params[2]
        odi = params[3]
        f1, f_iso = params[4], params[5]
        f2 = 1.0 - f1 - f_iso
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
            # Paper §3.2 DiSCo: D_∥ ~ Uniform(0.54, 0.66) × 10⁻³ mm²/s
            "d_par": (0.54e-9, 0.66e-9),
            "theta1": (0.0, float(jnp.pi)),
            "theta2": (0.0, float(jnp.pi)),
            "odi": (0.01, 0.30),
            # f1 = primary stick fraction within the (1 - f_iso) WM volume
            "f1": (0.1, 0.8),
            # f_iso = extracellular volume fraction. DiSCo strands are sparse:
            # GT Strand Intra Volume Fraction mean ≈ 0.18, so the per-voxel
            # extracellular fraction ranges roughly 0.4–0.95. Our pure-stick
            # WM compartment has no extra-axonal zeppelin (which FORCE uses),
            # so f_iso here doubles as the extracellular water proxy.
            "f_iso": (0.0, 0.95),
        },
        acquisition=acq,
    )


def build_disco_tuned_3d_two_stick_simulator(acq: JaxAcquisition) -> ModelSimulator:
    """Full 3D-orientation 2-stick + Bingham + iso simulator (Option B2).

    Extends the 6-param planar simulator (§20) to 8 params with
    independent (θ, φ) per fibre:

        ``[d_par, theta1, phi1, theta2, phi2, odi, f1, f_iso]``

    ``mu_i = [sin θ_i cos φ_i, sin θ_i sin φ_i, cos θ_i]``.

    The §20 simulator constrained all fibres to y=0 (φ=0 implicit),
    which works for scalar microstructure recovery but fundamentally
    breaks tractography since streamlines can only move in the +x/+z
    plane.
    """
    bingham = BinghamNODDI(grid_points=120)

    def forward_fn(params, acq):
        d_par = params[0]
        t1, p1 = params[1], params[2]
        t2, p2 = params[3], params[4]
        odi = params[5]
        f1, f_iso = params[6], params[7]
        f2 = 1.0 - f1 - f_iso
        kappa = 1.0 / jnp.tan(jnp.pi * odi / 2.0)
        mu1 = jnp.array([
            jnp.sin(t1) * jnp.cos(p1),
            jnp.sin(t1) * jnp.sin(p1),
            jnp.cos(t1),
        ])
        mu2 = jnp.array([
            jnp.sin(t2) * jnp.cos(p2),
            jnp.sin(t2) * jnp.sin(p2),
            jnp.cos(t2),
        ])
        s1 = bingham(acq.bvalues, acq.gradient_directions,
                     mu=mu1, kappa1=kappa, kappa2=kappa, lambda_par=d_par)
        s2 = bingham(acq.bvalues, acq.gradient_directions,
                     mu=mu2, kappa1=kappa, kappa2=kappa, lambda_par=d_par)
        s_iso = jnp.exp(-acq.bvalues * 3.0e-9)
        return f1 * s1 + f2 * s2 + f_iso * s_iso

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=[
            "d_par", "theta1", "phi1", "theta2", "phi2", "odi", "f1", "f_iso",
        ],
        parameter_ranges={
            "d_par": (0.54e-9, 0.66e-9),
            "theta1": (0.0, float(jnp.pi)),
            "phi1": (0.0, float(2 * jnp.pi)),
            "theta2": (0.0, float(jnp.pi)),
            "phi2": (0.0, float(2 * jnp.pi)),
            "odi": (0.01, 0.30),
            "f1": (0.1, 0.8),
            "f_iso": (0.0, 0.95),
        },
        acquisition=acq,
    )


def _bingham_dispersed_zeppelin(
    bingham: BinghamNODDI,
    bvalues, gradient_directions,
    mu, kappa1, kappa2,
    lambda_par, lambda_perp,
):
    """Bingham-averaged extra-axonal zeppelin signal.

    Mirrors :py:meth:`BinghamNODDI.__call__` but integrates a
    cylindrically symmetric tensor (zeppelin) over the same Bingham
    orientation grid instead of a stick. Lets the §22 simulator share
    one dispersion distribution between intra (stick) and extra
    (zeppelin) compartments per fibre — the standard NODDI / Bingham-
    NODDI convention.
    """
    mu_vec = mu / jnp.linalg.norm(mu)
    R = get_rotation_matrix_from_z_to_vector(mu_vec)
    grid_vectors_rotated = jnp.dot(bingham.grid_vectors_canonical, R.T)

    nx = bingham.grid_vectors_canonical[:, 0]
    ny = bingham.grid_vectors_canonical[:, 1]
    pdf_vals = jnp.exp(-kappa1 * nx ** 2 - kappa2 * ny ** 2)
    normalization = jnp.sum(pdf_vals * bingham.grid_weights)
    pdf_normalized = pdf_vals / normalization

    def signal_for_orientation(n_mu):
        return g2_zeppelin(
            bvalues, gradient_directions, n_mu, lambda_par, lambda_perp,
        )

    grid_signals = jax.vmap(signal_for_orientation)(grid_vectors_rotated)
    total_weights = bingham.grid_weights * pdf_normalized
    return jnp.sum(grid_signals * total_weights[:, None], axis=0)


def build_disco_tuned_3d_stick_zeppelin_simulator(
    acq: JaxAcquisition,
) -> ModelSimulator:
    """3D-orientation stick + zeppelin simulator (§22).

    Adds an extra-axonal zeppelin per fibre with NODDI-style tortuosity
    coupling on top of the §21 3D-stick-only simulator. The motivation
    is §20.5 / §21.10: FORCE's richer compartmental model produced a
    ~2× tighter scalar CCC; the analogous improvement here is whether
    the extra-axonal compartment closes the remaining 0.07 Pearson r
    gap between dmipy-JAX (r=0.82) and the FORCE paper (r=0.89) on
    DiSCo connectivity.

    Parameters: ``[d_par, θ1, φ1, θ2, φ2, ODI, v_ic, f1, f_iso]``.

    - ``v_ic`` ∈ (0.3, 0.95): per-fibre intra-axonal volume fraction.
      Same value for both fibres (NODDI shared-tortuosity convention).
    - ``d_perp = d_par * (1 - v_ic)`` (Szafer-Stanisz tortuosity).
    - Total signal: ``f1 [v_ic·stick(μ1) + (1-v_ic)·zep(μ1, d⊥)]``
      ``+ f2 [v_ic·stick(μ2) + (1-v_ic)·zep(μ2, d⊥)]``
      ``+ f_iso · ball(d_iso=3e-9)``.
    """
    bingham = BinghamNODDI(grid_points=120)

    def forward_fn(params, acq):
        d_par = params[0]
        t1, p1 = params[1], params[2]
        t2, p2 = params[3], params[4]
        odi = params[5]
        v_ic = params[6]
        f1, f_iso = params[7], params[8]
        f2 = 1.0 - f1 - f_iso
        kappa = 1.0 / jnp.tan(jnp.pi * odi / 2.0)
        d_perp = d_par * (1.0 - v_ic)

        mu1 = jnp.array([
            jnp.sin(t1) * jnp.cos(p1),
            jnp.sin(t1) * jnp.sin(p1),
            jnp.cos(t1),
        ])
        mu2 = jnp.array([
            jnp.sin(t2) * jnp.cos(p2),
            jnp.sin(t2) * jnp.sin(p2),
            jnp.cos(t2),
        ])

        # Intra-axonal (sticks) — d_perp implicitly 0 for sticks
        s1_intra = bingham(acq.bvalues, acq.gradient_directions,
                            mu=mu1, kappa1=kappa, kappa2=kappa,
                            lambda_par=d_par)
        s2_intra = bingham(acq.bvalues, acq.gradient_directions,
                            mu=mu2, kappa1=kappa, kappa2=kappa,
                            lambda_par=d_par)
        # Extra-axonal (zeppelins) — same Bingham distribution
        s1_extra = _bingham_dispersed_zeppelin(
            bingham, acq.bvalues, acq.gradient_directions,
            mu1, kappa, kappa, d_par, d_perp,
        )
        s2_extra = _bingham_dispersed_zeppelin(
            bingham, acq.bvalues, acq.gradient_directions,
            mu2, kappa, kappa, d_par, d_perp,
        )

        s_iso = jnp.exp(-acq.bvalues * 3.0e-9)

        s1 = v_ic * s1_intra + (1.0 - v_ic) * s1_extra
        s2 = v_ic * s2_intra + (1.0 - v_ic) * s2_extra
        return f1 * s1 + f2 * s2 + f_iso * s_iso

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=[
            "d_par", "theta1", "phi1", "theta2", "phi2",
            "odi", "v_ic", "f1", "f_iso",
        ],
        parameter_ranges={
            "d_par": (0.54e-9, 0.66e-9),
            "theta1": (0.0, float(jnp.pi)),
            "phi1": (0.0, float(2 * jnp.pi)),
            "theta2": (0.0, float(jnp.pi)),
            "phi2": (0.0, float(2 * jnp.pi)),
            "odi": (0.01, 0.30),
            "v_ic": (0.3, 0.95),
            "f1": (0.1, 0.8),
            "f_iso": (0.0, 0.95),
        },
        acquisition=acq,
    )


def _spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def dmipy_params_to_pam_single(
    params: np.ndarray,
    sphere,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a single voxel's 3D-2-stick params to (peak_dirs,
    peak_values, peak_indices) — the three fields ``PeaksAndMetrics``
    needs for tractography.

    Parameter layout: ``[d_par, θ1, φ1, θ2, φ2, ODI, f1, f_iso]``.
    Returns 5-slot arrays (dipy convention), with positions 2..4 zero.
    """
    p = np.asarray(params, dtype=np.float64)
    t1, p1, t2, p2 = p[1], p[2], p[3], p[4]
    f1, f_iso = float(p[6]), float(p[7])
    f2 = 1.0 - f1 - f_iso

    mu1 = _spherical_to_cartesian(t1, p1)
    mu2 = _spherical_to_cartesian(t2, p2)

    peak_dirs = np.zeros((5, 3), dtype=np.float64)
    peak_values = np.zeros(5, dtype=np.float64)
    peak_indices = -np.ones(5, dtype=np.int32)

    peak_dirs[0] = mu1
    peak_dirs[1] = mu2
    peak_values[0] = max(f1, 0.0)
    peak_values[1] = max(f2, 0.0)

    # Snap each peak direction to its nearest sphere vertex (antipodal-aware)
    vertices = sphere.vertices  # (N, 3)
    for k in (0, 1):
        if peak_values[k] > 0:
            dots = np.abs(vertices @ peak_dirs[k])
            peak_indices[k] = int(np.argmax(dots))
    return peak_dirs, peak_values, peak_indices


def dmipy_zeppelin_params_to_pam_single(
    params: np.ndarray,
    sphere,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§22 PAM adapter: 9-param stick+zeppelin layout to PAM triple.

    Layout: ``[d_par, θ1, φ1, θ2, φ2, ODI, v_ic, f1, f_iso]``. The
    zeppelin compartment does not contribute additional peaks — it
    shares the orientation of its matching stick — so peak_dirs and
    peak_indices are computed identically to the §21 8-param case,
    with v_ic skipped.
    """
    p = np.asarray(params, dtype=np.float64)
    eight = np.array([p[0], p[1], p[2], p[3], p[4], p[5], p[7], p[8]])
    return dmipy_params_to_pam_single(eight, sphere)


def fit_dmipy_dict_on_disco(
    subject: int = 1,
    snr: int = 30,
    library_size: int = 200_000,
    seed: int = 1,
    k_best: int = 10,
    _smoke_roi_subset: Optional[Iterable[int]] = None,
) -> dict:
    """Fit dmipy-JAX dict on DiSCo and return per-voxel parameter maps.

    Returns
    -------
    dict with keys: ``ndi``, ``fa_proxy``, ``odi``, ``params``, ``mask``,
    ``gtab``, ``snr``, ``library_size``.
    """
    from dmipy_jax.library.generator import LibraryGenerator
    from dmipy_jax.library.matcher import DictionaryMatcher
    from dmipy_jax.library.storage import SimulationLibrary

    from validation.lib.force_disco import load_disco_subject

    out = load_disco_subject(subject=subject, snr=snr, single_shell_b=1900)
    data = out["data"]
    mask = out["mask"]
    rois = out["rois"]
    # JaxAcquisition object reconstructed from gtab
    bvals = np.asarray(out["gtab"].bvals) * 1e6  # mm²/s → m²/s (SI)
    bvecs = np.asarray(out["gtab"].bvecs)
    acq = JaxAcquisition(bvalues=jnp.asarray(bvals), gradient_directions=jnp.asarray(bvecs))

    if _smoke_roi_subset is not None:
        fit_mask = np.isin(rois, list(_smoke_roi_subset)) & mask
    else:
        fit_mask = mask

    sim = build_disco_tuned_two_stick_simulator(acq)
    print(f"  generating dmipy-JAX library: {library_size:,} entries...")
    gen = LibraryGenerator(sim, chunk_size=20_000)
    params, signals = gen.generate(library_size, key=jax.random.PRNGKey(seed))
    jax.block_until_ready(signals)
    lib = SimulationLibrary(
        params=params, signals=signals,
        parameter_names=sim.parameter_names,
    )
    matcher = DictionaryMatcher(lib, k_best=k_best)

    # Volume-level matching: returns {param_name: (X,Y,Z)} maps
    print(f"  matching {int(fit_mask.sum()):,} voxels against library...")
    maps = matcher.match_volume(data, mask=fit_mask, batch_size=4096)

    ndi = np.zeros(mask.shape, dtype=np.float32)
    odi = np.zeros(mask.shape, dtype=np.float32)
    fa_proxy = np.zeros(mask.shape, dtype=np.float32)

    f1 = maps["f1"]
    f_iso = maps["f_iso"]
    # dmipy NDI = total stick fraction = f1 + f2 = 1 - f_iso (in our 2-stick
    # + iso model, the entire WM compartment is sticks).
    ndi_map = (1.0 - f_iso).astype(np.float32)
    ndi[fit_mask] = ndi_map[fit_mask]
    odi[fit_mask] = maps["odi"][fit_mask].astype(np.float32)

    # FA proxy from the angle between the two recovered fibres.
    # When fibres are aligned (single-fibre voxel) → high FA; when
    # orthogonal (crossing) → low FA. Not strictly FA but tracks the
    # tensor-FA pattern on this kind of phantom.
    t1 = maps["theta1"]
    t2 = maps["theta2"]
    cos_diff = np.cos(t1 - t2)
    f1m = maps["f1"]
    f2m = 1.0 - f1m - f_iso
    # Bingham concentration also matters; a single sharp fibre = high FA
    fa_proxy_arr = np.abs(cos_diff) * np.maximum(f1m, f2m)
    fa_proxy[fit_mask] = fa_proxy_arr[fit_mask].astype(np.float32)

    return {
        "ndi": ndi,
        "odi": odi,
        "fa_proxy": fa_proxy,
        "params": maps,
        "mask": fit_mask,
        "gtab": out["gtab"],
        "rois": rois,
        "snr": snr,
        "library_size": library_size,
    }

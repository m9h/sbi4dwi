"""
DIPY-side baseline adapters for the FORCE replication / 3-fibre benchmarks.

Each adapter takes a 1D measurement signal (numpy, shape ``(M,)``) and a
prepared model object, returns a list of unit-vector peak directions on the
sphere. Listing instead of array so callers can apply
:func:`validation.lib.force_helpers.best_two_peaks` /
:func:`best_three_peaks` uniformly.
"""

from __future__ import annotations

from typing import List

import numpy as np


def csd_peaks_from_signal(
    signal: np.ndarray,
    gtab,
    response,
    sphere,
    *,
    relative_peak_threshold: float = 0.5,
    min_separation_angle: float = 10.5,
    npeaks: int = 5,
) -> List[np.ndarray]:
    """Constrained Spherical Deconvolution -> peaks_from_model peaks."""
    from dipy.direction import peaks_from_model
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

    data4d = signal[None, None, None, :]
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=8)
    peaks = peaks_from_model(
        model=model, data=data4d, sphere=sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        return_odf=False, normalize_peaks=True, npeaks=npeaks,
    )
    return [peaks.peak_dirs[0, 0, 0, k] for k in range(peaks.peak_dirs.shape[-2])]


def gqi_peaks_from_signal(
    signal: np.ndarray,
    gtab,
    sphere,
    *,
    sampling_length: float = 1.2,
    relative_peak_threshold: float = 0.5,
    min_separation_angle: float = 10.5,
    npeaks: int = 5,
) -> List[np.ndarray]:
    """Generalized Q-Sampling Imaging -> peaks_from_model peaks."""
    from dipy.direction import peaks_from_model
    from dipy.reconst.gqi import GeneralizedQSamplingModel

    data4d = signal[None, None, None, :]
    model = GeneralizedQSamplingModel(gtab, sampling_length=sampling_length)
    peaks = peaks_from_model(
        model=model, data=data4d, sphere=sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        return_odf=False, normalize_peaks=True, npeaks=npeaks,
    )
    return [peaks.peak_dirs[0, 0, 0, k] for k in range(peaks.peak_dirs.shape[-2])]


def dipy_force_peaks_from_signal(signal: np.ndarray, force_model) -> List[np.ndarray]:
    """dipy upstream FORCE pipeline: ``model.fit(data)`` -> ``force_peaks``.

    Goes through SH-on-default_sphere postprocessing — angular resolution
    bounded by sphere vertex density and SH order.
    """
    from dipy.reconst.force import force_peaks

    data4d = signal[None, None, None, :].astype(np.float32)
    fit = force_model.fit(data4d)
    peaks = force_peaks(fit)
    return [peaks.peak_dirs[0, 0, 0, k] for k in range(peaks.peak_dirs.shape[-2])]


def dipy_force_label_directions_from_signal(
    signal: np.ndarray,
    force_model,
    sphere,
) -> List[np.ndarray]:
    """Bypass force_peaks: read FORCEFit.label off the matched sphere directly.

    FORCEFit.label is an ``(n_vertices,)`` integer array with nonzero entries
    at the sphere indices the matcher considers active fibres. Returns those
    vertex directions as-is — no SH conversion, no sphere extraction.
    """
    data4d = signal[None, None, None, :].astype(np.float32)
    fit = force_model.fit(data4d)[0, 0, 0]
    label = np.asarray(fit.label)
    nz = np.where(label > 0)[0]
    if nz.size == 0:
        return []
    return [sphere.vertices[i] for i in nz]

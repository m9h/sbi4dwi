"""
Helpers for the FORCE-paper-matched re-run (§13 of doc 004).

Three helpers, each unit-tested in
``tests/validation/test_force_matched_helpers.py``:

  - make_stanford_hardi_acquisition: 150 dirs × b=2000 single-shell as a
    JaxAcquisition. Reproduces the FORCE paper's synthetic experiment
    acquisition and the dipy in-tree tutorial demo.
  - odi_to_kappa: NODDI's ODI → Watson/Bingham concentration κ. Pins the
    convention so synthetic dispersion samples align with what the FORCE
    library was generated under.
  - dispersed_two_stick_signal: 2-stick + isotropic mixture where each
    stick is Bingham-dispersed around its principal axis with concentration
    κ. Numerical sphere integration via :class:`BinghamNODDI`.
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models.bingham import BinghamNODDI


# Single shared Bingham integrator — small grid for speed; tests verify
# correctness at the asymptotic limits.
_BINGHAM = BinghamNODDI(grid_points=200)


def make_stanford_hardi_acquisition() -> JaxAcquisition:
    """Load Stanford HARDI's gradient table as a JaxAcquisition.

    Stanford HARDI: 160 directions = 10 b=0 + 150 b=2000 s/mm². Single-shell.
    Matches the FORCE paper's synthetic-experiment acquisition (§3.1).
    """
    from dipy.data import get_fnames
    from dipy.io.gradients import read_bvals_bvecs

    _, bval_fname, bvec_fname = get_fnames(name="stanford_hardi")
    bvals_smm2, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
    # JaxAcquisition stores b-values in SI units (s/m^2)
    bvals_si = jnp.asarray(bvals_smm2, dtype=jnp.float32) * 1e6
    bvecs_jax = jnp.asarray(bvecs, dtype=jnp.float32)
    return JaxAcquisition(bvalues=bvals_si, gradient_directions=bvecs_jax)


def odi_to_kappa(odi: float) -> float:
    """NODDI ODI → Bingham/Watson concentration κ.

    Convention: ``ODI = (2/π) · arctan(1/κ)``  →  ``κ = 1 / tan(π · ODI / 2)``.

    ODI=0.5 ⇒ κ=1.0; ODI=0.01 ⇒ κ≈63.66; ODI=0.30 ⇒ κ≈1.96.
    """
    return float(1.0 / math.tan(math.pi * odi / 2.0))


def dispersed_two_stick_signal(
    acq: JaxAcquisition,
    mu1: jnp.ndarray,
    mu2: jnp.ndarray,
    kappa1: float,
    kappa2: float,
    f1: float,
    f_iso: float = 0.05,
    d_par: float = 1.7e-9,
    d_iso: float = 3.0e-9,
) -> jnp.ndarray:
    """2-stick + isotropic mixture with Bingham dispersion on each stick.

    Each fibre population's signal is the integral of the C1Stick attenuation
    weighted by a Bingham distribution centred on its principal axis. Uses
    :class:`BinghamNODDI` with axisymmetric concentration (kappa1 = kappa2 for
    each fibre) for the matched-FORCE re-run.

    Parameters
    ----------
    kappa1, kappa2
        Bingham concentrations of fibre 1 and fibre 2 (axisymmetric: pass
        the same κ for both axes when calling :class:`BinghamNODDI`).
    """
    # Per-fibre dispersed signals (lambda_par in m^2/s, JaxAcquisition is SI)
    s1 = _BINGHAM(
        acq.bvalues, acq.gradient_directions, mu=mu1,
        kappa1=kappa1, kappa2=kappa1, lambda_par=d_par,
    )
    s2 = _BINGHAM(
        acq.bvalues, acq.gradient_directions, mu=mu2,
        kappa1=kappa2, kappa2=kappa2, lambda_par=d_par,
    )
    s_iso = jnp.exp(-acq.bvalues * d_iso)
    f2 = 1.0 - f1 - f_iso
    return f1 * s1 + f2 * s2 + f_iso * s_iso

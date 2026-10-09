"""
Integration tests pinning the FORCE-paper-matched acquisition + dispersed
2-fibre forward model used by validate_force_matched.py.

The matched re-run (§13 of doc 004) needs three new helpers, each tested
here:

  - make_stanford_hardi_acquisition: load Stanford HARDI gtab as a
    JaxAcquisition (150 dirs × b=2000, single-shell). Reproduces the
    paper's synthetic-experiment acquisition.

  - odi_to_kappa: NODDI's standard ODI → Bingham/Watson concentration
    conversion. Pin the convention so the synthetic dispersion samples
    align with what the FORCE library uses.

  - dispersed_two_stick_signal: 2-stick + isotropic mixture where each
    stick is Bingham-dispersed around its principal axis with concentration
    κ. At κ → ∞ this must collapse to the sharp two_stick_signal; at
    κ → 0 it must collapse to the isotropic mean.
"""

import math

import numpy as np
import pytest


force_matched = pytest.importorskip("validation.lib.force_matched")


# --------------------------------------------------------------------------- #
# 1. Stanford HARDI acquisition
# --------------------------------------------------------------------------- #

class TestStanfordHardiAcquisition:
    def test_returns_150_directions_at_b2000(self):
        acq = force_matched.make_stanford_hardi_acquisition()
        bvals_si = np.asarray(acq.bvalues)
        bvals_smm2 = bvals_si / 1e6  # SI -> s/mm^2

        # 10 b=0 + 150 b=2000 in Stanford HARDI
        n_b0 = (bvals_smm2 < 50).sum()
        n_b2000 = (np.abs(bvals_smm2 - 2000) < 50).sum()
        assert n_b0 == 10, f"Expected 10 b=0; got {n_b0}"
        assert n_b2000 == 150, f"Expected 150 b=2000; got {n_b2000}"

    def test_total_directions_is_160(self):
        acq = force_matched.make_stanford_hardi_acquisition()
        assert np.asarray(acq.bvalues).shape == (160,)
        assert np.asarray(acq.gradient_directions).shape == (160, 3)

    def test_bvecs_are_unit_norm_at_nonzero_b(self):
        acq = force_matched.make_stanford_hardi_acquisition()
        bvals_si = np.asarray(acq.bvalues)
        bvecs = np.asarray(acq.gradient_directions)
        nonzero = bvals_si > 1e6  # b > 1 s/mm^2
        norms = np.linalg.norm(bvecs[nonzero], axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-3)


# --------------------------------------------------------------------------- #
# 2. ODI <-> kappa conversion (NODDI convention)
# --------------------------------------------------------------------------- #

class TestOdiToKappa:
    def test_odi_half_gives_kappa_one(self):
        """ODI = 0.5 corresponds to kappa = 1.0 (per NODDI: ODI = 2/π · arctan(1/κ))."""
        kappa = force_matched.odi_to_kappa(0.5)
        assert kappa == pytest.approx(1.0, rel=1e-3)

    def test_low_odi_high_kappa(self):
        """Sharp dispersion (low ODI) -> high κ."""
        k = force_matched.odi_to_kappa(0.01)
        assert k > 60.0  # ODI=0.01 gives κ ≈ 63.66

    def test_high_odi_low_kappa(self):
        """Broad dispersion (high ODI) -> low κ."""
        k = force_matched.odi_to_kappa(0.30)
        assert 1.0 < k < 3.0  # ODI=0.30 gives κ ≈ 1.96

    def test_monotonically_decreasing(self):
        odis = np.linspace(0.01, 0.99, 20)
        kappas = np.array([force_matched.odi_to_kappa(o) for o in odis])
        assert (np.diff(kappas) < 0).all(), "ODI->kappa should be strictly decreasing"


# --------------------------------------------------------------------------- #
# 3. Dispersed 2-stick forward model
# --------------------------------------------------------------------------- #

class TestDispersedTwoStickSignal:
    def setup_method(self):
        from validation.lib.two_fiber import make_multishell_acquisition
        # Use the v2 acquisition for cheap unit testing; production uses
        # Stanford HARDI but the helper is acquisition-agnostic.
        self.acq = make_multishell_acquisition()
        rad = math.radians(45)
        self.mu1 = np.array([0.0, 0.0, 1.0])
        self.mu2 = np.array([math.sin(rad), 0.0, math.cos(rad)])

    def test_b0_is_unity(self):
        import jax.numpy as jnp

        S = force_matched.dispersed_two_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            kappa1=10.0, kappa2=10.0, f1=0.45, f_iso=0.05,
        )
        b0 = np.asarray(self.acq.bvalues) == 0
        assert np.allclose(np.asarray(S)[b0], 1.0, atol=1e-3)

    def test_signal_in_unit_interval(self):
        import jax.numpy as jnp

        S = force_matched.dispersed_two_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            kappa1=10.0, kappa2=10.0, f1=0.45, f_iso=0.05,
        )
        S_np = np.asarray(S)
        assert (S_np >= 0).all() and (S_np <= 1.0 + 1e-3).all()

    def test_high_kappa_collapses_to_sharp_stick(self):
        """At very high concentration (sharp), the dispersed signal must
        approach the two_stick_signal output. Pins the asymptotic behaviour."""
        import jax.numpy as jnp
        from validation.lib.two_fiber import two_stick_signal

        S_dispersed = force_matched.dispersed_two_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            kappa1=200.0, kappa2=200.0,  # very sharp
            f1=0.45, f_iso=0.05,
        )
        S_sharp = two_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            f1=0.45, f_iso=0.05,
        )
        # At κ=200 the Bingham PDF is concentrated within ~5° of the axis;
        # average signal should be within 5% of sharp signal
        diff = np.max(np.abs(np.asarray(S_dispersed) - np.asarray(S_sharp)))
        assert diff < 0.05, (
            f"At κ=200, dispersed signal must approach sharp stick; "
            f"max diff = {diff:.4f}"
        )

    def test_low_kappa_gives_more_isotropic_signal(self):
        """At very low κ (broad dispersion), the angular dependency of the
        signal should weaken — std across gradient directions should drop
        compared to a sharp configuration."""
        import jax.numpy as jnp

        S_sharp = force_matched.dispersed_two_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            kappa1=200.0, kappa2=200.0, f1=0.45, f_iso=0.05,
        )
        S_dispersed = force_matched.dispersed_two_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            kappa1=1.0, kappa2=1.0, f1=0.45, f_iso=0.05,
        )
        bvals = np.asarray(self.acq.bvalues)
        # std across the b=2000 shell only
        b2k = np.abs(bvals - 2e9) < 5e7  # SI: 2e9 == b=2000
        std_sharp = np.std(np.asarray(S_sharp)[b2k])
        std_dispersed = np.std(np.asarray(S_dispersed)[b2k])
        assert std_dispersed < std_sharp

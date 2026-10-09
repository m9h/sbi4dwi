"""
Integration tests pinning the 2-fibre forward + library wiring used by
both validate_force_replication_v2.py and validate_force_snr_sweep.py.

Mirrors tests/validation/test_force_3fiber_integration.py. The 2-stick
forward and its dictionary simulator predate the 3-stick variant by one
commit; these tests retroactively pin them so the SNR sweep can build on
verified ground.
"""

import math

import numpy as np
import pytest


two_fiber = pytest.importorskip("validation.lib.two_fiber")


# --------------------------------------------------------------------------- #
# 1. Forward model
# --------------------------------------------------------------------------- #

class TestTwoStickSignal:
    def setup_method(self):
        self.acq = two_fiber.make_multishell_acquisition()
        rad = math.radians(45)
        self.mu1 = np.array([0.0, 0.0, 1.0])
        self.mu2 = np.array([math.sin(rad), 0.0, math.cos(rad)])

    def test_b0_signal_is_unity(self):
        import jax.numpy as jnp

        S = two_fiber.two_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            f1=0.45, f_iso=0.05,
        )
        b0 = np.asarray(self.acq.bvalues) == 0
        assert np.allclose(np.asarray(S)[b0], 1.0, atol=1e-6)

    def test_signal_is_bounded_unit_interval(self):
        import jax.numpy as jnp

        S = two_fiber.two_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            f1=0.45, f_iso=0.05,
        )
        S_np = np.asarray(S)
        assert (S_np >= 0).all() and (S_np <= 1.0 + 1e-6).all()

    def test_fractions_sum_to_one_in_signal(self):
        """f1 + f2 + f_iso = 1 implicitly; b=0 sum must equal 1.0 regardless
        of split."""
        import jax.numpy as jnp

        S = two_fiber.two_stick_signal(
            self.acq, jnp.asarray(self.mu1), jnp.asarray(self.mu2),
            f1=0.3, f_iso=0.15,  # f2 = 0.55 implicitly
        )
        b0 = np.asarray(self.acq.bvalues) == 0
        assert np.allclose(np.asarray(S)[b0], 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. dmipy-JAX 2-stick library self-recovery
# --------------------------------------------------------------------------- #

class TestTwoStickLibrarySelfRecovery:
    def test_clean_signal_recovered_from_library(self):
        import jax
        from dmipy_jax.library.generator import LibraryGenerator
        from dmipy_jax.library.matcher import DictionaryMatcher
        from dmipy_jax.library.storage import SimulationLibrary

        acq = two_fiber.make_multishell_acquisition()
        sim = two_fiber.build_two_stick_simulator(acq)
        gen = LibraryGenerator(sim, chunk_size=5_000)
        params, signals = gen.generate(5_000, key=jax.random.PRNGKey(11))
        lib = SimulationLibrary(
            params=params, signals=signals, parameter_names=sim.parameter_names,
        )
        matcher = DictionaryMatcher(lib, k_best=1)
        idx = 1234
        recovered, _ = matcher.match_single(signals[idx])
        assert np.allclose(np.asarray(recovered), np.asarray(params[idx]), atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. Rician noise helper monotonicity
# --------------------------------------------------------------------------- #

class TestRicianNoiseScaling:
    def test_higher_snr_lower_perturbation(self):
        """SNR=50 must produce a strictly smaller |noisy - clean| RMS than
        SNR=10 on the same signal and same key. Pins the SNR semantics so
        the SNR sweep's monotone-in-SNR expectation has ground truth.
        """
        import jax
        import jax.numpy as jnp

        acq = two_fiber.make_multishell_acquisition()
        rad = math.radians(45)
        mu1 = jnp.array([0.0, 0.0, 1.0])
        mu2 = jnp.array([math.sin(rad), 0.0, math.cos(rad)])
        S = two_fiber.two_stick_signal(acq, mu1, mu2, f1=0.45, f_iso=0.05)

        key = jax.random.PRNGKey(42)
        noisy_low = two_fiber.add_rician_noise(S, snr=10.0, key=key)
        noisy_high = two_fiber.add_rician_noise(S, snr=50.0, key=key)

        rms_low = float(jnp.sqrt(jnp.mean((noisy_low - S) ** 2)))
        rms_high = float(jnp.sqrt(jnp.mean((noisy_high - S) ** 2)))
        assert rms_high < rms_low
        # Also pin order-of-magnitude scaling: SNR=50 noise should be ~5x
        # smaller in std than SNR=10
        assert rms_low / max(rms_high, 1e-9) > 3.0

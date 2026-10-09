
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.simulation.scanner import bloch, objects
from dmipy_jax.simulation.scanner.sequences import GeneralSequence
from dmipy_jax.external import pulseq
from dmipy_jax.external import jemris

# Soft dependency check
try:
    import pypulseq as pp
    HAS_PYPULSEQ = True
except ImportError:
    HAS_PYPULSEQ = False

try:
    import h5py
    HAS_JEMRIS_DEPS = True
except ImportError:
    HAS_JEMRIS_DEPS = False

@pytest.mark.skipif(not HAS_PYPULSEQ, reason="pypulseq not installed")
class TestJemrisComparison:
    
    @pytest.fixture
    def simple_fid_sequence(self):
        """Creates a simple 90-degree pulse followed by delay (FID)."""
        system = pp.Opts(max_grad=30, grad_unit='mT/m', rise_time=0.3, slew_unit='T/m/s', rf_dead_time=100e-6, adc_dead_time=20e-6)
        seq = pp.Sequence(system)
        
        # 90 deg hard pulse
        rf90 = pp.make_block_pulse(flip_angle=np.pi/2, duration=1e-3, system=system)
        
        # ADC readout
        adc = pp.make_adc(num_samples=1, duration=10e-6, system=system, delay=5e-3) # Sample 5ms after
        
        seq.add_block(rf90)
        seq.add_block(adc)
        
        return seq

    @pytest.mark.xfail(reason="known failure, pre-existing, see docs/decisions/010 §7.3: Bloch simulate_acquisition raises inside JIT", strict=False)
    def test_fid_signal_match(self, simple_fid_sequence):
        """
        Validates that dmipy-jax produces purely transverse magnetization 
        after a 90 pulse, matching theoretical JEMRIS expectation.
        
        (Actual JEMRIS comparison requires running JEMRIS externally)
        """
        # 1. Convert to JAX sequence
        jax_seq = pulseq.pulseq_to_general(simple_fid_sequence, dt=10e-6)
        
        # 2. Define standard phantom (Single Spin on resonance)
        # T1, T2 long enough to ignore decay for this short test
        phantom = objects.IsochromatPhantom(
            positions=jnp.array([[0., 0., 0.]]),
            T1=jnp.array([1000.0]),
            T2=jnp.array([1000.0]),
            M0=jnp.array([1.0]),
            off_resonance=jnp.array([0.0])
        )
        
        # 3. Simulate (JAX)
        # Duration = total sequence duration
        duration = simple_fid_sequence.duration()[0] if isinstance(simple_fid_sequence.duration(), (list, tuple)) else simple_fid_sequence.duration()
        
        # Run bloch sim
        signal_jax = bloch.simulate_acquisition(phantom, jax_seq, duration)
        
        # Check basic physics first (Unit magnitude, Phase +90 for -y pulse or similar)
        # pypulseq default pulse phase is 0 (x-axis field) -> M rotates y to z? No, z to -y.
        # Torque = M x B. M=z, B=x. Torque = z cross x = y.
        # So M tips towards y.
        # Signal = Mx + iMy. Should be purely imaginary?
        
        mag = jnp.abs(signal_jax)
        # Expected ~1.0 (minus slight relaxation)
        assert jnp.allclose(mag, 1.0, atol=1e-2), f"Signal magnitude {mag} != 1.0"
        
    @pytest.mark.skip(reason="Requires external JEMRIS run & data file")
    def test_compare_against_file(self):
        """
        Skeleton for comparing against actual JEMRIS output file.
        Usage: 
            1. Run JEMRIS on 'test_seq.seq' -> 'jemris_out.h5'
            2. Point this test to the file.
        """
        jemris_file = "comparisons/jemris_fid.h5"
        
        # Load JEMRIS
        sig_jemris = jemris.load_jemris_signal(jemris_file)
        
        # Run JAX equivalent...
        # ...
        
        # Compare
        # nrmse = ...
        pass


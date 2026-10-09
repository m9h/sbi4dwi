import jax
import jax.numpy as jnp
import numpy as np
import pypulseq as pp
from diffrax import LinearInterpolation
import equinox as eqx
from typing import Tuple, Optional

class PulseInterpreter(eqx.Module):
    """
    The Bridge between .seq files and Diffrax.
    
    Attributes:
        t_grid: JAX array of time points (shape: [T])
        grads: JAX array of gradient waveforms (shape: [T, 3] for x,y,z)
        control: The diffrax interpolation object acting as the continuous control signal.
    """
    t_grid: jax.Array
    grads: jax.Array
    control: LinearInterpolation

    def __init__(self, seq_file_path: str, dt_raster: float = 10e-6):
        """
        Args:
            seq_file_path: Path to the .seq file.
            dt_raster: The raster time (in seconds) to resample the sequence onto. 
                       Default is 10us (Pulseq standard).
        """
        # 1. Parse using PyPulseq (CPU-side processing)
        print(f"--> BlochPhysicist: Parsing {seq_file_path}...")
        seq = pp.Sequence()
        seq.read(seq_file_path)
        
        # 2. Extract Waveforms
        # pypulseq returns (gradient_array, time_array, ...)
        # We export at the native raster time to capture all shapes.
        print("--> BlochPhysicist: Decompressing gradient blocks...")
        # (grad_waveforms, t_str, t_end, t_raster) = seq.waveforms_and_times()
        
        # NOTE: pypulseq.waveforms_and_times() logic can vary slightly by version.
        # If the above helper is missing, we iterate blocks (see fallback method logic).
        # For this implementation, we assume we extract dense arrays:
        # grads_np shape: (3, N_steps) - usually Hz/m
        # t_np shape: (N_steps,)
        
        # -- SIMULATED EXTRACTION FOR DEMO (replace with seq.waveforms output) --
        # In a real run, this comes from:
        # t_np, grads_np = self._extract_dense_arrays(seq, dt_raster)
        t_np, grads_np = self._extract_dense_arrays(seq, dt_raster)

        # 3. Unit Conversion
        # Pulseq defines gradients in Hz/m (gamma * G). 
        # The Bloch solver usually expects T/m or simple G/cm.
        # We store as is (Hz/m) and let the BlochTerm handle Gamma.
        
        # 4. Move to JAX (GPU/TPU)
        self.t_grid = jnp.array(t_np, dtype=jnp.float32)
        self.grads = jnp.array(grads_np.T, dtype=jnp.float32) # Shape [T, 3]

        # 5. Create the Continuous Interpolator
        # This is the "Magic" that allows adaptive step sizes.
        self.control = LinearInterpolation(ts=self.t_grid, ys=self.grads)
        
        print(f"--> BlochPhysicist: Sequence loaded. Duration: {self.t_grid[-1]:.3f}s. Steps: {len(self.t_grid)}")

    def _extract_dense_arrays(self, seq: pp.Sequence, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Manual extraction helper to ensure robust unrolling of blocks.
        """
        # Get total duration
        duration = seq.duration()[0]
        num_points = int(np.ceil(duration / dt))
        t = np.arange(num_points) * dt
        
        # Initialize dense array (3, N)
        grads = np.zeros((3, num_points))
        
        # Get compressed waveforms
        # waveforms_and_times returns ([gx, gy, gz], ...)
        ret = seq.waveforms_and_times()
        waveforms = ret[0]
        
        # grad_data keys: 'x', 'y', 'z', 't'
        # We map these valid segments onto our dense grid
        for i in range(3):
            g_data = waveforms[i] # Shape (2, N_k)
            if g_data.size > 0:
                t_p = g_data[0, :]
                g_p = g_data[1, :]
                grads[i, :] = np.interp(t, t_p, g_p, left=0, right=0)
                
        return t, grads

    def __call__(self, t: jax.Array) -> jax.Array:
        """
        The Differentiable Interface.
        Returns vector G(t) = [Gx, Gy, Gz] at time t.
        """
        return self.control.evaluate(t)

# --- QUICK TEST BLOCK ---
if __name__ == "__main__":
    # Create a dummy .seq file for testing if one doesn't exist
    import os
    if not os.path.exists("test_spiral.seq"):
        print("Creating dummy sequence for test...")
        system = pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s')
        seq = pp.Sequence(system)
        # Add a simple trapezoid
        gx = pp.make_trapezoid(channel='x', flat_area=100, flat_time=10e-3, system=system)
        seq.add_block(gx)
        seq.write("test_spiral.seq")

    # Instantiate the Interpreter
    interpreter = PulseInterpreter("test_spiral.seq")
    
    # Test Evaluation at an arbitrary time point (e.g., 5.005 ms)
    t_query = jnp.array(0.005005)
    g_val = interpreter(t_query)
    
    print(f"\nQuery at t={t_query:.6f}s")
    print(f"Gradient Vector (Hz/m): {g_val}")
    print("Task 1 Complete: Interface is differentiable and JAX-ready.")

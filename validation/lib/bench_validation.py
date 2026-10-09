
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import diffrax
import matplotlib.pyplot as plt
from typing import Tuple, List

# Imports from BENCH (assuming installed in environment)
# Imports from local patch (avoids Numba dependency)
try:
    from validation.lib import bench_diffusion_models_patch as bench_models
except ImportError:
    import sys
    sys.path.append(".") # Ensure dmipy root is in path
    from validation.lib import bench_diffusion_models_patch as bench_models
# Imports from dmipy_jax
from dmipy_jax.simulation.sde_models import CurvedTractSDE, solve_restricted_sde_batch
from dmipy_jax.simulation.simulator import accumulate_phase, trapezoidal_gradient_waveform
from dmipy_jax.constants import GYRO_MAGNETIC_RATIO

def get_pgse_params(bval: float, delta: float, Delta: float) -> float:
    """
    Calculate Gradient Amplitude G from b-value, delta, and Delta.
    b = (gamma * G * delta)^2 * (Delta - delta/3)
    G = sqrt( b / (Delta - delta/3) ) / (gamma * delta)
    """
    if bval == 0:
        return 0.0
    
    # bval is usually in s/mm^2 or similar. 
    # Gyro is in rad/(s*T).
    # We need consistent units.
    # SI Units:
    # bval [s/m^2]
    # G [T/m]
    # delta, Delta [s]
    # gamma [rad/(s*T)]
    
    numerator = bval / (Delta - delta/3.0)
    G = jnp.sqrt(numerator) / (GYRO_MAGNETIC_RATIO * delta)
    return G

class DiffraxSimulationWrapper(eqx.Module):
    """
    Wraps CurvedTractSDE to simulate PGSE signal.
    """
    sde_model: CurvedTractSDE
    
    def __init__(self, D_long, D_trans, affine=None):
        if affine is None:
            affine = jnp.eye(4)
            
        vector_field = jnp.zeros((3, 1, 1, 1))
        # Default Z-axis alignment
        vector_field = vector_field.at[2, ...].set(1.0) 
        
        potential_field = jnp.zeros((1, 1, 1))
        
        self.sde_model = CurvedTractSDE(
            vector_field=vector_field,
            potential_field=potential_field,
            affine=affine,
            diffusivity_long=D_long,
            diffusivity_trans=D_trans,
            k_confinement=0.0 # Free diffusion
        )
        
    def __call__(self, G_amps, gradients, delta, Delta, dt, N_particles, key):
        """
        Simulate for a list of gradients.
        G_amps: (N_meas,)
        gradients: (N_meas, 3) - Directions
        """
        T_max = Delta + delta + dt
        t_eval = jnp.arange(0, T_max, dt)
        save_at = diffrax.SaveAt(ts=t_eval)
        
        # Initial positions at 0 (or Gaussian? For free diffusion relative displacement matters)
        # Starting at 0 is fine as SDE adds noise.
        y0_batch = jnp.zeros((N_particles, 3))
        
        sol = solve_restricted_sde_batch(
            self.sde_model,
            (0.0, T_max),
            y0_batch,
            dt0=dt,
            key=key,
            save_at=save_at
        )
        # sol.ys shape: (N_particles, N_steps, 3)
        trajectories = sol.ys
        times = sol.ts 
        
        # Define simulation function for vmap
        def compute_single_signal(G, direction):
            # Re-interpolating waveform to trajectory times
            G_t = trapezoidal_gradient_waveform(times, G, direction, delta, Delta)
            
            # Phase accumulation
            phases = jax.vmap(accumulate_phase, in_axes=(0, None, None))(trajectories, G_t, dt)
            
            # Signal
            sig = jnp.mean(jnp.exp(1j * phases))
            return jnp.abs(sig)
            
        # Vmap over measurements
        signals = jax.vmap(compute_single_signal)(G_amps, gradients)
        
        return signals

def main():
    print("Running BENCH Validation Suite...")
    
    # --- Parameters (SI Units) ---
    D_val = 2.0e-9 # m^2/s
    
    # BENCH expects compatible units.
    # If we pass bval in s/mm^2, D should be in mm^2/s.
    # D_val_mm2s = 2.0e-9 * 1e6 = 2.0e-3 mm^2/s = 2.0 um^2/ms.
    D_bench = D_val * 1e6 
    
    # Acquisition
    bvals_s_mm2 = jnp.array([0, 1000, 2000, 3000], dtype=float)
    bvals_si = bvals_s_mm2 * 1e6 # s/m^2
    
    delta = 10e-3 # s
    Delta = 20e-3 # s
    dt = 0.5e-3   # s (0.5 ms)
    N_particles = 3000
    key = jax.random.PRNGKey(42)
    
    # --- 1. BALL (Isotropic) ---
    print("\n[VALIDATION] Ball (Isotropic)")
    
    # Diffrax
    G_amps = jax.vmap(get_pgse_params, in_axes=(0, None, None))(bvals_si, delta, Delta)
    grads = jnp.array([[1.0, 0.0, 0.0]] * len(bvals_si))
    
    wrapper_ball = DiffraxSimulationWrapper(D_val, D_val)
    sig_diffrax = wrapper_ball(G_amps, grads, delta, Delta, dt, N_particles, key)
    
    # BENCH
    # Pass bval in s/mm^2, D in mm^2/s
    sig_bench = bench_models.ball(bvals_s_mm2, grads, d_iso=D_bench)
    
    print(f"B-values (s/mm^2): {bvals_s_mm2}")
    print(f"Diffrax: {sig_diffrax}")
    print(f"BENCH:   {sig_bench}")
    
    mse = jnp.mean((sig_diffrax - sig_bench)**2)
    print(f"MSE: {mse:.2e}")
    if mse < 1e-3:
        print(">> PASS")
    else:
        print(">> FAIL")

    # --- 2. STICK (Anisotropic Z-axis) ---
    print("\n[VALIDATION] Stick (Anisotropic Z-axis)")
    theta, phi = 0.0, 0.0
    wrapper_stick = DiffraxSimulationWrapper(D_val, 0.0) 
    
    # Measurement Directions: Parallel (Z) and Perpendicular (X)
    meas_dirs = jnp.array([
        [0.0, 0.0, 1.0], # Parallel (Should decay)
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0], # Perpendicular (Should NOT decay)
        [1.0, 0.0, 0.0]
    ])
    meas_bvals = jnp.array([1000.0, 2000.0, 1000.0, 2000.0]) 
    meas_bvals_si = meas_bvals * 1e6
    meas_G = jax.vmap(get_pgse_params, in_axes=(0, None, None))(meas_bvals_si, delta, Delta)
    
    sig_stick_diffrax = wrapper_stick(meas_G, meas_dirs, delta, Delta, dt, N_particles, key)
    
    # BENCH
    sig_stick_bench = bench_models.stick(
        meas_bvals, 
        meas_dirs, 
        d_a=D_bench, 
        theta=theta, 
        phi=phi
    )
    
    print(f"B-values: {meas_bvals}")
    print(f"Dirs (Z=Par, X=Perp):\n{meas_dirs}")
    print(f"Diffrax: {sig_stick_diffrax}")
    print(f"BENCH:   {sig_stick_bench}")
    
    # Check Perpendicular (indices 2,3) -> Should be ~1.0
    # Check Parallel (indices 0,1) -> Should decay
    
    mse_stick = jnp.mean((sig_stick_diffrax - sig_stick_bench)**2)
    print(f"MSE: {mse_stick:.2e}")
    if mse_stick < 1e-3:
        print(">> PASS")
    else:
        print(">> FAIL")

    # --- 3. CIGAR (Finite Radial) ---
    print("\n[VALIDATION] Cigar (D_long=2, D_trans=0.5)")
    D_trans_val = 0.5e-9 
    D_trans_bench = D_trans_val * 1e6
    
    wrapper_cigar = DiffraxSimulationWrapper(D_val, D_trans_val)
    
    sig_cigar_diffrax = wrapper_cigar(meas_G, meas_dirs, delta, Delta, dt, N_particles, key)
    
    sig_cigar_bench = bench_models.cigar(
        meas_bvals,
        meas_dirs,
        d_a=D_bench,
        d_r=D_trans_bench,
        theta=theta,
        phi=phi
    )
    
    print(f"Diffrax: {sig_cigar_diffrax}")
    print(f"BENCH:   {sig_cigar_bench}")
    mse_cigar = jnp.mean((sig_cigar_diffrax - sig_cigar_bench)**2)
    print(f"MSE: {mse_cigar:.2e}")
    if mse_cigar < 1e-3:
        print(">> PASS")
    else:
        print(">> FAIL")

if __name__ == "__main__":
    main()

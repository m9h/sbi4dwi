
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from dmipy_jax.simulation.substrates.caterpillar import CATERPillarOracle
from dmipy_jax.simulation.monte_carlo import simulate_ground_truth
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.cylinder import C1Stick

def main():
    print("--- ReMiDi Connection: Model Mismatch Demo ---")
    
    # 1. Define Beading Parameters
    amplitudes = [0.0, 0.2, 0.4] # Beading amplitudes (0 = straight cylinder approx)
    
    # 2. Setup Acquisition (High b-value for sensitivity)
    bvals = jnp.array([1000.0, 3000.0, 5000.0]) * 1e6
    n_dirs = 32
    # Simple setup
    bvals_full = jnp.repeat(bvals, n_dirs)
    # Random dirs
    key = jax.random.PRNGKey(0)
    vecs = jax.random.normal(key, (len(bvals_full), 3))
    vecs = vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)
    
    # Waveform for simulation (simple PGSE approx)
    # Note: simulate_ground_truth takes a waveform (N_steps, 3)
    # We need to construct waveforms for every measurement.
    # For efficiency in this demo, let's simulate ONE direction with varying gradient strength (Stejskal-Tanner)
    # to show the signal decay change.
    
    # Or, we can use the `simulate_ground_truth` which takes a waveform.
    # We will simulate a Single Shell along X axis for simplicity of visualization?
    # No, let's do full 3D simulation for a few directions to get a mean signal.
    
    # Actually, let's stick to the prompt: Generate dataset of beading cylinders.
    # We will simulate for ONE b-value (3000) and ONE direction (Perpendicular to axon?) 
    # Beading effect is most visible perpendicular? Or Parallel?
    # Axons are usually aligned along Z in CATERPillar default?
    # Let's check config. alpha=4 means alignment.
    
    oracle = CATERPillarOracle()
    default_config = oracle.get_default_config()
    # Align axons along X (default seems to be roughly X or Z depending on distribution).
    # Default alpha=4 suggests concentration.
    
    results = []
    
    for amp in amplitudes:
        print(f"Generating Substrate with Beading Amplitude = {amp}...")
        config = default_config.copy()
        config['beading_variation'] = amp # Intensity of beading
        config['beading_period'] = 10.0 # 10um period
        config['vox_sizes'] = [20] # 20um box
        config['filename'] = f"beading_{amp}"
        
        # Run CATERPillar
        df = oracle.generate(config)
        
        # Setup Simulation
        # Convert to SI
        df_si = df.copy()
        for col in ['x', 'y', 'z', 'radius']:
            df_si[col] *= 1e-6
            
        sdf_func = oracle.get_sdf(df_si)
        
        # Init Func: Uniform in voxel
        voxel_size = 20e-6
        def init_func(key, n):
            return jax.random.uniform(key, (n, 3), minval=0.0, maxval=voxel_size)
            
        sim_fn = simulate_ground_truth(sdf_func, init_func)
        
        # Define Waveform (PGSE)
        # G = 80 mT/m
        # delta = 10 ms
        # Delta = 20 ms
        # b ~ 2600 s/mm^2
        G_val = 0.08 # T/m
        delta = 10e-3
        Delta = 20e-3
        TE = 0.05
        
        # Construct waveform (N_steps, 3)
        # Gradient along Y (perpendicular to X-axis axons)
        direction = jnp.array([0., 1., 0.])
        
        N_steps = 100
        dt = TE / N_steps
        
        # Simple block pulse
        # Rise for delta, wait, Fall (neg) for delta
        # Simplified:
        # 0..delta: G
        # delta..Delta: 0
        # Delta..Delta+delta: -G (if using 180 refoc is handled by signal eq? No, standard PGSE uses -G or effective G)
        # dmipy-jax simulator accumulates phase `gamma * x * g * dt`.
        # Standard PGSE (spin echo) effectively reverses phase accumulation.
        # We can mimic this by flipping gradient sign in 2nd lobe.
        
        times = jnp.linspace(0, TE, N_steps)
        waveform = jnp.zeros((N_steps, 3))
        
        for i, t in enumerate(times):
            if t < delta:
                waveform = waveform.at[i].set(direction * G_val)
            elif t > Delta and t < (Delta + delta):
                waveform = waveform.at[i].set(direction * -G_val)
                
        # Simulate
        key = jax.random.PRNGKey(42)
        print("Simulating Diffusion...")
        # 5000 walkers
        signal = sim_fn(waveform, 2.0e-9, dt, 5000, key)
        print(f"Signal (Amp={amp}): {signal}")
        
        results.append({'amplitude': amp, 'signal': float(signal)})
        
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv('experiments/remidi_connection/beading_results.csv', index=False)
    print("Results saved.")
    print(df_res)
    
if __name__ == '__main__':
    main()

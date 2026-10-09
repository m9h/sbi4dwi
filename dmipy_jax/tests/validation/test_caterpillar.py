
import pytest
import jax
import jax.numpy as jnp
import pandas as pd
from dmipy_jax.simulation.substrates.caterpillar import CATERPillarOracle
from dmipy_jax.simulation.monte_carlo import simulate_ground_truth

def test_caterpillar_integration():
    # 1. Generate substrate
    # Using explicit path if running from root
    binary_path = "/home/mhough/dev/dmipy/vendor/CATERPillar/Caterpillar"
    oracle = CATERPillarOracle(binary_path=binary_path)
    
    config = oracle.get_default_config()
    config['vox_sizes'] = [5.0] # Small voxel for speed
    config['axons_without_myelin_icvf'] = 0.1 # Low density
    config['nbr_threads'] = 1
    
    try:
        df = oracle.generate(config)
    except RuntimeError as e:
        pytest.fail(f"CATERPillar generation failed: {e}")

    assert not df.empty
    assert 'x' in df.columns
    assert 'radius' in df.columns
    
    print(f"Generated {len(df)} spheres.")
    
    # 2. SDF
    sdf_func = oracle.get_sdf(df)
    
    # Check SDF at center
    # Just ensure it runs and returns a scalar
    pos = jnp.array([2.5, 2.5, 2.5])
    dist = sdf_func(pos)
    assert dist.ndim == 0
    
    # 3. Simulation verification
    # Init walkers uniformly in [0, 5]^3
    def initialization_func(key, n_particles):
        return jax.random.uniform(key, (n_particles, 3), minval=0.0, maxval=5.0)
    
    # Compile the simulator
    sim_fn = simulate_ground_truth(sdf_func, initialization_func)
    
    # Create simple PGSE-like waveform
    # 100 steps, dt=0.5ms. Total 50ms.
    # Gradient on 10ms-20ms and 30ms-40ms (PGSE)
    dt = 0.5e-3
    steps = 100
    g_val = 0.04 # T/m approx
    waveform = jnp.zeros((steps, 3))
    # Pulse 1
    waveform = waveform.at[20:40, 0].set(g_val)
    # Pulse 2 (refocused effectively, but for phase accumulation we simulate gradients)
    # Standard MC usually simulates phase accumulation.
    # For PGSE, we have G, then -G (after 180) or effective gradient.
    # Let's just do a simple bipolar gradient: +G then -G.
    waveform = waveform.at[20:40, 0].set(g_val)
    waveform = waveform.at[60:80, 0].set(-g_val)
    
    key = jax.random.PRNGKey(42)
    # Run with small N for test speed
    signal = sim_fn(waveform, D=2.0e-9, dt=dt, N_particles=500, key=key) # D in m^2/s ~ 2 um^2/ms = 2e-9 m^2/s
    
    print(f"Simulated signal: {signal}")
    
    # Signal should be <= 1 (attenuation)
    assert signal <= 1.0 + 1e-6
    assert signal >= 0.0

if __name__ == "__main__":
    test_caterpillar_integration()

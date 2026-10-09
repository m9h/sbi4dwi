
import jax
import jax.numpy as jnp
import h5py
import numpy as np
import os
from dmipy_jax.simulation.substrates.caterpillar import CATERPillarOracle
from dmipy_jax.simulation.monte_carlo import simulate_trajectories

def main():
    print("Generating trajectories for KomaMRI...")
    # 1. Generate Substrate
    oracle = CATERPillarOracle()
    config = oracle.get_default_config()
    # Configure density
    # Default config usually is adequate.
    # Ensure config has reasonable size
    # Typically 10x10x10 um.
    df = oracle.generate(config)
    
    # 2. SDF (SI units)
    df_si = df.copy()
    columns_to_scale = ['x', 'y', 'z', 'radius']
    for col in columns_to_scale:
        df_si[col] *= 1e-6
        
    sdf_func = oracle.get_sdf(df_si)
    
    # 3. Simulate Trajectories
    # Init: 1000 walkers (demo)
    N_particles = 1000
    N_steps = 100
    dt = 1e-4 # 0.1 ms -> 10ms total
    D = 2.0e-9 
    
    voxel_size = 10e-6
    def init_func(key, n):
        # Sample anywhere in voxel
        return jax.random.uniform(key, (n, 3), minval=0.0, maxval=voxel_size)
    
    sim_traj_fn = simulate_trajectories(sdf_func, init_func)
    
    key = jax.random.PRNGKey(42)
    
    print(f"Simulating {N_particles} particles for {N_steps} steps (dt={dt}s)...")
    init_pos, trajectories = sim_traj_fn(D, dt, N_particles, N_steps, key)
    # trajectories: (N_steps, N_particles, 3)
    # init_pos: (N_particles, 3)
    
    # 4. Save to H5
    output_path = 'experiments/koma_integration/trajectories.h5'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving to {output_path}...")
    
    # Convert to numpy
    trajectories_np = np.array(trajectories)
    init_pos_np = np.array(init_pos)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('trajectories', data=trajectories_np)
        f.create_dataset('initial_positions', data=init_pos_np)
        f.attrs['dt'] = dt
        f.attrs['N_particles'] = N_particles
        f.attrs['N_steps'] = N_steps
        f.attrs['voxel_size'] = voxel_size
    
    print("Done.")

if __name__ == '__main__':
    main()

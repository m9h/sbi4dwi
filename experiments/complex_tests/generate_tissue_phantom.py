
import jax
import jax.numpy as jnp
import h5py
import numpy as np
import os
import pandas as pd
from dmipy_jax.simulation.substrates.caterpillar import CATERPillarOracle
from dmipy_jax.simulation.monte_carlo import simulate_trajectories
from dmipy_jax.simulation.sphere_sdf import MultiSphereSDF

def main():
    print("--- Multi-Compartment Tissue Phantom Generation ---")
    
    output_dir = "experiments/complex_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate Geometry (CATERPillar)
    oracle = CATERPillarOracle()
    config = oracle.get_default_config()
    # High density for interesting extra-axonal space
    config['vox_sizes'] = [15.0] 
    config['axons_without_myelin_icvf'] = 0.6  # High ICVF
    config['min_rad'] = 0.4
    
    print("Generating geometry...")
    df = oracle.generate(config)
    
    # Save geometry info
    df.to_csv(os.path.join(output_dir, "phantom_geometry.csv"), index=False)
    
    # Convert to SI
    df_si = df.copy()
    for col in ['x', 'y', 'z', 'radius']:
        df_si[col] *= 1e-6
    
    # Create SDF Func
    # CATERPillar provides centers/radii. MultiSphereSDF handles this.
    centers = df_si[['x', 'y', 'z']].values
    radii = df_si['radius'].values
    sdf_obj = MultiSphereSDF(centers, radii)
    sdf_func = sdf_obj.get_sdf_func()
    
    # 2. Simulation Setup
    N_particles_in = 2000
    N_particles_ex = 2000
    N_steps = 200
    dt = 1e-4 # 0.1ms -> 20ms total duration
    D_in = 1.0e-9 
    D_ex = 2.0e-9 # Faster in extra-axonal space (hindered)
    
    voxel_size = 15e-6
    
    # Initialization Function: Uniform in Voxel
    def init_func(key, n):
        return jax.random.uniform(key, (n, 3), minval=0.0, maxval=voxel_size)
    
    # Filter Init Positions
    # We need to ensure IN particles start INSIDE, and EX particles start OUTSIDE.
    # The current `simulate_trajectories` assumes initialization is valid.
    # Since we use uniform random, we might start invalid. Refection handles logic, 
    # but starting invalid might cause instant ejection or weirdness.
    # Better to filter.
    
    # We can handle filtering in Python before passing to simulation?
    # Or implement a smarter init func. 
    # Let's filter in Python for simplicity.
    
    print("Filtering initial positions...")
    key = jax.random.PRNGKey(42)
    # Generate excess candidates
    candidates = jax.random.uniform(key, (N_particles_in * 5, 3), minval=0.0, maxval=voxel_size)
    dists = jax.vmap(sdf_func)(candidates)
    
    # Inside: dist < 0
    mask_in = dists < 0
    pos_in = candidates[mask_in][:N_particles_in]
    if len(pos_in) < N_particles_in:
        raise ValueError(f"Not enough Inside particles found ({len(pos_in)} < {N_particles_in}). Increase candidates.")
        
    # Outside: dist > 0
    mask_ex = dists > 0
    pos_ex = candidates[mask_ex][:N_particles_ex]
    if len(pos_ex) < N_particles_ex:
        raise ValueError(f"Not enough Outside particles found ({len(pos_ex)} < {N_particles_ex}). Increase candidates.")
    
    # Define exact init funcs
    def init_in(key, n):
        # Ignore key, return pre-calculated (hack but works for fixed N)
        return pos_in
        
    def init_ex(key, n):
        return pos_ex

    # 3. Simulate Intra-Axonal (Restricted)
    print("Simulating Intra-Axonal (Restricted)...")
    sim_in = simulate_trajectories(sdf_func, init_in, confinement='inside')
    # N_particles determined by init_in return size implicitly?
    # No, passed as arg.
    _, traj_in = sim_in(D_in, dt, N_particles_in, N_steps, key)
    # traj_in: (N_steps, N_particles, 3)
    
    # 4. Simulate Extra-Axonal (Hindered)
    print("Simulating Extra-Axonal (Hindered)...")
    sim_ex = simulate_trajectories(sdf_func, init_ex, confinement='outside')
    _, traj_ex = sim_ex(D_ex, dt, N_particles_ex, N_steps, key)
    
    # 5. Combine and Export
    print("Exporting...")
    # Combine
    # Shape: (N_steps, N_total, 3)
    traj_all = jnp.concatenate([traj_in, traj_ex], axis=1) # (Steps, N, 3)
    init_all = jnp.concatenate([pos_in, pos_ex], axis=0)   # (N, 3)
    
    # Compartment IDs: 0 = Intra, 1 = Extra
    ids_in = np.zeros(N_particles_in, dtype=np.int32)
    ids_ex = np.ones(N_particles_ex, dtype=np.int32)
    ids_all = np.concatenate([ids_in, ids_ex])
    
    h5_path = os.path.join(output_dir, "tissue_phantom.h5")
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('trajectories', data=np.array(traj_all))
        f.create_dataset('initial_positions', data=np.array(init_all))
        f.create_dataset('compartment_ids', data=ids_all)
        f.attrs['dt'] = dt
        f.attrs['N_particles'] = N_particles_in + N_particles_ex
        f.attrs['N_steps'] = N_steps
        f.attrs['voxel_size'] = voxel_size
        
    print(f"Saved to {h5_path}")
    
if __name__ == '__main__':
    main()

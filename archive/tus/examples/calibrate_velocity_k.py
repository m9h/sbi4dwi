
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import argparse

from dmipy_jax.biophysics.network.connectome_mapper import ConnectomeMapper
from dmipy_jax.io.tms_loader import TMSLoader

def generate_synthetic_data():
    """
    Generates a minimal synthetic dataset for verification.
    Two regions connected by a single bundle of straight fibers.
    """
    print("Generating synthetic calibration data...")
    
    # Volume: 30x30x30
    shape = (30, 30, 30)
    affine = np.eye(4) # 1mm iso
    
    # Parcellation: ROI 1 centered at (10,10,10), ROI 2 at (20,10,10)
    # We make them 3x3x3 blocks to catch noisy streamline endpoints
    parcellation = np.zeros(shape, dtype=int)
    parcellation[9:12, 9:12, 9:12] = 1 # "L_M1"
    parcellation[19:22, 9:12, 9:12] = 2 # "R_M1"
    
    # Region Map
    region_names = {1: "L_M1", 2: "R_M1"}
    
    # Streamlines: 10 lines connecting ROI 1 and 2
    streamlines = []
    for i in range(10):
        # Slightly noisy lines from x=10 to x=20
        x = np.linspace(10, 20, 20)
        y = np.full_like(x, 10) + np.random.normal(0, 0.1, 20)
        z = np.full_like(x, 10) + np.random.normal(0, 0.1, 20)
        streamline = np.stack([x, y, z], axis=1) # (N, 3)
        streamlines.append(streamline)
        
    # Axon Diameter Map
    # Uniform 3.0 um diameter in the path
    diameter_map = np.zeros(shape)
    # Fill the corridor
    diameter_map[10:21, 9:12, 9:12] = 3.0
    
    # Ground Truth Physics
    # Let's say k_true = 6.0 m/s/um
    # v = 6.0 * 3.0 = 18.0 m/s.
    # Dist = 10 mm.
    # Time = 10 mm / 18 m/s = 10 / 18 ms = 0.555 ms.
    
    # Note: connectome_mapper logic checks endpoints vs parcellation.
    # The streamlines technically start exactly at the voxel coordinates, so they should map.
    
    empirical_latencies = {
        ("L_M1", "R_M1"): 0.5555
    }
    
    return streamlines, diameter_map, affine, parcellation, region_names, empirical_latencies

def objective_function(k: float, 
                       streamlines, 
                       diameter_map, 
                       affine, 
                       parcellation, 
                       region_names, 
                       empirical_latencies: Dict[Tuple[str, str], float]) -> float:
    """
    Loss = MSE(Simulated - Empirical)
    """
    if k <= 0: return 1e9
    
    # 1. Calculate Latency Matrix with current k
    # connectome_mapper uses 'base_velocity' arg as 'k' based on our implementation update
    n_regions = max(region_names.keys()) + 1
    
    simulated_delays = ConnectomeMapper.map_microstructure_to_velocity(
        streamlines=streamlines,
        diameter_map=diameter_map,
        affine=affine,
        parcellation=parcellation,
        n_regions=n_regions,
        base_velocity=float(k)
    )
    
    # 2. Compare with Empirical
    errors = []
    for (src_name, tgt_name), emp_lat in empirical_latencies.items():
        # Find IDs
        src_id = next((id for id, name in region_names.items() if name == src_name), None)
        tgt_id = next((id for id, name in region_names.items() if name == tgt_name), None)
        
        if src_id is None or tgt_id is None:
            continue
            
        # Matrix is 0-indexed, IDs are 1-based (usually)
        # connectome_mapper subtracts 1 for indices.
        sim_lat = simulated_delays[src_id-1, tgt_id-1]
        
        # If simulation found no connection (0), penalize?
        if sim_lat == 0:
            # Only penalize if we expected a connection
            # But maybe the tractography missed it. 
            # For calibration, we only look at existing tracts.
            continue
            
        errors.append((sim_lat - emp_lat)**2)
        
    if not errors:
        return 0.0
        
    mse = np.mean(errors)
    return mse

def main():
    parser = argparse.ArgumentParser(description="Calibrate Hursh-Rushton 'k' using TMS-EEG latencies.")
    parser.add_argument("--use-synthetic", action="store_true", default=False, help="Use synthetic data verification mode.")
    args = parser.parse_args()
    
    # In a real run, we would load data here.
    # For this agent execution, we default to synthetic if no valid path exists or if requested.
    # Since we have no real data paths configured yet:
    print("Initializing Calibration Agent...")
    
    # Force synthetic for now as default mechanism for the agent validation
    # Real loading logic would go here.
    streamlines, diameter_map, affine, parcellation, region_names, empirical_latencies = generate_synthetic_data()
    
    print(f"Empirical Target Latency: {empirical_latencies}")
    
    # Optimization
    print("Optimizing k...")
    result = scipy.optimize.minimize_scalar(
        objective_function, 
        bounds=(1.0, 15.0), # Physically plausible range for mammal brains
        args=(streamlines, diameter_map, affine, parcellation, region_names, empirical_latencies),
        method='bounded'
    )
    
    if result.success:
        print(f"Calibration Successful!")
        print(f"Optimal k: {result.x:.4f} m/s/um")
        
        # Verify
        k_opt = result.x
        n_regions = max(region_names.keys()) + 1
        final_delays = ConnectomeMapper.map_microstructure_to_velocity(
             streamlines, diameter_map, affine, parcellation, n_regions, base_velocity=k_opt
        )
        # Check specific link
        sim_val = final_delays[0, 1] # ID 1->2 is index 0->1
        print(f"Simulated Latency at optimum: {sim_val:.4f} ms")
        print(f"Target: {empirical_latencies[('L_M1', 'R_M1')]:.4f} ms")
        
        diff = abs(sim_val - empirical_latencies[('L_M1', 'R_M1')])
        if diff < 0.01:
            print("VERIFICATION PASS: Latency matched ground truth.")
        else:
            print(f"VERIFICATION FAIL: Residual error {diff:.4f} ms")
            
    else:
        print(f"Optimization Failed: {result.message}")

if __name__ == "__main__":
    main()

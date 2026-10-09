
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from validation.lib.histology import HistoDataset, HistologySimulator, histology_loss
from dmipy_jax.signal_models import cylinder_models, gaussian_models
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme

def main():
    print("Running Differentiable Histology Bridge Demo...")
    
    # 1. Load Data
    loader = HistoDataset()
    # Forces synthetic if download fails logic inside
    signal_measured, histo_gt = loader.load_data()
    
    print("Histology Ground Truth Loaded:")
    for k, v in histo_gt.items():
        print(f"  {k}: shape {v.shape}")

    # 2. Define Acquisition
    # Create a synthetic acquisition if none present
    bvals = jnp.array([0.0, 1000.0, 1000.0, 2000.0])
    bvecs = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    
    # delta (small_delta) and Delta (big_delta) are needed for RestrictedCylinder
    delta = jnp.full(bvals.shape, 0.01) # 10ms
    Delta = jnp.full(bvals.shape, 0.03) # 30ms
    
    acq = SimpleAcquisitionScheme(bvalues=bvals, gradient_directions=bvecs, delta=delta, Delta=Delta)

    # 3. Define Models
    # 'True' physical model for histology simulation
    sim_model = cylinder_models.RestrictedCylinder()
    histo_sim = HistologySimulator(model=sim_model)

    # 'Candidate' MRI model we want to validate
    mri_model = cylinder_models.RestrictedCylinder()
    
    # 4. Simulate 'True' Signal from Histology
    print("Simulating signals from Histology GT...")
    
    # Let's flatten
    radius_flat = histo_gt['radius'].reshape(-1)
    density_flat = histo_gt['density'].reshape(-1)
    
    # Helper to run model on single voxel params
    def run_sim(r, d):
        params = {
            "diameter": r * 2 * 1e-6, # Convert microns to meters
            "lambda_par": 1.7e-3, # 1.7e-3 mm^2/s (Standard units for dmipy) 
            "mu": jnp.array([0.0, 0.0]) # Orientation (theta, phi) - Aligned with Z-axis
        }
        # Unpack acquisition
        # RestrictedCylinder(bvals, bvecs, **kwargs)
        return sim_model(acq.bvalues, acq.gradient_directions, 
                         big_delta=acq.Delta[0], small_delta=acq.delta[0], 
                         **params)
    
    # Vectorize over voxels
    vmap_sim = jax.vmap(run_sim)
    s_histo_pred = vmap_sim(radius_flat, density_flat)
    
    print(f"Predicted Signal Shape: {s_histo_pred.shape}")
    
    # 5. Compute Loss against an MRI model fit (or just params)
    # Let's say we have some estimated MRI params and we want to see if they match histology predictions.
    # For demonstration, let's perturb the GT and calculate loss.
    
    mri_radius_est = radius_flat * 0.9 # Biased estimate
    mri_density_est = density_flat
    
    # Define MRI params dict
    mri_params = {
        "diameter": mri_radius_est * 2 * 1e-6,
        "lambda_par": jnp.full_like(mri_radius_est, 1.7e-3), # mm^2/s
        "mu": jnp.zeros(mri_radius_est.shape + (2,)) # Orientation per voxel
    }
    
    # MRI Model prediction:
    def run_mri(d, l, m):
        return mri_model(acq.bvalues, acq.gradient_directions, 
                         big_delta=acq.Delta[0], small_delta=acq.delta[0], 
                         diameter=d, lambda_par=l, mu=m)
        
    vmap_mri = jax.vmap(run_mri)
    s_mri_pred = vmap_mri(mri_params['diameter'], mri_params['lambda_par'], mri_params['mu'])
    
    loss = jnp.mean((s_mri_pred - s_histo_pred)**2)
    
    print(f"Validation Loss (MSE): {loss:.6f}")
    
    # 6. Gradient Check (Differentiability)
    print("Checking Differentiability...")
    
    def loss_fn(r_est):
        p = {
            "diameter": r_est * 2 * 1e-6,
            "lambda_par": jnp.full_like(r_est, 1.7e-3),
            "mu": jnp.zeros(r_est.shape + (2,))
        }
        # Predict mri
        s_m = vmap_mri(p['diameter'], p['lambda_par'], p['mu'])
        # Compare to fixed histo ground truth signal
        return jnp.mean((s_m - s_histo_pred)**2)
        
    grad_fn = jax.grad(loss_fn)
    grad_val = grad_fn(mri_radius_est)
    
    print(f"Gradient norm w.r.t radius: {jnp.linalg.norm(grad_val):.6f}")
    
    # 7. Visualization (if display available)
    try:
        if os.environ.get("DISPLAY"):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Histo Ground Truth Radius")
            plt.imshow(histo_gt['radius'][:,:,0])
            plt.colorbar()
            
            plt.subplot(1, 2, 2)
            plt.title("Signal Error (Abs)")
            # Reshape error
            err = jnp.abs(s_mri_pred - s_histo_pred).mean(axis=1).reshape(10,10)
            plt.imshow(err)
            plt.colorbar()
            plt.savefig("demo_histology_bridge.png")
            print("Saved demo_histology_bridge.png")
    except Exception as e:
        print(f"Skipping visualization: {e}")

    print("Success.")

if __name__ == "__main__":
    main()

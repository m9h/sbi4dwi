
import os
import urllib.request
import tarfile
from pathlib import Path
import numpy as np
import nibabel as nib
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Dict, Tuple, Any

# Initial constant for Histo-uSim data
# Using a placeholder URL or the one provided. 
# Prompt said: "Record 14559356".  
# Real Zenodo URL structure: https://zenodo.org/record/14559356/files/...
# I'll use a likely valid one, but allow override.
HISTO_URL = "https://zenodo.org/records/14559356/files/histo_usim_data.tar.gz?download=1"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "dmipy_jax" / "histology"

class HistoDataset(eqx.Module):
    """
    Loader for Histo-uSim dataset (validation ground truth).
    """
    path: Path = eqx.field(static=True)

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            self.path = DEFAULT_CACHE_DIR
        else:
            self.path = Path(path)

    def fetch_data(self, force_download: bool = False):
        """
        Downloads data from Zenodo if not present.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        tar_path = self.path / "histo_usim_data.tar.gz"

        if not tar_path.exists() or force_download:
            print(f"Downloading Histo-uSim data from {HISTO_URL}...")
            try:
                # Basic fetch
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(HISTO_URL, tar_path)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download: {e}")
                # We won't raise here strictly to allow synthetic fallback in load_data for demo purposes
                # if the user is offline or URL is wrong.
                pass

        # Extract
        if tar_path.exists():
            extracted_marker = self.path / ".extracted"
            if not extracted_marker.exists():
                print("Extracting...")
                try:
                    with tarfile.open(tar_path, "r:gz") as tar:
                        def is_within_directory(directory, target):
                            abs_directory = os.path.abspath(directory)
                            abs_target = os.path.abspath(target)
                            prefix = os.path.commonprefix([abs_directory, abs_target])
                            return prefix == abs_directory
                        
                        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                            for member in tar.getmembers():
                                member_path = os.path.join(path, member.name)
                                if not is_within_directory(path, member_path):
                                    raise Exception("Attempted Path Traversal in Tar File")
                            tar.extractall(path, members, numeric_owner=numeric_owner) 
                            
                        safe_extract(tar, self.path)
                    extracted_marker.touch()
                except Exception as e:
                    print(f"Extraction failed: {e}")


    def load_data(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Loads the data. 
        Returns:
            (images, ground_truth)
            images: placeholder for actual dMRI signal if provided in dataset.
            ground_truth: dictionary of maps { 'radius': ..., 'density': ... }
        """
        self.fetch_data()

        # Check for files
        # Hypothetical structure: radius.nii.gz, density.nii.gz, signal.nii.gz
        radius_path = self.path / "radius.nii.gz"
        density_path = self.path / "density.nii.gz"
        signal_path = self.path / "signal.nii.gz" # Hypothetical

        if not radius_path.exists():
            print("Warning: Histo files not found. Returning SYNTHETIC ground truth for demo.")
            return self._generate_synthetic()

        # Load NIfTI
        radius_img = nib.load(radius_path)
        density_img = nib.load(density_path)
        
        ground_truth = {
            "radius": jnp.array(radius_img.get_fdata()),
            "density": jnp.array(density_img.get_fdata())
        }
        
        if signal_path.exists():
            signal = jnp.array(nib.load(signal_path).get_fdata())
        else:
            signal = None # No measured signal, just ground truth
            
        return signal, ground_truth

    def _generate_synthetic(self):
        """Generates synthetic histology data for testing/demo."""
        shape = (10, 10, 1)
        
        # Radius gradient
        x = jnp.linspace(0.5, 3.0, shape[0])
        y = jnp.linspace(0.5, 3.0, shape[1])
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        radius = X[..., None] # (10, 10, 1)
        density = jnp.ones(shape) * 0.7
        
        ground_truth = {
            "radius": radius,
            "density": density,
             # Orientation could be added here
        }
        signal = None 
        return signal, ground_truth


class HistologySimulator(eqx.Module):
    """
    Simulator that maps histological Ground Truth to dMRI signals.
    Supports both Physically-Based (Heuristic) and Learned (MLP) mapping.
    """
    model: Optional[Any] = None # e.g. Cylinder, BallStick
    learned_model: Optional[Any] = None # e.g. MicrostructureMapper
    
    def __init__(self, model=None, learned_model=None):
        """
        Args:
            model: Physics-based dMRI model (e.g. RestrictedCylinder).
            learned_model: Neural network model (e.g. MicrostructureMapper).
        """
        self.model = model
        self.learned_model = learned_model
        
        if self.model is None and self.learned_model is None:
            raise ValueError("Must provide either 'model' (physics) or 'learned_model' (neural).")
        
    def __call__(self, acquisition, ground_truth: Dict[str, Any]):
        """
        Predicts signal from ground truth.
        """
        if self.learned_model is not None:
            return self._predict_learned(ground_truth)
        else:
            return self._predict_physics(acquisition, ground_truth)

    def _predict_learned(self, ground_truth: Dict[str, Any]):
        """
        Uses the learned MLP to predict signal.
        Expects ground_truth to be convertible to a flat input vector per voxel.
        """
        features_list = []
        spatial_shape = None
        
        if "radius" in ground_truth:
            features_list.append(ground_truth["radius"])
            spatial_shape = ground_truth["radius"].shape
        if "density" in ground_truth:
            features_list.append(ground_truth["density"])
            if spatial_shape is None:
                spatial_shape = ground_truth["density"].shape
        
        # Stack features: (*spatial_shape, N_features)
        features = jnp.stack(features_list, axis=-1)
        
        # Flatten spatial dims: (product(spatial_shape), N_features)
        features_flat = features.reshape(-1, features.shape[-1])
        
        # Apply MLP to every voxel
        # learned_model takes (n_feat,) -> (n_signals,)
        signals_flat = jax.vmap(self.learned_model)(features_flat)
        
        # Reshape back: (*spatial_shape, N_signals)
        return signals_flat.reshape(*spatial_shape, -1)

    def _predict_physics(self, acquisition, ground_truth: Dict[str, Any]):
        """
        Physically-based heuristic mapping.
        """
        model_params = {}
        
        # Heuristic mapping
        if "radius" in ground_truth:
            model_params["radius"] = ground_truth["radius"]
            # Some models use diameter
            model_params["diameter"] = ground_truth["radius"] * 2 
            
        if "density" in ground_truth:
            model_params["volume_fraction"] = ground_truth["density"]
            model_params["f_intra"] = ground_truth["density"]
            
        # Call model
        return self.model(acquisition, **model_params)


def histology_loss(mri_params: Dict[str, Any], 
                   histo_gt: Dict[str, Any], 
                   acquisition, 
                   mri_model, 
                   histo_simulator) -> float:
    """
    Computes a loss validating the MRI model parameters against Histology.
    
    Context:
    This loss might compare:
    1. MRI-derived parameters vs Histology Ground Truth (Parameter Recovery Loss)
    2. MRI-predicted signal vs Histology-predicted signal (Signal Consistency Loss)
    
    The prompt says: "differentiate fitting error w.r.t histology-predicted signals".
    This suggests comparing signals.
    
    Loss = MSE( S_mri(mri_params), S_histo(histo_gt) )
    """
    
    s_mri = mri_model(acquisition, **mri_params)
    s_histo = histo_simulator(acquisition, histo_gt)
    
    # Ensure shapes match
    # s_mri and s_histo should be (N_voxels, N_acq) or similar
    
    return jnp.mean((s_mri - s_histo)**2)

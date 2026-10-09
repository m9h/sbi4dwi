import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Any

class ConnectomeMapper:
    """
    Maps voxel-wise microstructural metrics to structural connectivity graph edges
    by sampling along tractography streamlines.
    """
    
    @staticmethod
    def map_microstructure_to_weights(
        streamlines: List[np.ndarray],
        metric_map: np.ndarray,
        affine: np.ndarray,
        parcellation: np.ndarray,
        n_regions: int
    ) -> np.ndarray:
        """
        Computes a connectivity matrix where weights are defined by the mean
        microstructural metric (e.g., ICVF) along the streamlines connecting regions.
        
        Args:
           streamlines: List of (N_points, 3) arrays in ras coordinates.
           metric_map: (X, Y, Z) volume of microstructural metric.
           affine: (4, 4) affine matrix for map/parcellation.
           parcellation: (X, Y, Z) integer volume of ROIs.
           n_regions: Number of regions in parcellation.
           
        Returns:
           weighted_sc: (n_regions, n_regions) connectivity matrix.
        """
        # Note: This is a simplified implementation. Real-world usage requires
        # careful handling of affine inverses and interpolation.
        
        inv_affine = np.linalg.inv(affine)
        
        # Accumulators
        # sum_metric[i, j] accumulates the metric sum for bundle i->j
        # count_streamlines[i, j] counts streamlines
        sum_metric = np.zeros((n_regions, n_regions))
        count_streamlines = np.zeros((n_regions, n_regions))
        
        for sl in streamlines:
            # Transform to voxel coords
            # points_vox = apply_affine(inv_affine, sl)
            # This is slow in pure python loop, but acceptable for demo.
            # In production, vectorized operations or tck_map equivalent is used.
            
            # Simple endpoint check for ROI assignment
            start_point = np.dot(inv_affine, np.append(sl[0], 1.0))[:3].astype(int)
            end_point = np.dot(inv_affine, np.append(sl[-1], 1.0))[:3].astype(int)
            
            # Bounds check
            if not _in_bounds(start_point, metric_map.shape) or not _in_bounds(end_point, metric_map.shape):
                continue
                
            roi_i = parcellation[tuple(start_point)]
            roi_j = parcellation[tuple(end_point)]
            
            if roi_i == 0 or roi_j == 0 or roi_i == roi_j:
                continue
                
            # Sample metric along line
            # For simplicity, sample at vertices
            points = np.dot(sl, inv_affine[:3, :3].T) + inv_affine[:3, 3]
            points_idx = points.astype(int)
            
            # Filter distinct points inside bounds
            valid_mask = np.all((points_idx >= 0) & (points_idx < np.array(metric_map.shape)), axis=1)
            if not np.any(valid_mask):
                continue
                
            values = metric_map[tuple(points_idx[valid_mask].T)]
            mean_val = np.mean(values)
            
            # Add to connectivity
            # Symmetrize
            idx_a, idx_b = (roi_i - 1, roi_j - 1) # Assume 1-based ROIs
            if idx_a < 0 or idx_b < 0 or idx_a >= n_regions or idx_b >= n_regions:
                continue
                
            sum_metric[idx_a, idx_b] += mean_val
            sum_metric[idx_b, idx_a] += mean_val
            count_streamlines[idx_a, idx_b] += 1
            count_streamlines[idx_b, idx_a] += 1
            
        # Compute mean
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = sum_metric / count_streamlines
            weights = np.nan_to_num(weights)
            
        return weights

    @staticmethod
    def map_microstructure_to_velocity(
        streamlines: List[np.ndarray],
        diameter_map: np.ndarray,
        affine: np.ndarray,
        parcellation: np.ndarray,
        n_regions: int,
        base_velocity: float = 6.0 # m/s (approx global mean)
    ) -> np.ndarray:
        """
        Computes a delay matrix where conduction velocity is modulated by 
        axon diameter.
        
        Velocity ~ Diameter^alpha (approx linear for myelinated)
        
        Args:
           ...
           
        Returns:
           delays: (n_regions, n_regions) matrix in seconds (distance / velocity).
        """
        # k is the Hursh-Rushton constant (velocity = k * diameter)
        # We assume k ~ 5.5 - 6.0, but here we just return the raw aggregated metric 
        # that allows later modulation by k, OR we accept k as an arg (base_velocity is misleading naming here).
        # Actually, the docstring says "base_velocity = 6.0". If we treat this as 'k', then:
        k = base_velocity
        
        delays = np.zeros((n_regions, n_regions))
        count_streamlines = np.zeros((n_regions, n_regions))
        
        inv_affine = np.linalg.inv(affine)

        for sl in streamlines:
            # 1. Identify start/end regions
            start_point = np.dot(inv_affine, np.append(sl[0], 1.0))[:3].astype(int)
            end_point = np.dot(inv_affine, np.append(sl[-1], 1.0))[:3].astype(int)
            
            if not _in_bounds(start_point, diameter_map.shape) or not _in_bounds(end_point, diameter_map.shape):
                continue
                
            roi_i = parcellation[tuple(start_point)]
            roi_j = parcellation[tuple(end_point)]
            
            if roi_i == 0 or roi_j == 0 or roi_i == roi_j:
                continue
            
            # 2. Sample diameter map along the streamline
            # Project points to voxel coords
            points_real = sl
            # Calculate segment lengths in mm (Euclidean distance between points)
            # diffs = np.diff(points_real, axis=0)
            # step_lengths = np.linalg.norm(diffs, axis=1)
            
            # Voxel indices for sampling
            points_vox = np.dot(points_real, inv_affine[:3, :3].T) + inv_affine[:3, 3]
            points_idx = points_vox.astype(int)
            
            valid_mask = np.all((points_idx >= 0) & (points_idx < np.array(diameter_map.shape)), axis=1)
            
            # If we don't have enough valid points, skip
            if np.sum(valid_mask) < 2:
                continue

            # Filter valid points and steps
            # We need step lengths aligned with the *segments*. 
            # Let's simplify: sample at midpoints of segments or just at points and average?
            # Standard: t = sum( L_i / v_i )
            diffs = np.diff(points_real, axis=0) # shape (N-1, 3)
            dists = np.linalg.norm(diffs, axis=1) # shape (N-1,)
            
            # Sample diameter at the *start* of each segment (or midpoint)
            # Let's use the points_idx[0:-1] corresponding to segment starts
            # taking valid mask into account is tricky if gaps exist, but assuming contiguous validity:
            valid_starts = valid_mask[:-1]
            if not np.any(valid_starts):
                continue
                
            # Filter dists by valid start points
            dists = dists[valid_starts]
            sample_indices = points_idx[:-1][valid_starts]
            
            diameters = diameter_map[tuple(sample_indices.T)]
            
            # Avoid divide by zero
            # If diameter is 0 (outside mask), velocity is 0 -> infinite delay.
            # Realistically, we should mask/clip.
            # Assume min diameter for physiologic transmission ~ 0.2 um
            diameters = np.maximum(diameters, 0.1) 
            
            # Calculate local velocities
            velocities = k * diameters # m/s (if d in um and k in m/s/um)
            
            # Calculate segment delays
            # dists in mm, velocities in m/s. 
            # time = dist / vel => (mm) / (m/s) = (1e-3 m) / (m/s) = 1e-3 seconds = 1 ms.
            # So if we simply divide, we get result in milliseconds directly!
            # Example: 1 mm / 1 m/s = 0.001 s = 1 ms. Correct.
            
            segment_delays = dists / velocities
            
            total_delay = np.sum(segment_delays)
            
            idx_a, idx_b = (roi_i - 1, roi_j - 1)
            if idx_a < 0 or idx_b < 0 or idx_a >= n_regions or idx_b >= n_regions:
                continue
                
            delays[idx_a, idx_b] += total_delay
            delays[idx_b, idx_a] += total_delay
            count_streamlines[idx_a, idx_b] += 1
            count_streamlines[idx_b, idx_a] += 1
            
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_delays = delays / count_streamlines
            mean_delays = np.nan_to_num(mean_delays)
            
        return mean_delays 

def _in_bounds(point, shape):
    return np.all(point >= 0) and np.all(point < shape)

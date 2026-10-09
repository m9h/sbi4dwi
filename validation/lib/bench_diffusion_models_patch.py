
"""
Patched version of bench/diffusion_models.py to remove Numba dependency.
Only contains models used for validation: ball, stick, cigar.
"""
import numpy as np
import warnings
from scipy import stats

# Removed numba imports and decorators

# helper functions:
def spherical2cart(theta, phi, r=1):
    """
    Converts spherical to cartesian coordinates
    :param theta: angel from z axis
    :param phi: angle from x axis
    :param r: radius
    :return: tuple [x, y, z]-coordinates
    """
    z = r * np.cos(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    return x, y, z


def cart2spherical(n):
    """
    Converts to spherical coordinates
    :param n: (:, 3) array containing vectors in (x,y,z) coordinates
    :return: tuple with (phi, theta, r)-coordinates
    """
    r = np.sqrt(np.sum(n ** 2, axis=1))
    theta = np.arccos(n[:, 2] / r)
    phi = np.arctan2(n[:, 1], n[:, 0])
    phi[r == 0] = 0
    theta[r == 0] = 0
    return r, phi, theta

# compartment definitions:
def ball(bval, bvec, d_iso, s0=1.):
    """
    Simulates diffusion signal for isotropic diffusion
    """
    s0, d_iso = [np.asanyarray(v)[..., np.newaxis] for v in (s0, d_iso)]
    if not np.all(s0 >= 0):
        warnings.warn('s0 cant be negative')
    if not np.all(d_iso >= 0):
        warnings.warn('d_iso cant be negative')
    if not np.isscalar(bval):
        # Handle broadcasting if bval is array
        if bval.ndim == 1 and d_iso.ndim == 2 and d_iso.shape[1] == 1:
             # bval (M,), d_iso (1,1) -> (M,1)? No.
             # If d_iso is scalar wrapped as (1,1), and bval is (M,).
             # We want output (M,).
             # bval * d_iso -> (M,1).
             pass
        # Original code:
        # bval = bval * np.ones(bvec.shape[0])
        pass

    return s0 * np.exp(-bval * d_iso)


def stick(bval, bvec, d_a, theta, phi, s0=1.0):
    """
    Simulates diffusion signal from single stick model
    """
    # Ensure inputs are arrays
    # s0, d_a, theta, phi are usually scalars or matching shape
    # If they are scalars, wrap them?
    # Original code: [np.asarray(v)[..., np.newaxis] for v in (s0, d_a, theta, phi)]
    # This adds a dimension. If bval is (M,), result is (M,1)?
    
    # Let's simplify for validation where params are scalars.
    
    s0 = np.atleast_1d(s0)
    d_a = np.atleast_1d(d_a)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    orientation = np.array(spherical2cart(theta, phi)).T # (N_params, 3)
    
    # bvec (M,3)
    # orientation (1, 3)
    # dot -> (M, 1) if broadcast?
    # orientation.dot(bvec.T) -> (1, 3) dot (3, M) -> (1, M).
    
    # Adjust for shapes
    if orientation.shape[0] == 1 and bvec.ndim == 2:
         # simple case
         dot_prod = bvec @ orientation[0] # (M,)
    else:
         # Generalized
         dot_prod = np.sum(orientation * bvec, axis=-1) # (M,) if orientation matches bvec? No.
         # For validation, we have 1 fibre, M measurements.
         dot_prod = bvec @ orientation.T # (M, N_param)
         
    term = (dot_prod ** 2) # (M, N_param)
    
    # bval (M,)
    # d_a (N_param,)
    
    # term (M, N_param)
    # bval[:, None] * d_a[None, :] * term ??
    
    # If N_param=1:
    if d_a.size == 1:
        return s0 * np.exp(-bval * d_a * term.flatten())
        
    return s0 * np.exp(-bval[:, None] * d_a * term)


def cigar(bval, bvec, d_a, d_r, theta=0., phi=0, s0=1.0):
    """
    Simulates diffusion signal from cigar model
    """
    s0 = np.atleast_1d(s0)
    d_a = np.atleast_1d(d_a)
    d_r = np.atleast_1d(d_r)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    
    orientation = np.array(spherical2cart(theta, phi)).T # (N_param, 3)
    
    # dot product (bvec . n)^2
    # bvec (M,3)
    if orientation.shape[0] == 1:
        dot_prod = bvec @ orientation[0] # (M,)
        dot_sq = dot_prod ** 2
        
        # d_a, d_r scalars
        diff_term = d_r + (d_a - d_r) * dot_sq
        return s0 * np.exp(-bval * diff_term)
        
    else:
        # Not needed for current validation
        pass


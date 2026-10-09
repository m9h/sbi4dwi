import pytest

meshio = pytest.importorskip("meshio")
import jax.numpy as jnp
import numpy as np
import os
from dmipy_jax.io.jinns_mesh import load_mesh_to_jax

def test_load_mesh_to_jax(tmp_path):
    """Test loading a synthetic mesh into JAX arrays."""
    
    # 1. Create a synthetic mesh
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    
    # Tetrahedra: one cell connecting 4 points
    cells = [
        ("tetra", np.array([[0, 1, 2, 3]]))
    ]
    
    mesh = meshio.Mesh(points, cells)
    test_file = tmp_path / "test_mesh.vtk"
    mesh.write(test_file)
    
    # 2. Load with our function
    mesh_jax = load_mesh_to_jax(test_file)
    
    # 3. Verify
    assert "points" in mesh_jax
    assert "cells" in mesh_jax
    
    # Check points
    assert isinstance(mesh_jax["points"], jnp.ndarray)
    assert mesh_jax["points"].shape == (4, 3)
    np.testing.assert_allclose(mesh_jax["points"], points)
    
    # Check cells
    assert "tetra" in mesh_jax["cells"]
    assert isinstance(mesh_jax["cells"]["tetra"], jnp.ndarray)
    assert mesh_jax["cells"]["tetra"].shape == (1, 4)
    np.testing.assert_array_equal(mesh_jax["cells"]["tetra"], np.array([[0, 1, 2, 3]]))

if __name__ == "__main__":
    # validation runner
    import sys
    sys.exit(pytest.main(["-v", __file__]))

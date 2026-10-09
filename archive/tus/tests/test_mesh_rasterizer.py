"""
Tests for tetrahedral mesh to regular grid rasterization.

RED phase: Define contracts for point-in-tetrahedron test and full mesh rasterization
needed to convert the SCI head model into a voxel grid for j-Wave / k-Wave simulation.
"""

import pytest
import numpy as np
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Phase 2.1: Point-in-Tetrahedron
# ---------------------------------------------------------------------------

class TestPointInTetrahedron:
    """Tests for barycentric point-in-tet classification."""

    def test_point_known_inside(self):
        """Centroid of unit tet is inside."""
        from dmipy_jax.biophysics.mesh_rasterizer import point_in_tetrahedron

        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
        centroid = verts.mean(axis=0)
        assert point_in_tetrahedron(centroid, verts) == True

    def test_point_known_outside(self):
        """A far-away point is outside."""
        from dmipy_jax.biophysics.mesh_rasterizer import point_in_tetrahedron

        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
        point = np.array([2.0, 2.0, 2.0])
        assert point_in_tetrahedron(point, verts) == False

    def test_point_on_vertex(self):
        """A vertex is inside (inclusive boundary)."""
        from dmipy_jax.biophysics.mesh_rasterizer import point_in_tetrahedron

        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
        assert point_in_tetrahedron(verts[0], verts) == True

    def test_point_on_edge(self):
        """Midpoint of an edge is inside (inclusive boundary)."""
        from dmipy_jax.biophysics.mesh_rasterizer import point_in_tetrahedron

        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
        midpoint = (verts[0] + verts[1]) / 2.0
        assert point_in_tetrahedron(midpoint, verts) == True

    def test_point_just_outside(self):
        """Point slightly beyond a face is outside."""
        from dmipy_jax.biophysics.mesh_rasterizer import point_in_tetrahedron

        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
        # Slightly past the slanted face x+y+z=1
        point = np.array([0.4, 0.4, 0.4])  # sum=1.2 > 1
        assert point_in_tetrahedron(point, verts) == False


# ---------------------------------------------------------------------------
# Phase 2.2: Full Mesh Rasterization
# ---------------------------------------------------------------------------

class TestRasterizeMesh:
    """Tests for rasterize_mesh() converting tet mesh to regular grid."""

    def test_rasterize_single_tet(self):
        """A single large tet rasterized should fill correct voxels."""
        from dmipy_jax.biophysics.mesh_rasterizer import rasterize_mesh

        points = np.array(
            [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float64
        )
        cells = np.array([[0, 1, 2, 3]])
        tissue = np.array([2])  # skull
        grid_shape = (11, 11, 11)
        grid_spacing = 1.0
        grid_origin = np.array([0.0, 0.0, 0.0])

        labels = rasterize_mesh(points, cells, tissue, grid_shape, grid_spacing, grid_origin)

        assert labels.shape == grid_shape
        # Interior voxel (1,1,1) should be label 2
        assert labels[1, 1, 1] == 2
        # Far corner outside the tet
        assert labels[10, 10, 10] == 0

    def test_rasterize_preserves_tissue_labels(self):
        """Multiple tissue types must appear in output."""
        from dmipy_jax.biophysics.mesh_rasterizer import rasterize_mesh

        # Two non-overlapping tets
        points = np.array(
            [
                [0, 0, 0],
                [5, 0, 0],
                [0, 5, 0],
                [0, 0, 5],
                [6, 6, 6],
                [10, 6, 6],
                [6, 10, 6],
                [6, 6, 10],
            ],
            dtype=np.float64,
        )
        cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        tissue = np.array([1, 3])  # scalp and CSF

        labels = rasterize_mesh(
            points, cells, tissue,
            grid_shape=(11, 11, 11),
            grid_spacing=1.0,
            grid_origin=np.array([0.0, 0.0, 0.0]),
        )

        unique = np.unique(labels)
        assert 1 in unique, "Scalp label missing"
        assert 3 in unique, "CSF label missing"
        assert 0 in unique, "Background label missing"

    def test_rasterize_output_shape(self):
        """Output shape must match requested grid_shape."""
        from dmipy_jax.biophysics.mesh_rasterizer import rasterize_mesh

        points = np.array(
            [[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float64
        )
        cells = np.array([[0, 1, 2, 3]])
        tissue = np.array([4])

        for shape in [(8, 8, 8), (16, 16, 16), (10, 20, 30)]:
            labels = rasterize_mesh(
                points, cells, tissue,
                grid_shape=shape,
                grid_spacing=1.0,
                grid_origin=np.array([0.0, 0.0, 0.0]),
            )
            assert labels.shape == shape

    def test_rasterize_background_default_zero(self):
        """Voxels outside all tetrahedra should be label 0."""
        from dmipy_jax.biophysics.mesh_rasterizer import rasterize_mesh

        # Tiny tet in corner of large grid
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
        cells = np.array([[0, 1, 2, 3]])
        tissue = np.array([5])

        labels = rasterize_mesh(
            points, cells, tissue,
            grid_shape=(20, 20, 20),
            grid_spacing=1.0,
            grid_origin=np.array([0.0, 0.0, 0.0]),
        )

        # Most voxels should be background
        n_background = np.sum(labels == 0)
        assert n_background > 0.9 * 20**3

    def test_rasterize_dtype_int(self):
        """Output must be integer dtype."""
        from dmipy_jax.biophysics.mesh_rasterizer import rasterize_mesh

        points = np.array(
            [[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float64
        )
        cells = np.array([[0, 1, 2, 3]])
        tissue = np.array([2])

        labels = rasterize_mesh(
            points, cells, tissue,
            grid_shape=(6, 6, 6),
            grid_spacing=1.0,
            grid_origin=np.array([0.0, 0.0, 0.0]),
        )

        assert np.issubdtype(labels.dtype, np.integer)


class TestRasterizeSCIHead:
    """Integration tests using the actual SCI head model."""

    @pytest.mark.slow
    def test_rasterize_sci_head_tissue_count(self):
        """SCI head at coarse resolution should produce 5+ tissue types."""
        from dmipy_jax.biophysics.mesh_rasterizer import rasterize_mesh

        try:
            from dmipy_jax.io.sci_head_loader import load_sci_head_mesh
        except ImportError:
            pytest.skip("SCI head loader not available")

        mesh_path = "data/SCI_headmodel/extracted/HeadMesh.mat"
        try:
            mesh = load_sci_head_mesh(mesh_path)
        except FileNotFoundError:
            pytest.skip(f"SCI head data not found at {mesh_path}")

        points = np.array(mesh["points"])
        cells = np.array(mesh["cells"]["tetra"])
        tissues = np.array(mesh["cell_data"]["tissue"])

        # Coarse grid (3mm) for speed
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
        extent = bounds_max - bounds_min
        spacing = 3.0
        grid_shape = tuple((extent / spacing).astype(int) + 1)

        labels = rasterize_mesh(
            points, cells, tissues,
            grid_shape=grid_shape,
            grid_spacing=spacing,
            grid_origin=bounds_min,
        )

        unique = np.unique(labels)
        # Should have background + at least 5 tissue types
        assert len(unique) >= 5, f"Only found {len(unique)} tissue types: {unique}"

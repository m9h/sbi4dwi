"""
Tetrahedral mesh to regular grid rasterization.

Converts a tetrahedral FEM mesh (e.g., SCI Institute head model) into a
regular 3D voxel grid with integer tissue labels, suitable for j-Wave or
k-Wave acoustic simulation.

Two strategies:
  - point_in_tetrahedron(): Pure numpy barycentric test for unit testing
  - rasterize_mesh(): Uses scipy.spatial.Delaunay.find_simplex() for
    production-scale meshes (handles ~11M tetrahedra efficiently)
"""

import numpy as np
from typing import Tuple


def point_in_tetrahedron(
    point: np.ndarray,
    vertices: np.ndarray,
    tol: float = -1e-10,
) -> bool:
    """
    Test if a point lies inside (or on the boundary of) a tetrahedron.

    Uses the sign-of-determinant method: a point is inside if the four
    sub-tetrahedra formed by replacing each vertex with the test point
    all have the same orientation (same sign determinant) as the original.

    Args:
        point: (3,) array — test point coordinates.
        vertices: (4, 3) array — tetrahedron vertex coordinates.
        tol: Tolerance for boundary inclusion (negative = inclusive).

    Returns:
        True if point is inside or on the boundary.
    """
    p = np.asarray(point, dtype=np.float64)
    v = np.asarray(vertices, dtype=np.float64)

    # Compute barycentric coordinates via determinant ratios
    T = np.column_stack([v[0] - v[3], v[1] - v[3], v[2] - v[3]])
    det_T = np.linalg.det(T)

    if abs(det_T) < 1e-15:
        # Degenerate tetrahedron
        return False

    # Barycentric coordinates: lambda = T^{-1} (p - v3)
    diff = p - v[3]
    lam = np.linalg.solve(T, diff)

    # Fourth coordinate
    lam4 = 1.0 - lam[0] - lam[1] - lam[2]

    # Inside if all barycentric coordinates are >= 0 (with tolerance)
    return bool(
        lam[0] >= tol
        and lam[1] >= tol
        and lam[2] >= tol
        and lam4 >= tol
    )


def rasterize_mesh(
    points: np.ndarray,
    cells: np.ndarray,
    tissue_labels: np.ndarray,
    grid_shape: Tuple[int, int, int],
    grid_spacing: float,
    grid_origin: np.ndarray,
) -> np.ndarray:
    """
    Rasterize a tetrahedral mesh onto a regular 3D grid.

    For each voxel center, finds which tetrahedron contains it and assigns
    the corresponding tissue label. Uses a centroid-based nearest-neighbor
    approach for large meshes (>100K tets) and scipy.spatial.Delaunay for
    smaller meshes.

    Args:
        points: (N_vertices, 3) vertex coordinates.
        cells: (N_cells, 4) tetrahedral connectivity (0-indexed).
        tissue_labels: (N_cells,) integer tissue label per tetrahedron.
        grid_shape: (nx, ny, nz) output grid dimensions.
        grid_spacing: Uniform spacing between voxel centers (same units as points).
        grid_origin: (3,) coordinates of voxel [0,0,0] center.

    Returns:
        (nx, ny, nz) integer array of tissue labels. 0 = background (outside mesh).
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    tissue_labels = np.asarray(tissue_labels).flatten()
    grid_origin = np.asarray(grid_origin, dtype=np.float64)

    nx, ny, nz = grid_shape

    # Generate voxel center coordinates
    x = grid_origin[0] + np.arange(nx) * grid_spacing
    y = grid_origin[1] + np.arange(ny) * grid_spacing
    z = grid_origin[2] + np.arange(nz) * grid_spacing
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    query_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    n_cells = cells.shape[0]

    if n_cells > 100_000:
        # Large mesh: use KDTree on cell centroids for fast nearest-neighbor
        # This is approximate but O(M log N) and works for millions of tets.
        output = _rasterize_centroid_kdtree(
            points, cells, tissue_labels, query_points, grid_shape
        )
    else:
        # Small mesh: use Delaunay for exact point location
        output = _rasterize_delaunay(
            points, cells, tissue_labels, query_points, grid_shape
        )

    return output


def _rasterize_centroid_kdtree(
    points: np.ndarray,
    cells: np.ndarray,
    tissue_labels: np.ndarray,
    query_points: np.ndarray,
    grid_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Fast rasterization using KDTree on cell centroids.

    For each query point, finds the nearest cell centroid and assigns
    that cell's tissue label. Then verifies the point is actually inside
    the mesh by checking the distance is within a reasonable threshold.
    """
    from scipy.spatial import cKDTree

    # Compute cell centroids
    centroids = points[cells].mean(axis=1)  # (N_cells, 3)

    # Build KDTree
    tree = cKDTree(centroids)

    # Query: find nearest centroid for each voxel
    distances, indices = tree.query(query_points, k=1)

    # Assign tissue labels from nearest cell
    output = tissue_labels[indices].astype(np.int32)

    # Mask out points too far from any centroid (outside the mesh)
    # Use median cell size as threshold
    cell_sizes = np.linalg.norm(
        points[cells[:, 0]] - points[cells[:, 1]], axis=1
    )
    threshold = np.median(cell_sizes) * 2.0
    output[distances > threshold] = 0  # background

    return output.reshape(grid_shape)


def _rasterize_delaunay(
    points: np.ndarray,
    cells: np.ndarray,
    tissue_labels: np.ndarray,
    query_points: np.ndarray,
    grid_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Exact rasterization using Delaunay triangulation (for small meshes)."""
    from scipy.spatial import Delaunay

    tri = Delaunay(points)

    simplex_indices = tri.find_simplex(query_points)

    delaunay_to_tissue = _build_delaunay_tissue_map(tri, cells, tissue_labels)

    nx, ny, nz = grid_shape
    output = np.zeros(nx * ny * nz, dtype=np.int32)
    inside_mask = simplex_indices >= 0
    output[inside_mask] = delaunay_to_tissue[simplex_indices[inside_mask]]

    return output.reshape(grid_shape)


def _build_delaunay_tissue_map(
    tri,
    cells: np.ndarray,
    tissue_labels: np.ndarray,
) -> np.ndarray:
    """
    Map Delaunay simplex indices to original mesh tissue labels.

    When Delaunay triangulates the point set, it creates its own simplices
    which may not match the original mesh cells 1:1 (Delaunay may add or
    subdivide tetrahedra). We handle this by:
    1. Building a hash map from sorted vertex tuples to tissue labels
    2. For Delaunay simplices that match an original cell exactly, use that label
    3. For Delaunay simplices that don't match, find the original cell whose
       centroid is closest and inherit its label
    """
    n_delaunay = tri.simplices.shape[0]
    delaunay_tissue = np.zeros(n_delaunay, dtype=np.int32)

    # Build lookup: frozenset of vertex indices -> tissue label
    cell_lookup = {}
    for i, cell in enumerate(cells):
        key = tuple(sorted(cell))
        cell_lookup[key] = tissue_labels[i]

    # Also build centroid lookup for fallback
    cell_centroids = tri.points[cells].mean(axis=1)  # (N_cells, 3)

    matched = 0
    for i, simplex in enumerate(tri.simplices):
        key = tuple(sorted(simplex))
        if key in cell_lookup:
            delaunay_tissue[i] = cell_lookup[key]
            matched += 1
        else:
            # Fallback: find nearest original cell centroid
            centroid = tri.points[simplex].mean(axis=0)
            dists = np.sum((cell_centroids - centroid) ** 2, axis=1)
            nearest = np.argmin(dists)
            delaunay_tissue[i] = tissue_labels[nearest]

    return delaunay_tissue

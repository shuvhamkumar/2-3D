"""Mesh cleanup, quality inspection, and format conversion.

Uses ``trimesh`` as the working container (it carries vertex-colors *and* UV
textures through edits and exports them to GLB), and ``open3d`` for robust
watertight/manifold checks. We **never auto-repair** non-manifold geometry --
we report it honestly so a human can judge printability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class MeshQuality:
    vertices: int
    faces: int
    components: int
    watertight: bool
    edge_manifold: bool
    vertex_manifold: bool

    def to_dict(self) -> dict:
        return asdict(self)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh as a single Trimesh, preserving texture/vertex colors."""
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a triangle mesh ({type(mesh).__name__}).")
    if len(mesh.faces) == 0:
        raise ValueError(f"{path} contains no faces (empty mesh).")
    return mesh


def _welded_face_labels(mesh: trimesh.Trimesh) -> np.ndarray:
    """Connected-component label per face, using position-welded connectivity.

    A UV-textured OBJ loads with per-corner *split* vertices (adjacent faces
    share no vertex index), which makes ``face_adjacency`` empty and every face
    its own "component". We weld coincident vertices on a copy purely to compute
    adjacency; ``merge_vertices`` keeps face count/order, so labels map back to
    the original faces 1:1 (and the original textured mesh is left untouched).
    """
    welded = mesh.copy()
    try:
        welded.merge_vertices(merge_tex=True, merge_norm=True)
    except TypeError:  # older trimesh signature
        welded.merge_vertices()
    return trimesh.graph.connected_component_labels(
        welded.face_adjacency, node_count=len(welded.faces)
    )


def component_sizes(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(face_counts_per_component, per_face_labels)``."""
    labels = _welded_face_labels(mesh)
    return np.bincount(labels), labels


def keep_largest_component(
    mesh: trimesh.Trimesh, min_fraction: float
) -> tuple[trimesh.Trimesh, int, int]:
    """Drop all but the largest connected component.

    Returns ``(mesh, num_components, faces_dropped)``. A warning-worthy second
    component (>= ``min_fraction`` of the largest) is left to the caller to flag.
    """
    counts, labels = component_sizes(mesh)
    if len(counts) <= 1:
        return mesh, int(len(counts)), 0
    largest = int(counts.argmax())
    face_mask = labels == largest
    dropped = int((~face_mask).sum())
    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()
    return mesh, int(len(counts)), dropped


def second_largest_fraction(mesh_counts: np.ndarray) -> float:
    if len(mesh_counts) < 2:
        return 0.0
    s = np.sort(mesh_counts)[::-1]
    return float(s[1] / s[0]) if s[0] else 0.0


def remove_statistical_outliers(
    mesh: trimesh.Trimesh, n_neighbors: int, std_ratio: float, max_fraction: float = 0.02
) -> tuple[trimesh.Trimesh, int]:
    """Remove vertices whose mean distance to k neighbours is an outlier.

    Genuine outliers are rare and far. If the ``mean + std_ratio·std`` threshold
    would flag more than ``max_fraction`` of vertices, the surface is simply
    near-uniform (no clear outliers) and trimming it would only punch holes /
    break watertightness — so we skip removal entirely in that case.
    """
    n = len(mesh.vertices)
    if n_neighbors <= 0 or n <= n_neighbors + 1:
        return mesh, 0
    from scipy.spatial import cKDTree

    tree = cKDTree(mesh.vertices)
    dist, _ = tree.query(mesh.vertices, k=n_neighbors + 1, workers=-1)
    mean_d = dist[:, 1:].mean(axis=1)  # exclude self (distance 0)
    threshold = mean_d.mean() + std_ratio * mean_d.std()
    keep = mean_d <= threshold
    removed = int((~keep).sum())
    if removed == 0 or removed >= n:
        return mesh, 0
    if max_fraction > 0 and removed / n > max_fraction:
        # Too many "outliers" => not outliers, just a uniform surface. Leave it.
        return mesh, 0
    mesh.update_vertices(keep)
    return mesh, removed


def inspect_quality(mesh: trimesh.Trimesh) -> MeshQuality:
    """Compute watertight/manifold flags using open3d on the geometry."""
    import open3d as o3d

    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    counts, _ = component_sizes(mesh)
    return MeshQuality(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        components=int(len(counts)),
        watertight=bool(om.is_watertight()),
        edge_manifold=bool(om.is_edge_manifold(allow_boundary_edges=False)),
        vertex_manifold=bool(om.is_vertex_manifold()),
    )


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> tuple[trimesh.Trimesh, bool]:
    """Reduce to ~target_faces via open3d quadric decimation (web use-case).

    Returns ``(mesh, texture_dropped)``. Quadric decimation does not preserve UV
    layouts, so a UV-textured mesh loses its texture (vertex colors survive).
    """
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh, False
    import open3d as o3d

    textured = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    om = om.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    out = trimesh.Trimesh(
        vertices=np.asarray(om.vertices), faces=np.asarray(om.triangles), process=False
    )
    return out, textured


def export(mesh: trimesh.Trimesh, path: Path) -> int:
    """Export to the format implied by ``path`` extension; return bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path.stat().st_size

"""Geometric object-isolation backstop: crop to the sparse object's OBB.

The sparse cloud (object-only when masks are used) defines the object's oriented
bounding box. Cropping the dense cloud (COLMAP backend) or the reconstructed mesh
(OpenMVS backend) to that box — plus a margin — removes surviving background and
turntable geometry even when masks are imperfect or absent. Works for either
backend and never depends on mask/undistortion alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .binaries import Environment, run_command


@dataclass
class CropResult:
    target: str  # "cloud" | "mesh"
    before: int
    after: int

    @property
    def removed(self) -> int:
        return self.before - self.after

    @property
    def fraction_removed(self) -> float:
        return self.removed / self.before if self.before else 0.0

    def to_metrics(self) -> dict:
        return {
            "target": self.target,
            f"{self.target}_before": self.before,
            f"{self.target}_after": self.after,
            "removed": self.removed,
            "fraction_removed": round(self.fraction_removed, 4),
        }


def export_sparse_points(
    colmap_path: str, sparse_model_dir: Path, out_ply: Path, env: Environment
) -> Path:
    """Export a COLMAP sparse model's 3D points to PLY for OBB computation.

    Reference command:
        colmap model_converter --input_path <sparse>/0 --output_path pts.ply \
            --output_type PLY
    """
    run_command(
        [colmap_path, "model_converter",
         "--input_path", str(sparse_model_dir),
         "--output_path", str(out_ply),
         "--output_type", "PLY"],
        env_required=env,
    )
    return out_ply


def compute_obb(points_ply: Path, margin: float):
    """Oriented bounding box of the sparse points, expanded by ``margin`` per side."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(points_ply))
    n = len(pcd.points)
    if n < 4:
        raise ValueError(f"Too few sparse points ({n}) to compute a bounding box.")
    obb = pcd.get_oriented_bounding_box()
    scale = 1.0 + 2.0 * float(margin)  # margin applied to each side
    return o3d.geometry.OrientedBoundingBox(obb.center, obb.R, obb.extent * scale)


def crop_cloud(in_ply: Path, obb, out_ply: Path) -> CropResult:
    """Crop a point cloud to ``obb`` (preserves normals/colors)."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(in_ply))
    before = len(pcd.points)
    cropped = pcd.crop(obb)
    after = len(cropped.points)
    o3d.io.write_point_cloud(str(out_ply), cropped)
    return CropResult("cloud", before, after)


def crop_mesh(in_ply: Path, obb, out_ply: Path) -> CropResult:
    """Crop a triangle mesh to ``obb``."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(in_ply))
    before = len(mesh.triangles)
    cropped = mesh.crop(obb)
    after = len(cropped.triangles)
    o3d.io.write_triangle_mesh(str(out_ply), cropped)
    return CropResult("mesh", before, after)

"""Tests for the OBB crop backstop (open3d; no external binaries)."""

from __future__ import annotations

import numpy as np

from engine.geometry import CropResult, compute_obb, crop_cloud, crop_mesh


def test_crop_result_math():
    r = CropResult("cloud", before=1000, after=750)
    assert r.removed == 250
    assert abs(r.fraction_removed - 0.25) < 1e-9
    m = r.to_metrics()
    assert m["cloud_before"] == 1000 and m["cloud_after"] == 750
    assert m["target"] == "cloud"


def _write_points_ply(path, pts):
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, dtype=float)))
    o3d.io.write_point_cloud(str(path), pc)


def test_compute_obb_and_crop_cloud_removes_far_points(tmp_path):
    rng = np.random.default_rng(0)
    # A compact object cluster ...
    cluster = rng.uniform(-1, 1, size=(2000, 3))
    sparse_ply = tmp_path / "sparse.ply"
    _write_points_ply(sparse_ply, cluster)

    # ... plus a dense cloud = cluster + far-away background points.
    far = rng.uniform(40, 50, size=(500, 3))
    dense_ply = tmp_path / "dense.ply"
    _write_points_ply(dense_ply, np.vstack([cluster, far]))

    obb = compute_obb(sparse_ply, margin=0.15)
    out = tmp_path / "cropped.ply"
    res = crop_cloud(dense_ply, obb, out)
    assert res.before == 2500
    # The 500 far points should be gone; the ~2000 cluster points retained.
    assert res.after >= 1900
    assert res.after <= 2100
    assert out.exists()


def test_crop_mesh_removes_detached_far_component(tmp_path):
    import open3d as o3d
    import trimesh

    main = trimesh.creation.box(extents=(2, 2, 2))
    far = trimesh.creation.box(extents=(1, 1, 1))
    far.apply_translation([60, 0, 0])
    combined = trimesh.util.concatenate([main, far])
    mesh_ply = tmp_path / "mesh.ply"
    combined.export(str(mesh_ply))

    # OBB from the main box's corners only.
    sparse_ply = tmp_path / "sparse.ply"
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(main.vertices)))
    o3d.io.write_point_cloud(str(sparse_ply), pc)

    obb = compute_obb(sparse_ply, margin=0.2)
    out = tmp_path / "mesh_cropped.ply"
    res = crop_mesh(mesh_ply, obb, out)
    assert res.target == "mesh"
    assert res.before == 24  # 12 + 12 faces
    assert res.after == 12   # only the main box survives

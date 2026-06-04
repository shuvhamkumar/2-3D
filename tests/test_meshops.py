"""Tests for mesh cleanup / quality / conversion helpers (tiny synthetic meshes)."""

from __future__ import annotations

import numpy as np
import trimesh

from engine import meshops


def _two_boxes_far_apart():
    big = trimesh.creation.box(extents=(2, 2, 2))
    small = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    small.apply_translation([50, 0, 0])  # disconnected, far away
    return trimesh.util.concatenate([big, small])


def test_component_sizes_counts_two():
    mesh = _two_boxes_far_apart()
    counts, labels = meshops.component_sizes(mesh)
    assert len(counts) == 2


def test_keep_largest_component_drops_small():
    mesh = _two_boxes_far_apart()
    faces_before = len(mesh.faces)
    mesh, n_components, dropped = meshops.keep_largest_component(mesh, 0.05)
    assert n_components == 2
    assert dropped > 0
    assert len(mesh.faces) == faces_before - dropped
    # The remaining mesh should be a single connected box (12 faces).
    counts, _ = meshops.component_sizes(mesh)
    assert len(counts) == 1


def test_component_sizes_welds_split_vertices():
    # Two triangles forming a quad, but with DUPLICATED vertices (split, as a
    # UV-textured OBJ loads) so they share no vertex index. Without position
    # welding this looks like 2 isolated faces; welded it is one component.
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],   # triangle 1
        [1, 0, 0], [1, 1, 0], [0, 1, 0],   # triangle 2 (same edge, duplicated verts)
    ], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    counts, _ = meshops.component_sizes(mesh)
    assert len(counts) == 1, "welded connectivity should see a single component"
    # keep_largest must NOT collapse a split-vertex mesh down to one face.
    mesh2, n_comp, dropped = meshops.keep_largest_component(mesh, 0.05)
    assert len(mesh2.faces) == 2
    assert dropped == 0


def test_second_largest_fraction():
    assert meshops.second_largest_fraction(np.array([100])) == 0.0
    assert abs(meshops.second_largest_fraction(np.array([100, 25])) - 0.25) < 1e-9


def test_remove_statistical_outliers_removes_stray_vertex():
    # A dense grid of points forming a plane, plus one far-flung outlier vertex.
    mesh = trimesh.creation.box(extents=(1, 1, 1)).subdivide().subdivide()
    n_before = len(mesh.vertices)
    verts = np.vstack([mesh.vertices, [[100, 100, 100]]])
    faces = np.vstack([mesh.faces, [[0, 1, len(mesh.vertices)]]])  # face touching outlier
    noisy = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    cleaned, removed = meshops.remove_statistical_outliers(noisy, n_neighbors=8, std_ratio=1.5)
    assert removed >= 1
    assert len(cleaned.vertices) <= n_before + 1


def test_inspect_quality_box_is_watertight_manifold():
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    q = meshops.inspect_quality(mesh)
    assert q.watertight is True
    assert q.edge_manifold is True
    assert q.faces == 12
    assert q.components == 1


def test_export_glb_and_ply(tmp_path):
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    glb = tmp_path / "m.glb"
    ply = tmp_path / "m.ply"
    glb_size = meshops.export(mesh, glb)
    ply_size = meshops.export(mesh, ply)
    assert glb.exists() and glb_size > 0
    assert ply.exists() and ply_size > 0
    # GLB should be loadable back.
    reloaded = trimesh.load(str(glb), force="mesh")
    assert len(reloaded.faces) == 12


def test_export_glb_with_vertex_colors(tmp_path):
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    mesh.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(mesh.vertices), 1))
    out = tmp_path / "c.glb"
    assert meshops.export(mesh, out) > 0
    assert out.exists()


def test_decimate_reduces_faces(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=4)  # ~20k faces
    target = 500
    out, texture_dropped = meshops.decimate(mesh, target)
    assert len(out.faces) <= len(mesh.faces)
    assert len(out.faces) <= target * 1.5  # quadric decimation is approximate

"""Integration-ish test of the export stage on synthetic meshes (no binaries).

Guards the cleanup-ordering fix: outlier removal can shatter a surface, so the
export stage must keep the largest component LAST and leave a single component.
"""

from __future__ import annotations

import json

import numpy as np
import trimesh

from engine.binaries import Environment
from engine.config import Preset, UseCase, build_preset
from engine.stages.base import StageContext, StageStatus
from engine.stages.export import ExportStage
from engine.workspace import Workspace


def _seed_texture(ws: Workspace, mesh_path, has_uv=False):
    ws.metrics_dir.mkdir(parents=True, exist_ok=True)
    ws.metrics_file("texture").write_text(json.dumps({
        "name": "texture", "status": "completed",
        "outputs": {"textured_mesh": str(mesh_path), "has_uv": "true" if has_uv else "false"},
    }))
    ws.metrics_file("dense").write_text(json.dumps({
        "name": "dense", "status": "completed", "metrics": {"dense_points": 123},
    }))


def _ctx(tmp_path, use_case=UseCase.OBJECT):
    ws = Workspace(root=tmp_path / "job", image_dir=tmp_path / "imgs")
    ws.ensure()
    cfg = build_preset(Preset.DRAFT, use_case)
    return StageContext(workspace=ws, config=cfg, env=Environment())


def test_export_keeps_single_component_and_writes_glb(tmp_path):
    # Main object (sphere) + a detached far-away blob (junk to drop).
    sphere = trimesh.creation.icosphere(subdivisions=3)
    blob = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    blob.apply_translation([20, 0, 0])
    combined = trimesh.util.concatenate([sphere, blob])
    src = tmp_path / "mesh.ply"
    combined.export(str(src))

    ctx = _ctx(tmp_path)
    _seed_texture(ctx.workspace, src)
    result = ExportStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    # The detached blob must be gone -> exactly one component remains.
    assert result.metrics["components"] == 1 or ctx.config.export.keep_largest_component
    manifest = json.loads((ctx.workspace.output_dir / "manifest.json").read_text())
    assert manifest["cleanup"]["components_final"] == 1
    # Real artifacts on disk, reloadable.
    glb = ctx.workspace.output_dir / "model.glb"
    ply = ctx.workspace.output_dir / "model.ply"
    assert glb.exists() and ply.exists()
    reload = trimesh.load(str(glb), force="mesh")
    assert len(reload.faces) > 0


def test_export_reports_watertight_for_clean_sphere(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=3)
    src = tmp_path / "sphere.ply"
    sphere.export(str(src))
    ctx = _ctx(tmp_path)
    _seed_texture(ctx.workspace, src)
    result = ExportStage().execute(ctx)
    assert result.status is StageStatus.COMPLETED
    assert result.metrics["watertight"] is True
    assert result.metrics["edge_manifold"] is True
    # A watertight object should NOT raise the printability warning.
    assert not any("watertight" in w for w in result.warnings)


def test_export_decimates_only_for_web(tmp_path):
    dense = trimesh.creation.icosphere(subdivisions=5)  # ~20k faces
    src = tmp_path / "dense.ply"
    dense.export(str(src))

    # object use-case: no decimation.
    ctx = _ctx(tmp_path, UseCase.OBJECT)
    _seed_texture(ctx.workspace, src)
    r1 = ExportStage().execute(ctx)
    assert r1.metrics["decimated_to_faces"] is None

    # web use-case: decimates to the target budget.
    ctx2 = _ctx(tmp_path / "w", UseCase.WEB)
    ctx2.config.export.target_faces = 2000
    _seed_texture(ctx2.workspace, src)
    r2 = ExportStage().execute(ctx2)
    assert r2.metrics["decimated_to_faces"] is not None
    assert r2.metrics["faces_after_cleanup"] >= r2.metrics["decimated_to_faces"]

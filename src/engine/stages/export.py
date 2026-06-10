"""Stage 9 — export & convert.

Cleans the mesh (largest connected component + statistical outlier removal),
reports watertight/manifold honestly (no auto-repair), optionally decimates for
the web use-case, converts to ``.glb`` (embedded texture for the UV path, vertex
colors for the colmap path), writes a ``.ply`` alongside, and copies final
artifacts into ``<job>/output/`` with a manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import UseCase
from ..logging import get_logger
from .. import geometry, meshops
from .base import Stage, StageContext, StageError


class ExportStage(Stage):
    name = "export"

    def validate_inputs(self, ctx: StageContext) -> None:
        src = self.prior_outputs(ctx, "texture").get("textured_mesh")
        if not src or not Path(src).exists():
            raise StageError("No textured mesh from 'texture' stage to export.")

    def run(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        ws = ctx.workspace
        logger = get_logger()
        ecfg = ctx.config.export
        warnings: list[str] = []

        tex = self.prior_outputs(ctx, "texture")
        src = Path(tex["textured_mesh"])
        has_uv = tex.get("has_uv") == "true"

        logger.info("Loading mesh for export: %s", src.name)
        mesh = meshops.load_mesh(src)
        before_v, before_f = len(mesh.vertices), len(mesh.faces)

        # cleanup — ORDER MATTERS. Statistical outlier removal deletes vertices
        # and so shatters the surface into many fragments; we therefore remove
        # outliers FIRST and keep the largest connected component LAST, leaving a
        # single connected object rather than a re-fragmented one.
        components_initial = int(len(meshops.component_sizes(mesh)[0]))
        dropped_faces = removed_verts = 0

        mesh, removed_verts = meshops.remove_statistical_outliers(
            mesh, ecfg.outlier_neighbors, ecfg.outlier_std_ratio, ecfg.outlier_max_fraction
        )
        if len(mesh.faces) == 0:
            raise StageError("Mesh is empty after outlier removal — nothing to export.")

        components = components_initial
        if ecfg.keep_largest_component:
            counts, _ = meshops.component_sizes(mesh)
            frac2 = meshops.second_largest_fraction(counts)
            mesh, components, dropped_faces = meshops.keep_largest_component(
                mesh, ecfg.min_component_fraction
            )
            if components_initial > 10 or components > 10:
                warnings.append(
                    f"Mesh was fragmented (≈{max(components_initial, components)} "
                    "components) before keeping the largest."
                )
            if frac2 >= ecfg.min_component_fraction:
                warnings.append(
                    f"2nd-largest component was {frac2:.0%} of the largest — cleanup "
                    "may have dropped real geometry; inspect the result."
                )

        if len(mesh.faces) == 0:
            raise StageError("Mesh is empty after cleanup — nothing to export.")

        # 2) quality (reported, never auto-repaired).
        quality = meshops.inspect_quality(mesh)
        if ctx.config.use_case is UseCase.OBJECT and not quality.watertight:
            warnings.append(
                "Mesh is NOT watertight — may not be directly 3D-printable without repair."
            )
        if not quality.edge_manifold:
            warnings.append("Mesh is NOT edge-manifold.")

        # 3) decimation (web use-case only).
        decimated_to = None
        texture_dropped = False
        if ctx.config.use_case is UseCase.WEB and ecfg.target_faces:
            mesh, texture_dropped = meshops.decimate(mesh, ecfg.target_faces)
            decimated_to = len(mesh.faces)
            if texture_dropped:
                warnings.append("Decimation dropped the UV texture (vertex colors retained).")

        # 3b) reorient to glTF +Y. COLMAP's gauge is arbitrary and y-down, so the
        # mesh otherwise appears upside-down; estimate the up axis from the camera
        # poses and rotate. Best-effort — never fail the export over orientation.
        reoriented_up = None
        colmap_tool = ctx.env.tools.get("colmap")
        if ecfg.reorient_up and colmap_tool and colmap_tool.available \
                and ws.metrics_file("sfm").exists():
            sparse_model = self.prior_outputs(ctx, "sfm").get("sparse_model")
            if sparse_model and Path(sparse_model).exists():
                try:
                    up = geometry.estimate_scene_up(
                        colmap_tool.path, Path(sparse_model), ctx.env
                    )
                    if up is not None:
                        meshops.reorient_to_up(mesh, up)
                        reoriented_up = [round(float(x), 4) for x in up]
                        logger.info("Reoriented mesh: scene up %s -> +Y (glTF).", reoriented_up)
                    else:
                        warnings.append("Up-axis reorientation skipped: could not estimate up.")
                except Exception as exc:  # noqa: BLE001 -- orientation is non-critical
                    warnings.append(f"Up-axis reorientation skipped: {exc}")

        # 3c) normalize the PBR material so the baked albedo texture renders at
        # full brightness (trimesh's OBJ->GLB leaves a dark baseColorFactor +
        # metallic=1, which makes the model look dim/black in PBR viewers).
        if meshops.normalize_pbr_material(mesh):
            logger.info("Normalized PBR material: baseColor=white, metallic=0, roughness=1.")

        # 4) convert + copy to output/.
        ws.output_dir.mkdir(parents=True, exist_ok=True)
        glb_path = ws.output_dir / "model.glb"
        ply_path = ws.output_dir / "model.ply"
        glb_size = meshops.export(mesh, glb_path)
        ply_size = meshops.export(mesh, ply_path)

        manifest = {
            "backend": ctx.config.mvs.backend.value,
            "use_case": ctx.config.use_case.value,
            "source_mesh": src.name,
            "has_uv_texture": has_uv and not texture_dropped,
            "reoriented_up": reoriented_up,
            "files": {
                "glb": {"path": glb_path.name, "bytes": glb_size},
                "ply": {"path": ply_path.name, "bytes": ply_size},
            },
            "cleanup": {
                "vertices_before": before_v,
                "faces_before": before_f,
                "components_initial": components_initial,
                "components_before_keep_largest": components,
                "components_final": quality.components,
                "faces_dropped_non_largest": dropped_faces,
                "outlier_vertices_removed": removed_verts,
                "vertices_after": quality.vertices,
                "faces_after": quality.faces,
                "decimated_to_faces": decimated_to,
            },
            "quality": quality.to_dict(),
        }
        (ws.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        logger.info(
            "Export: %d->%d faces after cleanup; watertight=%s manifold=%s; "
            "glb %.1f KB -> %s",
            before_f, quality.faces, quality.watertight, quality.edge_manifold,
            glb_size / 1024, glb_path,
        )

        # surface dense-point count from upstream into the export metrics summary
        dense_points = self.prior_metrics(ctx, "dense").get("dense_points")
        metrics: dict[str, Any] = {
            "dense_points": dense_points,
            "vertices_before_cleanup": before_v,
            "faces_before_cleanup": before_f,
            "vertices_after_cleanup": quality.vertices,
            "faces_after_cleanup": quality.faces,
            "components": components,
            "outlier_vertices_removed": removed_verts,
            "watertight": quality.watertight,
            "edge_manifold": quality.edge_manifold,
            "vertex_manifold": quality.vertex_manifold,
            "decimated_to_faces": decimated_to,
            "reoriented_up": reoriented_up,
            "glb_path": str(glb_path),
            "glb_bytes": glb_size,
            "ply_path": str(ply_path),
        }
        return metrics, warnings, {
            "glb": str(glb_path),
            "ply": str(ply_path),
            "manifest": str(ws.output_dir / "manifest.json"),
        }

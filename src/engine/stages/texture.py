"""Stage 8 — texturing.

* colmap : no-op pass-through. The poisson/delaunay mesh already carries vertex
           colors, which export embeds into the GLB. No UV atlas is produced.
* openmvs: ``TextureMesh`` builds a UV atlas + texture image (OBJ + MTL + PNG).

Reference command:
    TextureMesh <mesh>.mvs --export-type obj \
        --resolution-level <texture.resolution_level> \
        --max-texture-size <texture.texture_size>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..binaries import require, run_command
from ..config import MvsBackend
from ..logging import get_logger
from ..openmvs import openmvs_lib_env
from .base import Stage, StageContext, StageError


class TextureStage(Stage):
    name = "texture"

    def validate_inputs(self, ctx: StageContext) -> None:
        if ctx.config.mvs.backend is MvsBackend.OPENMVS:
            require(ctx.env, ["TextureMesh"])
            scene = self.prior_outputs(ctx, "dense").get("mvs_scene")
            mesh = self.prior_outputs(ctx, "mesh").get("mesh")
            if not scene or not Path(scene).exists():
                raise StageError("No OpenMVS dense scene (.mvs) from 'dense' stage.")
            if not mesh or not Path(mesh).exists():
                raise StageError("No mesh from 'mesh' stage to texture.")
        elif not self.prior_outputs(ctx, "mesh").get("mesh"):
            raise StageError("No mesh from 'mesh' stage.")

    def run(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        if ctx.config.mvs.backend is MvsBackend.COLMAP:
            return self._passthrough(ctx)
        return self._run_openmvs(ctx)

    # ---- colmap: carry vertex colors through ------------------------------ #
    def _passthrough(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        mesh = self.prior_outputs(ctx, "mesh")["mesh"]
        get_logger().info("Texture: colmap backend — carrying vertex colors (no UV atlas).")
        return (
            {"mode": "vertex_colors", "has_uv": False},
            [],
            {"textured_mesh": mesh, "has_uv": "false"},
        )

    # ---- openmvs: UV texture --------------------------------------------- #
    def _run_openmvs(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        ws = ctx.workspace
        envx = openmvs_lib_env(ctx.env)
        texturer = ctx.env.tools["TextureMesh"].path
        # Texture the reconstructed mesh against the dense scene (cameras+images).
        # This ReconstructMesh build emits only the mesh .ply (not a mesh .mvs),
        # so we pass the dense scene + --mesh-file rather than a mesh .mvs.
        scene = Path(self.prior_outputs(ctx, "dense")["mvs_scene"])  # scene_dense.mvs
        mesh_ply = Path(self.prior_outputs(ctx, "mesh")["mesh"])     # scene_dense_mesh(.ply)
        tcfg = ctx.config.texture

        run_command(
            [texturer, scene.name,
             "--mesh-file", mesh_ply.name,
             "-o", "scene_textured.mvs",
             "--export-type", "obj",
             "--resolution-level", str(tcfg.resolution_level),
             "--max-texture-size", str(tcfg.texture_size)],
            cwd=ws.dense_dir, env_required=ctx.env, extra_env=envx,
        )

        obj = ws.dense_dir / "scene_textured.obj"
        if not obj.exists():
            candidates = sorted(
                ws.dense_dir.glob("*textured*.obj"), key=lambda p: p.stat().st_mtime, reverse=True
            ) or sorted(ws.dense_dir.glob("*.obj"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise StageError(
                    f"TextureMesh produced no OBJ in {ws.dense_dir}. Texturing failed."
                )
            obj = candidates[0]

        # Locate the texture image referenced alongside the OBJ.
        images = sorted(
            list(ws.dense_dir.glob(f"{obj.stem}*.png")) + list(ws.dense_dir.glob(f"{obj.stem}*.jpg"))
        )
        tex_w = tex_h = 0
        if images:
            try:
                from PIL import Image

                with Image.open(images[0]) as im:
                    tex_w, tex_h = im.size
            except Exception:  # noqa: BLE001
                pass

        get_logger().info("Texture: UV atlas %s (%dx%d) -> %s",
                          images[0].name if images else "?", tex_w, tex_h, obj.name)
        metrics: dict[str, Any] = {
            "mode": "uv_texture",
            "has_uv": True,
            "texture_image": images[0].name if images else None,
            "texture_size_px": [tex_w, tex_h],
            "max_texture_size": tcfg.texture_size,
        }
        return metrics, [], {"textured_mesh": str(obj), "has_uv": "true"}

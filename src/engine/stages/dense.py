"""Stage 6 — dense reconstruction (MVS), two backends.

* colmap : ``patch_match_stereo`` -> ``stereo_fusion`` -> ``fused.ply``
* openmvs: ``InterfaceCOLMAP`` -> ``DensifyPointCloud`` -> ``scene_dense.ply``

Reference commands:
    # colmap
    colmap patch_match_stereo --workspace_path <dense> --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true --PatchMatchStereo.gpu_index 0 \
        --PatchMatchStereo.max_image_size <mvs.max_image_size>
    colmap stereo_fusion --workspace_path <dense> --workspace_format COLMAP \
        --input_type geometric --output_path <dense>/fused.ply

    # openmvs (run inside <dense>)
    InterfaceCOLMAP -i <dense> -o scene.mvs --image-folder <dense>/images
    DensifyPointCloud scene.mvs --resolution-level <mvs.resolution_level>

VRAM: the dense step is where a 6 GB GPU runs out first. CUDA-OOM is detected
from stderr and turned into an actionable message (lower mvs.max_image_size or
raise mvs.resolution_level). We do NOT silently retry smaller unless
``mvs.auto_downscale`` is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..binaries import CommandError, require, run_command
from ..config import MvsBackend
from ..logging import get_logger
from ..masks import write_openmvs_masks
from ..openmvs import (
    COLMAP_NO_CUDA_SIGNATURE,
    count_ply_points,
    cuda_device_arg,
    looks_like_cuda_error,
    looks_like_oom,
    openmvs_lib_env,
)
from .base import Stage, StageContext, StageError

# Optional hook for a future milestone (turntable masks); see project notes.
MASKS_DIRNAME = "masks"


def _oom_message(ctx: StageContext, backend: str) -> str:
    mvs = ctx.config.mvs
    return (
        f"CUDA out of memory during dense reconstruction ({backend}). "
        f"The GPU ran out of VRAM. Lower mvs.max_image_size (currently "
        f"{mvs.max_image_size}) or raise mvs.resolution_level (currently "
        f"{mvs.resolution_level}); or pass --auto-downscale to retry "
        "automatically at a coarser setting."
    )


class DenseStage(Stage):
    name = "dense"

    def validate_inputs(self, ctx: StageContext) -> None:
        backend = ctx.config.mvs.backend
        require(ctx.env, ["colmap"])
        if backend is MvsBackend.OPENMVS:
            require(ctx.env, ["InterfaceCOLMAP", "DensifyPointCloud"])
        dense_ws = self.prior_outputs(ctx, "undistortion").get("dense_workspace")
        if not dense_ws or not Path(dense_ws).exists():
            raise StageError("No dense workspace from 'undistortion'. Run it first.")

    def run(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        if ctx.config.mvs.backend is MvsBackend.COLMAP:
            return self._run_colmap(ctx)
        return self._run_openmvs(ctx)

    # ---- colmap backend --------------------------------------------------- #
    def _run_colmap(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        ws = ctx.workspace
        colmap = ctx.env.tools["colmap"].path
        gpu_index = 0 if (ctx.config.mvs.use_gpu and ctx.env.cuda_available) else -1

        pm = run_command(
            [colmap, "patch_match_stereo",
             "--workspace_path", str(ws.dense_dir),
             "--workspace_format", "COLMAP",
             "--PatchMatchStereo.geom_consistency", "true",
             "--PatchMatchStereo.gpu_index", str(gpu_index),
             "--PatchMatchStereo.max_image_size", str(ctx.config.mvs.max_image_size)],
            env_required=ctx.env, check=False,
        )
        combined = (pm.stdout + pm.stderr).lower()
        # COLMAP logs (and exits 0 on) this when built without CUDA.
        if COLMAP_NO_CUDA_SIGNATURE in combined:
            raise StageError(
                "COLMAP dense stereo requires a CUDA-enabled COLMAP build, but the "
                "installed colmap is CPU-only (no CUDA). Use '--mvs-backend openmvs' "
                "(OpenMVS provides its own CUDA dense reconstruction), or install a "
                "CUDA build of COLMAP."
            )
        if pm.returncode != 0:
            if looks_like_oom(combined):
                raise StageError(_oom_message(ctx, "colmap"))
            raise CommandError(pm.args, pm.returncode, pm.stdout, pm.stderr)

        fused = ws.dense_dir / "fused.ply"
        run_command(
            [colmap, "stereo_fusion",
             "--workspace_path", str(ws.dense_dir),
             "--workspace_format", "COLMAP",
             "--input_type", "geometric",
             "--output_path", str(fused)],
            env_required=ctx.env,
        )
        return self._finish(ctx, fused, "colmap")

    # ---- openmvs backend -------------------------------------------------- #
    def _run_openmvs(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        ws = ctx.workspace
        envx = openmvs_lib_env(ctx.env)
        iface = ctx.env.tools["InterfaceCOLMAP"].path
        densify = ctx.env.tools["DensifyPointCloud"].path
        cuda_dev = cuda_device_arg(ctx.env, ctx.config.mvs.use_gpu)

        # 1) Convert the COLMAP dense workspace into an OpenMVS scene.
        run_command(
            [iface, "-i", str(ws.dense_dir), "-o", "scene.mvs",
             "--image-folder", str(ws.dense_dir / "images")],
            cwd=ws.dense_dir, env_required=ctx.env, extra_env=envx,
        )

        # Optional: place .mask.png files next to the undistorted images so
        # DensifyPointCloud ignores masked-out (background) pixels. Best-effort —
        # the OBB crop is the alignment-independent backstop.
        warnings: list[str] = []
        # Masks are optional; only consult the features stage if it recorded one.
        mask_dir = None
        if ctx.workspace.metrics_file("features").exists():
            mask_dir = self.prior_outputs(ctx, "features").get("mask_dir")
        ignore_mask = False
        if mask_dir and Path(mask_dir).is_dir():
            n = write_openmvs_masks(Path(mask_dir), ws.dense_dir / "images")
            if n > 0:
                ignore_mask = True
                warnings.append(
                    f"OpenMVS densify masking enabled ({n} masks). Note: masks are not "
                    "undistorted, so this is approximate; the OBB crop is the backstop."
                )

        # 2) Densify, with optional auto-downscale retry on CUDA-OOM.
        res_level = ctx.config.mvs.resolution_level
        attempts = ctx.config.mvs.auto_downscale_attempts if ctx.config.mvs.auto_downscale else 0
        for attempt in range(attempts + 1):
            cmd = [densify, "scene.mvs",
                   "--resolution-level", str(res_level),
                   "--max-resolution", str(ctx.config.mvs.max_image_size),
                   "--cuda-device", str(cuda_dev)]
            if ignore_mask:
                cmd += ["--ignore-mask-label", "0"]  # 0 = background pixels to ignore
            dp = run_command(
                cmd, cwd=ws.dense_dir, env_required=ctx.env, extra_env=envx, check=False,
            )
            if dp.returncode == 0:
                break
            combined = (dp.stdout + dp.stderr).lower()
            if looks_like_oom(combined) and attempt < attempts:
                res_level += 1
                warnings.append(
                    f"CUDA-OOM at resolution-level {res_level - 1}; "
                    f"auto-downscaling to resolution-level {res_level} and retrying."
                )
                get_logger().warning(warnings[-1])
                continue
            if looks_like_oom(combined):
                raise StageError(_oom_message(ctx, "openmvs"))
            # CUDA init/symbol failure (not OOM): the OpenMVS CUDA build does not
            # match this GPU/driver. Actionable, not a raw crash.
            if cuda_dev >= 0 and looks_like_cuda_error(combined):
                raise StageError(
                    "OpenMVS DensifyPointCloud failed with a CUDA runtime error "
                    "(not out-of-memory) — the OpenMVS CUDA build does not match this "
                    "GPU/driver. Re-run with '--mvs-cpu' to densify on CPU, or rebuild "
                    "OpenMVS for your GPU architecture. Underlying error:\n"
                    + "\n".join(dp.stderr.strip().splitlines()[-5:])
                )
            raise CommandError(dp.args, dp.returncode, dp.stdout, dp.stderr)

        dense_ply = ws.dense_dir / "scene_dense.ply"
        metrics, w2, outputs = self._finish(ctx, dense_ply, "openmvs")
        metrics["resolution_level_used"] = res_level
        metrics["cuda_device"] = cuda_dev  # -1 == CPU
        outputs["mvs_scene"] = str(ws.dense_dir / "scene_dense.mvs")
        return metrics, warnings + w2, outputs

    # ---- shared tail ------------------------------------------------------ #
    def _finish(
        self, ctx: StageContext, cloud_path: Path, backend: str
    ) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        if not cloud_path.exists():
            raise StageError(
                f"Dense reconstruction did not produce {cloud_path.name}. "
                "Check the stage log above for the failing command."
            )
        points = count_ply_points(cloud_path)
        if points == 0:
            raise StageError(
                f"Dense point cloud {cloud_path.name} is empty — fusion failed."
            )
        get_logger().info("Dense (%s): %d points -> %s", backend, points, cloud_path.name)
        metrics: dict[str, Any] = {"backend": backend, "dense_points": points}
        return metrics, [], {"dense_cloud": str(cloud_path)}

"""Stage 5 — undistortion (COLMAP ``image_undistorter``).

Prepares a dense workspace from the verified sparse model: undistorts the
images and writes a COLMAP dense layout consumed by *both* MVS backends.

Reference command:
    colmap image_undistorter \
        --image_path <images> \
        --input_path <sparse>/0 \
        --output_path <dense> \
        --output_type COLMAP \
        --max_image_size <mvs.max_image_size>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..binaries import require, run_command
from ..imaging import discover_images
from .base import Stage, StageContext, StageError


class UndistortStage(Stage):
    name = "undistortion"

    def validate_inputs(self, ctx: StageContext) -> None:
        require(ctx.env, ["colmap"])
        sparse_model = self.prior_outputs(ctx, "sfm").get("sparse_model")
        if not sparse_model or not Path(sparse_model).exists():
            raise StageError(
                "No sparse model from the 'sfm' stage. Run SfM first."
            )

    def run(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        ws = ctx.workspace
        colmap = ctx.env.tools["colmap"].path
        sparse_model = Path(self.prior_outputs(ctx, "sfm")["sparse_model"])
        ws.dense_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            colmap, "image_undistorter",
            "--image_path", str(ws.image_dir),
            "--input_path", str(sparse_model),
            "--output_path", str(ws.dense_dir),
            "--output_type", "COLMAP",
            "--max_image_size", str(ctx.config.mvs.max_image_size),
        ]
        run_command(cmd, env_required=ctx.env)

        undist_images = ws.dense_dir / "images"
        n = len(discover_images(undist_images)) if undist_images.is_dir() else 0
        if n == 0:
            raise StageError(
                f"Undistortion produced no images in {undist_images}."
            )

        metrics: dict[str, Any] = {
            "max_image_size": ctx.config.mvs.max_image_size,
            "num_undistorted_images": n,
        }
        return metrics, [], {
            "dense_workspace": str(ws.dense_dir),
            "undistorted_images": str(undist_images),
        }

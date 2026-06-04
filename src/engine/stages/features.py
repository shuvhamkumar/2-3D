"""Stage 2 — feature extraction (COLMAP ``feature_extractor``).

Reference command:
    colmap feature_extractor \
        --database_path database.db \
        --image_path images/ \
        --ImageReader.camera_model OPENCV \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 1 \
        --SiftExtraction.max_image_size 3200 \
        --SiftExtraction.max_num_features 8192
"""

from __future__ import annotations

from typing import Any

from ..binaries import colmap_gpu_flag, require, run_command
from ..colmap import read_database_stats
from ..logging import get_logger
from ..masks import prepare_masks
from .base import Stage, StageContext, StageError


class FeatureExtractionStage(Stage):
    name = "features"

    def validate_inputs(self, ctx: StageContext) -> None:
        require(ctx.env, ["colmap"])
        if not any(ctx.workspace.image_dir.iterdir()):
            raise StageError(f"No images in {ctx.workspace.image_dir}")

    def run(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        ws = ctx.workspace
        fcfg = ctx.config.feature
        colmap = ctx.env.tools["colmap"].path
        gpu = colmap_gpu_flag(ctx.env, fcfg.use_gpu)
        warnings: list[str] = []

        # Object isolation: restrict feature extraction to the masked region so
        # the sparse cloud is object-only (masks applied to original images).
        mask_info = prepare_masks(ws.image_dir, ws.root, ctx.config.masking)

        cmd = [
            colmap, "feature_extractor",
            "--database_path", str(ws.db_path),
            "--image_path", str(ws.image_dir),
            "--ImageReader.camera_model", fcfg.camera_model.value,
            "--ImageReader.single_camera", "1" if fcfg.single_camera else "0",
            "--SiftExtraction.use_gpu", str(gpu),
            "--SiftExtraction.max_image_size", str(fcfg.max_image_size),
            "--SiftExtraction.max_num_features", str(fcfg.max_num_features),
        ]
        if mask_info is not None:
            cmd += ["--ImageReader.mask_path", str(mask_info.mask_dir)]
            get_logger().info(
                "Masking ON: %d/%d images (%.0f%% coverage), source=%s",
                mask_info.n_masks, mask_info.n_images, mask_info.coverage * 100,
                mask_info.source,
            )
            if mask_info.coverage < 1.0:
                warnings.append(
                    f"Mask coverage only {mask_info.coverage:.0%} "
                    f"({mask_info.n_masks}/{mask_info.n_images} images)."
                )
        run_command(cmd, env_required=ctx.env)

        stats = read_database_stats(ws.db_path)
        if stats.num_descriptors_images == 0:
            raise StageError(
                "Feature extraction produced no keypoints for any image — "
                "images may be blank, tiny, or (if masking) fully masked out."
            )

        if stats.num_descriptors_images < stats.num_images:
            warnings.append(
                f"{stats.num_images - stats.num_descriptors_images} image(s) got no "
                "features and will not register."
            )

        outputs = {"database": str(ws.db_path)}
        metrics: dict[str, Any] = {
            "camera_model": fcfg.camera_model.value,
            "single_camera": fcfg.single_camera,
            "gpu": bool(gpu),
            "num_images": stats.num_images,
            "num_images_with_features": stats.num_descriptors_images,
            "total_keypoints": stats.num_keypoints,
            "mean_keypoints_per_image": round(
                stats.num_keypoints / max(1, stats.num_descriptors_images), 1
            ),
            "masking": mask_info.to_metrics() if mask_info else None,
        }
        if mask_info is not None:
            outputs["mask_dir"] = str(mask_info.mask_dir)
        return metrics, warnings, outputs

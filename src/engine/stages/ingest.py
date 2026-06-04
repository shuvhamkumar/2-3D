"""Stage 1 — ingest & validate.

Scans the image folder and rejects the job *early* with specifics if it is
unlikely to reconstruct: too few images, wildly inconsistent resolutions,
mostly-blurry frames, or missing EXIF focal priors (when required).

Produces an ingest report (per-image dimensions / sharpness / EXIF) plus
aggregate stats. This stage runs no external binaries.
"""

from __future__ import annotations

from typing import Any

from ..imaging import ImageInfo, discover_images, inspect_image
from ..logging import get_logger
from .base import Stage, StageContext, StageError


class IngestStage(Stage):
    name = "ingest"

    def validate_inputs(self, ctx: StageContext) -> None:
        image_dir = ctx.workspace.image_dir
        if not image_dir.exists() or not image_dir.is_dir():
            raise StageError(f"Image directory does not exist: {image_dir}")

    def run(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        cfg = ctx.config.ingest
        logger = get_logger()
        image_dir = ctx.workspace.image_dir

        paths = discover_images(image_dir)
        if len(paths) < cfg.min_images:
            raise StageError(
                f"Only {len(paths)} usable image(s) found in {image_dir} "
                f"(minimum {cfg.min_images}). Supported extensions: "
                "jpg/jpeg/png/tif/tiff/bmp."
            )

        logger.info("Inspecting %d images …", len(paths))
        infos: list[ImageInfo] = []
        unreadable: list[str] = []
        for p in paths:
            try:
                infos.append(inspect_image(p))
            except Exception as exc:  # corrupt/unsupported file
                unreadable.append(f"{p.name}: {exc}")

        if len(infos) < cfg.min_images:
            raise StageError(
                f"Only {len(infos)} of {len(paths)} images were readable "
                f"(minimum {cfg.min_images}). Unreadable: {unreadable}"
            )

        warnings: list[str] = []
        if unreadable:
            warnings.append(f"{len(unreadable)} unreadable file(s) skipped: {unreadable}")

        # ---- resolution consistency -------------------------------------- #
        areas = [i.area for i in infos]
        min_area, max_area = min(areas), max(areas)
        res_ratio = (max_area / min_area) if min_area else float("inf")
        if res_ratio > cfg.max_resolution_ratio:
            raise StageError(
                f"Inconsistent image resolutions: largest/smallest area ratio "
                f"{res_ratio:.1f} exceeds limit {cfg.max_resolution_ratio}. "
                "Mixed resolutions/orientations often indicate a bad capture set."
            )

        # ---- sharpness / blur -------------------------------------------- #
        sharps = [i.sharpness for i in infos]
        blurry = [i for i in infos if i.sharpness < cfg.blur_threshold]
        blurry_fraction = len(blurry) / len(infos)
        if blurry_fraction > cfg.max_blurry_fraction:
            raise StageError(
                f"{len(blurry)}/{len(infos)} images "
                f"({blurry_fraction:.0%}) are below the sharpness threshold "
                f"({cfg.blur_threshold}); limit is {cfg.max_blurry_fraction:.0%}. "
                "Capture appears mostly blurry."
            )
        if blurry:
            warnings.append(
                f"{len(blurry)} image(s) below sharpness threshold "
                f"{cfg.blur_threshold} (kept): "
                + ", ".join(sorted(b.path.name for b in blurry)[:10])
                + ("…" if len(blurry) > 10 else "")
            )

        # ---- EXIF focal priors ------------------------------------------- #
        with_focal = [i for i in infos if i.has_focal_exif]
        focal_fraction = len(with_focal) / len(infos)
        if cfg.require_exif_focal and focal_fraction < 1.0:
            raise StageError(
                f"EXIF focal length required but only {len(with_focal)}/{len(infos)} "
                "images carry it. Disable ingest.require_exif_focal or supply EXIF."
            )
        if not cfg.require_exif_focal and focal_fraction < 1.0:
            warnings.append(
                f"{len(infos) - len(with_focal)}/{len(infos)} images lack EXIF focal "
                "length; COLMAP will estimate intrinsics from scratch."
            )

        metrics: dict[str, Any] = {
            "image_dir": str(image_dir),
            "num_discovered": len(paths),
            "num_readable": len(infos),
            "num_unreadable": len(unreadable),
            "resolution": {
                "min_wh": [min(i.width for i in infos), min(i.height for i in infos)],
                "max_wh": [max(i.width for i in infos), max(i.height for i in infos)],
                "area_ratio": round(res_ratio, 3),
            },
            "sharpness": {
                "min": round(min(sharps), 2),
                "mean": round(sum(sharps) / len(sharps), 2),
                "max": round(max(sharps), 2),
                "threshold": cfg.blur_threshold,
                "num_below_threshold": len(blurry),
            },
            "exif_focal": {
                "num_with_focal": len(with_focal),
                "fraction": round(focal_fraction, 3),
            },
        }
        logger.info(
            "Ingest OK: %d images, sharpness mean %.0f (%d blurry), %.0f%% with EXIF focal.",
            len(infos), metrics["sharpness"]["mean"], len(blurry), focal_fraction * 100,
        )
        return metrics, warnings, {}

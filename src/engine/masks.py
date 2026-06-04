"""Object isolation via masks.

Two complementary mechanisms (see also the OBB backstop in :mod:`engine.geometry`):

1. **Masked SfM** — COLMAP ``--ImageReader.mask_path`` restricts feature
   extraction to the object, so the *sparse* cloud is object-only. This is the
   reliable path: masks are applied to the original (distorted) images, so there
   is no undistortion-alignment problem.
2. **Masked densify** — OpenMVS reads ``<image_stem>.mask.png`` files next to the
   (undistorted) images, gated by ``--ignore-mask-label``. Best-effort: the
   undistorter does not warp masks, so this is accurate only for low-distortion
   cameras. The OBB crop is the alignment-independent backstop.

Masks are binary PNGs: 0 = ignore, 255 = use.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .imaging import discover_images
from .logging import get_logger


class MaskError(RuntimeError):
    """Raised for mask configuration / generation problems."""


@dataclass
class MaskInfo:
    mask_dir: Path
    n_images: int
    n_masks: int
    source: str  # "external" | "auto(rembg)" | "sibling"

    @property
    def coverage(self) -> float:
        return self.n_masks / self.n_images if self.n_images else 0.0

    def to_metrics(self) -> dict:
        return {
            "mask_dir": str(self.mask_dir),
            "source": self.source,
            "images": self.n_images,
            "masks": self.n_masks,
            "coverage": round(self.coverage, 3),
        }


def colmap_mask_filename(image_name: str) -> str:
    """COLMAP looks for ``<mask_path>/<image_name>.png``."""
    return f"{image_name}.png"


def openmvs_mask_filename(image_name: str) -> str:
    """OpenMVS looks for ``<image_stem>.mask.png`` beside the image."""
    return f"{Path(image_name).stem}.mask.png"


def _count_coverage(image_dir: Path, mask_dir: Path) -> tuple[int, int]:
    images = discover_images(image_dir)
    have = sum(1 for img in images if (mask_dir / colmap_mask_filename(img.name)).exists())
    return len(images), have


def resolve_mask_dir(image_dir: Path, job_root: Path, cfg) -> tuple[Path | None, str]:
    """Decide where masks come from, without generating them.

    Returns ``(mask_dir_or_None, source_label)``.
    """
    if cfg.mask_dir is not None:
        return Path(cfg.mask_dir), "external"
    if cfg.auto_mask:
        return job_root / "masks", "auto(rembg)"
    sibling = image_dir.parent / "masks"
    if sibling.is_dir():
        return sibling, "sibling"
    return None, "none"


def prepare_masks(image_dir: Path, job_root: Path, cfg) -> MaskInfo | None:
    """Resolve (and if needed generate) masks; return coverage info or None."""
    mask_dir, source = resolve_mask_dir(image_dir, job_root, cfg)
    if mask_dir is None:
        return None

    if cfg.auto_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)
        n_img, n_have = _count_coverage(image_dir, mask_dir)
        if n_have < n_img:  # generate the missing ones
            generate_masks_rembg(image_dir, mask_dir, dilate_frac=cfg.mask_dilate_frac)

    if not mask_dir.is_dir():
        raise MaskError(
            f"Mask directory does not exist: {mask_dir}. Provide --masks <dir> with "
            f"binary PNGs named '<image>.png' (0=ignore, 255=use), or use --auto-mask."
        )

    # External masks + dilation: materialise dilated copies into a job-local dir
    # so the originals are never modified. (Auto-masks are dilated at generation.)
    if cfg.mask_dilate_frac > 0 and source != "auto(rembg)":
        dilated_dir = job_root / "masks_dilated"
        _materialise_dilated(image_dir, mask_dir, dilated_dir, cfg.mask_dilate_frac)
        mask_dir, source = dilated_dir, f"{source}+dilated"

    n_img, n_have = _count_coverage(image_dir, mask_dir)
    if n_have == 0:
        raise MaskError(
            f"No masks found in {mask_dir} matching images in {image_dir}. "
            "Expected COLMAP-named masks '<image_name>.png' (e.g. 'frame001.JPG.png')."
        )
    info = MaskInfo(mask_dir=mask_dir, n_images=n_img, n_masks=n_have, source=source)
    if info.coverage < 1.0:
        get_logger().warning(
            "Mask coverage %.0f%% (%d/%d images). Unmasked images keep all features.",
            info.coverage * 100, n_have, n_img,
        )
    return info


def dilate_mask(arr: "np.ndarray", px: int) -> "np.ndarray":
    """Grow the foreground (>127) of a binary mask by ``px`` pixels."""
    import numpy as np

    fg = arr > 127
    if px <= 0 or not fg.any():
        return (fg.astype(np.uint8)) * 255
    from scipy.ndimage import distance_transform_edt

    # distance from each background pixel to the nearest foreground pixel
    dist = distance_transform_edt(~fg)
    grown = fg | (dist <= px)
    return grown.astype(np.uint8) * 255


def _materialise_dilated(image_dir: Path, src_dir: Path, dst_dir: Path, frac: float) -> None:
    import numpy as np
    from PIL import Image

    dst_dir.mkdir(parents=True, exist_ok=True)
    for img in discover_images(image_dir):
        src = src_dir / colmap_mask_filename(img.name)
        if not src.exists():
            continue
        with Image.open(src) as m:
            arr = np.asarray(m.convert("L"))
        px = int(round(frac * min(arr.shape)))
        Image.fromarray(dilate_mask(arr, px), "L").save(dst_dir / colmap_mask_filename(img.name))


def generate_masks_rembg(image_dir: Path, mask_dir: Path, dilate_frac: float = 0.0) -> None:
    """Generate per-image binary masks with rembg (optional dependency)."""
    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise MaskError(
            "--auto-mask requires the optional 'rembg' package: pip install rembg. "
            "Alternatively supply external masks with --masks <dir>."
        ) from exc
    import numpy as np
    from PIL import Image

    mask_dir.mkdir(parents=True, exist_ok=True)
    session = new_session()
    logger = get_logger()
    images = discover_images(image_dir)
    logger.info("Auto-masking %d images with rembg (dilate_frac=%.3f) …", len(images), dilate_frac)
    for img_path in images:
        out = mask_dir / colmap_mask_filename(img_path.name)
        if out.exists():
            continue
        with Image.open(img_path) as im:
            rgba = remove(im.convert("RGB"), session=session, only_mask=True)
        # rembg only_mask returns an L image (alpha); binarise to 0/255.
        arr = np.asarray(rgba.convert("L"))
        if dilate_frac > 0:
            binary = dilate_mask(arr, int(round(dilate_frac * min(arr.shape))))
        else:
            binary = (arr > 127).astype("uint8") * 255
        Image.fromarray(binary, mode="L").save(out)


def write_openmvs_masks(mask_dir: Path, dense_images_dir: Path) -> int:
    """Place ``<stem>.mask.png`` files next to the undistorted images.

    Copies each COLMAP-named mask to the OpenMVS naming convention so
    DensifyPointCloud picks them up. Returns the number written.
    """
    if not dense_images_dir.is_dir() or not mask_dir.is_dir():
        return 0
    written = 0
    for img in discover_images(dense_images_dir):
        src = mask_dir / colmap_mask_filename(img.name)
        if src.exists():
            shutil.copyfile(src, dense_images_dir / openmvs_mask_filename(img.name))
            written += 1
    return written

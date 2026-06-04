"""Image inspection helpers: discovery, sharpness, resolution, EXIF.

Dependency-light on purpose -- uses Pillow + numpy only (no OpenCV), so the
ingest stage works in a minimal install.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ExifTags

# Extensions COLMAP/OpenMVS can read.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Long edge to which images are downscaled before computing sharpness, so the
# variance-of-Laplacian threshold is comparable across differing resolutions.
_SHARPNESS_LONG_EDGE = 1024

_EXIF_FOCAL = {ExifTags.Base.FocalLength.value, ExifTags.Base.FocalLengthIn35mmFilm.value}


@dataclass
class ImageInfo:
    path: Path
    width: int
    height: int
    sharpness: float  # variance of Laplacian (downscaled grayscale)
    has_focal_exif: bool

    @property
    def area(self) -> int:
        return self.width * self.height


def discover_images(image_dir: Path) -> list[Path]:
    """Return sorted image paths under ``image_dir`` (non-recursive)."""
    files = [
        p
        for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return files


def _variance_of_laplacian(gray: np.ndarray) -> float:
    """4-neighbour discrete Laplacian variance on the image interior."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = (
        -4.0 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def _has_focal_exif(img: Image.Image) -> bool:
    try:
        exif = img.getexif()
    except Exception:
        return False
    if not exif:
        return False
    return any(tag in exif for tag in _EXIF_FOCAL)


def inspect_image(path: Path) -> ImageInfo:
    """Open one image and extract dimensions, sharpness, and EXIF-focal flag."""
    with Image.open(path) as img:
        width, height = img.size
        has_focal = _has_focal_exif(img)
        # Downscale + grayscale for a resolution-comparable sharpness measure.
        gray = img.convert("L")
        long_edge = max(gray.size)
        if long_edge > _SHARPNESS_LONG_EDGE:
            scale = _SHARPNESS_LONG_EDGE / long_edge
            new_size = (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
            gray = gray.resize(new_size, Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float64)
    sharpness = _variance_of_laplacian(arr)
    return ImageInfo(
        path=path,
        width=width,
        height=height,
        sharpness=sharpness,
        has_focal_exif=has_focal,
    )

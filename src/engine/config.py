"""Configuration model and presets.

A single :class:`EngineConfig` (pydantic v2) drives the whole pipeline. It is
loadable from YAML/JSON and overridable by CLI flags. Three presets --
``draft``, ``balanced``, ``high`` -- trade speed for quality.

Defaults here are tuned for the project's primary use case:
small-object scanning for 3D printing, captured as ordered/video frames,
exported to ``.glb``. See README for what each preset changes.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Enumerations                                                                 #
# --------------------------------------------------------------------------- #
class Preset(str, Enum):
    DRAFT = "draft"
    BALANCED = "balanced"
    HIGH = "high"


class UseCase(str, Enum):
    OBJECT = "object"  # small-object / 3D-printing (project default)
    OUTDOOR = "outdoor"  # survey / large scene
    WEB = "web"  # lightweight web-viewable assets


class CameraModel(str, Enum):
    SIMPLE_RADIAL = "SIMPLE_RADIAL"
    RADIAL = "RADIAL"
    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    PINHOLE = "PINHOLE"


class MatcherStrategy(str, Enum):
    AUTO = "auto"  # choose by image count / use case at runtime
    EXHAUSTIVE = "exhaustive"
    SEQUENTIAL = "sequential"
    VOCAB_TREE = "vocab_tree"
    SPATIAL = "spatial"


class MvsBackend(str, Enum):
    COLMAP = "colmap"
    OPENMVS = "openmvs"


class MeshBackend(str, Enum):
    # COLMAP meshers
    POISSON = "poisson"
    DELAUNAY = "delaunay"
    # OpenMVS
    OPENMVS = "openmvs"


class OutputFormat(str, Enum):
    GLB = "glb"
    OBJ = "obj"
    PLY = "ply"


# --------------------------------------------------------------------------- #
# Stage sub-configs                                                            #
# --------------------------------------------------------------------------- #
class IngestConfig(BaseModel):
    min_images: int = Field(8, ge=2, description="Reject jobs with fewer images.")
    # Variance-of-Laplacian below this => frame flagged as blurry.
    blur_threshold: float = Field(100.0, gt=0)
    # If more than this fraction of frames are blurry, reject the job.
    max_blurry_fraction: float = Field(0.5, ge=0, le=1)
    # Reject if the ratio of largest:smallest image area exceeds this.
    max_resolution_ratio: float = Field(4.0, ge=1)
    require_exif_focal: bool = Field(
        False, description="If True, fail when EXIF focal-length priors are absent."
    )


class FeatureConfig(BaseModel):
    camera_model: CameraModel = CameraModel.OPENCV
    single_camera: bool = Field(
        True, description="Share intrinsics across all images (one physical camera)."
    )
    max_image_size: int = Field(
        3200, gt=0, description="Downscale long edge to this before extraction."
    )
    max_num_features: int = Field(8192, gt=0)
    use_gpu: bool = True


class MatchingConfig(BaseModel):
    strategy: MatcherStrategy = MatcherStrategy.AUTO
    use_gpu: bool = True
    # For sequential matcher: number of neighbouring frames to match.
    sequential_overlap: int = Field(10, ge=1)
    # Threshold for AUTO -> exhaustive vs vocab_tree on unordered sets.
    exhaustive_max_images: int = Field(150, ge=1)
    vocab_tree_path: Path | None = None


class SfmConfig(BaseModel):
    # Quality gates -- degenerate results are flagged, not hidden.
    min_registered_fraction: float = Field(0.6, ge=0, le=1)
    max_mean_reproj_error: float = Field(1.5, gt=0, description="pixels")
    # When the mapper yields multiple disconnected models, keep the largest.
    keep_largest_model_only: bool = True


class MvsConfig(BaseModel):
    backend: MvsBackend = MvsBackend.OPENMVS
    # Long-edge cap for dense step; lower = faster/coarser.
    max_image_size: int = Field(2000, gt=0)
    # OpenMVS DensifyPointCloud resolution-level (0=full res, higher=coarser).
    resolution_level: int = Field(1, ge=0)
    use_gpu: bool = True
    # On a CUDA out-of-memory failure, retry at a coarser setting instead of
    # aborting. Off by default: we fail with actionable guidance rather than
    # silently producing a lower-quality result the user didn't ask for.
    auto_downscale: bool = False
    # Max coarsening retries when auto_downscale is on.
    auto_downscale_attempts: int = Field(2, ge=1, le=4)


class MeshConfig(BaseModel):
    backend: MeshBackend = MeshBackend.OPENMVS
    # OpenMVS RefineMesh -- expensive; off for draft.
    refine: bool = False
    # COLMAP poisson depth (higher = finer, slower).
    poisson_depth: int = Field(11, ge=5, le=14)
    # COLMAP poisson trim: higher = trims more low-density (extrapolated) area.
    poisson_trim: int = Field(7, ge=0, le=20)


class TextureConfig(BaseModel):
    # OpenMVS TextureMesh resolution-level (0=full res).
    resolution_level: int = Field(0, ge=0)
    # Max texture atlas dimension (maps to OpenMVS --max-texture-size).
    texture_size: int = Field(8192, gt=0)


class MaskingConfig(BaseModel):
    """Object isolation for turntable/orbit captures.

    Masks restrict SfM + dense reconstruction to the object (binary PNG,
    0 = ignore / 255 = use). The OBB crop is a geometric backstop that removes
    surviving background even when masks are imperfect or absent.
    """

    # Directory of externally-provided masks (the primary, always-available
    # path). COLMAP convention: ``<mask_dir>/<image_name>.png``.
    mask_dir: Path | None = None
    # Generate per-frame masks with rembg (optional extra dependency).
    auto_mask: bool = False
    # Dilate masks by this fraction of the image's smaller side before use.
    # Tight masks on a small-in-frame object starve SfM (few features -> poor
    # registration); a small dilation keeps surrounding context for matching
    # while still excluding most background. 0 = no dilation.
    mask_dilate_frac: float = Field(0.0, ge=0, le=0.25)
    # Geometric backstop: crop to the oriented bbox of the sparse object points.
    bbox_crop: bool = True
    # Fractional margin added to each OBB half-extent before cropping.
    bbox_margin: float = Field(0.15, ge=0, le=2.0)


class ExportConfig(BaseModel):
    format: OutputFormat = OutputFormat.GLB
    # Optional decimation target (number of faces). None = no decimation.
    target_faces: int | None = None
    # Mesh cleanup (export stage): keep largest connected component and strip
    # statistical outliers. Reported honestly; meshes are never auto-repaired.
    keep_largest_component: bool = True
    # Statistical outlier removal: k neighbours + std-dev multiplier.
    outlier_neighbors: int = Field(20, ge=0)
    outlier_std_ratio: float = Field(2.0, gt=0)
    # Safety cap: if more than this fraction would be flagged, the surface is
    # near-uniform (no real outliers) and removal is skipped to avoid holes.
    outlier_max_fraction: float = Field(0.02, ge=0, le=1)
    # Components smaller than this fraction of the largest are dropped as junk.
    min_component_fraction: float = Field(0.05, ge=0, le=1)


# --------------------------------------------------------------------------- #
# Top-level config                                                             #
# --------------------------------------------------------------------------- #
class EngineConfig(BaseModel):
    """Complete pipeline configuration."""

    preset: Preset = Preset.BALANCED
    use_case: UseCase = UseCase.OBJECT

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    sfm: SfmConfig = Field(default_factory=SfmConfig)
    mvs: MvsConfig = Field(default_factory=MvsConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    texture: TextureConfig = Field(default_factory=TextureConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    model_config = {"extra": "forbid", "validate_assignment": True}

    # ---- loaders ---------------------------------------------------------- #
    @classmethod
    def from_file(cls, path: str | Path) -> "EngineConfig":
        """Load config from a YAML or JSON file.

        Honors an optional top-level ``preset:`` key: the preset is applied
        first, then any other keys in the file override the preset values.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            data: dict[str, Any] = yaml.safe_load(text) or {}
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            raise ValueError(
                f"Unsupported config extension {path.suffix!r}; use .yaml/.yml/.json"
            )
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must contain a mapping at top level.")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        """Build a config from a dict, applying a named preset as the base."""
        data = dict(data)  # shallow copy; don't mutate caller's dict
        preset_name = data.pop("preset", None)
        if preset_name is not None:
            base = build_preset(Preset(preset_name))
            merged = _deep_merge(base.model_dump(mode="python"), data)
            return cls.model_validate(merged)
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Preset construction                                                          #
# --------------------------------------------------------------------------- #
def build_preset(preset: Preset, use_case: UseCase = UseCase.OBJECT) -> EngineConfig:
    """Return a fully-populated config for a named preset.

    What each preset changes (speed <-> quality):

    * ``draft``     -- small images, coarse MVS, no mesh refine, no texturing
                       polish. Fast smoke-test quality.
    * ``balanced``  -- moderate resolution, OpenMVS dense+mesh, light refine.
    * ``high``      -- full resolution, fine MVS, mesh refinement on, large
                       textures. Slow, best quality.
    """
    cfg = EngineConfig(preset=preset, use_case=use_case)

    if preset is Preset.DRAFT:
        cfg.feature.max_image_size = 1600
        cfg.feature.max_num_features = 4096
        cfg.mvs.max_image_size = 1000
        cfg.mvs.resolution_level = 3
        cfg.mesh.refine = False
        cfg.mesh.poisson_depth = 9
        cfg.texture.resolution_level = 2
        cfg.texture.texture_size = 2048
    elif preset is Preset.BALANCED:
        cfg.feature.max_image_size = 3200
        cfg.feature.max_num_features = 8192
        cfg.mvs.max_image_size = 2000
        cfg.mvs.resolution_level = 1
        cfg.mesh.refine = True
        cfg.mesh.poisson_depth = 11
        cfg.texture.resolution_level = 1
        cfg.texture.texture_size = 4096
    elif preset is Preset.HIGH:
        cfg.feature.max_image_size = 5000
        cfg.feature.max_num_features = 16384
        cfg.mvs.max_image_size = 3200
        cfg.mvs.resolution_level = 0
        cfg.mesh.refine = True
        cfg.mesh.poisson_depth = 13
        cfg.texture.resolution_level = 0
        cfg.texture.texture_size = 8192

    _apply_use_case_defaults(cfg, use_case)
    return cfg


def _apply_use_case_defaults(cfg: EngineConfig, use_case: UseCase) -> None:
    """Adjust matcher/camera defaults based on capture style."""
    if use_case is UseCase.OBJECT:
        # Turntable/orbit video of one object: ordered frames, shared intrinsics.
        cfg.matching.strategy = MatcherStrategy.SEQUENTIAL
        cfg.feature.single_camera = True
    elif use_case is UseCase.OUTDOOR:
        cfg.matching.strategy = MatcherStrategy.SPATIAL
        cfg.feature.single_camera = False
        cfg.ingest.require_exif_focal = True
    elif use_case is UseCase.WEB:
        cfg.matching.strategy = MatcherStrategy.AUTO
        # Lightweight output: decimate to a web-friendly budget.
        if cfg.export.target_faces is None:
            cfg.export.target_faces = 100_000


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out

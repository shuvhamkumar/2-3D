"""Pipeline stages, in execution order.

Later milestones append undistort -> dense -> mesh -> texture -> export.
"""

from __future__ import annotations

from .base import Stage, StageContext, StageError, StageResult, StageStatus
from .dense import DenseStage
from .export import ExportStage
from .features import FeatureExtractionStage
from .ingest import IngestStage
from .matching import MatchingStage
from .mesh import MeshStage
from .sfm import SfmStage
from .texture import TextureStage
from .undistort import UndistortStage

#: Ordered list of pipeline stages (single source of truth for sequencing).
ALL_STAGES: list[Stage] = [
    IngestStage(),
    FeatureExtractionStage(),
    MatchingStage(),
    SfmStage(),
    UndistortStage(),
    DenseStage(),
    MeshStage(),
    TextureStage(),
    ExportStage(),
]

STAGE_NAMES: list[str] = [s.name for s in ALL_STAGES]

__all__ = [
    "Stage",
    "StageContext",
    "StageError",
    "StageResult",
    "StageStatus",
    "ALL_STAGES",
    "STAGE_NAMES",
    "IngestStage",
    "FeatureExtractionStage",
    "MatchingStage",
    "SfmStage",
    "UndistortStage",
    "DenseStage",
    "MeshStage",
    "TextureStage",
    "ExportStage",
]

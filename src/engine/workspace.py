"""Deterministic job-directory layout.

All stage outputs live at predictable paths under the job root so that stages
are resumable: a stage is "complete" iff its metrics file exists. The input
image directory is kept *separate* from the job root and never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    root: Path  # the --output job directory
    image_dir: Path  # user-provided input images (read-only)

    # ---- COLMAP / OpenMVS artifact locations ----------------------------- #
    @property
    def db_path(self) -> Path:
        return self.root / "database.db"

    @property
    def sparse_dir(self) -> Path:
        return self.root / "sparse"

    @property
    def dense_dir(self) -> Path:
        return self.root / "dense"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def report_path(self) -> Path:
        return self.root / "report.json"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    def metrics_file(self, stage: str) -> Path:
        return self.metrics_dir / f"{stage}.json"

    def ensure(self) -> None:
        for d in (self.root, self.metrics_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

"""Tests for the ingest/validate stage using synthetic images (no binaries)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from engine.binaries import Environment
from engine.config import Preset, UseCase, build_preset
from engine.stages.base import StageContext, StageError, StageStatus
from engine.stages.ingest import IngestStage
from engine.workspace import Workspace


def _write_image(path, size=(640, 480), kind="sharp"):
    rng = np.random.default_rng(abs(hash(str(path))) % (2**32))
    if kind == "sharp":
        arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    else:  # blurry: smooth gradient, near-zero high-frequency content
        x = np.linspace(0, 60, size[0])
        row = np.tile(x, (size[1], 1))
        arr = np.stack([row, row, row], axis=-1).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _ctx(image_dir, tmp_path, use_case=UseCase.OBJECT):
    ws = Workspace(root=tmp_path / "job", image_dir=image_dir)
    ws.ensure()
    cfg = build_preset(Preset.DRAFT, use_case)
    return StageContext(workspace=ws, config=cfg, env=Environment())


def test_ingest_accepts_good_set(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(10):
        _write_image(img_dir / f"img_{i:02d}.png", kind="sharp")

    result = IngestStage().execute(_ctx(img_dir, tmp_path))
    assert result.status is StageStatus.COMPLETED
    assert result.metrics["num_readable"] == 10
    assert result.metrics["sharpness"]["mean"] > 0


def test_ingest_rejects_too_few(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        _write_image(img_dir / f"img_{i}.png")
    result = IngestStage().execute(_ctx(img_dir, tmp_path))
    assert result.status is StageStatus.FAILED
    assert "minimum" in result.error


def test_ingest_rejects_inconsistent_resolution(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(8):
        _write_image(img_dir / f"img_{i}.png", size=(640, 480))
    # One wildly larger image -> area ratio blows past the limit.
    _write_image(img_dir / "huge.png", size=(4000, 3000))
    result = IngestStage().execute(_ctx(img_dir, tmp_path))
    assert result.status is StageStatus.FAILED
    assert "resolution" in result.error.lower()


def test_ingest_rejects_mostly_blurry(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(10):
        _write_image(img_dir / f"blur_{i}.png", kind="blurry")
    result = IngestStage().execute(_ctx(img_dir, tmp_path))
    assert result.status is StageStatus.FAILED
    assert "blurry" in result.error.lower()


def test_ingest_warns_on_missing_exif(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(10):
        _write_image(img_dir / f"img_{i}.png")  # PNGs carry no EXIF focal
    result = IngestStage().execute(_ctx(img_dir, tmp_path))
    assert result.status is StageStatus.COMPLETED
    assert any("EXIF focal" in w for w in result.warnings)


def test_ingest_requires_exif_when_outdoor(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(10):
        _write_image(img_dir / f"img_{i}.png")
    # OUTDOOR preset sets require_exif_focal=True.
    result = IngestStage().execute(_ctx(img_dir, tmp_path, UseCase.OUTDOOR))
    assert result.status is StageStatus.FAILED
    assert "EXIF focal" in result.error

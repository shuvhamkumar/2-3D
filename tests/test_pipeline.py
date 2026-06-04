"""Tests for stage sequencing and resume/skip logic (no binaries needed)."""

from __future__ import annotations

import pytest

from engine.binaries import Environment
from engine.config import Preset, build_preset
from engine.pipeline import _resolve_range
from engine.stages import STAGE_NAMES
from engine.stages.base import Stage, StageContext, StageStatus
from engine.workspace import Workspace


def test_stage_order_is_expected():
    assert STAGE_NAMES == [
        "ingest", "features", "matching", "sfm",
        "undistortion", "dense", "mesh", "texture", "export",
    ]


def test_resolve_range_full():
    stages = _resolve_range(None, None)
    assert [s.name for s in stages] == STAGE_NAMES


def test_resolve_range_to_sfm_stops_after_sfm():
    stages = _resolve_range(None, "sfm")
    assert [s.name for s in stages] == ["ingest", "features", "matching", "sfm"]


def test_resolve_range_to_dense_stops_partway():
    stages = _resolve_range(None, "dense")
    assert [s.name for s in stages][-1] == "dense"
    assert "export" not in [s.name for s in stages]


def test_resolve_range_to_export_is_full_chain():
    stages = _resolve_range(None, "export")
    assert [s.name for s in stages] == STAGE_NAMES


def test_resolve_range_to_ingest_only():
    stages = _resolve_range(None, "ingest")
    assert [s.name for s in stages] == ["ingest"]


def test_resolve_range_from_dense_to_texture():
    stages = _resolve_range("dense", "texture")
    assert [s.name for s in stages] == ["dense", "mesh", "texture"]


def test_resolve_range_from_matching():
    stages = _resolve_range("matching", None)
    assert [s.name for s in stages][0] == "matching"
    assert [s.name for s in stages][-1] == "export"


def test_resolve_range_unknown_stage():
    with pytest.raises(ValueError, match="not a known stage"):
        _resolve_range(None, "bogus")


def test_resolve_range_from_after_to():
    with pytest.raises(ValueError, match="comes after"):
        _resolve_range("sfm", "ingest")


# --- resume / skip ---------------------------------------------------------- #
class _CountingStage(Stage):
    name = "ingest"  # reuse a real name so metrics path resolves

    def __init__(self):
        self.run_count = 0

    def validate_inputs(self, ctx):  # noqa: D401
        pass

    def run(self, ctx):
        self.run_count += 1
        return {"ran": self.run_count}, [], {}


def _ctx(tmp_path, force=False):
    ws = Workspace(root=tmp_path / "job", image_dir=tmp_path / "imgs")
    ws.ensure()
    return StageContext(workspace=ws, config=build_preset(Preset.DRAFT),
                        env=Environment(), force=force)


def test_stage_runs_then_skips_on_resume(tmp_path):
    stage = _CountingStage()
    ctx = _ctx(tmp_path)

    r1 = stage.execute(ctx)
    assert r1.status is StageStatus.COMPLETED
    assert stage.run_count == 1

    # Second run: metrics file exists -> skipped, run() not called again.
    r2 = stage.execute(ctx)
    assert r2.status is StageStatus.SKIPPED
    assert stage.run_count == 1
    assert r2.metrics == {"ran": 1}  # metrics preserved from disk


def test_force_reruns_completed_stage(tmp_path):
    stage = _CountingStage()
    ctx = _ctx(tmp_path)
    stage.execute(ctx)
    ctx.force = True
    r2 = stage.execute(ctx)
    assert r2.status is StageStatus.COMPLETED
    assert stage.run_count == 2


class _FailingStage(Stage):
    name = "features"

    def validate_inputs(self, ctx):
        pass

    def run(self, ctx):
        raise RuntimeError("boom")


def test_failed_stage_records_error_and_is_not_complete(tmp_path):
    stage = _FailingStage()
    ctx = _ctx(tmp_path)
    result = stage.execute(ctx)
    assert result.status is StageStatus.FAILED
    assert "boom" in result.error
    # A failed stage must not count as complete on resume.
    assert stage.is_complete(ctx) is False

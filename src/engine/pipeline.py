"""Pipeline orchestrator: sequence stages, honour --from/--to, resume, report.

This is the importable Python API that the CLI wraps. Example::

    from engine.pipeline import run_pipeline
    report = run_pipeline(image_dir, output_dir, config=build_preset(Preset.DRAFT),
                          to_stage="sfm")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .binaries import Environment, detect_environment
from .config import EngineConfig
from .logging import get_logger
from .report import build_report, write_report
from .stages import ALL_STAGES, STAGE_NAMES
from .stages.base import StageContext, StageResult, StageStatus
from .workspace import Workspace


def _resolve_range(from_stage: str | None, to_stage: str | None) -> list:
    """Return the slice of stages to run, validating the names."""
    for label, name in (("--from", from_stage), ("--to", to_stage)):
        if name is not None and name not in STAGE_NAMES:
            raise ValueError(
                f"{label} {name!r} is not a known stage. Choose from: {', '.join(STAGE_NAMES)}"
            )
    start = STAGE_NAMES.index(from_stage) if from_stage else 0
    end = STAGE_NAMES.index(to_stage) + 1 if to_stage else len(STAGE_NAMES)
    if start >= end:
        raise ValueError(f"--from {from_stage!r} comes after --to {to_stage!r}.")
    return ALL_STAGES[start:end]


def run_pipeline(
    image_dir: str | Path,
    output_dir: str | Path,
    config: EngineConfig,
    *,
    env: Environment | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run the selected stages and return the aggregated report dict.

    The report is also persisted to ``<output_dir>/report.json`` after *each*
    stage, so an interrupted job still leaves an up-to-date report.
    """
    logger = get_logger()
    env = env or detect_environment()
    workspace = Workspace(root=Path(output_dir).resolve(), image_dir=Path(image_dir).resolve())
    workspace.ensure()

    # Snapshot the resolved config alongside the outputs (provenance).
    workspace.config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    stages = _resolve_range(from_stage, to_stage)
    logger.info("Running stages: %s", " -> ".join(s.name for s in stages))

    ctx = StageContext(workspace=workspace, config=config, env=env, force=force)
    results: list[StageResult] = []
    status = "completed"

    for stage in stages:
        result = stage.execute(ctx)
        results.append(result)
        # Persist incrementally so interruption still yields a report.
        write_report(workspace, build_report(workspace, config.preset.value, results, "partial"))
        if result.status is StageStatus.FAILED:
            status = "failed"
            logger.error("Pipeline stopped at failed stage '%s'.", stage.name)
            break

    # Final report covers ALL persisted stages (this run + earlier resumed runs),
    # not just the stages executed in this invocation.
    from .report import collect_stage_results

    all_results = collect_stage_results(workspace)
    report = build_report(workspace, config.preset.value, all_results or results, status)
    write_report(workspace, report)
    return report

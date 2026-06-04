"""The :class:`Stage` abstraction.

Every stage follows the same contract: ``validate inputs -> run -> record
metrics -> mark complete``. Completion is persisted as a metrics JSON file, so
re-running a job skips already-completed stages unless ``force`` is set.

Stages never silently swallow failures: an exception during ``run`` is captured
into a ``failed`` :class:`StageResult` (with the full error text) and surfaced
to the orchestrator, which stops the pipeline.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..binaries import Environment
from ..config import EngineConfig
from ..logging import get_logger, stage_timer
from ..workspace import Workspace


class StageStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"  # already complete, reused from a previous run
    FAILED = "failed"


@dataclass
class StageContext:
    workspace: Workspace
    config: EngineConfig
    env: Environment
    force: bool = False


@dataclass
class StageResult:
    name: str
    status: StageStatus
    elapsed_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "elapsed_s": round(self.elapsed_s, 3),
            "metrics": self.metrics,
            "warnings": self.warnings,
            "outputs": self.outputs,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageResult":
        return cls(
            name=data["name"],
            status=StageStatus(data["status"]),
            elapsed_s=data.get("elapsed_s", 0.0),
            metrics=data.get("metrics", {}),
            warnings=data.get("warnings", []),
            outputs=data.get("outputs", {}),
            error=data.get("error"),
        )

    @property
    def ok(self) -> bool:
        return self.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)


class StageError(RuntimeError):
    """Raised by a stage's ``validate``/``run`` to signal a clean failure."""


class Stage(ABC):
    """Base class for pipeline stages."""

    #: unique stage name; also the metrics filename and the --from/--to token.
    name: str

    # ---- subclass hooks --------------------------------------------------- #
    @abstractmethod
    def validate_inputs(self, ctx: StageContext) -> None:
        """Raise :class:`StageError` if prerequisites are missing/invalid."""

    @abstractmethod
    def run(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        """Do the work. Return ``(metrics, warnings, outputs)``."""

    # ---- reading upstream results ---------------------------------------- #
    @staticmethod
    def _read_metrics_file(ctx: StageContext, stage_name: str) -> dict[str, Any]:
        path = ctx.workspace.metrics_file(stage_name)
        if not path.exists():
            raise StageError(
                f"Required upstream stage '{stage_name}' has not run "
                f"(missing {path.name}). Run it first."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def prior_outputs(self, ctx: StageContext, stage_name: str) -> dict[str, str]:
        """Return the ``outputs`` recorded by a previously-run stage."""
        return self._read_metrics_file(ctx, stage_name).get("outputs", {})

    def prior_metrics(self, ctx: StageContext, stage_name: str) -> dict[str, Any]:
        """Return the ``metrics`` recorded by a previously-run stage."""
        return self._read_metrics_file(ctx, stage_name).get("metrics", {})

    # ---- completion bookkeeping ------------------------------------------ #
    def is_complete(self, ctx: StageContext) -> bool:
        path = ctx.workspace.metrics_file(self.name)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("status") == StageStatus.COMPLETED.value
        except (json.JSONDecodeError, OSError):
            return False

    def _load_result(self, ctx: StageContext) -> StageResult:
        data = json.loads(ctx.workspace.metrics_file(self.name).read_text(encoding="utf-8"))
        result = StageResult.from_dict(data)
        result.status = StageStatus.SKIPPED  # reused, not re-run
        return result

    def _persist(self, ctx: StageContext, result: StageResult) -> None:
        ctx.workspace.metrics_dir.mkdir(parents=True, exist_ok=True)
        ctx.workspace.metrics_file(self.name).write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )

    # ---- the uniform driver ---------------------------------------------- #
    def execute(self, ctx: StageContext) -> StageResult:
        logger = get_logger()
        if self.is_complete(ctx) and not ctx.force:
            logger.info("[dim]⏭ %s already complete — skipping (use --force to re-run)[/]",
                        self.name, extra={"markup": True})
            return self._load_result(ctx)

        try:
            self.validate_inputs(ctx)
            with stage_timer(self.name) as t:
                metrics, warnings, outputs = self.run(ctx)
            result = StageResult(
                name=self.name,
                status=StageStatus.COMPLETED,
                elapsed_s=t["elapsed_s"],
                metrics=metrics,
                warnings=warnings,
                outputs=outputs,
            )
            for w in warnings:
                logger.warning("[%s] %s", self.name, w)
            self._persist(ctx, result)
            return result
        except Exception as exc:  # noqa: BLE001 -- surface everything, hide nothing
            result = StageResult(
                name=self.name,
                status=StageStatus.FAILED,
                error=str(exc),
            )
            self._persist(ctx, result)
            logger.error("[bold red]FAILED %s[/]\n%s", self.name, exc, extra={"markup": True})
            return result

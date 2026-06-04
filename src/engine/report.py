"""Aggregate per-stage results into a single ``report.json`` + pretty summary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.table import Table

from .stages.base import StageResult, StageStatus
from .workspace import Workspace


def collect_stage_results(workspace: Workspace) -> list[StageResult]:
    """Load every persisted stage result from ``metrics/`` in pipeline order.

    This makes a job's report reflect *all* completed stages, even across
    separate (resumed) ``engine run`` invocations — not just the stages run in
    the most recent call.
    """
    # Imported here to avoid a circular import at module load.
    from .stages import STAGE_NAMES

    results: list[StageResult] = []
    for name in STAGE_NAMES:
        path = workspace.metrics_file(name)
        if path.exists():
            try:
                results.append(StageResult.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError, KeyError):
                continue
    return results


def build_report(
    workspace: Workspace,
    preset: str,
    results: list[StageResult],
    status: str,
) -> dict[str, Any]:
    warnings = [f"[{r.name}] {w}" for r in results for w in r.warnings]
    summary: dict[str, Any] = {}
    by_name = {r.name: r.metrics for r in results if r.metrics}

    # Surface the headline numbers from each phase at the top level when present.
    if "sfm" in by_name:
        m = by_name["sfm"]
        summary["sfm"] = {
            "registered_images": m.get("registered_images"),
            "total_images": m.get("total_images"),
            "registered_fraction": m.get("registered_fraction"),
            "mean_reprojection_error_px": m.get("mean_reprojection_error_px"),
            "mean_track_length": m.get("mean_track_length"),
            "total_sparse_points": m.get("total_sparse_points"),
            "quality_ok": m.get("quality_ok"),
        }
    if by_name.get("features", {}).get("masking"):
        summary["masking"] = by_name["features"]["masking"]
    if "dense" in by_name:
        summary["dense_points"] = by_name["dense"].get("dense_points")
    if "mesh" in by_name:
        summary["mesh_faces"] = by_name["mesh"].get("mesh_faces")
        if by_name["mesh"].get("bbox_crop"):
            summary["bbox_crop"] = by_name["mesh"]["bbox_crop"]
    if "export" in by_name:
        m = by_name["export"]
        summary["export"] = {
            "faces_after_cleanup": m.get("faces_after_cleanup"),
            "watertight": m.get("watertight"),
            "edge_manifold": m.get("edge_manifold"),
            "glb_path": m.get("glb_path"),
            "glb_bytes": m.get("glb_bytes"),
        }

    return {
        "job": {
            "image_dir": str(workspace.image_dir),
            "output": str(workspace.root),
            "preset": preset,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "status": status,
        "summary": summary,
        "warnings": warnings,
        "stages": [r.to_dict() for r in results],
    }


def write_report(workspace: Workspace, report: dict[str, Any]) -> None:
    workspace.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_report(workspace: Workspace) -> dict[str, Any]:
    return json.loads(workspace.report_path.read_text(encoding="utf-8"))


def print_summary(report: dict[str, Any], console: Console) -> None:
    """Render a human-readable summary of a (partial or finished) job."""
    job = report.get("job", {})
    status = report.get("status", "unknown")
    colour = {"completed": "green", "failed": "red", "partial": "yellow"}.get(status, "white")
    console.rule(f"[bold {colour}]Job {status.upper()}[/]")
    console.print(f"images: {job.get('image_dir')}")
    console.print(f"output: {job.get('output')}    preset: {job.get('preset')}")

    # Per-stage timing/status table.
    table = Table(title="Stages")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Time (s)", justify="right")
    for s in report.get("stages", []):
        st = s["status"]
        mark = {
            StageStatus.COMPLETED.value: "[green]completed[/]",
            StageStatus.SKIPPED.value: "[cyan]skipped[/]",
            StageStatus.FAILED.value: "[red]failed[/]",
        }.get(st, st)
        table.add_row(s["name"], mark, f"{s.get('elapsed_s', 0):.1f}")
    console.print(table)

    summary = report.get("summary") or {}
    sfm = summary.get("sfm") or {}
    if sfm:
        flag = sfm.get("quality_ok")
        badge = "[green]OK[/]" if flag else "[yellow]REVIEW[/]" if flag is not None else ""
        console.print(f"\n[bold]SfM[/] {badge}")
        console.print(
            f"  registered: {sfm.get('registered_images')}/{sfm.get('total_images')} "
            f"({(sfm.get('registered_fraction') or 0) * 100:.0f}%)"
        )
        console.print(f"  mean reprojection error: {sfm.get('mean_reprojection_error_px')} px")
        console.print(f"  mean track length:       {sfm.get('mean_track_length')}")
        console.print(f"  sparse points:           {sfm.get('total_sparse_points')}")

    mask = summary.get("masking")
    if mask:
        console.print(
            f"\n[bold]Masking[/] ({mask.get('source')}): "
            f"{mask.get('masks')}/{mask.get('images')} images "
            f"({(mask.get('coverage') or 0) * 100:.0f}% coverage)"
        )
    crop = summary.get("bbox_crop")
    if crop:
        tgt = crop.get("target", "?")
        console.print(
            f"[bold]OBB crop[/] ({tgt}): "
            f"{crop.get(f'{tgt}_before')} → {crop.get(f'{tgt}_after')} "
            f"({(crop.get('fraction_removed') or 0) * 100:.1f}% removed)"
        )

    if "dense_points" in summary or "mesh_faces" in summary or "export" in summary:
        console.print("\n[bold]Dense → mesh → export[/]")
        if summary.get("dense_points") is not None:
            console.print(f"  dense points:  {summary['dense_points']:,}")
        if summary.get("mesh_faces") is not None:
            console.print(f"  mesh faces:    {summary['mesh_faces']:,}")
        exp = summary.get("export") or {}
        if exp:
            faces = exp.get("faces_after_cleanup")
            console.print(f"  final faces:   {faces:,}" if faces is not None else "  final faces:   ?")
            console.print(
                f"  watertight: {exp.get('watertight')}   "
                f"edge-manifold: {exp.get('edge_manifold')}"
            )
            mb = (exp.get("glb_bytes") or 0) / 1e6
            console.print(f"  glb: {exp.get('glb_path')} ({mb:.1f} MB)")

    warnings = report.get("warnings", [])
    if warnings:
        console.print(f"\n[bold yellow]Warnings ({len(warnings)})[/]")
        for w in warnings:
            console.print(f"  • {w}")

    # If a stage failed, surface its error verbatim.
    for s in report.get("stages", []):
        if s["status"] == StageStatus.FAILED.value and s.get("error"):
            console.print(f"\n[bold red]Error in {s['name']}:[/]\n{s['error']}")

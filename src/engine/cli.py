"""Command-line interface (``engine``).

Milestone 1 implements the environment/config-facing commands:

* ``engine doctor``        -- report detected binaries + CUDA.
* ``engine show-config``   -- resolve a preset (+ optional file) and print it.

``run`` / ``validate`` / ``report`` are declared here so the surface is stable,
but their stage logic lands in later milestones; they currently exit with a
clear "not yet implemented" message rather than pretending to work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from .binaries import detect_environment
from .config import EngineConfig, Preset, UseCase, build_preset
from .logging import configure_logging, get_console

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Photogrammetry pipeline: photos -> textured 3D model (COLMAP + OpenMVS).",
)


def _load_config(preset: Preset, use_case: UseCase, config_file: Optional[Path]) -> EngineConfig:
    if config_file is not None:
        return EngineConfig.from_file(config_file)
    return build_preset(preset, use_case)


@app.command()
def doctor() -> None:
    """Report detected external binaries and CUDA availability."""
    configure_logging()
    console = get_console()
    env = detect_environment()

    table = Table(title="External tools", show_lines=False)
    table.add_column("Tool")
    table.add_column("Status")
    table.add_column("Path")
    for name, status in env.tools.items():
        mark = "[green]found[/]" if status.available else "[red]MISSING[/]"
        table.add_row(name, mark, status.path or "—")
    console.print(table)

    if env.cuda_available:
        console.print(f"[green]CUDA:[/] available ({len(env.cuda_devices)} device(s))")
        for dev in env.cuda_devices:
            console.print(f"  • {dev}")
        colmap_gpu = "[green]yes[/]" if env.colmap_cuda else "[yellow]no (CPU-only build)[/]"
        console.print(f"  COLMAP CUDA build: {colmap_gpu}")
    else:
        console.print("[yellow]CUDA:[/] not detected — GPU stages will fall back to CPU")

    console.print(
        f"\nCOLMAP ready: {'[green]yes[/]' if env.has_colmap() else '[red]no[/]'}    "
        f"OpenMVS ready: {'[green]yes[/]' if env.has_openmvs() else '[red]no[/]'}"
    )


@app.command(name="show-config")
def show_config(
    preset: Preset = typer.Option(Preset.BALANCED, help="Preset to resolve."),
    use_case: UseCase = typer.Option(UseCase.OBJECT, help="Capture style."),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML/JSON config file (overrides preset)."
    ),
) -> None:
    """Resolve a preset (and optional config file) and print the full config."""
    cfg = _load_config(preset, use_case, config_file)
    get_console().print_json(json.dumps(cfg.to_dict()))


@app.command()
def validate(
    image_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    preset: Preset = typer.Option(Preset.BALANCED),
    use_case: UseCase = typer.Option(UseCase.OBJECT),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run only ingest/validation against a temp workspace and print the report."""
    from .pipeline import run_pipeline

    configure_logging()
    cfg = _load_config(preset, use_case, config_file)
    # Ingest is read-only; use a transient metrics dir next to the images.
    out = image_dir.resolve().parent / f".{image_dir.name}.validate"
    report = run_pipeline(image_dir, out, cfg, to_stage="ingest", force=True)
    _print_report(report)
    raise typer.Exit(0 if report["status"] == "completed" else 1)


@app.command()
def run(
    image_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output", "-o"),
    preset: Preset = typer.Option(Preset.BALANCED),
    use_case: UseCase = typer.Option(UseCase.OBJECT),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
    from_stage: Optional[str] = typer.Option(None, "--from", help="Resume from stage."),
    to_stage: Optional[str] = typer.Option(None, "--to", help="Stop after stage."),
    force: bool = typer.Option(False, "--force", help="Re-run completed stages."),
    matcher: Optional[str] = typer.Option(
        None, "--matcher", help="Override matcher (auto|exhaustive|sequential|vocab_tree|spatial)."
    ),
    mvs_backend: Optional[str] = typer.Option(
        None, "--mvs-backend", help="Override MVS backend (colmap|openmvs)."
    ),
    auto_downscale: bool = typer.Option(
        False, "--auto-downscale", help="On CUDA-OOM in dense, retry at a coarser resolution."
    ),
    mvs_cpu: bool = typer.Option(
        False, "--mvs-cpu", help="Run OpenMVS dense/refine on CPU (--cuda-device -1)."
    ),
    masks: Optional[Path] = typer.Option(
        None, "--masks", help="Directory of binary masks '<image>.png' (0=ignore, 255=use)."
    ),
    auto_mask: bool = typer.Option(
        False, "--auto-mask", help="Generate per-frame object masks with rembg (optional dep)."
    ),
    mask_dilate: Optional[float] = typer.Option(
        None, "--mask-dilate",
        help="Dilate masks by this fraction of the image's short side (e.g. 0.04) so SfM "
             "keeps context around a small object.",
    ),
    bbox_margin: Optional[float] = typer.Option(
        None, "--bbox-margin", help="Margin for the sparse-OBB crop backstop (e.g. 0.15)."
    ),
    no_bbox_crop: bool = typer.Option(
        False, "--no-bbox-crop", help="Disable the OBB geometric crop backstop."
    ),
) -> None:
    """Run the pipeline (use --to/--from to limit the stage range)."""
    from .config import MatcherStrategy, MvsBackend
    from .pipeline import run_pipeline

    configure_logging()
    cfg = _load_config(preset, use_case, config_file)
    if matcher is not None:
        cfg.matching.strategy = MatcherStrategy(matcher)
    if mvs_backend is not None:
        cfg.mvs.backend = MvsBackend(mvs_backend)
    if auto_downscale:
        cfg.mvs.auto_downscale = True
    if mvs_cpu:
        cfg.mvs.use_gpu = False
    if masks is not None:
        cfg.masking.mask_dir = masks
    if auto_mask:
        cfg.masking.auto_mask = True
    if mask_dilate is not None:
        cfg.masking.mask_dilate_frac = mask_dilate
    if bbox_margin is not None:
        cfg.masking.bbox_margin = bbox_margin
    if no_bbox_crop:
        cfg.masking.bbox_crop = False

    report = run_pipeline(
        image_dir, output, cfg,
        from_stage=from_stage, to_stage=to_stage, force=force,
    )
    _print_report(report)
    raise typer.Exit(0 if report["status"] != "failed" else 1)


@app.command()
def report(job_dir: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """Print the metrics summary for a finished/partial job."""
    from .report import build_report, collect_stage_results, load_report
    from .workspace import Workspace

    configure_logging()
    ws = Workspace(root=job_dir, image_dir=job_dir)
    # Prefer rebuilding from per-stage metrics so the summary covers the whole
    # job (across resumed runs); fall back to a saved report.json.
    results = collect_stage_results(ws)
    if results:
        preset = "unknown"
        if ws.config_path.exists():
            try:
                preset = json.loads(ws.config_path.read_text()).get("preset", "unknown")
            except (json.JSONDecodeError, OSError):
                pass
        status = "failed" if any(r.status.value == "failed" for r in results) else "completed"
        _print_report(build_report(ws, preset, results, status))
    elif ws.report_path.exists():
        _print_report(load_report(ws))
    else:
        get_console().print(f"[red]No report.json or metrics in {job_dir}[/]")
        raise typer.Exit(1)


def _print_report(report: dict) -> None:
    from .report import print_summary

    print_summary(report, get_console())


if __name__ == "__main__":  # pragma: no cover
    app()

"""Stage 4 — sparse reconstruction (COLMAP ``mapper``).

Recovers camera poses + a sparse point cloud and runs bundle adjustment
internally. The mapper may yield several disconnected models; we keep the
largest (most registered images) and warn about the rest.

Reference command:
    colmap mapper \
        --database_path database.db \
        --image_path images/ \
        --output_path sparse/
"""

from __future__ import annotations

from typing import Any

from ..binaries import require, run_command
from ..colmap import read_database_stats, select_largest_model
from ..logging import get_logger
from .base import Stage, StageContext, StageError


class SfmStage(Stage):
    name = "sfm"

    def validate_inputs(self, ctx: StageContext) -> None:
        require(ctx.env, ["colmap"])
        if not ctx.workspace.db_path.exists():
            raise StageError(
                f"Feature database not found: {ctx.workspace.db_path}. "
                "Run features + matching first."
            )

    def run(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        ws = ctx.workspace
        colmap = ctx.env.tools["colmap"].path
        ws.sparse_dir.mkdir(parents=True, exist_ok=True)

        cmd = [colmap, "mapper",
               "--database_path", str(ws.db_path),
               "--image_path", str(ws.image_dir),
               "--output_path", str(ws.sparse_dir)]
        run_command(cmd, env_required=ctx.env)

        best_dir, best, summaries = select_largest_model(colmap, ws.sparse_dir, ctx.env)

        warnings: list[str] = []
        if len(summaries) > 1:
            others = [
                f"{s['model']} ({s.get('registered_images', 0)} imgs)"
                for s in summaries if s["model"] != best_dir.name
            ]
            warnings.append(
                f"Mapper produced {len(summaries)} disconnected models; kept "
                f"'{best_dir.name}' ({best.get('registered_images', 0)} imgs). "
                f"Discarded: {', '.join(others)}."
            )

        registered = int(best.get("registered_images", 0)) or int(best.get("images", 0))
        # Total = images in the database (all inputs that got features), NOT the
        # largest model's image count — otherwise a partial reconstruction looks
        # like "100% registered" when most images never registered.
        total = read_database_stats(ws.db_path).num_images or registered
        reg_fraction = registered / total if total else 0.0
        reproj = float(best.get("mean_reproj_error", 0.0))
        if registered < total:
            warnings.append(
                f"{total - registered} of {total} images did not register into the "
                f"largest model (registered {registered}/{total} = {reg_fraction:.0%})."
            )

        # ---- degeneracy flags (warnings, not silent) --------------------- #
        scfg = ctx.config.sfm
        if reg_fraction < scfg.min_registered_fraction:
            warnings.append(
                f"DEGENERATE: only {reg_fraction:.0%} of images registered "
                f"({registered}/{total}); threshold {scfg.min_registered_fraction:.0%}. "
                "Reconstruction is likely partial/unreliable."
            )
        if reproj > scfg.max_mean_reproj_error:
            warnings.append(
                f"High mean reprojection error {reproj:.2f}px "
                f"(threshold {scfg.max_mean_reproj_error}px) — geometry may be inaccurate."
            )

        metrics: dict[str, Any] = {
            "best_model": best_dir.name,
            "num_models": len(summaries),
            "registered_images": registered,
            "total_images": total,
            "registered_fraction": round(reg_fraction, 3),
            "mean_reprojection_error_px": round(reproj, 3),
            "mean_track_length": round(float(best.get("mean_track_length", 0.0)), 3),
            "total_sparse_points": int(best.get("points", 0)),
            "observations": int(best.get("observations", 0)),
            "models": summaries,
            "quality_ok": reg_fraction >= scfg.min_registered_fraction
            and reproj <= scfg.max_mean_reproj_error,
        }
        get_logger().info(
            "SfM: %d/%d images registered (%.0f%%), %d points, "
            "reproj %.2fpx, track len %.2f",
            registered, total, reg_fraction * 100, metrics["total_sparse_points"],
            reproj, metrics["mean_track_length"],
        )
        return metrics, warnings, {"sparse_model": str(best_dir)}

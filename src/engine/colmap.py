"""Thin helpers around COLMAP artifacts.

We drive COLMAP through its CLI (via :func:`engine.binaries.run_command`) so
every reference command is visible and reproducible by hand. These helpers only
*read* COLMAP outputs to compute metrics:

* SQLite database counts (images, keypoints, matched pairs).
* ``model_analyzer`` summary parsing (registered images, reprojection error,
  track length, points).
* sparse-model enumeration + largest-model selection.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .binaries import Environment, run_command


# --------------------------------------------------------------------------- #
# Database introspection                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class DatabaseStats:
    num_images: int
    num_keypoints: int
    num_descriptors_images: int  # images that actually got features
    num_matched_pairs: int  # pairs in two_view_geometries (geometrically verified)
    num_inlier_matches: int


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def read_database_stats(db_path: Path) -> DatabaseStats:
    """Read counts from a COLMAP SQLite database."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        num_images = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]

        num_keypoints = 0
        num_desc_images = 0
        if _table_exists(con, "keypoints"):
            # keypoints.rows = number of features for that image.
            for (rows,) in con.execute("SELECT rows FROM keypoints"):
                if rows:
                    num_keypoints += rows
                    num_desc_images += 1

        num_pairs = 0
        num_inliers = 0
        if _table_exists(con, "two_view_geometries"):
            for (rows,) in con.execute("SELECT rows FROM two_view_geometries"):
                if rows and rows > 0:
                    num_pairs += 1
                    num_inliers += rows
        return DatabaseStats(
            num_images=num_images,
            num_keypoints=num_keypoints,
            num_descriptors_images=num_desc_images,
            num_matched_pairs=num_pairs,
            num_inlier_matches=num_inliers,
        )
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Sparse model enumeration / selection                                         #
# --------------------------------------------------------------------------- #
def list_sparse_models(sparse_dir: Path) -> list[Path]:
    """Return reconstruction sub-dirs (``0``, ``1``, …) that look like models."""
    if not sparse_dir.is_dir():
        return []
    models = []
    for sub in sorted(sparse_dir.iterdir()):
        if sub.is_dir() and (
            (sub / "cameras.bin").exists() or (sub / "cameras.txt").exists()
        ):
            models.append(sub)
    return models


_ANALYZER_PATTERNS: dict[str, re.Pattern[str]] = {
    "cameras": re.compile(r"Cameras:\s*(\d+)"),
    "images": re.compile(r"Images:\s*(\d+)"),
    "registered_images": re.compile(r"Registered images:\s*(\d+)"),
    "points": re.compile(r"Points:\s*(\d+)"),
    "observations": re.compile(r"Observations:\s*(\d+)"),
    "mean_track_length": re.compile(r"Mean track length:\s*([\d.]+)"),
    "mean_obs_per_image": re.compile(r"Mean observations per image:\s*([\d.]+)"),
    "mean_reproj_error": re.compile(r"Mean reprojection error:\s*([\d.]+)"),
}


def parse_model_analyzer(text: str) -> dict[str, float]:
    """Parse ``colmap model_analyzer`` output (it logs to stderr via glog)."""
    out: dict[str, float] = {}
    for key, pat in _ANALYZER_PATTERNS.items():
        m = pat.search(text)
        if m:
            val = m.group(1)
            out[key] = float(val) if "." in val else int(val)
    return out


def analyze_model(colmap_path: str, model_dir: Path, env: Environment) -> dict[str, Any]:
    """Run ``colmap model_analyzer`` on a model and return parsed metrics.

    Reference command:
        colmap model_analyzer --path <model_dir>
    """
    proc = run_command(
        [colmap_path, "model_analyzer", "--path", str(model_dir)],
        check=True,
        env_required=env,
    )
    # model_analyzer prints stats to stderr (glog); include stdout to be safe.
    return parse_model_analyzer(proc.stderr + "\n" + proc.stdout)


def select_largest_model(
    colmap_path: str, sparse_dir: Path, env: Environment
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Pick the model with the most registered images.

    Returns ``(best_dir, best_metrics, all_metrics)``. ``all_metrics`` carries a
    per-model summary so the orchestrator can warn about discarded models.
    """
    models = list_sparse_models(sparse_dir)
    if not models:
        raise FileNotFoundError(
            f"No sparse reconstruction found under {sparse_dir}. "
            "The mapper failed to register any images."
        )
    summaries: list[dict[str, Any]] = []
    for m in models:
        stats = analyze_model(colmap_path, m, env)
        summaries.append({"model": m.name, **stats})
    best_idx = max(
        range(len(summaries)),
        key=lambda i: summaries[i].get("registered_images", 0),
    )
    return models[best_idx], summaries[best_idx], summaries

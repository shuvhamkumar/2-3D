"""Stage 3 — feature matching.

Auto-selects a matcher by job type / size unless overridden:

* ``exhaustive_matcher``  — small unordered sets (≲ exhaustive_max_images).
* ``sequential_matcher``  — video / ordered captures.
* ``vocab_tree_matcher``  — large unordered sets (needs a vocab tree).
* ``spatial_matcher``     — geo-tagged captures.

Reference commands:
    colmap exhaustive_matcher  --database_path database.db --SiftMatching.use_gpu 1
    colmap sequential_matcher  --database_path database.db --SiftMatching.use_gpu 1 \
        --SequentialMatching.overlap 10
    colmap vocab_tree_matcher  --database_path database.db --SiftMatching.use_gpu 1 \
        --VocabTreeMatching.vocab_tree_path tree.bin
    colmap spatial_matcher     --database_path database.db --SiftMatching.use_gpu 1
"""

from __future__ import annotations

from typing import Any

from ..binaries import colmap_gpu_flag, require, run_command
from ..colmap import read_database_stats
from ..config import MatcherStrategy
from ..logging import get_logger
from .base import Stage, StageContext, StageError


def select_strategy(ctx: StageContext, num_images: int) -> MatcherStrategy:
    """Resolve the AUTO strategy to a concrete matcher."""
    requested = ctx.config.matching.strategy
    if requested is not MatcherStrategy.AUTO:
        return requested
    # AUTO: small sets -> exhaustive; otherwise vocab_tree if a tree is set,
    # else fall back to exhaustive with a warning handled by the caller.
    if num_images <= ctx.config.matching.exhaustive_max_images:
        return MatcherStrategy.EXHAUSTIVE
    if ctx.config.matching.vocab_tree_path is not None:
        return MatcherStrategy.VOCAB_TREE
    return MatcherStrategy.EXHAUSTIVE


class MatchingStage(Stage):
    name = "matching"

    def validate_inputs(self, ctx: StageContext) -> None:
        require(ctx.env, ["colmap"])
        if not ctx.workspace.db_path.exists():
            raise StageError(
                f"Feature database not found: {ctx.workspace.db_path}. "
                "Run the 'features' stage first."
            )

    def run(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        ws = ctx.workspace
        mcfg = ctx.config.matching
        colmap = ctx.env.tools["colmap"].path
        gpu = colmap_gpu_flag(ctx.env, mcfg.use_gpu)
        warnings: list[str] = []

        pre = read_database_stats(ws.db_path)
        strategy = select_strategy(ctx, pre.num_images)
        if (
            mcfg.strategy is MatcherStrategy.AUTO
            and pre.num_images > mcfg.exhaustive_max_images
            and mcfg.vocab_tree_path is None
        ):
            warnings.append(
                f"AUTO chose exhaustive matching for {pre.num_images} images "
                f"(> {mcfg.exhaustive_max_images}) because no vocab tree is "
                "configured; this is O(n²) and may be slow."
            )

        cmd = [colmap, f"{strategy.value}_matcher",
               "--database_path", str(ws.db_path),
               "--SiftMatching.use_gpu", str(gpu)]
        if strategy is MatcherStrategy.SEQUENTIAL:
            cmd += ["--SequentialMatching.overlap", str(mcfg.sequential_overlap)]
        elif strategy is MatcherStrategy.VOCAB_TREE:
            if mcfg.vocab_tree_path is None:
                raise StageError("vocab_tree strategy requires matching.vocab_tree_path.")
            cmd += ["--VocabTreeMatching.vocab_tree_path", str(mcfg.vocab_tree_path)]

        get_logger().info("Matching strategy: %s", strategy.value)
        run_command(cmd, env_required=ctx.env)

        post = read_database_stats(ws.db_path)
        if post.num_matched_pairs == 0:
            raise StageError(
                "Matching produced zero geometrically-verified image pairs — "
                "images may not overlap, or features are too weak to match. "
                "SfM cannot proceed."
            )

        metrics: dict[str, Any] = {
            "strategy": strategy.value,
            "gpu": bool(gpu),
            "num_images": post.num_images,
            "matched_pairs": post.num_matched_pairs,
            "total_inlier_matches": post.num_inlier_matches,
            "mean_inliers_per_pair": round(
                post.num_inlier_matches / max(1, post.num_matched_pairs), 1
            ),
        }
        return metrics, warnings, {}

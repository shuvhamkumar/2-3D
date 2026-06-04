"""Stage 7 — meshing, two backends.

* colmap : ``poisson_mesher`` (or ``delaunay_mesher``) on the fused cloud
           -> a vertex-colored mesh.
* openmvs: ``ReconstructMesh`` (+ ``RefineMesh`` when ``mesh.refine``)
           -> an untextured mesh the texture stage will UV-texture.

Reference commands:
    colmap poisson_mesher --input_path <dense>/fused.ply \
        --output_path <dense>/meshed-poisson.ply \
        --PoissonMeshing.depth <mesh.poisson_depth> --PoissonMeshing.trim <mesh.poisson_trim>

    ReconstructMesh scene_dense.mvs
    RefineMesh scene_dense_mesh.mvs --resolution-level 1   # only if mesh.refine
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..binaries import require, run_command
from ..config import MeshBackend, MvsBackend
from ..geometry import compute_obb, crop_cloud, crop_mesh, export_sparse_points
from ..logging import get_logger
from ..openmvs import cuda_device_arg, openmvs_lib_env, read_ply_element_counts
from .base import Stage, StageContext, StageError


class MeshStage(Stage):
    name = "mesh"

    def validate_inputs(self, ctx: StageContext) -> None:
        require(ctx.env, ["colmap"])
        if ctx.config.mvs.backend is MvsBackend.OPENMVS:
            require(ctx.env, ["ReconstructMesh"])
            if ctx.config.mesh.refine:
                require(ctx.env, ["RefineMesh"])
        dense = self.prior_outputs(ctx, "dense")
        if not dense.get("dense_cloud"):
            raise StageError("No dense cloud from 'dense' stage. Run it first.")

    def run(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        if ctx.config.mvs.backend is MvsBackend.COLMAP:
            return self._run_colmap(ctx)
        return self._run_openmvs(ctx)

    # ---- OBB backstop ----------------------------------------------------- #
    def _object_obb(self, ctx: StageContext):
        """Oriented bbox of the sparse object points (+ margin), or None."""
        if not ctx.config.masking.bbox_crop:
            return None
        colmap = ctx.env.tools["colmap"].path
        sparse_model = Path(self.prior_outputs(ctx, "sfm")["sparse_model"])
        sparse_ply = ctx.workspace.dense_dir / "sparse_points.ply"
        export_sparse_points(colmap, sparse_model, sparse_ply, ctx.env)
        return compute_obb(sparse_ply, ctx.config.masking.bbox_margin)

    # ---- colmap backend --------------------------------------------------- #
    def _run_colmap(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        ws = ctx.workspace
        colmap = ctx.env.tools["colmap"].path
        fused = Path(self.prior_outputs(ctx, "dense")["dense_cloud"])
        mcfg = ctx.config.mesh
        warnings: list[str] = []

        # OBB backstop: crop the dense cloud to the sparse object box before meshing.
        crop_metrics = None
        obb = self._object_obb(ctx)
        if obb is not None:
            cropped = ws.dense_dir / "fused_cropped.ply"
            res = crop_cloud(fused, obb, cropped)
            crop_metrics = res.to_metrics()
            get_logger().info(
                "OBB crop (cloud): %d -> %d points (%.1f%% removed)",
                res.before, res.after, res.fraction_removed * 100,
            )
            if res.after == 0:
                raise StageError(
                    "OBB crop removed all dense points — the sparse model and dense "
                    "cloud may be misaligned, or bbox_margin is too small."
                )
            if res.fraction_removed > 0.6:
                warnings.append(
                    f"OBB crop removed {res.fraction_removed:.0%} of dense points; "
                    "check masking / bbox_margin if this is unexpected."
                )
            fused = cropped

        use_delaunay = mcfg.backend is MeshBackend.DELAUNAY
        if mcfg.backend is MeshBackend.OPENMVS:
            warnings.append(
                "mesh.backend=openmvs but mvs.backend=colmap; using poisson_mesher."
            )
        if use_delaunay:
            out = ws.dense_dir / "meshed-delaunay.ply"
            run_command(
                [colmap, "delaunay_mesher",
                 "--input_path", str(ws.dense_dir),
                 "--input_type", "dense",
                 "--output_path", str(out)],
                env_required=ctx.env,
            )
        else:
            out = ws.dense_dir / "meshed-poisson.ply"
            run_command(
                [colmap, "poisson_mesher",
                 "--input_path", str(fused),
                 "--output_path", str(out),
                 "--PoissonMeshing.depth", str(mcfg.poisson_depth),
                 "--PoissonMeshing.trim", str(mcfg.poisson_trim)],
                env_required=ctx.env,
            )
        return self._finish(ctx, out, has_uv=False, extra=warnings, crop_metrics=crop_metrics)

    # ---- openmvs backend -------------------------------------------------- #
    def _run_openmvs(self, ctx: StageContext) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        ws = ctx.workspace
        envx = openmvs_lib_env(ctx.env)
        reconstruct = ctx.env.tools["ReconstructMesh"].path

        run_command([reconstruct, "scene_dense.mvs"],
                    cwd=ws.dense_dir, env_required=ctx.env, extra_env=envx)
        mesh_mvs = ws.dense_dir / "scene_dense_mesh.mvs"
        mesh_ply = ws.dense_dir / "scene_dense_mesh.ply"
        warnings: list[str] = []

        if ctx.config.mesh.refine:
            # RefineMesh consumes a scene .mvs that embeds the mesh. Some
            # ReconstructMesh builds emit only the mesh .ply; if so, fail loudly
            # rather than silently skipping the requested refinement.
            if not mesh_mvs.exists():
                raise StageError(
                    "mesh.refine is on, but ReconstructMesh did not emit "
                    f"{mesh_mvs.name} (only the .ply). RefineMesh needs the scene "
                    ".mvs. Disable refinement (draft preset) or rebuild OpenMVS so "
                    "ReconstructMesh writes the mesh scene."
                )
            refine = ctx.env.tools["RefineMesh"].path
            run_command(
                [refine, "scene_dense_mesh.mvs", "--resolution-level", "1",
                 "--cuda-device", str(cuda_device_arg(ctx.env, ctx.config.mvs.use_gpu))],
                cwd=ws.dense_dir, env_required=ctx.env, extra_env=envx,
            )
            refined_ply = ws.dense_dir / "scene_dense_mesh_refine.ply"
            if refined_ply.exists():
                mesh_ply = refined_ply

        # OBB backstop: ReconstructMesh has no external-cloud input here, so the
        # crop is applied to the reconstructed (untextured) mesh instead.
        crop_metrics = None
        obb = self._object_obb(ctx)
        if obb is not None:
            cropped = ws.dense_dir / f"{mesh_ply.stem}_cropped.ply"
            res = crop_mesh(mesh_ply, obb, cropped)
            crop_metrics = res.to_metrics()
            get_logger().info(
                "OBB crop (mesh): %d -> %d faces (%.1f%% removed)",
                res.before, res.after, res.fraction_removed * 100,
            )
            if res.after == 0:
                raise StageError(
                    "OBB crop removed the entire mesh — sparse/dense misalignment or "
                    "bbox_margin too small."
                )
            if res.fraction_removed > 0.6:
                warnings.append(
                    f"OBB crop removed {res.fraction_removed:.0%} of mesh faces; "
                    "check masking / bbox_margin if unexpected."
                )
            mesh_ply = cropped

        metrics, w2, outputs = self._finish(ctx, mesh_ply, has_uv=False, crop_metrics=crop_metrics)
        metrics["refined"] = ctx.config.mesh.refine
        return metrics, warnings + w2, outputs

    # ---- shared tail ------------------------------------------------------ #
    def _finish(
        self, ctx: StageContext, mesh_path: Path, *, has_uv: bool,
        extra: list[str] | None = None, crop_metrics: dict | None = None,
    ) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        counts = read_ply_element_counts(mesh_path)
        verts, faces = counts.get("vertex", 0), counts.get("face", 0)
        if not mesh_path.exists() or faces == 0:
            raise StageError(
                f"Meshing produced an empty/missing mesh ({mesh_path.name}, "
                f"{faces} faces). The dense cloud may be too sparse."
            )
        get_logger().info("Mesh: %d vertices, %d faces -> %s", verts, faces, mesh_path.name)
        metrics: dict[str, Any] = {
            "mesh_vertices": verts,
            "mesh_faces": faces,
            "has_uv": has_uv,
            "bbox_crop": crop_metrics,
        }
        return metrics, (extra or []), {"mesh": str(mesh_path)}

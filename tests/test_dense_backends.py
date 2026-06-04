"""Tests for MVS backend selection, CUDA/OOM diagnosis (mocked subprocess)."""

from __future__ import annotations

import json
import subprocess

import pytest

from engine import binaries
from engine.binaries import Environment, ToolStatus
from engine.config import MvsBackend, Preset, UseCase, build_preset
from engine.openmvs import cuda_device_arg, looks_like_oom, openmvs_lib_env
from engine.stages.base import StageContext, StageError, StageStatus
from engine.stages.dense import DenseStage
from engine.workspace import Workspace


def _env(colmap_cuda=False, cuda=True):
    tools = {
        b: ToolStatus(b, f"/usr/bin/{b}")
        for b in binaries.COLMAP_BINARIES + binaries.OPENMVS_BINARIES
    }
    return Environment(tools=tools, cuda_available=cuda, colmap_cuda=colmap_cuda)


def _ctx(tmp_path, backend, env=None, auto_downscale=False):
    ws = Workspace(root=tmp_path / "job", image_dir=tmp_path / "imgs")
    ws.ensure()
    ws.dense_dir.mkdir(parents=True, exist_ok=True)
    (ws.dense_dir / "images").mkdir(parents=True, exist_ok=True)
    # Seed the undistortion stage output so validate_inputs passes.
    ws.metrics_file("undistortion").write_text(json.dumps({
        "name": "undistortion", "status": "completed",
        "outputs": {"dense_workspace": str(ws.dense_dir)},
    }))
    cfg = build_preset(Preset.DRAFT, UseCase.OBJECT)
    cfg.mvs.backend = backend
    cfg.mvs.auto_downscale = auto_downscale
    return StageContext(workspace=ws, config=cfg, env=env or _env(), force=False)


# --- helpers ---------------------------------------------------------------- #
def test_looks_like_oom():
    assert looks_like_oom("CUDA error: out of memory")
    assert looks_like_oom("std::bad_alloc")
    assert not looks_like_oom("everything is fine")


def test_cuda_device_arg():
    assert cuda_device_arg(_env(cuda=True), True) == 0
    assert cuda_device_arg(_env(cuda=False), True) == -1
    assert cuda_device_arg(_env(cuda=True), False) == -1


def test_openmvs_lib_env_derives_lib_dir(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin" / "OpenMVS"
    lib_dir = tmp_path / "lib" / "OpenMVS"
    bin_dir.mkdir(parents=True)
    lib_dir.mkdir(parents=True)
    env = Environment(tools={"DensifyPointCloud": ToolStatus("DensifyPointCloud", str(bin_dir / "DensifyPointCloud"))})
    out = openmvs_lib_env(env)
    assert str(lib_dir) in out["LD_LIBRARY_PATH"]


# --- colmap backend: CPU-only build fails with actionable message ----------- #
def test_colmap_backend_without_cuda_fails_actionably(tmp_path, monkeypatch):
    def fake_run(cmd, cwd, capture_output, text, env=None):
        # COLMAP CPU-only build: logs "requires CUDA" and exits 0.
        return subprocess.CompletedProcess(
            cmd, 0, "", "Dense stereo reconstruction requires CUDA, which is not available."
        )

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    result = DenseStage().execute(_ctx(tmp_path, MvsBackend.COLMAP))
    assert result.status is StageStatus.FAILED
    assert "--mvs-backend openmvs" in result.error
    assert "CUDA" in result.error


# --- openmvs backend: success + OOM paths ----------------------------------- #
def _write_fake_ply(path, n_points):
    path.write_text(f"ply\nformat ascii 1.0\nelement vertex {n_points}\nend_header\n")


def test_openmvs_backend_success(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, MvsBackend.OPENMVS)

    def fake_run(cmd, cwd, capture_output, text, env=None):
        # DensifyPointCloud writes scene_dense.ply
        if "DensifyPointCloud" in cmd[0]:
            _write_fake_ply(ctx.workspace.dense_dir / "scene_dense.ply", 12345)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    result = DenseStage().execute(ctx)
    assert result.status is StageStatus.COMPLETED
    assert result.metrics["dense_points"] == 12345
    assert result.metrics["backend"] == "openmvs"


def test_openmvs_oom_without_autodownscale_fails_actionably(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, MvsBackend.OPENMVS, auto_downscale=False)

    def fake_run(cmd, cwd, capture_output, text, env=None):
        if "DensifyPointCloud" in cmd[0]:
            return subprocess.CompletedProcess(cmd, 1, "", "CUDA error: out of memory")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    result = DenseStage().execute(ctx)
    assert result.status is StageStatus.FAILED
    assert "out of memory" in result.error.lower()
    assert "auto-downscale" in result.error.lower()


def test_openmvs_cuda_symbol_error_is_actionable(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, MvsBackend.OPENMVS)

    def fake_run(cmd, cwd, capture_output, text, env=None):
        if "DensifyPointCloud" in cmd[0]:
            return subprocess.CompletedProcess(
                cmd, 1, "", "CUDA error at UtilCUDADevice.h:74: invalid device symbol (code 13)"
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    result = DenseStage().execute(ctx)
    assert result.status is StageStatus.FAILED
    assert "--mvs-cpu" in result.error
    assert "out of memory" not in result.error.lower()


def test_openmvs_cpu_uses_device_minus_one(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, MvsBackend.OPENMVS)
    ctx.config.mvs.use_gpu = False  # --mvs-cpu
    seen_args = []

    def fake_run(cmd, cwd, capture_output, text, env=None):
        seen_args.append(cmd)
        if "DensifyPointCloud" in cmd[0]:
            _write_fake_ply(ctx.workspace.dense_dir / "scene_dense.ply", 7)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    result = DenseStage().execute(ctx)
    assert result.status is StageStatus.COMPLETED
    assert result.metrics["cuda_device"] == -1
    densify_cmd = next(c for c in seen_args if "DensifyPointCloud" in c[0])
    assert "-1" in densify_cmd  # --cuda-device -1


def test_openmvs_oom_with_autodownscale_retries_then_succeeds(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, MvsBackend.OPENMVS, auto_downscale=True)
    calls = {"densify": 0}

    def fake_run(cmd, cwd, capture_output, text, env=None):
        if "DensifyPointCloud" in cmd[0]:
            calls["densify"] += 1
            if calls["densify"] == 1:  # first attempt OOMs
                return subprocess.CompletedProcess(cmd, 1, "", "out of memory")
            _write_fake_ply(ctx.workspace.dense_dir / "scene_dense.ply", 999)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    result = DenseStage().execute(ctx)
    assert result.status is StageStatus.COMPLETED
    assert calls["densify"] == 2  # retried once
    assert result.metrics["dense_points"] == 999
    assert any("auto-downscal" in w.lower() for w in result.warnings)

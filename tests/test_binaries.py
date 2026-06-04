"""Tests for binary detection, CUDA detection, and the command runner.

subprocess and shutil.which are mocked so these run with no real binaries.
"""

from __future__ import annotations

import subprocess

import pytest

from engine import binaries
from engine.binaries import (
    BinaryNotFoundError,
    CommandError,
    Environment,
    ToolStatus,
    detect_cuda,
    detect_environment,
    gpu_flag,
    require,
    run_command,
)


def _fake_which(mapping):
    return lambda name: mapping.get(name)


def test_detect_environment_all_present(monkeypatch):
    present = {b: f"/usr/bin/{b}" for b in
               binaries.COLMAP_BINARIES + binaries.OPENMVS_BINARIES}
    monkeypatch.setattr(binaries, "_which", _fake_which(present))
    monkeypatch.setattr(binaries, "detect_cuda", lambda: (True, ["GPU 0: Test"]))
    env = detect_environment()
    assert env.has_colmap()
    assert env.has_openmvs()
    assert env.cuda_available


def test_detect_environment_all_missing(monkeypatch):
    monkeypatch.setattr(binaries, "_which", _fake_which({}))
    monkeypatch.setattr(binaries, "detect_cuda", lambda: (False, []))
    env = detect_environment()
    assert not env.has_colmap()
    assert not env.has_openmvs()
    assert env.missing(["colmap"]) == ["colmap"]


def test_require_raises_with_install_hint():
    env = Environment(tools={"colmap": ToolStatus("colmap", None)})
    with pytest.raises(BinaryNotFoundError) as exc:
        require(env, ["colmap"])
    msg = str(exc.value)
    assert "colmap" in msg
    assert "install" in msg.lower()


def test_require_passes_when_present():
    env = Environment(tools={"colmap": ToolStatus("colmap", "/usr/bin/colmap")})
    require(env, ["colmap"])  # should not raise


def test_openmvs_tools_share_one_install_hint():
    env = Environment(tools={b: ToolStatus(b, None) for b in binaries.OPENMVS_BINARIES})
    with pytest.raises(BinaryNotFoundError) as exc:
        require(env, list(binaries.OPENMVS_BINARIES))
    # All five share one hint -> hint text appears exactly once (deduped).
    assert str(exc.value).lower().count("openmvs") >= 1


def test_gpu_flag_falls_back_without_cuda():
    env = Environment(cuda_available=False)
    assert gpu_flag(env, prefer_gpu=True) == 0


def test_gpu_flag_enabled_with_cuda():
    env = Environment(cuda_available=True, cuda_devices=["GPU 0"])
    assert gpu_flag(env, prefer_gpu=True) == 1
    assert gpu_flag(env, prefer_gpu=False) == 0


def test_detect_cuda_no_smi(monkeypatch):
    monkeypatch.setattr(binaries, "_which", _fake_which({}))
    assert detect_cuda() == (False, [])


def test_detect_cuda_parses_devices(monkeypatch):
    monkeypatch.setattr(binaries, "_which", _fake_which({"nvidia-smi": "/usr/bin/nvidia-smi"}))

    def fake_run(cmd, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 0, "GPU 0: Test (UUID: x)\nGPU 1: Test2\n", "")

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    ok, devices = detect_cuda()
    assert ok
    assert len(devices) == 2


def test_run_command_success(monkeypatch):
    def fake_run(cmd, cwd, capture_output, text, env=None):
        return subprocess.CompletedProcess(cmd, 0, "hello", "")

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    proc = run_command(["echo", "hello"])
    assert proc.returncode == 0
    assert proc.stdout == "hello"


def test_run_command_failure_raises_commanderror(monkeypatch):
    def fake_run(cmd, cwd, capture_output, text, env=None):
        return subprocess.CompletedProcess(cmd, 2, "", "boom: bad arg")

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    with pytest.raises(CommandError) as exc:
        run_command(["colmap", "explode"])
    assert exc.value.returncode == 2
    assert "boom" in str(exc.value)


def test_run_command_missing_binary_raises(monkeypatch):
    def fake_run(cmd, cwd, capture_output, text, env=None):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    with pytest.raises(BinaryNotFoundError):
        run_command(["colmap", "foo"])


def test_run_command_check_false_returns_nonzero(monkeypatch):
    def fake_run(cmd, cwd, capture_output, text, env=None):
        return subprocess.CompletedProcess(cmd, 1, "", "warn")

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    proc = run_command(["colmap", "x"], check=False)
    assert proc.returncode == 1

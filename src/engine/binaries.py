"""External-binary and CUDA detection.

Constraint #2: fail *loudly* with install guidance when a required tool is
missing -- never proceed and produce a misleading partial result.

This module also runs external commands through a single ``run_command``
helper so that every invocation is logged verbatim and its exit code / stderr
is surfaced (never swallowed).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .logging import format_command, get_logger

# Tools grouped by the capability they provide.
COLMAP_BINARIES = ("colmap",)
OPENMVS_BINARIES = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
    "RefineMesh",
    "TextureMesh",
)

# Per-tool install hints surfaced when a binary is missing.
_INSTALL_HINTS: dict[str, str] = {
    "colmap": (
        "Install COLMAP: `apt install colmap` (Ubuntu), `brew install colmap` (macOS), "
        "or build from https://colmap.github.io/install.html. A CUDA build is "
        "recommended for dense reconstruction."
    ),
    "_openmvs": (
        "Install OpenMVS (provides InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, "
        "RefineMesh, TextureMesh). See https://github.com/cdcseacave/openMVS/wiki/Building. "
        "On Ubuntu the `openmvs` package or a from-source build places these on PATH."
    ),
}


class BinaryNotFoundError(RuntimeError):
    """Raised when a required external tool is not on PATH."""


class CommandError(RuntimeError):
    """Raised when an external command exits non-zero.

    Carries the full diagnostic context so callers can present it.
    """

    def __init__(self, cmd: Sequence[str], returncode: int, stdout: str, stderr: str):
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-15:]) or "(no stderr)"
        super().__init__(
            f"Command failed with exit code {returncode}:\n"
            f"  {format_command(cmd)}\n"
            f"--- stderr (last lines) ---\n{tail}"
        )


@dataclass(frozen=True)
class ToolStatus:
    name: str
    path: str | None

    @property
    def available(self) -> bool:
        return self.path is not None


@dataclass
class Environment:
    """Snapshot of available tools + hardware, taken at startup."""

    tools: dict[str, ToolStatus] = field(default_factory=dict)
    cuda_available: bool = False
    cuda_devices: list[str] = field(default_factory=list)
    # Whether the *colmap binary* was compiled with CUDA. A machine can have a
    # CUDA GPU while colmap itself is a CPU-only build (e.g. the Ubuntu apt
    # package) -- in that case colmap's SIFT GPU path is unavailable.
    colmap_cuda: bool = False

    def has(self, name: str) -> bool:
        status = self.tools.get(name)
        return bool(status and status.available)

    def has_colmap(self) -> bool:
        return all(self.has(b) for b in COLMAP_BINARIES)

    def has_openmvs(self) -> bool:
        return all(self.has(b) for b in OPENMVS_BINARIES)

    def missing(self, names: Sequence[str]) -> list[str]:
        return [n for n in names if not self.has(n)]


def _which(name: str) -> str | None:
    return shutil.which(name)


def detect_cuda() -> tuple[bool, list[str]]:
    """Detect CUDA GPUs via ``nvidia-smi`` (best-effort, never raises)."""
    smi = _which("nvidia-smi")
    if not smi:
        return False, []
    try:
        proc = subprocess.run(
            [smi, "-L"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return False, []
    if proc.returncode != 0:
        return False, []
    devices = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return (len(devices) > 0), devices


def detect_colmap_cuda(colmap_path: str | None) -> bool:
    """Best-effort: does this colmap binary report a CUDA build?

    COLMAP prints e.g. ``COLMAP 3.9.1 (Commit ... without CUDA)`` in its banner;
    a CUDA build omits ``without CUDA``. Never raises.
    """
    if not colmap_path:
        return False
    try:
        proc = subprocess.run(
            [colmap_path, "-h"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return False
    banner = (proc.stdout + proc.stderr).lower()
    if "colmap" not in banner:
        return False
    return "without cuda" not in banner


def detect_environment(extra: Sequence[str] = ()) -> Environment:
    """Probe PATH for all known tools and detect CUDA.

    Detection never raises -- callers decide which tools are *required* for the
    work they're about to do and call :func:`require` accordingly.
    """
    names = list(COLMAP_BINARIES) + list(OPENMVS_BINARIES) + list(extra)
    tools = {n: ToolStatus(name=n, path=_which(n)) for n in names}
    cuda_available, cuda_devices = detect_cuda()
    colmap_cuda = detect_colmap_cuda(tools.get("colmap", ToolStatus("colmap", None)).path)
    return Environment(
        tools=tools,
        cuda_available=cuda_available,
        cuda_devices=cuda_devices,
        colmap_cuda=colmap_cuda,
    )


def install_hint(name: str) -> str:
    if name in OPENMVS_BINARIES:
        return _INSTALL_HINTS["_openmvs"]
    return _INSTALL_HINTS.get(name, f"Ensure `{name}` is installed and on PATH.")


def require(env: Environment, names: Sequence[str]) -> None:
    """Raise :class:`BinaryNotFoundError` if any named tool is unavailable."""
    missing = env.missing(names)
    if not missing:
        return
    lines = [f"Required tool(s) not found on PATH: {', '.join(missing)}", ""]
    seen_hints: set[str] = set()
    for name in missing:
        hint = install_hint(name)
        if hint not in seen_hints:
            lines.append(f"  • {hint}")
            seen_hints.add(hint)
    raise BinaryNotFoundError("\n".join(lines))


def gpu_flag(env: Environment, prefer_gpu: bool) -> int:
    """Return 1/0 for COLMAP/OpenMVS ``use_gpu`` flags, warning on fallback."""
    if prefer_gpu and not env.cuda_available:
        get_logger().warning(
            "GPU requested but no CUDA device detected — falling back to CPU "
            "(this will be substantially slower)."
        )
        return 0
    return 1 if (prefer_gpu and env.cuda_available) else 0


def colmap_gpu_flag(env: Environment, prefer_gpu: bool) -> int:
    """GPU flag for COLMAP stages, honouring whether colmap has a CUDA build.

    Distinct from :func:`gpu_flag` because a CUDA GPU can be present while the
    colmap binary itself is CPU-only -- in which case we must run on CPU and
    say so loudly (the GPU is still usable by OpenMVS later).
    """
    if not prefer_gpu:
        return 0
    if not env.cuda_available:
        get_logger().warning(
            "GPU requested but no CUDA device detected — COLMAP will run on CPU."
        )
        return 0
    if not env.colmap_cuda:
        get_logger().warning(
            "GPU requested and a CUDA device is present, but this COLMAP binary "
            "is a CPU-only build (no CUDA) — running COLMAP SfM on CPU. "
            "The GPU will still be used by OpenMVS for dense reconstruction."
        )
        return 0
    return 1


def run_command(
    cmd: Sequence[str],
    *,
    cwd: str | Path | None = None,
    check: bool = True,
    env_required: Environment | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an external command, logging it verbatim and surfacing failures.

    * Logs the exact command before running (copy-pasteable).
    * On non-zero exit (with ``check``), raises :class:`CommandError` carrying
      exit code + stderr -- nothing is swallowed.
    * ``extra_env`` is merged over ``os.environ`` for this call only (e.g. to add
      OpenMVS's shared-library dir to ``LD_LIBRARY_PATH``).
    """
    cmd = [str(c) for c in cmd]
    logger = get_logger()

    if env_required is not None:
        # cmd[0] may be an absolute path (resolved from PATH); check by tool name.
        require(env_required, [Path(cmd[0]).name])

    proc_env = None
    if extra_env:
        proc_env = {**os.environ, **extra_env}

    logger.info("[dim]$ %s[/]", format_command(cmd), extra={"markup": True})
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, env=proc_env
        )
    except FileNotFoundError as exc:  # binary vanished between detect and run
        raise BinaryNotFoundError(
            f"{cmd[0]!r} not found when executing.\n  {install_hint(cmd[0])}"
        ) from exc

    if proc.stderr.strip():
        logger.debug("stderr from %s:\n%s", cmd[0], proc.stderr.strip())
    if check and proc.returncode != 0:
        raise CommandError(cmd, proc.returncode, proc.stdout, proc.stderr)
    return proc

"""Helpers for the OpenMVS backend.

OpenMVS tools are driven via the CLI (so reference commands stay reproducible).
This module only sets up their runtime environment and reads counts back out of
their outputs.
"""

from __future__ import annotations

from pathlib import Path

from .binaries import Environment

# stderr/stdout signatures that indicate a CUDA out-of-memory condition.
OOM_SIGNATURES = (
    "out of memory",
    "cudaerrormemoryallocation",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "failed to allocate",
    "std::bad_alloc",
)

# COLMAP's message when dense stereo is requested on a non-CUDA build.
COLMAP_NO_CUDA_SIGNATURE = "requires cuda"

# CUDA runtime/init failures that are NOT out-of-memory: usually a build whose
# kernels don't match the installed GPU architecture or driver.
CUDA_ERROR_SIGNATURES = (
    "invalid device symbol",
    "invalid device function",
    "no cuda-capable device",
    "cuda error at",
    "cuda driver version is insufficient",
    "forward compatibility was attempted",
)


def looks_like_oom(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in OOM_SIGNATURES)


def looks_like_cuda_error(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in CUDA_ERROR_SIGNATURES)


def openmvs_lib_env(env: Environment) -> dict[str, str]:
    """Return env additions so OpenMVS finds its shared libs.

    OpenMVS is often installed with binaries in ``.../bin/OpenMVS`` and shared
    libraries in ``.../lib/OpenMVS``. We derive the lib dir from the resolved
    binary path and prepend it to LD_LIBRARY_PATH for the call. The inherited
    environment (e.g. a user's ~/.bashrc) still takes effect; this is a robust
    fallback, not a replacement.
    """
    import os

    status = env.tools.get("DensifyPointCloud")
    if not status or not status.path:
        return {}
    bin_dir = Path(status.path).parent
    candidates = []
    # .../bin/OpenMVS -> .../lib/OpenMVS, and .../bin -> .../lib
    parts = bin_dir.parts
    if "bin" in parts:
        idx = len(parts) - 1 - parts[::-1].index("bin")
        lib_dir = Path(*parts[:idx], "lib", *parts[idx + 1 :])
        candidates.append(lib_dir)
    candidates.append(bin_dir)  # libs sometimes sit next to the binaries

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [str(c) for c in candidates if c.exists()]
    if not paths:
        return {}
    combined = ":".join(paths + ([existing] if existing else []))
    return {"LD_LIBRARY_PATH": combined}


def cuda_device_arg(env: Environment, use_gpu: bool) -> int:
    """OpenMVS --cuda-device value: device index, or -1 for CPU."""
    if use_gpu and env.cuda_available:
        return 0
    return -1


def read_ply_element_counts(ply_path: Path) -> dict[str, int]:
    """Read ``element <name> <count>`` lines from a PLY header (no full load)."""
    counts: dict[str, int] = {}
    if not ply_path.exists():
        return counts
    with open(ply_path, "rb") as fh:
        for _ in range(200):  # header is short; bail out quickly
            line = fh.readline()
            if not line:
                break
            text = line.decode("latin-1").strip()
            if text.startswith("element"):
                parts = text.split()
                if len(parts) >= 3 and parts[2].isdigit():
                    counts[parts[1]] = int(parts[2])
            elif text == "end_header":
                break
    return counts


def count_ply_points(ply_path: Path) -> int:
    """Read the vertex count from a PLY header without loading the whole file."""
    return read_ply_element_counts(ply_path).get("vertex", 0)

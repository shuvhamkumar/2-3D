"""Structured logging + per-stage timing built on ``rich``.

Every external command is echoed verbatim *before* it runs (constraint #2:
never silently swallow anything). Timing is recorded per stage so the final
report can show where the wall-clock went.
"""

from __future__ import annotations

import logging
import shlex
import time
from contextlib import contextmanager
from collections.abc import Iterator, Sequence

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)
_CONFIGURED = False


def get_console() -> Console:
    return _console


def configure_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Install a rich handler once; return the engine logger."""
    global _CONFIGURED
    logger = logging.getLogger("engine")
    if not _CONFIGURED:
        handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            show_path=False,
            omit_repeated_times=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.handlers = [handler]
        logger.propagate = False
        _CONFIGURED = True
    logger.setLevel(level)
    return logger


def get_logger() -> logging.Logger:
    return configure_logging()


def format_command(cmd: Sequence[str]) -> str:
    """Render an argv list as a copy-pasteable shell command."""
    return " ".join(shlex.quote(str(c)) for c in cmd)


@contextmanager
def stage_timer(name: str) -> Iterator[dict[str, float]]:
    """Time a block; yields a dict that gets ``elapsed_s`` filled in.

    Usage::

        with stage_timer("sfm") as t:
            ...
        # t["elapsed_s"] now set
    """
    logger = get_logger()
    timing: dict[str, float] = {}
    start = time.perf_counter()
    logger.info("[bold cyan]▶ %s[/] starting", name, extra={"markup": True})
    try:
        yield timing
    except BaseException:
        elapsed = time.perf_counter() - start
        timing["elapsed_s"] = elapsed
        logger.info("[yellow]✗ %s aborted after %.1fs[/]", name, elapsed, extra={"markup": True})
        raise
    else:
        elapsed = time.perf_counter() - start
        timing["elapsed_s"] = elapsed
        logger.info(
            "[bold green]✔ %s[/] finished in %.1fs", name, elapsed, extra={"markup": True}
        )

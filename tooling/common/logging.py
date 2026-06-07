"""Logging helpers for development tooling."""

from __future__ import annotations

import logging
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path


def configure_tool_logging(log_file: Path | None = None) -> None:
    """Configure process logging for a tooling entrypoint.

    Args:
        log_file: Optional file receiving the same log stream as stderr.

    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

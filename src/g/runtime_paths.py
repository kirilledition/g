"""Runtime path policy helpers."""

from __future__ import annotations

from pathlib import Path

from g import _core


def default_local_cache_directory(directory_name: str) -> Path:
    """Build a default cache directory under the platform temporary directory.

    Args:
        directory_name: Cache directory name to place under the user-specific root.

    Returns:
        Default temporary cache directory path.

    """
    return Path(_core.default_local_cache_directory_value(directory_name))

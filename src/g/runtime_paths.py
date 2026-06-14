"""Runtime path policy helpers."""

from __future__ import annotations

import getpass
import tempfile
from pathlib import Path

DEFAULT_LOCAL_TEMPORARY_ROOT = Path(tempfile.gettempdir()).expanduser()


def default_local_cache_directory(directory_name: str) -> Path:
    """Build a default cache directory under the platform temporary directory.

    Args:
        directory_name: Cache directory name to place under the user-specific root.

    Returns:
        Default temporary cache directory path.

    """
    user_name = getpass.getuser() or "unknown"
    return DEFAULT_LOCAL_TEMPORARY_ROOT / user_name / directory_name

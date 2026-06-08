"""Runtime path policy helpers."""

from __future__ import annotations

import getpass
from pathlib import Path

DEFAULT_NODE_LOCAL_CACHE_ROOT = Path("/tmp")


def default_node_local_cache_directory(directory_name: str) -> Path:
    """Build a default cache directory under node-local temporary storage.

    Args:
        directory_name: Cache directory name to place under the user-specific root.

    Returns:
        Node-local cache directory path.

    """
    user_name = getpass.getuser() or "unknown"
    return DEFAULT_NODE_LOCAL_CACHE_ROOT / user_name / directory_name


def path_is_beegfs(path: Path) -> bool:
    """Return whether a path is on the BeeGFS mount used by this project.

    Args:
        path: Path to classify.

    Returns:
        Whether the expanded path starts with the BeeGFS mount prefix.

    """
    expanded_path = path.expanduser()
    return str(expanded_path).startswith("/mnt/beegfs/")


def path_is_node_local(path: Path) -> bool:
    """Return whether a path is on node-local temporary storage.

    Args:
        path: Path to classify.

    Returns:
        Whether the expanded path is under `/tmp`.

    """
    expanded_path = path.expanduser()
    return str(expanded_path).startswith("/tmp/") or str(expanded_path) == "/tmp"

"""Repository path helpers for development tooling."""

from __future__ import annotations

import os
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    import collections.abc

DATA_DIRECTORY_ENVIRONMENT_VARIABLE = "GWAS_ENGINE_DATA_DIR"


def find_repository_root(start_path: Path | None = None) -> Path:
    """Find the repository root from a starting path.

    Args:
        start_path: Optional path inside the repository.

    Returns:
        Repository root when project metadata is found, otherwise the resolved
        current working directory.

    """
    search_path = (start_path or Path.cwd()).resolve()
    if search_path.is_file():
        search_path = search_path.parent
    for candidate_path in (search_path, *search_path.parents):
        has_python_package = (candidate_path / "src" / "g").is_dir()
        has_project_metadata = (candidate_path / "pyproject.toml").is_file()
        if has_python_package and has_project_metadata:
            return candidate_path
    return Path.cwd().resolve()


def resolve_repo_relative_path(path: Path, repository_root: Path | None = None) -> Path:
    """Resolve a path relative to the repository root when needed.

    Args:
        path: Absolute path or repository-relative path.
        repository_root: Optional pre-resolved repository root.

    Returns:
        Absolute path.

    """
    if path.is_absolute():
        return path
    return (repository_root or find_repository_root()) / path


def configured_data_directory(environment: collections.abc.Mapping[str, str] | None = None) -> Path:
    """Return the configured data directory without changing relative paths.

    Args:
        environment: Optional environment mapping for tests.

    Returns:
        Raw configured data directory path.

    """
    environment_values = environment if environment is not None else os.environ
    return Path(environment_values.get(DATA_DIRECTORY_ENVIRONMENT_VARIABLE, "data"))


def resolve_data_directory(
    repository_root: Path | None = None,
    environment: collections.abc.Mapping[str, str] | None = None,
) -> Path:
    """Resolve the local data directory with environment override support.

    Args:
        repository_root: Optional pre-resolved repository root.
        environment: Optional environment mapping for tests.

    Returns:
        Absolute data directory path.

    """
    resolved_repository_root = repository_root or find_repository_root()
    return resolve_repo_relative_path(configured_data_directory(environment), resolved_repository_root)


def resolve_data_path(data_directory: Path, path: Path) -> Path:
    """Resolve a benchmark input path relative to a data directory when needed.

    Args:
        data_directory: Base data directory.
        path: Absolute path or data-directory-relative path.

    Returns:
        Absolute input path.

    """
    if path.is_absolute():
        return path
    return data_directory / path

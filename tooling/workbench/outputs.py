"""Validate completed native outputs before a Workbench attempt is published."""

from __future__ import annotations

import hashlib
import json
import stat
import typing
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputFile:
    """Content identity of a validated output file.

    Attributes:
        relative_path: File path relative to the attempt's output directory.
        sha256: SHA-256 digest of the complete file contents.
        size_bytes: Observed file length.

    """

    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ChunkCommit:
    """Committed native variant interval and its containing Parquet part.

    Attributes:
        chunk_identifier: First variant index of the logical chunk.
        variant_start_index: Inclusive first variant index.
        variant_stop_index: Exclusive last variant index.
        row_count: Number of output rows in the logical chunk.
        chunk_file_name: Safe basename of the containing Parquet part.

    """

    chunk_identifier: int
    variant_start_index: int
    variant_stop_index: int
    row_count: int
    chunk_file_name: str


def require_json_object(value: object, context: str) -> dict[str, object]:
    """Read a JSON object without silently coercing malformed fields."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object.")
    return typing.cast("dict[str, object]", value)


def read_json_object(path: Path) -> dict[str, object]:
    """Load a native manifest as a strictly structured JSON object."""
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read output manifest {path}: {error}") from error
    return require_json_object(value, str(path))


def require_integer(value: object, context: str) -> int:
    """Require an actual JSON integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer.")
    return value


def read_chunk_commits(value: object, context: str) -> tuple[ChunkCommit, ...]:
    """Read logical commits with the same geometry as the native writer."""
    if not isinstance(value, list):
        raise ValueError(f"{context} committed_chunks must be a list.")
    commits: list[ChunkCommit] = []
    identifiers: set[int] = set()
    for entry in typing.cast("list[object]", value):
        payload = require_json_object(entry, f"{context} chunk commit")
        file_name = payload.get("chunk_file_name")
        if (
            not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or "/" in file_name
            or "\\" in file_name
            or "\x00" in file_name
            or not file_name.endswith(".parquet")
        ):
            raise ValueError(f"{context} has an unsafe Parquet part name.")
        commit = ChunkCommit(
            chunk_identifier=require_integer(payload.get("chunk_identifier"), f"{context} chunk_identifier"),
            variant_start_index=require_integer(payload.get("variant_start_index"), f"{context} variant_start_index"),
            variant_stop_index=require_integer(payload.get("variant_stop_index"), f"{context} variant_stop_index"),
            row_count=require_integer(payload.get("row_count"), f"{context} row_count"),
            chunk_file_name=file_name,
        )
        if commit.chunk_identifier in identifiers:
            raise ValueError(f"{context} has duplicate chunk identifiers.")
        if (
            commit.chunk_identifier != commit.variant_start_index
            or commit.variant_start_index < 0
            or commit.row_count <= 0
            or commit.variant_stop_index - commit.variant_start_index != commit.row_count
        ):
            raise ValueError(f"{context} has inconsistent chunk range or row_count.")
        identifiers.add(commit.chunk_identifier)
        commits.append(commit)
    return tuple(sorted(commits, key=lambda commit: commit.variant_start_index))


def collect_output_files(output_root: Path) -> tuple[Path, ...]:
    """Enumerate ordinary output files while rejecting filesystem links."""
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("Output root must be an existing ordinary directory without symlinks.")
    directories = [output_root]
    files: list[Path] = []
    while directories:
        directory = directories.pop()
        for path in sorted(directory.iterdir()):
            file_mode = path.lstat().st_mode
            if stat.S_ISLNK(file_mode):
                raise ValueError(f"Output contains a symlink: {path}")
            if stat.S_ISDIR(file_mode):
                directories.append(path)
            elif stat.S_ISREG(file_mode):
                files.append(path)
            else:
                raise ValueError(f"Output contains a non-regular file: {path}")
    return tuple(sorted(files))


def validate_parquet_part(path: Path, expected_commits: tuple[ChunkCommit, ...]) -> None:
    """Check Parquet framing, footer rows and native chunk commit metadata.

    PyArrow is imported only at the output-validation boundary. Planning and
    input preflight therefore remain usable without loading native libraries.
    """
    size_bytes = path.stat().st_size
    if size_bytes < 12:
        raise ValueError(f"Parquet part is empty or truncated: {path}")
    with path.open("rb") as source:
        header = source.read(4)
        source.seek(-8, 2)
        trailer = source.read(8)
    footer_length = int.from_bytes(trailer[:4], "little")
    if header != b"PAR1" or trailer[4:] != b"PAR1" or not 0 < footer_length <= size_bytes - 12:
        raise ValueError(f"Parquet part has invalid magic or footer length: {path}")

    try:
        import pyarrow as pa
        import pyarrow.parquet
    except ImportError as error:
        raise RuntimeError("Output validation requires pyarrow; install the Workbench tooling dependencies.") from error

    try:
        metadata = pyarrow.parquet.read_metadata(path)
    except (OSError, pa.ArrowException) as error:
        raise ValueError(f"Cannot read Parquet footer {path}: {error}") from error
    expected_rows = sum(commit.row_count for commit in expected_commits)
    if metadata.num_rows != expected_rows:
        raise ValueError(f"Parquet footer row count differs from committed rows: {path}")
    footer_metadata = metadata.metadata or {}
    commit_text = footer_metadata.get(b"g.output.chunk_commits")
    if commit_text is None:
        raise ValueError(f"Parquet footer is missing g.output.chunk_commits: {path}")
    try:
        footer_value: object = json.loads(commit_text)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Parquet footer has malformed chunk commits: {path}") from error
    footer_commits = read_chunk_commits(footer_value, str(path))
    if footer_commits != expected_commits:
        raise ValueError(f"Parquet footer chunk commits differ from the manifest: {path}")


def hash_output_file(path: Path, output_root: Path) -> OutputFile:
    """Hash an ordinary output file and detect changes during hashing."""
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Output is no longer an ordinary file: {path}")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    after = path.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"Output changed during hashing: {path}")
    return OutputFile(relative_path=path.relative_to(output_root).as_posix(), sha256=digest, size_bytes=after.st_size)


def validate_completed_outputs(
    output_root: Path,
    phenotype_columns: tuple[str, ...],
    variant_count: int,
) -> tuple[OutputFile, ...]:
    """Require complete native results and return content identities for upload.

    Args:
        output_root: Attempt directory containing the native trait run directories.
        phenotype_columns: Exact phenotype names expected from this attempt.
        variant_count: Number of source variants expected in every phenotype output.

    Returns:
        Content identities for every ordinary output file, sorted by relative path.

    Raises:
        ValueError: Outputs are incomplete, inconsistent or unsafe to publish.
        RuntimeError: The output-validation dependency is unavailable.

    """
    if not phenotype_columns or len(set(phenotype_columns)) != len(phenotype_columns):
        raise ValueError("Expected phenotype names must be nonempty and unique.")
    if isinstance(variant_count, bool) or variant_count <= 0:
        raise ValueError("Expected variant_count must be a positive integer.")
    files = collect_output_files(output_root)
    manifests = tuple(path for path in files if path.name == "run_manifest.json")
    if len(manifests) != len(phenotype_columns):
        raise ValueError("Output manifest count differs from the expected phenotype count.")
    observed_phenotypes: set[str] = set()
    referenced_parts: set[Path] = set()
    for manifest_path in manifests:
        manifest = read_json_object(manifest_path)
        if manifest.get("status") != "completed":
            raise ValueError(f"Output manifest is not completed: {manifest_path}")
        execution_plan = require_json_object(manifest.get("execution_plan"), f"{manifest_path} execution_plan")
        phenotype_name = execution_plan.get("phenotype_name")
        if (
            not isinstance(phenotype_name, str)
            or phenotype_name not in phenotype_columns
            or phenotype_name in observed_phenotypes
        ):
            raise ValueError(f"Output manifest has an unexpected or duplicate phenotype: {manifest_path}")
        if require_integer(execution_plan.get("variant_count"), f"{manifest_path} variant_count") != variant_count:
            raise ValueError(f"Output manifest variant_count differs from the expected input: {manifest_path}")
        observed_phenotypes.add(phenotype_name)
        commits = read_chunk_commits(manifest.get("committed_chunks"), str(manifest_path))
        next_variant_index = 0
        commits_by_part: dict[str, list[ChunkCommit]] = {}
        for commit in commits:
            if commit.variant_start_index != next_variant_index:
                raise ValueError(f"Committed chunks have gaps or overlaps: {manifest_path}")
            next_variant_index = commit.variant_stop_index
            commits_by_part.setdefault(commit.chunk_file_name, []).append(commit)
        if next_variant_index != variant_count:
            raise ValueError(f"Committed chunks do not cover the full variant range: {manifest_path}")
        parts_directory = manifest_path.parent / "parts"
        if not parts_directory.is_dir():
            raise ValueError(f"Output parts directory is missing: {parts_directory}")
        observed_parts = {path for path in files if path.parent == parts_directory}
        expected_parts = {parts_directory / file_name for file_name in commits_by_part}
        if observed_parts != expected_parts:
            raise ValueError(f"Output parts are missing or unexplained: {parts_directory}")
        for file_name, part_commits in commits_by_part.items():
            part_path = parts_directory / file_name
            validate_parquet_part(part_path, tuple(part_commits))
            referenced_parts.add(part_path)
    observed_parquet = {path for path in files if path.suffix.lower() == ".parquet"}
    if observed_parquet != referenced_parts:
        raise ValueError("Output contains unexplained Parquet files.")
    identities = tuple(hash_output_file(path, output_root) for path in files)
    if collect_output_files(output_root) != files:
        raise ValueError("Output file set changed during validation.")
    return identities

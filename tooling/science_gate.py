"""Deterministic source identity for scientific qualification."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path

SCIENCE_SOURCE_FINGERPRINT_DOMAIN = b"g-science-source-v1\0"
SCIENCE_SOURCE_DIRECTORY_PREFIXES = (
    "crates/",
    "native/",
    "src/",
    "tests/parity/",
    "vendor/openxla/",
)
SCIENCE_SOURCE_EXACT_PATHS = frozenset(
    {
        ".cargo/config.toml",
        ".cargo/config",
        ".gitattributes",
        ".gitignore",
        ".python-version",
        "Cargo.lock",
        "Cargo.toml",
        "Justfile",
        "pyproject.toml",
        "rust-toolchain.toml",
        "tests/conftest.py",
        "tests/numerical.py",
        "tests/parity/__init__.py",
        "tests/parity/golden_metadata.json",
        "tests/parity/harness.py",
        "tests/test_regenie2_parity.py",
        "tooling/__init__.py",
        "tooling/science_gate.py",
        "tooling/server/exact_parity_bootstrap.sh",
        "tooling/server/server_env.sh",
        "uv.lock",
    }
)
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ScienceSourceEntry:
    """One tracked input to the scientific source identity."""

    relative_path: str
    git_mode: str
    content: bytes
    blob_identifier: str | None = None


@dataclass(frozen=True)
class GitIndexEntry:
    """One stage-zero path recorded in the Git index."""

    relative_path: str
    git_mode: str
    blob_identifier: str


@dataclass(frozen=True)
class WorkingTreeSource:
    """Bytes and Git-compatible mode observed for one working-tree path."""

    content: bytes
    git_mode: str


@dataclass(frozen=True)
class ScienceSourceState:
    """Clean repository state accepted for exact-source qualification."""

    git_commit: str
    science_source_sha256: str


def is_science_source_path(relative_path: str) -> bool:
    """Return whether a tracked path can affect production science or its gate."""
    return (
        relative_path in SCIENCE_SOURCE_EXACT_PATHS
        or relative_path.startswith(SCIENCE_SOURCE_DIRECTORY_PREFIXES)
        or ("/" not in relative_path and relative_path.endswith(".py"))
    )


def canonical_science_content(relative_path: str, content: bytes) -> bytes:
    """Return source bytes with mutable qualification claims removed."""
    if relative_path != "tests/parity/golden_metadata.json":
        return content
    payload = typing.cast("dict[str, object]", json.loads(content))
    workflow_payloads = typing.cast("list[dict[str, object]]", payload["workflows"])
    for workflow_payload in workflow_payloads:
        workflow_payload.pop("gate_status", None)
        workflow_payload.pop("qualification", None)
    return f"{json.dumps(payload, separators=(',', ':'), sort_keys=True)}\n".encode()


def science_source_fingerprint(entries: typing.Iterable[ScienceSourceEntry]) -> str:
    """Hash an ordered, path-bound set of production and science inputs."""
    ordered_entries = sorted(entries, key=lambda entry: entry.relative_path)
    relative_paths = [entry.relative_path for entry in ordered_entries]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Science-source entries contain duplicate relative paths")
    digest = hashlib.sha256(SCIENCE_SOURCE_FINGERPRINT_DOMAIN)
    for entry in ordered_entries:
        path_bytes = entry.relative_path.encode()
        mode_bytes = entry.git_mode.encode()
        content = canonical_science_content(entry.relative_path, entry.content)
        for field in (path_bytes, mode_bytes, content):
            digest.update(len(field).to_bytes(8, byteorder="big"))
            digest.update(field)
    return digest.hexdigest()


def git_output(repository_root: Path, *arguments: str) -> bytes:
    """Return bytes emitted by one read-only Git query."""
    completed_process = subprocess.run(
        ["git", "-C", str(repository_root), "--no-replace-objects", *arguments],
        check=True,
        capture_output=True,
    )
    return completed_process.stdout


def committed_blob_contents(repository_root: Path, blob_identifiers: tuple[str, ...]) -> tuple[bytes, ...]:
    """Read committed blobs in one replacement-disabled Git batch."""
    if not blob_identifiers:
        return ()
    completed_process = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "--no-replace-objects",
            "cat-file",
            "--batch",
        ],
        input="".join(f"{blob_identifier}\n" for blob_identifier in blob_identifiers).encode(),
        check=True,
        capture_output=True,
    )
    output = completed_process.stdout
    contents: list[bytes] = []
    offset = 0
    for expected_identifier in blob_identifiers:
        header_end = output.find(b"\n", offset)
        if header_end < 0:
            raise AssertionError("Git cat-file returned a truncated blob header")
        header_fields = output[offset:header_end].decode().split()
        if len(header_fields) != 3:
            raise AssertionError(f"Git cat-file returned an invalid blob header: {header_fields}")
        observed_identifier, object_type, size_text = header_fields
        if observed_identifier != expected_identifier or object_type != "blob":
            raise AssertionError(
                f"Git cat-file returned the wrong object: "
                f"expected {expected_identifier} blob, observed {observed_identifier} {object_type}"
            )
        content_start = header_end + 1
        content_end = content_start + int(size_text)
        if output[content_end : content_end + 1] != b"\n":
            raise AssertionError("Git cat-file returned a truncated blob payload")
        contents.append(output[content_start:content_end])
        offset = content_end + 1
    if offset != len(output):
        raise AssertionError("Git cat-file returned unexpected trailing output")
    return tuple(contents)


def committed_science_source_entries(repository_root: Path) -> tuple[ScienceSourceEntry, ...]:
    """Load production and science inputs directly from the committed HEAD tree."""
    tree_output = git_output(repository_root, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    tree_records: list[tuple[str, str, str]] = []
    for tree_record in tree_output.split(b"\0"):
        if not tree_record:
            continue
        metadata, relative_path_bytes = tree_record.split(b"\t", maxsplit=1)
        git_mode, object_type, blob_identifier = metadata.decode().split()
        relative_path = relative_path_bytes.decode()
        if not is_science_source_path(relative_path):
            continue
        if object_type != "blob":
            raise AssertionError(f"Science-source path is not a committed blob: {relative_path}")
        if git_mode not in {"100644", "100755"}:
            raise AssertionError(f"Science-source path is not a committed regular file: {relative_path} ({git_mode})")
        tree_records.append((relative_path, git_mode, blob_identifier))
    if not tree_records:
        raise AssertionError(f"No committed science-source inputs found below {repository_root}")
    blob_contents = committed_blob_contents(
        repository_root,
        tuple(blob_identifier for _relative_path, _git_mode, blob_identifier in tree_records),
    )
    return tuple(
        ScienceSourceEntry(
            relative_path=relative_path,
            git_mode=git_mode,
            content=content,
            blob_identifier=blob_identifier,
        )
        for (relative_path, git_mode, blob_identifier), content in zip(tree_records, blob_contents, strict=True)
    )


def index_science_source_entries(repository_root: Path) -> tuple[GitIndexEntry, ...]:
    """Load stage-zero production and science paths from the Git index."""
    index_output = git_output(repository_root, "ls-files", "--stage", "-z")
    entries: list[GitIndexEntry] = []
    for index_record in index_output.split(b"\0"):
        if not index_record:
            continue
        metadata, relative_path_bytes = index_record.split(b"\t", maxsplit=1)
        git_mode, blob_identifier, stage = metadata.decode().split()
        relative_path = relative_path_bytes.decode()
        if not is_science_source_path(relative_path):
            continue
        if stage != "0":
            raise AssertionError(f"Science-source index contains an unmerged path: {relative_path}")
        entries.append(
            GitIndexEntry(
                relative_path=relative_path,
                git_mode=git_mode,
                blob_identifier=blob_identifier,
            )
        )
    return tuple(entries)


def assert_science_source_index_flags_clear(repository_root: Path) -> None:
    """Reject index flags that allow Git status to ignore working-tree changes."""
    for flag_argument, flag_name in (
        ("-v", "assume-unchanged or skip-worktree"),
        ("-f", "fsmonitor-valid or skip-worktree"),
    ):
        tagged_output = git_output(repository_root, "ls-files", flag_argument, "-z")
        for tagged_record in tagged_output.split(b"\0"):
            if not tagged_record:
                continue
            tag_bytes, relative_path_bytes = tagged_record.split(b" ", maxsplit=1)
            relative_path = relative_path_bytes.decode()
            if is_science_source_path(relative_path) and tag_bytes != b"H":
                raise AssertionError(
                    f"Science-source index has forbidden {flag_name} flags: "
                    f"{relative_path} ({tag_bytes.decode(errors='replace')})"
                )


def assert_index_matches_committed_sources(
    committed_entries: tuple[ScienceSourceEntry, ...],
    index_entries: tuple[GitIndexEntry, ...],
) -> None:
    """Require index paths, modes, and blobs to equal the committed HEAD tree."""
    committed_by_path = {entry.relative_path: entry for entry in committed_entries}
    index_by_path = {entry.relative_path: entry for entry in index_entries}
    if set(committed_by_path) != set(index_by_path):
        missing_paths = sorted(set(committed_by_path).difference(index_by_path))
        unexpected_paths = sorted(set(index_by_path).difference(committed_by_path))
        raise AssertionError(
            f"Science-source index paths differ from HEAD: missing={missing_paths}, unexpected={unexpected_paths}"
        )
    for relative_path, committed_entry in committed_by_path.items():
        index_entry = index_by_path[relative_path]
        if (
            index_entry.git_mode != committed_entry.git_mode
            or index_entry.blob_identifier != committed_entry.blob_identifier
        ):
            raise AssertionError(
                f"Science-source index differs from HEAD for {relative_path}: "
                f"expected {committed_entry.git_mode}/{committed_entry.blob_identifier}, "
                f"observed {index_entry.git_mode}/{index_entry.blob_identifier}"
            )


def read_working_tree_source(repository_root: Path, relative_path: str) -> WorkingTreeSource:
    """Read one on-disk regular file without following symbolic links."""
    source_path = repository_root / relative_path
    source_status = source_path.lstat()
    if stat.S_ISLNK(source_status.st_mode):
        raise AssertionError(f"Science-source working-tree path is a symbolic link: {relative_path}")
    if not stat.S_ISREG(source_status.st_mode):
        raise AssertionError(f"Science-source working-tree path is not a regular file: {relative_path}")
    git_mode = "100755" if source_status.st_mode & 0o111 else "100644"
    return WorkingTreeSource(content=source_path.read_bytes(), git_mode=git_mode)


def assert_working_tree_matches_committed_sources(
    repository_root: Path,
    committed_entries: tuple[ScienceSourceEntry, ...],
) -> None:
    """Byte-compare every committed production/science blob with disk."""
    for committed_entry in committed_entries:
        try:
            working_source = read_working_tree_source(
                repository_root,
                committed_entry.relative_path,
            )
        except FileNotFoundError as error:
            raise AssertionError(
                f"Science-source working-tree path is missing: {committed_entry.relative_path}"
            ) from error
        if working_source.git_mode != committed_entry.git_mode:
            raise AssertionError(
                f"Science-source working-tree mode differs from HEAD for "
                f"{committed_entry.relative_path}: "
                f"expected {committed_entry.git_mode}, observed {working_source.git_mode}"
            )
        if working_source.content != committed_entry.content:
            raise AssertionError(
                f"Science-source working-tree bytes differ from HEAD for {committed_entry.relative_path}"
            )


def repository_science_source_fingerprint(repository_root: Path) -> str:
    """Return the committed HEAD tree's deterministic scientific identity."""
    return science_source_fingerprint(committed_science_source_entries(repository_root))


def repository_git_commit(repository_root: Path) -> str:
    """Return the full commit identifier checked out at repository HEAD."""
    git_commit = git_output(repository_root, "rev-parse", "HEAD").decode().strip()
    if GIT_COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise AssertionError(f"Git returned an invalid HEAD commit identifier: {git_commit}")
    return git_commit


def repository_working_tree_status(repository_root: Path) -> bytes:
    """Return porcelain status including non-ignored untracked paths."""
    return git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )


def assert_clean_exact_source(repository_root: Path, expected_git_commit: str) -> ScienceSourceState:
    """Require a clean checkout at the scheduler-selected exact commit.

    Args:
        repository_root: Root of the qualification checkout.
        expected_git_commit: Full commit selected by the qualification job.

    Returns:
        Exact commit and deterministic science-source identity.

    Raises:
        AssertionError: If the expected commit is malformed, HEAD differs, or
            tracked/non-ignored working-tree state is dirty.

    """
    if GIT_COMMIT_PATTERN.fullmatch(expected_git_commit) is None:
        raise AssertionError(f"Expected a full 40-character Git commit, got {expected_git_commit!r}")
    observed_git_commit = repository_git_commit(repository_root)
    if observed_git_commit != expected_git_commit:
        raise AssertionError(
            f"Qualification checkout is at wrong Git commit: "
            f"expected {expected_git_commit}, observed {observed_git_commit}"
        )
    working_tree_status = repository_working_tree_status(repository_root)
    if working_tree_status:
        status_text = working_tree_status.decode(errors="replace").rstrip()
        raise AssertionError(f"Qualification checkout is dirty:\n{status_text}")
    committed_entries = committed_science_source_entries(repository_root)
    assert_science_source_index_flags_clear(repository_root)
    assert_index_matches_committed_sources(
        committed_entries,
        index_science_source_entries(repository_root),
    )
    assert_working_tree_matches_committed_sources(repository_root, committed_entries)
    return ScienceSourceState(
        git_commit=observed_git_commit,
        science_source_sha256=science_source_fingerprint(committed_entries),
    )


def main() -> int:
    """Print the current repository science-source fingerprint."""
    repository_root = Path(__file__).resolve().parents[1]
    print(repository_science_source_fingerprint(repository_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic source identity for scientific qualification."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path

SCIENCE_SOURCE_FINGERPRINT_DOMAIN = b"g-science-source-v1\0"
SCIENCE_SOURCE_DIRECTORY_PREFIXES = (
    "crates/",
    "native/",
    "src/",
    "vendor/openxla/",
)
SCIENCE_SOURCE_EXACT_PATHS = frozenset(
    {
        ".cargo/config.toml",
        ".python-version",
        "Cargo.lock",
        "Cargo.toml",
        "Justfile",
        "pyproject.toml",
        "rust-toolchain.toml",
        "tests/numerical.py",
        "tests/parity/golden_metadata.json",
        "tests/parity/harness.py",
        "tests/test_regenie2_parity.py",
        "tooling/science_gate.py",
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


@dataclass(frozen=True)
class ScienceSourceState:
    """Clean repository state accepted for exact-source qualification."""

    git_commit: str
    science_source_sha256: str


def is_science_source_path(relative_path: str) -> bool:
    """Return whether a tracked path can affect production science or its gate."""
    return relative_path in SCIENCE_SOURCE_EXACT_PATHS or relative_path.startswith(SCIENCE_SOURCE_DIRECTORY_PREFIXES)


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
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
    )
    return completed_process.stdout


def tracked_science_source_entries(repository_root: Path) -> tuple[ScienceSourceEntry, ...]:
    """Load tracked production and science inputs from the working tree."""
    index_output = git_output(repository_root, "ls-files", "--stage", "-z")
    entries: list[ScienceSourceEntry] = []
    for index_record in index_output.split(b"\0"):
        if not index_record:
            continue
        metadata, relative_path_bytes = index_record.split(b"\t", maxsplit=1)
        git_mode, _blob_identifier, stage = metadata.decode().split()
        if stage != "0":
            raise AssertionError("Science-source fingerprint cannot use an unmerged Git index")
        relative_path = relative_path_bytes.decode()
        if not is_science_source_path(relative_path):
            continue
        entries.append(
            ScienceSourceEntry(
                relative_path=relative_path,
                git_mode=git_mode,
                content=(repository_root / relative_path).read_bytes(),
            )
        )
    if not entries:
        raise AssertionError(f"No tracked science-source inputs found below {repository_root}")
    return tuple(entries)


def repository_science_source_fingerprint(repository_root: Path) -> str:
    """Return the current working tree's deterministic scientific identity."""
    return science_source_fingerprint(tracked_science_source_entries(repository_root))


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
    return ScienceSourceState(
        git_commit=observed_git_commit,
        science_source_sha256=repository_science_source_fingerprint(repository_root),
    )


def main() -> int:
    """Print the current repository science-source fingerprint."""
    repository_root = Path(__file__).resolve().parents[1]
    print(repository_science_source_fingerprint(repository_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

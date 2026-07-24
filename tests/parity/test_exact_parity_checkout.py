"""Adversarial tests for exact qualification checkout isolation."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKOUT_HELPER = REPOSITORY_ROOT / "tooling" / "server" / "exact_parity_checkout.sh"


@dataclass(frozen=True)
class SourceRepositoryFixture:
    """One local source repository with a selected commit."""

    path: Path
    git_commit: str


def run_git(repository_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a deterministic local Git command."""
    return subprocess.run(
        ["/usr/bin/git", "-C", str(repository_path), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "HOME": str(repository_path.parent),
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )


def create_source_repository(directory_path: Path) -> SourceRepositoryFixture:
    """Create one committed SHA-1 source repository."""
    directory_path.mkdir()
    run_git(directory_path, "init", "--quiet", "--initial-branch=main")
    run_git(directory_path, "config", "user.name", "Parity Test")
    run_git(directory_path, "config", "user.email", "parity@example.invalid")
    (directory_path / ".gitattributes").write_text("payload.txt filter=adversarial\n", encoding="utf-8")
    (directory_path / "payload.txt").write_text("selected commit bytes\n", encoding="utf-8")
    run_git(directory_path, "add", ".gitattributes", "payload.txt")
    run_git(directory_path, "commit", "--quiet", "-m", "selected source")
    git_commit = run_git(directory_path, "rev-parse", "HEAD").stdout.strip()
    return SourceRepositoryFixture(path=directory_path, git_commit=git_commit)


def checkout_process(
    fixture: SourceRepositoryFixture,
    scratch_root: Path,
    *,
    selected_object_identifier: str,
) -> subprocess.CompletedProcess[str]:
    """Execute the production checkout helper."""
    scratch_root.mkdir(mode=0o700)
    return subprocess.run(
        [
            str(CHECKOUT_HELPER),
            str(fixture.path),
            selected_object_identifier,
            str(scratch_root / "checkout"),
            str(scratch_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            "HOME": str(scratch_root),
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )


def install_source_config_attack(fixture: SourceRepositoryFixture, marker_directory: Path) -> tuple[Path, ...]:
    """Install source-local commands that must remain outside the helper trust boundary."""
    marker_directory.mkdir()
    marker_paths = tuple(
        marker_directory / marker_name
        for marker_name in (
            "alternate-refs",
            "pack-objects",
            "filter",
        )
    )
    command_paths: list[Path] = []
    for marker_path in marker_paths:
        command_path = marker_directory / f"{marker_path.name}.sh"
        command_path.write_text(
            f"#!/usr/bin/bash\n/usr/bin/touch {marker_path}\nexit 1\n",
            encoding="utf-8",
        )
        command_path.chmod(0o700)
        command_paths.append(command_path)
    included_config_path = marker_directory / "included.config"
    included_config_path.write_text(
        "[uploadpack]\n"
        f"\tpackObjectsHook = {command_paths[1]}\n"
        '[filter "adversarial"]\n'
        f"\tsmudge = {command_paths[2]}\n"
        f"\tprocess = {command_paths[2]}\n",
        encoding="utf-8",
    )
    run_git(fixture.path, "config", "include.path", str(included_config_path))
    run_git(fixture.path, "config", "core.alternateRefsCommand", str(command_paths[0]))
    run_git(fixture.path, "config", "core.hooksPath", str(marker_directory))
    return marker_paths


def test_checkout_ignores_executable_source_config_and_owns_object_closure(tmp_path: Path) -> None:
    """Require checkout materialization to consume only a config-free shadow."""
    fixture = create_source_repository(tmp_path / "source")
    marker_paths = install_source_config_attack(fixture, tmp_path / "markers")
    scratch_root = tmp_path / "scratch"

    completed_process = checkout_process(
        fixture,
        scratch_root,
        selected_object_identifier=fixture.git_commit,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert all(not marker_path.exists() for marker_path in marker_paths)
    checkout_path = scratch_root / "checkout"
    assert (checkout_path / "payload.txt").read_text(encoding="utf-8") == "selected commit bytes\n"
    assert run_git(checkout_path, "rev-parse", "HEAD").stdout.strip() == fixture.git_commit
    assert run_git(checkout_path, "status", "--short").stdout == ""
    assert run_git(checkout_path, "remote").stdout == ""
    target_alternates_path = checkout_path / ".git" / "objects" / "info" / "alternates"
    assert not os.path.lexists(target_alternates_path)

    fixture.path.rename(tmp_path / "source-removed")
    assert run_git(checkout_path, "cat-file", "-t", fixture.git_commit).stdout.strip() == "commit"
    run_git(checkout_path, "fsck", "--full", "--strict")


@pytest.mark.parametrize("object_kind", ["annotated_tag", "tree", "blob"])
def test_checkout_rejects_noncommit_object_identifiers(tmp_path: Path, object_kind: str) -> None:
    """Reject object identifiers that peel to, contain, or accompany a commit."""
    fixture = create_source_repository(tmp_path / "source")
    if object_kind == "annotated_tag":
        run_git(fixture.path, "tag", "-a", "selected-tag", "-m", "annotated")
        object_identifier = run_git(fixture.path, "rev-parse", "selected-tag^{tag}").stdout.strip()
    elif object_kind == "tree":
        object_identifier = run_git(fixture.path, "rev-parse", "HEAD^{tree}").stdout.strip()
    else:
        object_identifier = run_git(fixture.path, "rev-parse", "HEAD:payload.txt").stdout.strip()

    completed_process = checkout_process(
        fixture,
        tmp_path / f"scratch-{object_kind}",
        selected_object_identifier=object_identifier,
    )

    assert completed_process.returncode != 0
    assert "must identify a commit object exactly" in completed_process.stderr


def test_checkout_rejects_source_object_alternates(tmp_path: Path) -> None:
    """Reject source object stores whose closure depends on another database."""
    fixture = create_source_repository(tmp_path / "source")
    alternate_objects_path = tmp_path / "alternate-objects"
    alternate_objects_path.mkdir()
    source_alternates_path = fixture.path / ".git" / "objects" / "info" / "alternates"
    source_alternates_path.write_text(f"{alternate_objects_path}\n", encoding="utf-8")

    completed_process = checkout_process(
        fixture,
        tmp_path / "scratch",
        selected_object_identifier=fixture.git_commit,
    )

    assert completed_process.returncode != 0
    assert "self-contained source object database" in completed_process.stderr

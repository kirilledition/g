from __future__ import annotations

import subprocess
import typing

import tooling.symphony_sync_main as symphony_sync_main

if typing.TYPE_CHECKING:
    from pathlib import Path


def run_git(repository_path: Path, *arguments: str) -> str:
    completed_process = subprocess.run(
        ["git", "-C", str(repository_path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed_process.stdout.strip()


def initialize_repository(repository_path: Path) -> None:
    repository_path.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main", str(repository_path)], check=True, capture_output=True, text=True)
    run_git(repository_path, "config", "user.email", "codex@example.invalid")
    run_git(repository_path, "config", "user.name", "Codex")


def commit_file(repository_path: Path, file_name: str, file_contents: str, message: str) -> str:
    (repository_path / file_name).write_text(file_contents)
    run_git(repository_path, "add", file_name)
    run_git(repository_path, "commit", "-m", message)
    return run_git(repository_path, "rev-parse", "HEAD")


def setup_origin_backed_repository(tmp_path: Path) -> tuple[Path, Path]:
    origin_path = tmp_path / "origin.git"
    repository_path = tmp_path / "repository"
    subprocess.run(["git", "init", "--bare", str(origin_path)], check=True, capture_output=True, text=True)
    initialize_repository(repository_path)
    commit_file(repository_path, "README.md", "initial\n", "Initial commit")
    run_git(repository_path, "remote", "add", "origin", str(origin_path))
    run_git(repository_path, "push", "-u", "origin", "main")
    subprocess.run(
        ["git", "--git-dir", str(origin_path), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
        capture_output=True,
        text=True,
    )
    return repository_path, origin_path


def push_remote_main_commit(tmp_path: Path, origin_path: Path, file_contents: str) -> str:
    clone_path = tmp_path / f"remote-{len(file_contents)}"
    subprocess.run(["git", "clone", str(origin_path), str(clone_path)], check=True, capture_output=True, text=True)
    run_git(clone_path, "config", "user.email", "codex@example.invalid")
    run_git(clone_path, "config", "user.name", "Codex")
    remote_head = commit_file(clone_path, "remote.txt", file_contents, "Remote update")
    run_git(clone_path, "push", "origin", "main")
    return remote_head


def test_parse_worktree_branch_records_accepts_branch_records(tmp_path: Path) -> None:
    porcelain_output = f"""worktree {tmp_path / "repository"}
HEAD abc123
branch refs/heads/main

worktree {tmp_path / "issue"}
HEAD def456
branch refs/heads/symphony/GLA-1
"""

    records = symphony_sync_main.parse_worktree_branch_records(porcelain_output)

    assert symphony_sync_main.find_branch_worktree(records, "main") == tmp_path / "repository"


def test_sync_local_main_fast_forwards_clean_main(tmp_path: Path) -> None:
    repository_path, origin_path = setup_origin_backed_repository(tmp_path)
    remote_head = push_remote_main_commit(tmp_path, origin_path, "remote\n")

    result = symphony_sync_main.sync_local_main(repository_path)

    assert result.status == symphony_sync_main.SyncMainStatus.UPDATED
    assert result.final_local_head == remote_head
    assert run_git(repository_path, "rev-parse", "HEAD") == remote_head


def test_sync_local_main_finds_main_worktree_from_issue_worktree(tmp_path: Path) -> None:
    repository_path, origin_path = setup_origin_backed_repository(tmp_path)
    issue_worktree_path = tmp_path / "issue-worktree"
    run_git(repository_path, "worktree", "add", "-b", "symphony/GLA-1", str(issue_worktree_path), "main")
    remote_head = push_remote_main_commit(tmp_path, origin_path, "remote\n")

    result = symphony_sync_main.sync_local_main(issue_worktree_path)

    assert result.status == symphony_sync_main.SyncMainStatus.UPDATED
    assert result.target_repository == repository_path
    assert run_git(repository_path, "rev-parse", "HEAD") == remote_head


def test_sync_local_main_skips_dirty_main(tmp_path: Path) -> None:
    repository_path, origin_path = setup_origin_backed_repository(tmp_path)
    original_head = run_git(repository_path, "rev-parse", "HEAD")
    push_remote_main_commit(tmp_path, origin_path, "remote\n")
    (repository_path / "dirty.txt").write_text("dirty\n")

    result = symphony_sync_main.sync_local_main(repository_path)

    assert result.status == symphony_sync_main.SyncMainStatus.SKIPPED
    assert result.detail == "Local main worktree has uncommitted changes."
    assert run_git(repository_path, "rev-parse", "HEAD") == original_head


def test_sync_local_main_skips_diverged_main(tmp_path: Path) -> None:
    repository_path, origin_path = setup_origin_backed_repository(tmp_path)
    push_remote_main_commit(tmp_path, origin_path, "remote\n")
    local_head = commit_file(repository_path, "local.txt", "local\n", "Local update")

    result = symphony_sync_main.sync_local_main(repository_path)

    assert result.status == symphony_sync_main.SyncMainStatus.SKIPPED
    assert result.detail == "Local main has diverged from origin/main; manual reconciliation required."
    assert run_git(repository_path, "rev-parse", "HEAD") == local_head

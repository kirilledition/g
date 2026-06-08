"""Safe local main synchronization for Symphony direct merges."""

from __future__ import annotations

import dataclasses
import enum
import typing
from pathlib import Path

from tooling.common import commands as tooling_commands

if typing.TYPE_CHECKING:
    import collections.abc


DEFAULT_BRANCH_NAME = "main"
DEFAULT_REMOTE_NAME = "origin"
LOCAL_BRANCH_REFERENCE_PREFIX = "refs/heads/"


class SyncMainStatus(enum.StrEnum):
    """Outcome of a local main synchronization attempt."""

    UPDATED = "updated"
    UP_TO_DATE = "up_to_date"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True)
class WorktreeBranchRecord:
    """Git worktree branch record.

    Attributes:
        path: Worktree path.
        branch_reference: Full local branch reference checked out there.

    """

    path: Path
    branch_reference: str | None


@dataclasses.dataclass(frozen=True)
class SyncMainResult:
    """Result of a safe local main synchronization attempt.

    Attributes:
        status: High-level synchronization outcome.
        requested_repository: Repository or worktree path passed by the caller.
        target_repository: Main worktree path used for local synchronization.
        branch_name: Local branch name that should be synchronized.
        remote_branch_reference: Remote branch reference used as the source.
        previous_local_head: Local branch head before synchronization.
        remote_head: Remote branch head after fetch.
        final_local_head: Local branch head after synchronization.
        detail: Human-readable outcome detail.
        command_outputs: Git command outputs captured during the attempt.

    """

    status: SyncMainStatus
    requested_repository: Path
    target_repository: Path
    branch_name: str
    remote_branch_reference: str
    previous_local_head: str | None
    remote_head: str | None
    final_local_head: str | None
    detail: str
    command_outputs: tuple[tooling_commands.CommandOutput, ...]

    @property
    def exit_code(self) -> int:
        """Return the process exit code for CLI use."""
        if self.status == SyncMainStatus.FAILED:
            return 1
        return 0


def run_git_command(repository_path: Path, arguments: collections.abc.Sequence[str]) -> tooling_commands.CommandOutput:
    """Run a git command inside a repository or worktree.

    Args:
        repository_path: Repository or worktree path for `git -C`.
        arguments: Git arguments after `git -C <path>`.

    Returns:
        Captured command output.

    """
    return tooling_commands.run_captured_command(["git", "-C", str(repository_path), *arguments])


def command_succeeded(command_output: tooling_commands.CommandOutput) -> bool:
    """Return whether a captured command succeeded."""
    return command_output.return_code == 0


def command_message(command_output: tooling_commands.CommandOutput) -> str:
    """Return the most useful single-line command message."""
    standard_error = command_output.stderr.strip()
    if standard_error:
        return standard_error
    standard_output = command_output.stdout.strip()
    if standard_output:
        return standard_output
    return f"{' '.join(command_output.command_arguments)} failed."


def parse_worktree_branch_records(porcelain_output: str) -> tuple[WorktreeBranchRecord, ...]:
    """Parse branch-bearing records from `git worktree list --porcelain`.

    Args:
        porcelain_output: Porcelain worktree list output.

    Returns:
        Parsed worktree records.

    """
    records: list[WorktreeBranchRecord] = []
    current_path: Path | None = None
    current_branch_reference: str | None = None

    for line in porcelain_output.splitlines():
        if not line:
            if current_path is not None:
                records.append(WorktreeBranchRecord(path=current_path, branch_reference=current_branch_reference))
            current_path = None
            current_branch_reference = None
            continue
        if line.startswith("worktree "):
            if current_path is not None:
                records.append(WorktreeBranchRecord(path=current_path, branch_reference=current_branch_reference))
            current_path = Path(line.removeprefix("worktree "))
            current_branch_reference = None
        elif line.startswith("branch "):
            current_branch_reference = line.removeprefix("branch ")

    if current_path is not None:
        records.append(WorktreeBranchRecord(path=current_path, branch_reference=current_branch_reference))
    return tuple(records)


def find_branch_worktree(records: collections.abc.Iterable[WorktreeBranchRecord], branch_name: str) -> Path | None:
    """Find the worktree path where a local branch is checked out.

    Args:
        records: Parsed worktree records.
        branch_name: Local branch name to locate.

    Returns:
        Worktree path when the branch is checked out, otherwise None.

    """
    branch_reference = f"{LOCAL_BRANCH_REFERENCE_PREFIX}{branch_name}"
    for record in records:
        if record.branch_reference == branch_reference:
            return record.path
    return None


def build_result(
    *,
    status: SyncMainStatus,
    requested_repository: Path,
    target_repository: Path,
    branch_name: str,
    remote_branch_reference: str,
    previous_local_head: str | None,
    remote_head: str | None,
    final_local_head: str | None,
    detail: str,
    command_outputs: collections.abc.Sequence[tooling_commands.CommandOutput],
) -> SyncMainResult:
    """Build a synchronization result from command outputs.

    Args:
        status: High-level synchronization outcome.
        requested_repository: Repository or worktree path passed by the caller.
        target_repository: Main worktree path used for synchronization.
        branch_name: Local branch name that should be synchronized.
        remote_branch_reference: Remote branch reference used as the source.
        previous_local_head: Local branch head before synchronization.
        remote_head: Remote branch head after fetch.
        final_local_head: Local branch head after synchronization.
        detail: Human-readable outcome detail.
        command_outputs: Captured git command outputs.

    Returns:
        Structured synchronization result.

    """
    return SyncMainResult(
        status=status,
        requested_repository=requested_repository,
        target_repository=target_repository,
        branch_name=branch_name,
        remote_branch_reference=remote_branch_reference,
        previous_local_head=previous_local_head,
        remote_head=remote_head,
        final_local_head=final_local_head,
        detail=detail,
        command_outputs=tuple(command_outputs),
    )


def sync_local_main(
    requested_repository: Path,
    *,
    branch_name: str = DEFAULT_BRANCH_NAME,
    remote_name: str = DEFAULT_REMOTE_NAME,
) -> SyncMainResult:
    """Fast-forward the local main worktree to the fetched remote branch when safe.

    The function is safe to call from a Symphony issue worktree. It first locates
    the worktree where `main` is checked out, then updates that checkout only if
    it is clean and can be fast-forwarded to `origin/main`.

    Args:
        requested_repository: Repository or worktree path used to locate git metadata.
        branch_name: Local branch name to synchronize.
        remote_name: Remote name to fetch from.

    Returns:
        Structured synchronization result.

    """
    command_outputs: list[tooling_commands.CommandOutput] = []
    resolved_requested_repository = requested_repository.resolve()
    remote_branch_reference = f"{remote_name}/{branch_name}"

    worktree_output = run_git_command(resolved_requested_repository, ["worktree", "list", "--porcelain"])
    command_outputs.append(worktree_output)
    if not command_succeeded(worktree_output):
        return build_result(
            status=SyncMainStatus.FAILED,
            requested_repository=resolved_requested_repository,
            target_repository=resolved_requested_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=None,
            remote_head=None,
            final_local_head=None,
            detail=command_message(worktree_output),
            command_outputs=command_outputs,
        )

    target_repository = find_branch_worktree(parse_worktree_branch_records(worktree_output.stdout), branch_name)
    if target_repository is None:
        target_repository = resolved_requested_repository

    branch_output = run_git_command(target_repository, ["branch", "--show-current"])
    command_outputs.append(branch_output)
    if not command_succeeded(branch_output):
        return build_result(
            status=SyncMainStatus.FAILED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=None,
            remote_head=None,
            final_local_head=None,
            detail=command_message(branch_output),
            command_outputs=command_outputs,
        )

    current_branch_name = branch_output.stdout.strip()
    if current_branch_name != branch_name:
        return build_result(
            status=SyncMainStatus.SKIPPED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=None,
            remote_head=None,
            final_local_head=None,
            detail=f"Target worktree is on {current_branch_name or 'detached HEAD'}, not {branch_name}.",
            command_outputs=command_outputs,
        )

    status_output = run_git_command(target_repository, ["status", "--porcelain"])
    command_outputs.append(status_output)
    if not command_succeeded(status_output):
        return build_result(
            status=SyncMainStatus.FAILED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=None,
            remote_head=None,
            final_local_head=None,
            detail=command_message(status_output),
            command_outputs=command_outputs,
        )
    if status_output.stdout.strip():
        return build_result(
            status=SyncMainStatus.SKIPPED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=None,
            remote_head=None,
            final_local_head=None,
            detail=f"Local {branch_name} worktree has uncommitted changes.",
            command_outputs=command_outputs,
        )

    previous_head_output = run_git_command(target_repository, ["rev-parse", "HEAD"])
    command_outputs.append(previous_head_output)
    if not command_succeeded(previous_head_output):
        return build_result(
            status=SyncMainStatus.FAILED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=None,
            remote_head=None,
            final_local_head=None,
            detail=command_message(previous_head_output),
            command_outputs=command_outputs,
        )
    previous_local_head = previous_head_output.stdout.strip()

    fetch_output = run_git_command(target_repository, ["fetch", remote_name, branch_name])
    command_outputs.append(fetch_output)
    if not command_succeeded(fetch_output):
        return build_result(
            status=SyncMainStatus.FAILED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=previous_local_head,
            remote_head=None,
            final_local_head=previous_local_head,
            detail=command_message(fetch_output),
            command_outputs=command_outputs,
        )

    remote_head_output = run_git_command(target_repository, ["rev-parse", remote_branch_reference])
    command_outputs.append(remote_head_output)
    if not command_succeeded(remote_head_output):
        return build_result(
            status=SyncMainStatus.FAILED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=previous_local_head,
            remote_head=None,
            final_local_head=previous_local_head,
            detail=command_message(remote_head_output),
            command_outputs=command_outputs,
        )
    remote_head = remote_head_output.stdout.strip()

    if previous_local_head == remote_head:
        return build_result(
            status=SyncMainStatus.UP_TO_DATE,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=previous_local_head,
            remote_head=remote_head,
            final_local_head=previous_local_head,
            detail=f"Local {branch_name} already matches {remote_branch_reference}.",
            command_outputs=command_outputs,
        )

    ancestor_output = run_git_command(
        target_repository,
        ["merge-base", "--is-ancestor", previous_local_head, remote_head],
    )
    command_outputs.append(ancestor_output)
    if ancestor_output.return_code == 1:
        return build_result(
            status=SyncMainStatus.SKIPPED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=previous_local_head,
            remote_head=remote_head,
            final_local_head=previous_local_head,
            detail=f"Local {branch_name} has diverged from {remote_branch_reference}; manual reconciliation required.",
            command_outputs=command_outputs,
        )
    if not command_succeeded(ancestor_output):
        return build_result(
            status=SyncMainStatus.FAILED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=previous_local_head,
            remote_head=remote_head,
            final_local_head=previous_local_head,
            detail=command_message(ancestor_output),
            command_outputs=command_outputs,
        )

    merge_output = run_git_command(target_repository, ["merge", "--ff-only", remote_branch_reference])
    command_outputs.append(merge_output)
    if not command_succeeded(merge_output):
        return build_result(
            status=SyncMainStatus.FAILED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=previous_local_head,
            remote_head=remote_head,
            final_local_head=previous_local_head,
            detail=command_message(merge_output),
            command_outputs=command_outputs,
        )

    final_head_output = run_git_command(target_repository, ["rev-parse", "HEAD"])
    command_outputs.append(final_head_output)
    if not command_succeeded(final_head_output):
        return build_result(
            status=SyncMainStatus.FAILED,
            requested_repository=resolved_requested_repository,
            target_repository=target_repository,
            branch_name=branch_name,
            remote_branch_reference=remote_branch_reference,
            previous_local_head=previous_local_head,
            remote_head=remote_head,
            final_local_head=None,
            detail=command_message(final_head_output),
            command_outputs=command_outputs,
        )

    final_local_head = final_head_output.stdout.strip()
    return build_result(
        status=SyncMainStatus.UPDATED,
        requested_repository=resolved_requested_repository,
        target_repository=target_repository,
        branch_name=branch_name,
        remote_branch_reference=remote_branch_reference,
        previous_local_head=previous_local_head,
        remote_head=remote_head,
        final_local_head=final_local_head,
        detail=f"Fast-forwarded local {branch_name} to {remote_branch_reference}.",
        command_outputs=command_outputs,
    )


def short_commit(commit_sha: str | None) -> str:
    """Return a short commit SHA for reports."""
    if commit_sha is None:
        return "-"
    return commit_sha[:12]


def render_sync_main_result(result: SyncMainResult) -> str:
    """Render a synchronization result for humans.

    Args:
        result: Synchronization result.

    Returns:
        Human-readable report.

    """
    return "\n".join(
        (
            f"Local main sync: {result.status.value}",
            f"Requested repository: {result.requested_repository}",
            f"Target repository: {result.target_repository}",
            f"Remote branch: {result.remote_branch_reference}",
            f"Previous local HEAD: {short_commit(result.previous_local_head)}",
            f"Remote HEAD: {short_commit(result.remote_head)}",
            f"Final local HEAD: {short_commit(result.final_local_head)}",
            f"Detail: {result.detail}",
            "",
        )
    )

from __future__ import annotations

import typing

import pytest

import tooling.symphony_cleanup as symphony_cleanup

if typing.TYPE_CHECKING:
    from pathlib import Path


def test_classify_worktree_path_accepts_matching_direct_child(tmp_path: Path) -> None:
    worktree_root = tmp_path / "symphony"
    worktree_path = worktree_root / "GLA-16"
    worktree_path.mkdir(parents=True)

    classification = symphony_cleanup.classify_worktree_path(worktree_path, worktree_root, "GLA-16")

    assert classification.category == symphony_cleanup.WorktreePathCategory.SELECTABLE


def test_classify_worktree_path_rejects_outside_root(tmp_path: Path) -> None:
    worktree_root = tmp_path / "symphony"
    worktree_path = tmp_path / "other" / "GLA-16"
    worktree_path.mkdir(parents=True)

    classification = symphony_cleanup.classify_worktree_path(worktree_path, worktree_root, "GLA-16")

    assert classification.category == symphony_cleanup.WorktreePathCategory.OUTSIDE_ROOT


def test_classify_worktree_path_rejects_root_itself(tmp_path: Path) -> None:
    worktree_root = tmp_path / "symphony"
    worktree_root.mkdir()

    classification = symphony_cleanup.classify_worktree_path(worktree_root, worktree_root, None)

    assert classification.category == symphony_cleanup.WorktreePathCategory.ROOT_PATH


def test_classify_worktree_path_rejects_nested_child(tmp_path: Path) -> None:
    worktree_root = tmp_path / "symphony"
    worktree_path = worktree_root / "nested" / "GLA-16"
    worktree_path.mkdir(parents=True)

    classification = symphony_cleanup.classify_worktree_path(worktree_path, worktree_root, "GLA-16")

    assert classification.category == symphony_cleanup.WorktreePathCategory.NESTED_PATH


@pytest.mark.parametrize("protected_name", ["data", "results", ".cache", ".pytest_cache", "build-cache"])
def test_classify_worktree_path_rejects_protected_names(tmp_path: Path, protected_name: str) -> None:
    worktree_root = tmp_path / "symphony"
    worktree_path = worktree_root / protected_name
    worktree_path.mkdir(parents=True)

    classification = symphony_cleanup.classify_worktree_path(worktree_path, worktree_root, None)

    assert classification.category == symphony_cleanup.WorktreePathCategory.PROTECTED_NAME


def test_classify_worktree_path_rejects_issue_name_mismatch(tmp_path: Path) -> None:
    worktree_root = tmp_path / "symphony"
    worktree_path = worktree_root / "unrelated"
    worktree_path.mkdir(parents=True)

    classification = symphony_cleanup.classify_worktree_path(worktree_path, worktree_root, "GLA-16")

    assert classification.category == symphony_cleanup.WorktreePathCategory.ISSUE_NAME_MISMATCH


@pytest.mark.parametrize(
    ("state_name", "state_type", "expected_category"),
    [
        ("In Progress", "started", symphony_cleanup.IssueStateCategory.ACTIVE),
        ("Todo", "unstarted", symphony_cleanup.IssueStateCategory.ACTIVE),
        ("Done", "completed", symphony_cleanup.IssueStateCategory.COMPLETED),
        ("Canceled", "canceled", symphony_cleanup.IssueStateCategory.CANCELED),
        ("Cancelled", "canceled", symphony_cleanup.IssueStateCategory.CANCELED),
        ("Duplicate", "duplicate", symphony_cleanup.IssueStateCategory.CANCELED),
        (None, None, symphony_cleanup.IssueStateCategory.UNKNOWN),
    ],
)
def test_categorize_linear_state(
    state_name: str | None,
    state_type: str | None,
    expected_category: symphony_cleanup.IssueStateCategory,
) -> None:
    assert symphony_cleanup.categorize_linear_state(state_name, state_type) == expected_category


def test_issue_identifier_from_branch_accepts_local_and_remote_symphony_branches() -> None:
    assert symphony_cleanup.issue_identifier_from_branch("symphony/GLA-16") == "GLA-16"
    assert symphony_cleanup.issue_identifier_from_branch("origin/symphony/GLA-16") == "GLA-16"
    assert symphony_cleanup.issue_identifier_from_branch("main") is None


def test_classify_git_worktree_selects_clean_completed_issue(tmp_path: Path) -> None:
    worktree_root = tmp_path / "symphony"
    worktree_path = worktree_root / "GLA-16"
    worktree_path.mkdir(parents=True)
    git_worktree = symphony_cleanup.GitWorktreeRecord(
        path=worktree_path,
        head="abc123",
        branch_name="symphony/GLA-16",
    )
    issue_state = symphony_cleanup.IssueState(
        issue_identifier="GLA-16",
        state_name="Done",
        state_type="completed",
        category=symphony_cleanup.IssueStateCategory.COMPLETED,
        detail="test",
    )

    classified_worktree = symphony_cleanup.classify_git_worktree(
        git_worktree,
        worktree_root,
        {"GLA-16": issue_state},
        symphony_cleanup.WorktreeDeletionReadiness(ready=True, reason="clean"),
        include_unknown=False,
    )

    assert classified_worktree.stale_candidate is True
    assert classified_worktree.retain_reason == ""


def test_classify_git_worktree_retains_active_issue(tmp_path: Path) -> None:
    worktree_root = tmp_path / "symphony"
    worktree_path = worktree_root / "GLA-16"
    worktree_path.mkdir(parents=True)
    git_worktree = symphony_cleanup.GitWorktreeRecord(
        path=worktree_path,
        head="abc123",
        branch_name="symphony/GLA-16",
    )
    issue_state = symphony_cleanup.IssueState(
        issue_identifier="GLA-16",
        state_name="In Progress",
        state_type="started",
        category=symphony_cleanup.IssueStateCategory.ACTIVE,
        detail="test",
    )

    classified_worktree = symphony_cleanup.classify_git_worktree(
        git_worktree,
        worktree_root,
        {"GLA-16": issue_state},
        symphony_cleanup.WorktreeDeletionReadiness(ready=True, reason="clean"),
        include_unknown=False,
    )

    assert classified_worktree.stale_candidate is False
    assert classified_worktree.retain_reason == "Linear issue state is active."


def test_classify_git_worktree_retains_unknown_by_default(tmp_path: Path) -> None:
    worktree_root = tmp_path / "symphony"
    worktree_path = worktree_root / "GLA-16"
    worktree_path.mkdir(parents=True)
    git_worktree = symphony_cleanup.GitWorktreeRecord(
        path=worktree_path,
        head="abc123",
        branch_name="symphony/GLA-16",
    )
    issue_state = symphony_cleanup.unknown_issue_state("GLA-16", "test")

    classified_worktree = symphony_cleanup.classify_git_worktree(
        git_worktree,
        worktree_root,
        {"GLA-16": issue_state},
        symphony_cleanup.WorktreeDeletionReadiness(ready=True, reason="clean"),
        include_unknown=False,
    )

    assert classified_worktree.stale_candidate is False
    assert classified_worktree.retain_reason == "Linear issue state is unknown."


def test_unsafe_protected_children_allows_symlinked_data(tmp_path: Path) -> None:
    worktree_path = tmp_path / "GLA-16"
    real_data_path = tmp_path / "real-data"
    data_symlink = worktree_path / "data"
    worktree_path.mkdir()
    real_data_path.mkdir()
    data_symlink.symlink_to(real_data_path, target_is_directory=True)

    assert symphony_cleanup.unsafe_protected_children(worktree_path) == ()

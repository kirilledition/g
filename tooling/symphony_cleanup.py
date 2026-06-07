"""Safe cleanup planning for Symphony worktrees and branches."""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import re
import typing
import urllib.error
import urllib.request
from pathlib import Path

from tooling.common import commands as tooling_commands

if typing.TYPE_CHECKING:
    import collections.abc

LINEAR_GRAPHQL_ENDPOINT = "https://api.linear.app/graphql"
LINEAR_API_KEY_ENVIRONMENT_VARIABLE = "LINEAR_API_KEY"
SYMPHONY_WORKTREE_ROOT_ENVIRONMENT_VARIABLE = "SYMPHONY_WORKTREE_ROOT"
DEFAULT_SYMPHONY_WORKTREE_ROOT = Path("/mnt/beegfs/kirill/Projects/g-worktrees/symphony")
SYMPHONY_BRANCH_PREFIX = "symphony/"
ORIGIN_REMOTE_PREFIX = "origin/"
ISSUE_IDENTIFIER_PATTERN = re.compile(r"[A-Z][A-Z0-9]+-\d+")
LOCAL_BRANCH_REFERENCE_PREFIX = "refs/heads/"
PROTECTED_PATH_NAMES = frozenset(
    {
        "data",
        "results",
        "cache",
        "caches",
        ".cache",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".ty_cache",
        "__pycache__",
    }
)


class CleanupError(RuntimeError):
    """Raised when cleanup planning cannot inspect repository state."""


class IssueStateCategory(enum.StrEnum):
    """Issue state category used by cleanup selection."""

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELED = "canceled"
    UNKNOWN = "unknown"


class WorktreePathCategory(enum.StrEnum):
    """Safety category for a candidate worktree path."""

    SELECTABLE = "selectable"
    OUTSIDE_ROOT = "outside_root"
    ROOT_PATH = "root_path"
    NESTED_PATH = "nested_path"
    PROTECTED_NAME = "protected_name"
    ISSUE_NAME_MISMATCH = "issue_name_mismatch"


class BranchKind(enum.StrEnum):
    """Git branch namespace represented by a cleanup record."""

    LOCAL = "local"
    REMOTE = "remote"


@dataclasses.dataclass(frozen=True)
class IssueState:
    """Linear state summary for one issue.

    Attributes:
        issue_identifier: Linear issue identifier.
        state_name: Linear state name, when known.
        state_type: Linear state type, when known.
        category: Cleanup category derived from Linear state fields.
        detail: Human-readable explanation for unknown or fallback states.

    """

    issue_identifier: str
    state_name: str | None
    state_type: str | None
    category: IssueStateCategory
    detail: str


@dataclasses.dataclass(frozen=True)
class LinearIssueStateResult:
    """Result of querying Linear issue states.

    Attributes:
        issue_states: States keyed by issue identifier.
        reachable: Whether Linear was reachable and returned a usable response.
        detail: Human-readable API status.

    """

    issue_states: dict[str, IssueState]
    reachable: bool
    detail: str


@dataclasses.dataclass(frozen=True)
class GitWorktreeRecord:
    """Worktree entry reported by git.

    Attributes:
        path: Worktree path.
        head: Current HEAD object name reported by git, when present.
        branch_name: Local branch name checked out by the worktree, when present.

    """

    path: Path
    head: str | None
    branch_name: str | None


@dataclasses.dataclass(frozen=True)
class GitBranchRecord:
    """Git branch entry considered by cleanup.

    Attributes:
        name: Branch name as reported by git.
        kind: Local or remote branch namespace.

    """

    name: str
    kind: BranchKind


@dataclasses.dataclass(frozen=True)
class WorktreePathClassification:
    """Safety classification for a worktree path.

    Attributes:
        category: Safety category.
        resolved_path: Resolved worktree path.
        reason: Human-readable reason.

    """

    category: WorktreePathCategory
    resolved_path: Path
    reason: str


@dataclasses.dataclass(frozen=True)
class WorktreeDeletionReadiness:
    """Whether git worktree removal should be attempted.

    Attributes:
        ready: Whether the worktree has no detected blocking condition.
        reason: Human-readable readiness reason.

    """

    ready: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class ClassifiedWorktree:
    """Worktree record classified for cleanup.

    Attributes:
        git_worktree: Raw git worktree record.
        issue_identifier: Linear issue identifier inferred from path or branch.
        issue_state: Linear issue state summary.
        path_classification: Safety classification for the worktree path.
        deletion_readiness: Whether deletion is safe to attempt.
        stale_candidate: Whether this worktree is selected by the current plan.
        retain_reason: Human-readable reason when the worktree is not selected.

    """

    git_worktree: GitWorktreeRecord
    issue_identifier: str | None
    issue_state: IssueState
    path_classification: WorktreePathClassification
    deletion_readiness: WorktreeDeletionReadiness
    stale_candidate: bool
    retain_reason: str


@dataclasses.dataclass(frozen=True)
class ClassifiedBranch:
    """Branch record classified for cleanup.

    Attributes:
        git_branch: Raw git branch record.
        issue_identifier: Linear issue identifier inferred from the branch.
        issue_state: Linear issue state summary.
        stale_candidate: Whether this branch is selected by the current plan.
        retain_reason: Human-readable reason when the branch is not selected.

    """

    git_branch: GitBranchRecord
    issue_identifier: str | None
    issue_state: IssueState
    stale_candidate: bool
    retain_reason: str


@dataclasses.dataclass(frozen=True)
class CleanupPlan:
    """Planned stale Symphony cleanup actions.

    Attributes:
        repository_root: Repository path used for git commands.
        worktree_root: Configured Symphony worktree root.
        linear_result: Linear reachability and issue state lookup result.
        worktrees: Classified worktree records.
        local_branches: Classified local branch records.
        remote_branches: Classified remote branch records.

    """

    repository_root: Path
    worktree_root: Path
    linear_result: LinearIssueStateResult
    worktrees: tuple[ClassifiedWorktree, ...]
    local_branches: tuple[ClassifiedBranch, ...]
    remote_branches: tuple[ClassifiedBranch, ...]


@dataclasses.dataclass(frozen=True)
class CleanupExecutionResult:
    """Result of applying a cleanup plan.

    Attributes:
        command_outputs: Git command results in execution order.

    """

    command_outputs: tuple[tooling_commands.CommandOutput, ...]

    @property
    def exit_code(self) -> int:
        """Return the aggregate process exit code."""
        for command_output in self.command_outputs:
            if command_output.return_code not in (0, None):
                return int(command_output.return_code)
            if command_output.return_code is None:
                return 127
        return 0


def configured_worktree_root(environment: collections.abc.Mapping[str, str] | None = None) -> Path:
    """Return the configured Symphony worktree root.

    Args:
        environment: Optional environment mapping for tests.

    Returns:
        Configured worktree root path.

    """
    environment_values = environment if environment is not None else os.environ
    return Path(environment_values.get(SYMPHONY_WORKTREE_ROOT_ENVIRONMENT_VARIABLE, DEFAULT_SYMPHONY_WORKTREE_ROOT))


def normalize_issue_identifier(value: str) -> str | None:
    """Return a normalized Linear issue identifier when the value matches one.

    Args:
        value: Candidate issue identifier.

    Returns:
        Uppercase Linear issue identifier, or None when the value does not match.

    """
    candidate_identifier = value.strip().upper()
    if ISSUE_IDENTIFIER_PATTERN.fullmatch(candidate_identifier) is None:
        return None
    return candidate_identifier


def issue_identifier_from_branch(branch_name: str | None) -> str | None:
    """Infer a Linear issue identifier from a Symphony branch name.

    Args:
        branch_name: Local or remote branch name.

    Returns:
        Linear issue identifier, or None when the branch is not a Symphony issue branch.

    """
    if branch_name is None:
        return None
    normalized_branch_name = branch_name.removeprefix(ORIGIN_REMOTE_PREFIX)
    if not normalized_branch_name.startswith(SYMPHONY_BRANCH_PREFIX):
        return None
    return normalize_issue_identifier(normalized_branch_name.removeprefix(SYMPHONY_BRANCH_PREFIX))


def issue_identifier_from_worktree(git_worktree: GitWorktreeRecord) -> str | None:
    """Infer a Linear issue identifier from a worktree path or branch.

    Args:
        git_worktree: Worktree record.

    Returns:
        Linear issue identifier, or None when neither path nor branch matches.

    """
    branch_issue_identifier = issue_identifier_from_branch(git_worktree.branch_name)
    if branch_issue_identifier is not None:
        return branch_issue_identifier
    return normalize_issue_identifier(git_worktree.path.name)


def unknown_issue_state(issue_identifier: str | None, detail: str) -> IssueState:
    """Build an unknown issue state placeholder.

    Args:
        issue_identifier: Optional issue identifier.
        detail: Human-readable reason.

    Returns:
        Unknown issue state.

    """
    return IssueState(
        issue_identifier=issue_identifier or "unknown",
        state_name=None,
        state_type=None,
        category=IssueStateCategory.UNKNOWN,
        detail=detail,
    )


def categorize_linear_state(state_name: str | None, state_type: str | None) -> IssueStateCategory:
    """Categorize a Linear workflow state for cleanup.

    Args:
        state_name: Linear state name.
        state_type: Linear state type.

    Returns:
        Cleanup issue state category.

    """
    normalized_state_name = (state_name or "").casefold()
    normalized_state_type = (state_type or "").casefold()
    if normalized_state_type == "completed" or normalized_state_name in {"done", "closed"}:
        return IssueStateCategory.COMPLETED
    if normalized_state_type in {"canceled", "duplicate"} or normalized_state_name in {
        "canceled",
        "cancelled",
        "duplicate",
    }:
        return IssueStateCategory.CANCELED
    if not normalized_state_name and not normalized_state_type:
        return IssueStateCategory.UNKNOWN
    return IssueStateCategory.ACTIVE


def issue_state_from_linear_payload(issue_identifier: str, issue_payload: object) -> IssueState:
    """Build an issue state from one Linear GraphQL issue payload.

    Args:
        issue_identifier: Requested Linear issue identifier.
        issue_payload: GraphQL issue payload.

    Returns:
        Issue state summary.

    """
    if not isinstance(issue_payload, dict):
        return unknown_issue_state(issue_identifier, "Issue was not found in Linear.")
    typed_issue_payload = typing.cast("dict[str, object]", issue_payload)
    state_payload = typed_issue_payload.get("state")
    if not isinstance(state_payload, dict):
        return unknown_issue_state(issue_identifier, "Linear response did not include an issue state.")
    typed_state_payload = typing.cast("dict[str, object]", state_payload)
    state_name_value = typed_state_payload.get("name")
    state_type_value = typed_state_payload.get("type")
    state_name = state_name_value if isinstance(state_name_value, str) else None
    state_type = state_type_value if isinstance(state_type_value, str) else None
    category = categorize_linear_state(state_name, state_type)
    return IssueState(
        issue_identifier=issue_identifier,
        state_name=state_name,
        state_type=state_type,
        category=category,
        detail="Linear state resolved.",
    )


def build_linear_graphql_query(issue_identifiers: tuple[str, ...]) -> str:
    """Build a GraphQL query for issue state lookup.

    Args:
        issue_identifiers: Issue identifiers to query.

    Returns:
        GraphQL query text.

    """
    variable_declarations = ", ".join(
        f"$issue_identifier_{issue_index}: String!" for issue_index in range(len(issue_identifiers))
    )
    issue_selections = "\n".join(
        f"  issue_{issue_index}: issue(id: $issue_identifier_{issue_index}) {{ identifier state {{ name type }} }}"
        for issue_index in range(len(issue_identifiers))
    )
    return f"query SymphonyCleanupIssueStates({variable_declarations}) {{\n{issue_selections}\n}}"


def fetch_linear_issue_states(
    issue_identifiers: collections.abc.Iterable[str],
    *,
    environment: collections.abc.Mapping[str, str] | None = None,
    endpoint: str = LINEAR_GRAPHQL_ENDPOINT,
    timeout_seconds: float = 15.0,
) -> LinearIssueStateResult:
    """Fetch Linear issue states by identifier.

    Args:
        issue_identifiers: Issue identifiers to query.
        environment: Optional environment mapping for tests.
        endpoint: Linear GraphQL endpoint.
        timeout_seconds: HTTP timeout in seconds.

    Returns:
        Linear issue state lookup result. Missing credentials or API failures
        return unknown states instead of raising.

    """
    sorted_issue_identifiers = tuple(sorted(set(issue_identifiers)))
    if not sorted_issue_identifiers:
        return LinearIssueStateResult(issue_states={}, reachable=False, detail="No issue identifiers to query.")
    environment_values = environment if environment is not None else os.environ
    linear_api_key = environment_values.get(LINEAR_API_KEY_ENVIRONMENT_VARIABLE)
    if not linear_api_key:
        return LinearIssueStateResult(
            issue_states={
                issue_identifier: unknown_issue_state(issue_identifier, "LINEAR_API_KEY is not set.")
                for issue_identifier in sorted_issue_identifiers
            },
            reachable=False,
            detail="LINEAR_API_KEY is not set.",
        )

    variables = {
        f"issue_identifier_{issue_index}": issue_identifier
        for issue_index, issue_identifier in enumerate(sorted_issue_identifiers)
    }
    request_body = json.dumps(
        {
            "query": build_linear_graphql_query(sorted_issue_identifiers),
            "variables": variables,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={
            "Authorization": linear_api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.URLError as error:
        return linear_error_result(sorted_issue_identifiers, f"Linear request failed: {error}")
    except TimeoutError as error:
        return linear_error_result(sorted_issue_identifiers, f"Linear request timed out: {error}")

    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        return linear_error_result(sorted_issue_identifiers, f"Linear response was not JSON: {error}")
    if not isinstance(response_payload, dict):
        return linear_error_result(sorted_issue_identifiers, "Linear response was not an object.")
    if response_payload.get("errors") is not None:
        return linear_error_result(
            sorted_issue_identifiers, f"Linear returned GraphQL errors: {response_payload['errors']}"
        )
    data_payload = response_payload.get("data")
    if not isinstance(data_payload, dict):
        return linear_error_result(sorted_issue_identifiers, "Linear response did not include data.")

    issue_states: dict[str, IssueState] = {}
    for issue_index, issue_identifier in enumerate(sorted_issue_identifiers):
        issue_payload = data_payload.get(f"issue_{issue_index}")
        issue_states[issue_identifier] = issue_state_from_linear_payload(issue_identifier, issue_payload)
    return LinearIssueStateResult(
        issue_states=issue_states,
        reachable=True,
        detail="Linear issue states resolved.",
    )


def linear_error_result(issue_identifiers: tuple[str, ...], detail: str) -> LinearIssueStateResult:
    """Build a Linear lookup result where all issue states are unknown.

    Args:
        issue_identifiers: Issue identifiers that could not be resolved.
        detail: Human-readable error detail.

    Returns:
        Linear lookup result.

    """
    return LinearIssueStateResult(
        issue_states={
            issue_identifier: unknown_issue_state(issue_identifier, detail) for issue_identifier in issue_identifiers
        },
        reachable=False,
        detail=detail,
    )


def normalize_local_branch_reference(branch_reference: str | None) -> str | None:
    """Normalize a local git branch reference.

    Args:
        branch_reference: Branch reference from `git worktree list --porcelain`.

    Returns:
        Short local branch name when present.

    """
    if branch_reference is None:
        return None
    return branch_reference.removeprefix(LOCAL_BRANCH_REFERENCE_PREFIX)


def parse_git_worktree_porcelain(porcelain_output: str) -> tuple[GitWorktreeRecord, ...]:
    """Parse `git worktree list --porcelain` output.

    Args:
        porcelain_output: Porcelain output from git.

    Returns:
        Parsed git worktree records.

    """
    records: list[GitWorktreeRecord] = []
    current_fields: dict[str, str] = {}
    for output_line in porcelain_output.splitlines():
        if not output_line:
            append_git_worktree_record(records, current_fields)
            current_fields = {}
            continue
        field_name, separator, field_value = output_line.partition(" ")
        current_fields[field_name] = field_value if separator else ""
    append_git_worktree_record(records, current_fields)
    return tuple(records)


def append_git_worktree_record(records: list[GitWorktreeRecord], fields: dict[str, str]) -> None:
    """Append one parsed worktree record when fields contain a worktree path.

    Args:
        records: Mutable record list.
        fields: Parsed fields for one worktree.

    """
    worktree_value = fields.get("worktree")
    if worktree_value is None:
        return
    records.append(
        GitWorktreeRecord(
            path=Path(worktree_value),
            head=fields.get("HEAD"),
            branch_name=normalize_local_branch_reference(fields.get("branch")),
        )
    )


def read_git_worktrees(repository_root: Path) -> tuple[GitWorktreeRecord, ...]:
    """Read git worktrees for the repository.

    Args:
        repository_root: Repository path used for git commands.

    Returns:
        Git worktree records.

    Raises:
        CleanupError: If git cannot list worktrees.

    """
    command_output = tooling_commands.run_captured_command(
        ["git", "-C", str(repository_root), "worktree", "list", "--porcelain"]
    )
    if command_output.return_code != 0:
        raise CleanupError(command_output.stderr.strip() or "git worktree list failed.")
    return parse_git_worktree_porcelain(command_output.stdout)


def read_git_branches(
    repository_root: Path, reference_prefix: str, branch_kind: BranchKind
) -> tuple[GitBranchRecord, ...]:
    """Read branches from a git reference namespace.

    Args:
        repository_root: Repository path used for git commands.
        reference_prefix: Git reference prefix.
        branch_kind: Branch namespace represented by the records.

    Returns:
        Git branch records.

    Raises:
        CleanupError: If git cannot list branches.

    """
    command_output = tooling_commands.run_captured_command(
        ["git", "-C", str(repository_root), "for-each-ref", "--format=%(refname:short)", reference_prefix]
    )
    if command_output.return_code != 0:
        raise CleanupError(command_output.stderr.strip() or f"git for-each-ref failed for {reference_prefix}.")
    return tuple(
        GitBranchRecord(name=branch_name, kind=branch_kind)
        for branch_name in command_output.stdout.splitlines()
        if branch_name
    )


def is_protected_path_name(path_name: str) -> bool:
    """Return whether a path name is protected from cleanup selection.

    Args:
        path_name: Basename to inspect.

    Returns:
        Whether the name represents data, results, or cache content.

    """
    normalized_path_name = path_name.casefold()
    return (
        normalized_path_name in PROTECTED_PATH_NAMES
        or normalized_path_name.endswith("_cache")
        or normalized_path_name.endswith("-cache")
        or normalized_path_name.endswith(".cache")
    )


def classify_worktree_path(
    worktree_path: Path,
    worktree_root: Path,
    issue_identifier: str | None,
) -> WorktreePathClassification:
    """Classify whether a worktree path may be selected.

    Args:
        worktree_path: Candidate worktree path.
        worktree_root: Configured Symphony worktree root.
        issue_identifier: Issue identifier inferred for the worktree.

    Returns:
        Path safety classification.

    """
    resolved_worktree_root = worktree_root.resolve(strict=False)
    resolved_worktree_path = worktree_path.resolve(strict=False)
    if resolved_worktree_path == resolved_worktree_root:
        return WorktreePathClassification(
            category=WorktreePathCategory.ROOT_PATH,
            resolved_path=resolved_worktree_path,
            reason="Refusing to select the configured worktree root itself.",
        )
    if not resolved_worktree_path.is_relative_to(resolved_worktree_root):
        return WorktreePathClassification(
            category=WorktreePathCategory.OUTSIDE_ROOT,
            resolved_path=resolved_worktree_path,
            reason="Path is outside the configured Symphony worktree root.",
        )
    if resolved_worktree_path.parent != resolved_worktree_root:
        return WorktreePathClassification(
            category=WorktreePathCategory.NESTED_PATH,
            resolved_path=resolved_worktree_path,
            reason="Only direct children of the Symphony worktree root are selectable.",
        )
    if is_protected_path_name(resolved_worktree_path.name):
        return WorktreePathClassification(
            category=WorktreePathCategory.PROTECTED_NAME,
            resolved_path=resolved_worktree_path,
            reason="Path name is protected data, results, or cache content.",
        )
    if issue_identifier is not None and resolved_worktree_path.name.upper() != issue_identifier:
        return WorktreePathClassification(
            category=WorktreePathCategory.ISSUE_NAME_MISMATCH,
            resolved_path=resolved_worktree_path,
            reason="Worktree directory name does not match the inferred Linear issue identifier.",
        )
    return WorktreePathClassification(
        category=WorktreePathCategory.SELECTABLE,
        resolved_path=resolved_worktree_path,
        reason="Path is a direct child of the configured Symphony worktree root.",
    )


def inspect_worktree_deletion_readiness(worktree_path: Path) -> WorktreeDeletionReadiness:
    """Inspect whether a worktree is ready for non-forced git removal.

    Args:
        worktree_path: Worktree path.

    Returns:
        Deletion readiness result.

    """
    if not worktree_path.exists():
        return WorktreeDeletionReadiness(ready=False, reason="Worktree path does not exist.")
    if not worktree_path.is_dir():
        return WorktreeDeletionReadiness(ready=False, reason="Worktree path is not a directory.")
    protected_children = unsafe_protected_children(worktree_path)
    if protected_children:
        child_list = ", ".join(str(child_path) for child_path in protected_children)
        return WorktreeDeletionReadiness(
            ready=False,
            reason=f"Worktree contains non-symlink protected paths: {child_list}",
        )
    command_output = tooling_commands.run_captured_command(["git", "-C", str(worktree_path), "status", "--porcelain"])
    if command_output.return_code != 0:
        return WorktreeDeletionReadiness(
            ready=False,
            reason=command_output.stderr.strip() or "git status failed for worktree.",
        )
    if command_output.stdout.strip():
        return WorktreeDeletionReadiness(ready=False, reason="Worktree has uncommitted or untracked changes.")
    return WorktreeDeletionReadiness(ready=True, reason="Worktree is clean for non-forced git removal.")


def deletion_readiness_for_plan(git_worktree: GitWorktreeRecord, worktree_root: Path) -> WorktreeDeletionReadiness:
    """Inspect deletion readiness only for selectable worktree paths.

    Args:
        git_worktree: Git worktree record.
        worktree_root: Configured Symphony worktree root.

    Returns:
        Deletion readiness result.

    """
    issue_identifier = issue_identifier_from_worktree(git_worktree)
    path_classification = classify_worktree_path(git_worktree.path, worktree_root, issue_identifier)
    if path_classification.category != WorktreePathCategory.SELECTABLE:
        return WorktreeDeletionReadiness(ready=False, reason=path_classification.reason)
    return inspect_worktree_deletion_readiness(git_worktree.path)


def unsafe_protected_children(worktree_path: Path) -> tuple[Path, ...]:
    """Return protected child paths that are real paths rather than symlinks.

    Args:
        worktree_path: Worktree directory to inspect.

    Returns:
        Protected child paths that cleanup must not remove.

    """
    protected_children: list[Path] = []
    for child_path in worktree_path.iterdir():
        if is_protected_path_name(child_path.name) and not child_path.is_symlink():
            protected_children.append(child_path)
    return tuple(protected_children)


def should_select_issue_state(issue_state: IssueState, *, include_unknown: bool) -> bool:
    """Return whether an issue state is selected by cleanup.

    Args:
        issue_state: Issue state to inspect.
        include_unknown: Whether unknown issue states may be selected.

    Returns:
        Whether cleanup may select this state.

    """
    if issue_state.category in {IssueStateCategory.COMPLETED, IssueStateCategory.CANCELED}:
        return True
    return include_unknown and issue_state.category == IssueStateCategory.UNKNOWN


def classify_git_worktree(
    git_worktree: GitWorktreeRecord,
    worktree_root: Path,
    issue_states: collections.abc.Mapping[str, IssueState],
    deletion_readiness: WorktreeDeletionReadiness,
    *,
    include_unknown: bool,
) -> ClassifiedWorktree:
    """Classify one git worktree for cleanup.

    Args:
        git_worktree: Git worktree record.
        worktree_root: Configured Symphony worktree root.
        issue_states: Linear issue states keyed by identifier.
        deletion_readiness: Non-forced removal readiness.
        include_unknown: Whether unknown issue states may be selected.

    Returns:
        Classified worktree.

    """
    issue_identifier = issue_identifier_from_worktree(git_worktree)
    if issue_identifier is None:
        issue_state = unknown_issue_state(issue_identifier, "No Linear state is available for this worktree.")
    else:
        issue_state = issue_states.get(
            issue_identifier,
            unknown_issue_state(issue_identifier, "No Linear state is available for this worktree."),
        )
    path_classification = classify_worktree_path(git_worktree.path, worktree_root, issue_identifier)
    retain_reason = worktree_retain_reason(
        issue_identifier=issue_identifier,
        issue_state=issue_state,
        path_classification=path_classification,
        deletion_readiness=deletion_readiness,
        include_unknown=include_unknown,
    )
    return ClassifiedWorktree(
        git_worktree=git_worktree,
        issue_identifier=issue_identifier,
        issue_state=issue_state,
        path_classification=path_classification,
        deletion_readiness=deletion_readiness,
        stale_candidate=retain_reason == "",
        retain_reason=retain_reason,
    )


def worktree_retain_reason(
    *,
    issue_identifier: str | None,
    issue_state: IssueState,
    path_classification: WorktreePathClassification,
    deletion_readiness: WorktreeDeletionReadiness,
    include_unknown: bool,
) -> str:
    """Return why a worktree should be retained, or an empty string when selected.

    Args:
        issue_identifier: Inferred issue identifier.
        issue_state: Linear issue state summary.
        path_classification: Worktree path safety classification.
        deletion_readiness: Non-forced removal readiness.
        include_unknown: Whether unknown issue states may be selected.

    Returns:
        Empty string for selected candidates, otherwise a retain reason.

    """
    if path_classification.category != WorktreePathCategory.SELECTABLE:
        return path_classification.reason
    if issue_identifier is None:
        return "No Symphony issue identifier found in worktree path or branch."
    if not deletion_readiness.ready:
        return deletion_readiness.reason
    if should_select_issue_state(issue_state, include_unknown=include_unknown):
        return ""
    if issue_state.category == IssueStateCategory.UNKNOWN:
        return "Linear issue state is unknown."
    return f"Linear issue state is {issue_state.category.value}."


def classify_git_branch(
    git_branch: GitBranchRecord,
    issue_states: collections.abc.Mapping[str, IssueState],
    *,
    include_unknown: bool,
) -> ClassifiedBranch:
    """Classify one git branch for cleanup.

    Args:
        git_branch: Git branch record.
        issue_states: Linear issue states keyed by identifier.
        include_unknown: Whether unknown issue states may be selected.

    Returns:
        Classified branch.

    """
    issue_identifier = issue_identifier_from_branch(git_branch.name)
    if issue_identifier is None:
        issue_state = unknown_issue_state(issue_identifier, "No Linear state is available for this branch.")
    else:
        issue_state = issue_states.get(
            issue_identifier,
            unknown_issue_state(issue_identifier, "No Linear state is available for this branch."),
        )
    retain_reason = branch_retain_reason(
        issue_identifier=issue_identifier,
        issue_state=issue_state,
        include_unknown=include_unknown,
    )
    return ClassifiedBranch(
        git_branch=git_branch,
        issue_identifier=issue_identifier,
        issue_state=issue_state,
        stale_candidate=retain_reason == "",
        retain_reason=retain_reason,
    )


def branch_retain_reason(
    *,
    issue_identifier: str | None,
    issue_state: IssueState,
    include_unknown: bool,
) -> str:
    """Return why a branch should be retained, or an empty string when selected.

    Args:
        issue_identifier: Inferred issue identifier.
        issue_state: Linear issue state summary.
        include_unknown: Whether unknown issue states may be selected.

    Returns:
        Empty string for selected candidates, otherwise a retain reason.

    """
    if issue_identifier is None:
        return "Branch is not a Symphony issue branch."
    if should_select_issue_state(issue_state, include_unknown=include_unknown):
        return ""
    if issue_state.category == IssueStateCategory.UNKNOWN:
        return "Linear issue state is unknown."
    return f"Linear issue state is {issue_state.category.value}."


def build_cleanup_plan(
    repository_root: Path,
    worktree_root: Path,
    *,
    include_unknown: bool = False,
    environment: collections.abc.Mapping[str, str] | None = None,
) -> CleanupPlan:
    """Build a stale Symphony cleanup plan.

    Args:
        repository_root: Repository path used for git commands.
        worktree_root: Configured Symphony worktree root.
        include_unknown: Whether unknown issue states may be selected.
        environment: Optional environment mapping for Linear lookup.

    Returns:
        Cleanup plan.

    Raises:
        CleanupError: If git state cannot be inspected.

    """
    resolved_repository_root = repository_root.resolve(strict=False)
    resolved_worktree_root = worktree_root.resolve(strict=False)
    git_worktrees = read_git_worktrees(resolved_repository_root)
    local_branches = read_git_branches(resolved_repository_root, "refs/heads/symphony", BranchKind.LOCAL)
    remote_branches = read_git_branches(resolved_repository_root, "refs/remotes/origin/symphony", BranchKind.REMOTE)
    issue_identifiers = issue_identifiers_for_lookup(git_worktrees, local_branches, remote_branches)
    linear_result = fetch_linear_issue_states(issue_identifiers, environment=environment)
    classified_worktrees = tuple(
        classify_git_worktree(
            git_worktree,
            resolved_worktree_root,
            linear_result.issue_states,
            deletion_readiness_for_plan(git_worktree, resolved_worktree_root),
            include_unknown=include_unknown,
        )
        for git_worktree in git_worktrees
    )
    classified_local_branches = tuple(
        classify_git_branch(git_branch, linear_result.issue_states, include_unknown=include_unknown)
        for git_branch in local_branches
    )
    classified_remote_branches = tuple(
        classify_git_branch(git_branch, linear_result.issue_states, include_unknown=include_unknown)
        for git_branch in remote_branches
    )
    return CleanupPlan(
        repository_root=resolved_repository_root,
        worktree_root=resolved_worktree_root,
        linear_result=linear_result,
        worktrees=classified_worktrees,
        local_branches=classified_local_branches,
        remote_branches=classified_remote_branches,
    )


def issue_identifiers_for_lookup(
    git_worktrees: collections.abc.Iterable[GitWorktreeRecord],
    local_branches: collections.abc.Iterable[GitBranchRecord],
    remote_branches: collections.abc.Iterable[GitBranchRecord],
) -> tuple[str, ...]:
    """Collect issue identifiers that need Linear state lookup.

    Args:
        git_worktrees: Git worktree records.
        local_branches: Local branch records.
        remote_branches: Remote branch records.

    Returns:
        Sorted issue identifiers.

    """
    issue_identifiers: set[str] = set()
    for git_worktree in git_worktrees:
        issue_identifier = issue_identifier_from_worktree(git_worktree)
        if issue_identifier is not None:
            issue_identifiers.add(issue_identifier)
    for git_branch in (*tuple(local_branches), *tuple(remote_branches)):
        issue_identifier = issue_identifier_from_branch(git_branch.name)
        if issue_identifier is not None:
            issue_identifiers.add(issue_identifier)
    return tuple(sorted(issue_identifiers))


def apply_cleanup_plan(
    cleanup_plan: CleanupPlan,
    *,
    delete_local_branches: bool = False,
    delete_remote_branches: bool = False,
) -> CleanupExecutionResult:
    """Apply a cleanup plan using non-forced git commands.

    Args:
        cleanup_plan: Plan to apply.
        delete_local_branches: Whether to delete selected local branches.
        delete_remote_branches: Whether to delete selected remote branches.

    Returns:
        Cleanup execution result.

    """
    command_outputs: list[tooling_commands.CommandOutput] = []
    for classified_worktree in cleanup_plan.worktrees:
        if classified_worktree.stale_candidate:
            command_outputs.append(
                tooling_commands.run_captured_command(
                    [
                        "git",
                        "-C",
                        str(cleanup_plan.repository_root),
                        "worktree",
                        "remove",
                        str(classified_worktree.git_worktree.path),
                    ]
                )
            )
    if delete_local_branches:
        for classified_branch in cleanup_plan.local_branches:
            if classified_branch.stale_candidate:
                command_outputs.append(
                    tooling_commands.run_captured_command(
                        [
                            "git",
                            "-C",
                            str(cleanup_plan.repository_root),
                            "branch",
                            "-d",
                            classified_branch.git_branch.name,
                        ]
                    )
                )
    if delete_remote_branches:
        for classified_branch in cleanup_plan.remote_branches:
            if classified_branch.stale_candidate:
                command_outputs.append(
                    tooling_commands.run_captured_command(
                        [
                            "git",
                            "-C",
                            str(cleanup_plan.repository_root),
                            "push",
                            "origin",
                            "--delete",
                            classified_branch.git_branch.name.removeprefix(ORIGIN_REMOTE_PREFIX),
                        ]
                    )
                )
    return CleanupExecutionResult(command_outputs=tuple(command_outputs))


def issue_state_label(issue_state: IssueState) -> str:
    """Format an issue state for reports.

    Args:
        issue_state: Issue state to format.

    Returns:
        Human-readable issue state label.

    """
    if issue_state.state_name is None:
        return f"unknown ({issue_state.category.value})"
    return f"{issue_state.state_name} ({issue_state.category.value})"


def render_cleanup_plan(
    cleanup_plan: CleanupPlan,
    *,
    apply_changes: bool,
    include_unknown: bool,
    delete_local_branches: bool,
    delete_remote_branches: bool,
) -> str:
    """Render a cleanup plan.

    Args:
        cleanup_plan: Plan to render.
        apply_changes: Whether destructive application was requested.
        include_unknown: Whether unknown issue states may be selected.
        delete_local_branches: Whether local branch deletion was requested.
        delete_remote_branches: Whether remote branch deletion was requested.

    Returns:
        Human-readable plan text.

    """
    linear_reachability = "reachable" if cleanup_plan.linear_result.reachable else "unreachable"
    local_branch_control = "will delete candidates" if apply_changes and delete_local_branches else "not deleting"
    remote_branch_control = "will delete candidates" if apply_changes and delete_remote_branches else "not deleting"
    lines = [
        f"Mode: {'apply' if apply_changes else 'dry-run'}",
        f"Repository: {cleanup_plan.repository_root}",
        f"Symphony worktree root: {cleanup_plan.worktree_root}",
        f"Linear: {linear_reachability} - {cleanup_plan.linear_result.detail}",
        f"Unknown issue selection: {'enabled' if include_unknown else 'disabled'}",
        "",
    ]
    render_worktree_section(lines, "Candidate worktrees", tuple(candidate_worktrees(cleanup_plan.worktrees)))
    render_branch_section(lines, "Candidate local branches", tuple(candidate_branches(cleanup_plan.local_branches)))
    render_branch_section(lines, "Candidate remote branches", tuple(candidate_branches(cleanup_plan.remote_branches)))
    retained_worktrees = tuple(retained_classified_worktrees(cleanup_plan.worktrees))
    retained_branches = tuple(
        retained_classified_branches((*cleanup_plan.local_branches, *cleanup_plan.remote_branches))
    )
    render_worktree_section(lines, "Retained or skipped worktrees", retained_worktrees)
    render_branch_section(lines, "Retained branches", retained_branches)
    lines.extend(
        [
            "Deletion controls:",
            f"- worktrees: {'will remove candidates' if apply_changes else 'dry-run only'}",
            f"- local branches: {local_branch_control}",
            f"- remote branches: {remote_branch_control}",
        ]
    )
    if not apply_changes:
        lines.append(
            "No changes were made. Re-run with --apply, or use just symphony-cleanup-apply, to remove candidates."
        )
    return "\n".join(lines) + "\n"


def candidate_worktrees(
    classified_worktrees: collections.abc.Iterable[ClassifiedWorktree],
) -> collections.abc.Iterator[ClassifiedWorktree]:
    """Yield selected worktree candidates."""
    for classified_worktree in classified_worktrees:
        if classified_worktree.stale_candidate:
            yield classified_worktree


def retained_classified_worktrees(
    classified_worktrees: collections.abc.Iterable[ClassifiedWorktree],
) -> collections.abc.Iterator[ClassifiedWorktree]:
    """Yield retained worktree records."""
    for classified_worktree in classified_worktrees:
        if not classified_worktree.stale_candidate:
            yield classified_worktree


def candidate_branches(
    classified_branches: collections.abc.Iterable[ClassifiedBranch],
) -> collections.abc.Iterator[ClassifiedBranch]:
    """Yield selected branch candidates."""
    for classified_branch in classified_branches:
        if classified_branch.stale_candidate:
            yield classified_branch


def retained_classified_branches(
    classified_branches: collections.abc.Iterable[ClassifiedBranch],
) -> collections.abc.Iterator[ClassifiedBranch]:
    """Yield retained branch records."""
    for classified_branch in classified_branches:
        if not classified_branch.stale_candidate:
            yield classified_branch


def render_worktree_section(
    lines: list[str],
    section_title: str,
    classified_worktrees: tuple[ClassifiedWorktree, ...],
) -> None:
    """Render a worktree section into an existing line list.

    Args:
        lines: Mutable report line list.
        section_title: Section title.
        classified_worktrees: Worktrees to render.

    """
    lines.append(f"{section_title}:")
    if not classified_worktrees:
        lines.append("- none")
        lines.append("")
        return
    for classified_worktree in classified_worktrees:
        issue_label = classified_worktree.issue_identifier or "unknown"
        line = (
            f"- {classified_worktree.git_worktree.path} "
            f"branch={classified_worktree.git_worktree.branch_name or 'detached'} "
            f"issue={issue_label} state={issue_state_label(classified_worktree.issue_state)}"
        )
        if classified_worktree.retain_reason:
            line = f"{line} reason={classified_worktree.retain_reason}"
        lines.append(line)
    lines.append("")


def render_branch_section(
    lines: list[str],
    section_title: str,
    classified_branches: tuple[ClassifiedBranch, ...],
) -> None:
    """Render a branch section into an existing line list.

    Args:
        lines: Mutable report line list.
        section_title: Section title.
        classified_branches: Branches to render.

    """
    lines.append(f"{section_title}:")
    if not classified_branches:
        lines.append("- none")
        lines.append("")
        return
    for classified_branch in classified_branches:
        issue_label = classified_branch.issue_identifier or "unknown"
        line = (
            f"- {classified_branch.git_branch.name} kind={classified_branch.git_branch.kind.value} "
            f"issue={issue_label} state={issue_state_label(classified_branch.issue_state)}"
        )
        if classified_branch.retain_reason:
            line = f"{line} reason={classified_branch.retain_reason}"
        lines.append(line)
    lines.append("")


def render_execution_result(execution_result: CleanupExecutionResult) -> str:
    """Render cleanup command execution output.

    Args:
        execution_result: Cleanup execution result.

    Returns:
        Human-readable execution summary.

    """
    lines = ["Execution result:"]
    if not execution_result.command_outputs:
        lines.append("- no candidate deletion commands were executed")
        return "\n".join(lines) + "\n"
    for command_output in execution_result.command_outputs:
        command_text = " ".join(command_output.command_arguments)
        lines.append(f"- rc={command_output.return_code} command={command_text}")
        if command_output.stdout.strip():
            lines.append(f"  stdout={command_output.stdout.strip()}")
        if command_output.stderr.strip():
            lines.append(f"  stderr={command_output.stderr.strip()}")
    return "\n".join(lines) + "\n"

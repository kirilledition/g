#!/usr/bin/env python3
"""Manage Codex task worktrees generated from docs/code-review.md."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import enum
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import typing
from pathlib import Path

JsonObject = dict[str, typing.Any]

REPO_RELATIVE_SOURCE_PATH = Path("docs/code-review.md")
REPO_RELATIVE_MANIFEST_PATH = Path("docs/code-review.tasks.json")
REPO_RELATIVE_STATE_DIRECTORY = Path(".codex-task-worktrees")
DEFAULT_WORKTREE_ROOT = "../g-worktrees"
DEFAULT_INTEGRATION_WORKTREE = "../g-worktrees/integration-code-review"
DEFAULT_INTEGRATION_BRANCH = "integration/code-review"
MANIFEST_VERSION = 1

PRESERVED_MANUAL_KEYS = {"dependencies", "status", "enabled", "assignee", "notes", "manual_expected_paths"}


class TaskStatus(enum.StrEnum):
    """Known lifecycle states for task farm records."""

    READY = "ready"
    RUNNING = "running"
    IMPLEMENTED = "implemented"
    REVIEWED = "reviewed"
    INTEGRATING = "integrating"
    MERGED = "merged"
    BLOCKED = "blocked"


@dataclasses.dataclass(frozen=True)
class ParsedTask:
    """Task parsed from the markdown source."""

    identifier: int
    title: str
    slug: str
    category: str
    source_start_line: int
    source_end_line: int
    body_markdown: str
    guidance_markdown: str
    expected_paths: list[str]


@dataclasses.dataclass(frozen=True)
class TaskSections:
    """Split markdown body sections for one task."""

    body_markdown: str
    guidance_markdown: str


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """Result of a completed command."""

    command_arguments: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclasses.dataclass(frozen=True)
class WorkerLaunch:
    """Detached or foreground worker process."""

    task_identifier: int
    process_identifier: int
    process: subprocess.Popen[str]


@dataclasses.dataclass(frozen=True)
class WorkerCompletion:
    """Classification of a finished worker."""

    status: TaskStatus
    final_message_exists: bool
    worktree_clean: bool
    branch_ahead: bool
    exit_code: int | None
    verification: str
    reason: str


@dataclasses.dataclass(frozen=True)
class DoctorCheck:
    """One doctor check result."""

    name: str
    passed: bool
    warning: bool
    message: str


def repository_root() -> Path:
    """Resolve the git repository root."""
    completed_process = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed_process.stdout.strip())


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    """Convert a task title to a stable slug."""
    without_inline_code = value.replace("`", "")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", without_inline_code.lower()).strip("-")
    return slug or "task"


def read_json_object(path: Path) -> JsonObject:
    """Read a JSON object from disk."""
    with path.open() as input_file:
        loaded_value = json.load(input_file)
    if not isinstance(loaded_value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return typing.cast("JsonObject", loaded_value)


def write_json_object(path: Path, value: JsonObject) -> None:
    """Write a JSON object with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def split_task_sections(task_markdown: str) -> TaskSections:
    """Split task prose from the guidance block."""
    marker = "\n**Guidance**\n"
    if marker not in task_markdown:
        stripped_markdown = strip_trailing_task_delimiter(task_markdown)
        return TaskSections(body_markdown=stripped_markdown, guidance_markdown="")
    body_markdown, guidance_markdown = task_markdown.split(marker, maxsplit=1)
    return TaskSections(
        body_markdown=strip_trailing_task_delimiter(body_markdown),
        guidance_markdown=strip_trailing_task_delimiter(guidance_markdown),
    )


def strip_trailing_task_delimiter(markdown_text: str) -> str:
    """Remove the horizontal rule used between tasks."""
    stripped_lines = markdown_text.strip().splitlines()
    while stripped_lines and stripped_lines[-1].strip() in {"---", ""}:
        stripped_lines.pop()
    return "\n".join(stripped_lines).strip()


def extract_expected_paths(task_markdown: str) -> list[str]:
    """Extract likely repository paths from inline-code references."""
    candidate_paths = re.findall(r"(?<!`)`([^`\n]+)`(?!`)", task_markdown)
    expected_paths: list[str] = []
    for candidate_path in candidate_paths:
        if " " in candidate_path:
            continue
        if candidate_path.startswith(("src/", "tests/", "docs/", "scripts/", "Cargo.", "pyproject.toml", "Justfile")):
            expected_paths.append(candidate_path)
    return sorted(set(expected_paths))


def parse_review_tasks(markdown_text: str) -> list[ParsedTask]:
    """Parse numbered review tasks from a markdown document."""
    lines = markdown_text.splitlines()
    category = "Uncategorized"
    inside_fenced_code = False
    current_task_identifier: int | None = None
    current_task_title = ""
    current_task_category = ""
    current_task_start_line = 0
    current_task_lines: list[str] = []
    tasks: list[ParsedTask] = []

    def flush_task(end_line: int) -> None:
        nonlocal current_task_identifier
        nonlocal current_task_title
        nonlocal current_task_category
        nonlocal current_task_start_line
        nonlocal current_task_lines
        if current_task_identifier is None:
            return
        task_markdown = "\n".join(current_task_lines).strip()
        sections = split_task_sections(task_markdown)
        tasks.append(
            ParsedTask(
                identifier=current_task_identifier,
                title=current_task_title,
                slug=slugify(current_task_title),
                category=current_task_category,
                source_start_line=current_task_start_line,
                source_end_line=end_line,
                body_markdown=sections.body_markdown,
                guidance_markdown=sections.guidance_markdown,
                expected_paths=extract_expected_paths(task_markdown),
            )
        )
        current_task_identifier = None
        current_task_title = ""
        current_task_category = ""
        current_task_start_line = 0
        current_task_lines = []

    for line_number, line in enumerate(lines, start=1):
        stripped_line = line.strip()
        if stripped_line.startswith("```"):
            inside_fenced_code = not inside_fenced_code

        if not inside_fenced_code and line.startswith("# "):
            flush_task(line_number - 1)
            category = line.removeprefix("# ").strip()
            continue

        task_heading_match = re.match(r"^## ([0-9]+)\. (.+)$", line)
        if not inside_fenced_code and task_heading_match is not None:
            flush_task(line_number - 1)
            current_task_identifier = int(task_heading_match.group(1))
            current_task_title = task_heading_match.group(2).strip()
            current_task_category = category
            current_task_start_line = line_number
            current_task_lines = [line]
            continue

        if current_task_identifier is not None:
            current_task_lines.append(line)

    flush_task(len(lines))
    return tasks


def default_manifest() -> JsonObject:
    """Create the default manifest shell."""
    return {
        "version": MANIFEST_VERSION,
        "source_path": REPO_RELATIVE_SOURCE_PATH.as_posix(),
        "defaults": {
            "base_branch": "main",
            "worktree_root": DEFAULT_WORKTREE_ROOT,
            "state_directory": REPO_RELATIVE_STATE_DIRECTORY.as_posix(),
            "worker_model": "gpt-5.5",
            "worker_reasoning_effort": "high",
            "reviewer_model": "gpt-5.5",
            "reviewer_reasoning_effort": "xhigh",
            "integrator_model": "gpt-5.5",
            "integrator_reasoning_effort": "xhigh",
            "worker_dangerously_bypass_approvals": False,
            "integrator_dangerously_bypass_approvals": False,
            "integration_worktree": DEFAULT_INTEGRATION_WORKTREE,
            "integration_branch": DEFAULT_INTEGRATION_BRANCH,
            "jobs": 5,
        },
        "tasks": [],
    }


def task_branch(parsed_task: ParsedTask) -> str:
    """Return the branch name for a task."""
    return f"codex/review-{parsed_task.identifier:02d}-{parsed_task.slug}"


def task_worktree(defaults: JsonObject, parsed_task: ParsedTask) -> str:
    """Return the manifest worktree path for a task."""
    worktree_root = str(defaults.get("worktree_root", DEFAULT_WORKTREE_ROOT))
    return f"{worktree_root}/review-{parsed_task.identifier:02d}-{parsed_task.slug}"


def build_task_record(defaults: JsonObject, parsed_task: ParsedTask, existing_task: JsonObject | None) -> JsonObject:
    """Build one task manifest record while preserving manual metadata."""
    task_record: JsonObject = {
        "id": parsed_task.identifier,
        "slug": parsed_task.slug,
        "title": parsed_task.title,
        "category": parsed_task.category,
        "source_start_line": parsed_task.source_start_line,
        "source_end_line": parsed_task.source_end_line,
        "body_markdown": parsed_task.body_markdown,
        "guidance_markdown": parsed_task.guidance_markdown,
        "kind": "implementation",
        "priority": parsed_task.category,
        "expected_paths": parsed_task.expected_paths,
        "dependencies": [],
        "status": TaskStatus.READY.value,
        "enabled": True,
        "branch": task_branch(parsed_task),
        "worktree": task_worktree(defaults, parsed_task),
    }
    if existing_task is None:
        return task_record
    for key in PRESERVED_MANUAL_KEYS:
        if key in existing_task:
            task_record[key] = existing_task[key]
    return task_record


def sync_manifest(repository_directory: Path) -> JsonObject:
    """Synchronize the task manifest from the markdown source."""
    source_path = repository_directory / REPO_RELATIVE_SOURCE_PATH
    manifest_path = repository_directory / REPO_RELATIVE_MANIFEST_PATH
    existing_manifest = read_json_object(manifest_path) if manifest_path.exists() else default_manifest()
    manifest = default_manifest()
    if isinstance(existing_manifest.get("defaults"), dict):
        manifest["defaults"].update(typing.cast("JsonObject", existing_manifest["defaults"]))
    defaults = typing.cast("JsonObject", manifest["defaults"])
    existing_tasks_by_identifier: dict[int, JsonObject] = {}
    for existing_task in existing_manifest.get("tasks", []):
        if isinstance(existing_task, dict) and isinstance(existing_task.get("id"), int):
            existing_task_identifier = typing.cast("int", existing_task["id"])
            existing_tasks_by_identifier[existing_task_identifier] = typing.cast("JsonObject", existing_task)

    parsed_tasks = parse_review_tasks(source_path.read_text())
    manifest["tasks"] = [
        build_task_record(defaults, parsed_task, existing_tasks_by_identifier.get(parsed_task.identifier))
        for parsed_task in parsed_tasks
    ]
    write_json_object(manifest_path, manifest)
    return manifest


def load_manifest(repository_directory: Path) -> JsonObject:
    """Load the task manifest."""
    manifest_path = repository_directory / REPO_RELATIVE_MANIFEST_PATH
    if not manifest_path.exists():
        return sync_manifest(repository_directory)
    return read_json_object(manifest_path)


def save_manifest(repository_directory: Path, manifest: JsonObject) -> None:
    """Save the task manifest."""
    write_json_object(repository_directory / REPO_RELATIVE_MANIFEST_PATH, manifest)


def state_directory(repository_directory: Path, manifest: JsonObject) -> Path:
    """Resolve the state directory for runtime logs."""
    defaults = typing.cast("JsonObject", manifest.get("defaults", {}))
    return repository_directory / str(defaults.get("state_directory", REPO_RELATIVE_STATE_DIRECTORY.as_posix()))


def resolve_manifest_path(repository_directory: Path, manifest_path: str) -> Path:
    """Resolve a path stored in the manifest."""
    path = Path(manifest_path)
    if path.is_absolute():
        return path
    return (repository_directory / path).resolve()


def selected_tasks(manifest: JsonObject, identifiers: list[int]) -> list[JsonObject]:
    """Return selected manifest task records."""
    tasks = [task for task in manifest.get("tasks", []) if isinstance(task, dict)]
    typed_tasks = [typing.cast("JsonObject", task) for task in tasks]
    if not identifiers:
        return typed_tasks
    selected_identifiers = set(identifiers)
    return [task for task in typed_tasks if task.get("id") in selected_identifiers]


def run_command(command_arguments: list[str], *, cwd: Path) -> CommandResult:
    """Run a command and capture output."""
    completed_process = subprocess.run(
        command_arguments,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return CommandResult(
        command_arguments=command_arguments,
        returncode=completed_process.returncode,
        stdout=completed_process.stdout,
        stderr=completed_process.stderr,
    )


def ensure_success(command_result: CommandResult) -> None:
    """Raise if a command failed."""
    if command_result.returncode == 0:
        return
    command_text = " ".join(command_result.command_arguments)
    error_message = command_result.stderr.strip() or command_result.stdout.strip()
    raise RuntimeError(f"Command failed ({command_result.returncode}): {command_text}\n{error_message}")


def git_branch_exists(repository_directory: Path, branch: str) -> bool:
    """Return whether a local branch exists."""
    command_result = run_command(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=repository_directory,
    )
    return command_result.returncode == 0


def ensure_task_worktree(repository_directory: Path, task: JsonObject, base_branch: str) -> Path:
    """Create a task worktree if it is missing."""
    branch = str(task["branch"])
    worktree_path = resolve_manifest_path(repository_directory, str(task["worktree"]))
    if worktree_path.exists():
        return worktree_path
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    if git_branch_exists(repository_directory, branch):
        command_arguments = ["git", "worktree", "add", str(worktree_path), branch]
    else:
        command_arguments = ["git", "worktree", "add", "-b", branch, str(worktree_path), base_branch]
    ensure_success(run_command(command_arguments, cwd=repository_directory))
    return worktree_path


def ensure_integration_worktree(repository_directory: Path, defaults: JsonObject) -> Path:
    """Create the integration worktree if it is missing."""
    base_branch = str(defaults.get("base_branch", "main"))
    integration_branch = str(defaults.get("integration_branch", DEFAULT_INTEGRATION_BRANCH))
    integration_worktree = str(defaults.get("integration_worktree", DEFAULT_INTEGRATION_WORKTREE))
    worktree_path = resolve_manifest_path(repository_directory, integration_worktree)
    if worktree_path.exists():
        return worktree_path
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    if git_branch_exists(repository_directory, integration_branch):
        command_arguments = ["git", "worktree", "add", str(worktree_path), integration_branch]
    else:
        command_arguments = ["git", "worktree", "add", "-b", integration_branch, str(worktree_path), base_branch]
    ensure_success(run_command(command_arguments, cwd=repository_directory))
    return worktree_path


def git_worktree_is_clean(worktree_path: Path) -> bool:
    """Return whether a worktree has no uncommitted changes."""
    if not worktree_path.exists():
        return False
    command_result = run_command(["git", "status", "--short"], cwd=worktree_path)
    return command_result.returncode == 0 and not command_result.stdout.strip()


def git_branch_ahead_count(worktree_path: Path, base_branch: str) -> int:
    """Return how many commits HEAD is ahead of the base branch."""
    command_result = run_command(["git", "rev-list", "--count", f"{base_branch}..HEAD"], cwd=worktree_path)
    if command_result.returncode != 0:
        return 0
    stripped_stdout = command_result.stdout.strip()
    if not stripped_stdout:
        return 0
    return int(stripped_stdout)


def read_worker_exit_code(exit_code_path: Path) -> int | None:
    """Read a worker wrapper exit code if present."""
    if not exit_code_path.exists():
        return None
    stripped_text = exit_code_path.read_text().strip()
    if not stripped_text:
        return None
    return int(stripped_text)


def classify_worker_completion(
    repository_directory: Path,
    manifest: JsonObject,
    task: JsonObject,
) -> WorkerCompletion:
    """Classify a finished worker from final output, git state, and wrapper exit code."""
    defaults = typing.cast("JsonObject", manifest["defaults"])
    task_identifier = int(task["id"])
    run_directory = state_directory(repository_directory, manifest) / "runs" / f"{task_identifier:02d}"
    final_message_exists = (run_directory / "worker-final.md").exists()
    worktree_path = resolve_manifest_path(repository_directory, str(task["worktree"]))
    worktree_clean = git_worktree_is_clean(worktree_path)
    branch_ahead = git_branch_ahead_count(worktree_path, str(defaults.get("base_branch", "main"))) > 0
    exit_code_path = run_directory / "exit-code.txt"
    try:
        exit_code = read_worker_exit_code(exit_code_path)
    except ValueError:
        return WorkerCompletion(
            status=TaskStatus.BLOCKED,
            final_message_exists=final_message_exists,
            worktree_clean=worktree_clean,
            branch_ahead=branch_ahead,
            exit_code=None,
            verification="invalid-exit-code",
            reason="worker exit-code.txt is not an integer",
        )
    if exit_code is not None and exit_code != 0:
        return WorkerCompletion(
            status=TaskStatus.BLOCKED,
            final_message_exists=final_message_exists,
            worktree_clean=worktree_clean,
            branch_ahead=branch_ahead,
            exit_code=exit_code,
            verification="failed",
            reason=f"worker exited with {exit_code}",
        )
    if not final_message_exists:
        return WorkerCompletion(
            status=TaskStatus.BLOCKED,
            final_message_exists=False,
            worktree_clean=worktree_clean,
            branch_ahead=branch_ahead,
            exit_code=exit_code,
            verification="missing-final",
            reason="worker final message is missing",
        )
    if not worktree_clean:
        return WorkerCompletion(
            status=TaskStatus.BLOCKED,
            final_message_exists=True,
            worktree_clean=False,
            branch_ahead=branch_ahead,
            exit_code=exit_code,
            verification="dirty-worktree",
            reason="task worktree has uncommitted changes",
        )
    if not branch_ahead:
        return WorkerCompletion(
            status=TaskStatus.BLOCKED,
            final_message_exists=True,
            worktree_clean=True,
            branch_ahead=False,
            exit_code=exit_code,
            verification="no-task-commit",
            reason="task branch has no commits ahead of base",
        )
    if exit_code is None:
        return WorkerCompletion(
            status=TaskStatus.IMPLEMENTED,
            final_message_exists=True,
            worktree_clean=True,
            branch_ahead=True,
            exit_code=None,
            verification="legacy-unverified",
            reason="legacy run has no wrapper exit-code.txt",
        )
    return WorkerCompletion(
        status=TaskStatus.IMPLEMENTED,
        final_message_exists=True,
        worktree_clean=True,
        branch_ahead=True,
        exit_code=exit_code,
        verification="verified",
        reason="worker completed cleanly",
    )


def write_worker_wrapper(
    wrapper_path: Path,
    prompt_path: Path,
    exit_code_path: Path,
    command_arguments: list[str],
) -> None:
    """Write the per-task worker wrapper."""
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set +e",
            f"{shlex.join(command_arguments)} < {shlex.quote(str(prompt_path))}",
            "worker_exit_code=$?",
            f"printf '%s\\n' \"${{worker_exit_code}}\" > {shlex.quote(str(exit_code_path))}",
            "exit \"${worker_exit_code}\"",
            "",
        ]
    )
    wrapper_path.write_text(script)
    wrapper_path.chmod(0o755)


def build_worker_prompt(task: JsonObject) -> str:
    """Build the implementation prompt for a worker agent."""
    return f"""You are a Codex implementation worker in a dedicated git worktree.

Read AGENTS.md, docs/STYLEGUIDE.md, and Justfile before editing code.
If the worktree already contains edits, inspect them and continue from them unless they are clearly wrong.
Implement exactly this task and keep the change narrow.
Commit logical intermediate steps and leave a clean worktree when done.
Run relevant tests through `nix develop --command just ...` when feasible.
Never commit files under data/ or .codex-task-worktrees/.

Task {task["id"]}: {task["title"]}
Category: {task["category"]}
Expected paths: {", ".join(typing.cast("list[str]", task.get("expected_paths", []))) or "not precomputed"}

{task["body_markdown"]}

Guidance:
{task.get("guidance_markdown", "")}

Final response must include changed files, commits created, tests run, and any remaining blockers.
"""


def build_review_prompt(task: JsonObject) -> str:
    """Build the review prompt for a task branch."""
    return f"""Review task branch {task["branch"]} against main.

Task {task["id"]}: {task["title"]}

Review for correctness, behavioral regressions, styleguide compliance, missing tests, and whether the implementation
actually satisfies docs/code-review.md.
Lead with findings ordered by severity. If there are no findings, say that explicitly and mention remaining risk.
"""


def build_integration_prompt(task: JsonObject, review_report_path: Path | None, integration_branch: str) -> str:
    """Build the integration prompt for the main agent."""
    review_instruction = (
        f"Read the review report at {review_report_path} before merging."
        if review_report_path is not None and review_report_path.exists()
        else "Run your own review before merging because no review report is available."
    )
    return f"""You are the main integration agent for this repository.

Integrate task branch {task["branch"]} into the integration branch {integration_branch}.

Task {task["id"]}: {task["title"]}
{review_instruction}

Requirements:
- Start by checking that the integration worktree is clean and on {integration_branch}.
- Inspect the task commits and diff against the base branch.
- Resolve merge conflicts if they occur.
- Fix concrete review findings before committing.
- Run the narrow relevant tests first, then the broadest feasible project check.
- Commit the merge with a clear message if the result is acceptable.
- If the branch is not acceptable or conflicts cannot be resolved safely, abort the merge and report the blocker.

Do not merge unrelated branches. Do not touch data/. Do not revert user changes unrelated to this task.
"""


def build_worker_command(
    worktree_path: Path,
    model: str,
    reasoning_effort: str,
    final_message_path: Path,
    *,
    dangerously_bypass_approvals: bool,
) -> list[str]:
    """Build the worker Codex command."""
    command_arguments = [
        "codex",
        "--cd",
        str(worktree_path),
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "exec",
        "--json",
        "-o",
        str(final_message_path),
        "-",
    ]
    if dangerously_bypass_approvals:
        command_arguments.insert(7, "--dangerously-bypass-approvals-and-sandbox")
    return command_arguments


def build_review_command(
    worktree_path: Path,
    model: str,
    reasoning_effort: str,
    final_message_path: Path,
) -> list[str]:
    """Build the Codex review command."""
    return [
        "codex",
        "--cd",
        str(worktree_path),
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "never",
        "exec",
        "--json",
        "-o",
        str(final_message_path),
        "-",
    ]


def build_integration_command(
    integration_worktree_path: Path,
    worktree_path: Path,
    model: str,
    reasoning_effort: str,
    final_message_path: Path,
    *,
    dangerously_bypass_approvals: bool,
) -> list[str]:
    """Build the integration Codex command."""
    command_arguments = [
        "codex",
        "--cd",
        str(integration_worktree_path),
        "--add-dir",
        str(worktree_path),
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "exec",
        "--json",
        "-o",
        str(final_message_path),
        "-",
    ]
    if dangerously_bypass_approvals:
        command_arguments.insert(9, "--dangerously-bypass-approvals-and-sandbox")
    return command_arguments


def running_process_exists(process_identifier: int) -> bool:
    """Return whether a process id appears to be alive."""
    try:
        os.kill(process_identifier, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_state(state_path: Path) -> JsonObject:
    """Load runtime state."""
    if not state_path.exists():
        return {"runs": {}}
    return read_json_object(state_path)


def save_state(state_path: Path, state: JsonObject) -> None:
    """Save runtime state."""
    write_json_object(state_path, state)


def runtime_task_status(state: JsonObject, task: JsonObject) -> str:
    """Return the current runtime status for a task."""
    statuses = typing.cast("JsonObject", state.get("statuses", {}))
    task_identifier = str(task["id"])
    status = statuses.get(task_identifier, task.get("status", TaskStatus.READY.value))
    return str(status)


def set_runtime_task_status(state: JsonObject, task: JsonObject, status: TaskStatus) -> None:
    """Set a task status in ignored runtime state."""
    statuses = typing.cast("JsonObject", state.setdefault("statuses", {}))
    statuses[str(task["id"])] = status.value
    status_updates = typing.cast("JsonObject", state.setdefault("status_updated_at", {}))
    status_updates[str(task["id"])] = utc_timestamp()


def record_worker_completion(state: JsonObject, task: JsonObject, completion: WorkerCompletion) -> None:
    """Record worker completion details in runtime state."""
    set_runtime_task_status(state, task, completion.status)
    worker_results = typing.cast("JsonObject", state.setdefault("worker_results", {}))
    worker_results[str(task["id"])] = {
        "branch_ahead": completion.branch_ahead,
        "exit_code": completion.exit_code,
        "final_message_exists": completion.final_message_exists,
        "reason": completion.reason,
        "verification": completion.verification,
        "worktree_clean": completion.worktree_clean,
    }


def tasks_by_identifier(manifest: JsonObject) -> dict[int, JsonObject]:
    """Return manifest tasks keyed by task id."""
    return {int(task["id"]): task for task in selected_tasks(manifest, [])}


def dependency_identifiers(task: JsonObject) -> list[int]:
    """Return normalized dependency task identifiers."""
    dependencies = task.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ValueError(f"Task {task['id']} dependencies must be a list.")
    identifiers: list[int] = []
    for dependency in dependencies:
        if not isinstance(dependency, int):
            raise ValueError(f"Task {task['id']} has non-integer dependency {dependency!r}.")
        identifiers.append(dependency)
    return identifiers


def ensure_dependencies_ready(
    manifest: JsonObject,
    state: JsonObject,
    tasks: list[JsonObject],
    allowed_statuses: set[str],
    operation: str,
) -> None:
    """Raise if selected tasks have dependencies that do not satisfy an operation."""
    indexed_tasks = tasks_by_identifier(manifest)
    for task in tasks:
        for dependency_identifier in dependency_identifiers(task):
            dependency_task = indexed_tasks.get(dependency_identifier)
            if dependency_task is None:
                raise ValueError(
                    f"Task {task['id']} cannot {operation}: dependency {dependency_identifier} is missing.",
                )
            if not bool(dependency_task.get("enabled", True)):
                continue
            dependency_status = runtime_task_status(state, dependency_task)
            if dependency_status not in allowed_statuses:
                raise ValueError(
                    f"Task {task['id']} cannot {operation}: dependency {dependency_identifier} is "
                    f"{dependency_status}, expected one of {sorted(allowed_statuses)}.",
                )


def refresh_runtime_statuses(repository_directory: Path, manifest: JsonObject) -> None:
    """Refresh task statuses from detached worker state."""
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    runs = typing.cast("JsonObject", state.get("runs", {}))
    changed = False
    for task in selected_tasks(manifest, []):
        task_identifier = int(task["id"])
        run = runs.get(str(task_identifier), {})
        if not isinstance(run, dict):
            continue
        pid = run.get("pid")
        if not isinstance(pid, int):
            continue
        if running_process_exists(pid):
            if runtime_task_status(state, task) == TaskStatus.READY.value:
                set_runtime_task_status(state, task, TaskStatus.RUNNING)
                changed = True
            continue
        if runtime_task_status(state, task) != TaskStatus.RUNNING.value:
            continue
        run_directory = state_directory(repository_directory, manifest) / "runs" / f"{task_identifier:02d}"
        if not run_directory.exists():
            set_runtime_task_status(state, task, TaskStatus.BLOCKED)
            changed = True
            continue
        completion = classify_worker_completion(repository_directory, manifest, task)
        record_worker_completion(state, task, completion)
        changed = True
    if changed:
        save_state(state_path, state)


def launch_worker(
    *,
    repository_directory: Path,
    manifest: JsonObject,
    task: JsonObject,
    dangerously_bypass_approvals: bool,
) -> WorkerLaunch:
    """Launch one Codex worker."""
    defaults = typing.cast("JsonObject", manifest["defaults"])
    base_branch = str(defaults.get("base_branch", "main"))
    worktree_path = ensure_task_worktree(repository_directory, task, base_branch)
    task_identifier = int(task["id"])
    run_directory = state_directory(repository_directory, manifest) / "runs" / f"{task_identifier:02d}"
    run_directory.mkdir(parents=True, exist_ok=True)
    final_message_path = run_directory / "worker-final.md"
    jsonl_log_path = run_directory / "worker.jsonl"
    stderr_log_path = run_directory / "worker.stderr.log"
    prompt_path = run_directory / "worker-prompt.md"
    wrapper_path = run_directory / "worker-wrapper.sh"
    exit_code_path = run_directory / "exit-code.txt"
    prompt = build_worker_prompt(task)
    prompt_path.write_text(prompt)
    command_arguments = build_worker_command(
        worktree_path=worktree_path,
        model=str(defaults.get("worker_model", "gpt-5.5")),
        reasoning_effort=str(defaults.get("worker_reasoning_effort", "high")),
        final_message_path=final_message_path,
        dangerously_bypass_approvals=dangerously_bypass_approvals,
    )
    write_worker_wrapper(wrapper_path, prompt_path, exit_code_path, command_arguments)
    with jsonl_log_path.open("w") as stdout_log, stderr_log_path.open("w") as stderr_log:
        process = subprocess.Popen(
            ["bash", str(wrapper_path)],
            cwd=repository_directory,
            stdout=stdout_log,
            stderr=stderr_log,
            text=True,
            start_new_session=True,
        )
        return WorkerLaunch(
            task_identifier=task_identifier,
            process_identifier=process.pid,
            process=process,
        )


def make_doctor_check(name: str, *, passed: bool, message: str, warning: bool = False) -> DoctorCheck:
    """Create a doctor check."""
    return DoctorCheck(name=name, passed=passed, warning=warning, message=message)


def doctor_command_check(command_name: str, *, warning: bool = False) -> DoctorCheck:
    """Check that a command exists on PATH."""
    if shutil.which(command_name) is not None:
        return make_doctor_check(command_name, passed=True, message=f"{command_name} found", warning=False)
    return make_doctor_check(command_name, passed=False, message=f"missing command: {command_name}", warning=warning)


def doctor_file_check(repository_directory: Path, relative_path: Path) -> DoctorCheck:
    """Check that a required repository file exists."""
    path = repository_directory / relative_path
    return make_doctor_check(
        relative_path.as_posix(),
        passed=path.exists(),
        message=f"{relative_path.as_posix()} exists",
    )


def collect_doctor_checks(repository_directory: Path, manifest: JsonObject, *, strict: bool) -> list[DoctorCheck]:
    """Collect task-farm environment checks."""
    defaults = typing.cast("JsonObject", manifest["defaults"])
    base_branch = str(defaults.get("base_branch", "main"))
    checks = [
        doctor_command_check("git"),
        doctor_command_check("codex"),
        doctor_command_check("nix", warning=not strict),
        make_doctor_check(
            "repo root",
            passed=(repository_directory / ".git").exists(),
            message=f"repo root: {repository_directory}",
        ),
        doctor_file_check(repository_directory, REPO_RELATIVE_SOURCE_PATH),
        doctor_file_check(repository_directory, Path("Justfile")),
        doctor_file_check(repository_directory, Path("AGENTS.md")),
        doctor_file_check(repository_directory, Path("docs/STYLEGUIDE.md")),
    ]
    base_branch_exists = git_branch_exists(repository_directory, base_branch)
    checks.append(
        make_doctor_check(
            "base branch",
            passed=base_branch_exists,
            message=f"base branch {base_branch} exists",
        )
    )
    current_branch_result = run_command(["git", "branch", "--show-current"], cwd=repository_directory)
    current_branch = current_branch_result.stdout.strip()
    clean_result = run_command(["git", "status", "--short"], cwd=repository_directory)
    main_checkout_clean = (
        current_branch_result.returncode == 0
        and clean_result.returncode == 0
        and current_branch == base_branch
        and not clean_result.stdout.strip()
    )
    checks.append(
        make_doctor_check(
            "clean main checkout",
            passed=main_checkout_clean,
            message=f"current branch={current_branch or '<detached>'}, dirty={bool(clean_result.stdout.strip())}",
        )
    )
    return checks


def command_doctor(arguments: argparse.Namespace) -> int:
    """Handle doctor."""
    repository_directory = repository_root()
    manifest = load_manifest(repository_directory)
    checks = collect_doctor_checks(repository_directory, manifest, strict=bool(arguments.strict))
    exit_code = 0
    for check in checks:
        if check.passed:
            prefix = "ok"
        elif check.warning:
            prefix = "warning"
        else:
            prefix = "fail"
            exit_code = 1
        print(f"{prefix}: {check.name}: {check.message}")
    return exit_code


def command_sync_manifest(arguments: argparse.Namespace) -> int:
    """Handle sync-manifest."""
    repository_directory = repository_root()
    manifest = sync_manifest(repository_directory)
    print(f"Synced {len(manifest.get('tasks', []))} tasks into {REPO_RELATIVE_MANIFEST_PATH}.")
    return 0


def command_list(arguments: argparse.Namespace) -> int:
    """Handle list."""
    repository_directory = repository_root()
    manifest = load_manifest(repository_directory)
    refresh_runtime_statuses(repository_directory, manifest)
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    for task in selected_tasks(manifest, typing.cast("list[int]", arguments.task)):
        print(f"{int(task['id']):02d}  {runtime_task_status(state, task):12}  {task['branch']}  {task['title']}")
    return 0


def command_run(arguments: argparse.Namespace) -> int:
    """Handle run."""
    repository_directory = repository_root()
    manifest = load_manifest(repository_directory)
    refresh_runtime_statuses(repository_directory, manifest)
    defaults = typing.cast("JsonObject", manifest["defaults"])
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    jobs = int(arguments.jobs if arguments.jobs is not None else defaults.get("jobs", 5))
    wait_for_completion = bool(arguments.wait)
    force = bool(arguments.force)
    dangerously_bypass_approvals = bool(
        arguments.dangerous or defaults.get("worker_dangerously_bypass_approvals", False),
    )
    candidates = selected_tasks(manifest, typing.cast("list[int]", arguments.task))
    runnable_tasks: list[JsonObject] = []
    for task in candidates:
        if len(runnable_tasks) >= jobs:
            break
        if not bool(task.get("enabled", True)):
            continue
        runnable_statuses = {TaskStatus.READY.value}
        if force or runtime_task_status(state, task) in runnable_statuses:
            runnable_tasks.append(task)
    if not runnable_tasks:
        print("No runnable tasks selected.")
        return 0
    ensure_dependencies_ready(
        manifest,
        state,
        runnable_tasks,
        {TaskStatus.MERGED.value},
        "run",
    )

    runs = typing.cast("JsonObject", state.setdefault("runs", {}))
    exit_code = 0
    launched_workers: list[WorkerLaunch] = []
    for task in runnable_tasks:
        set_runtime_task_status(state, task, TaskStatus.RUNNING)
        save_state(state_path, state)
        launched_worker = launch_worker(
            repository_directory=repository_directory,
            manifest=manifest,
            task=task,
            dangerously_bypass_approvals=dangerously_bypass_approvals,
        )
        task_identifier = int(task["id"])
        runs[str(task_identifier)] = {
            "pid": launched_worker.process_identifier,
            "returncode": None,
            "started_at": utc_timestamp(),
            "branch": task["branch"],
            "worktree": task["worktree"],
        }
        launched_workers.append(launched_worker)
        print(f"Launched task {task_identifier:02d}: {task['branch']}")
        save_state(state_path, state)
    if wait_for_completion:
        for launched_worker in launched_workers:
            returncode = launched_worker.process.wait()
            task = next(task for task in runnable_tasks if int(task["id"]) == launched_worker.task_identifier)
            run = runs.get(str(launched_worker.task_identifier), {})
            if isinstance(run, dict):
                run["returncode"] = returncode
                run["finished_at"] = utc_timestamp()
            completion = classify_worker_completion(repository_directory, manifest, task)
            record_worker_completion(state, task, completion)
            if completion.status == TaskStatus.BLOCKED and exit_code == 0:
                exit_code = returncode or 1
            save_state(state_path, state)
    save_state(state_path, state)
    return exit_code


def command_status(arguments: argparse.Namespace) -> int:
    """Handle status."""
    repository_directory = repository_root()
    manifest = load_manifest(repository_directory)
    refresh_runtime_statuses(repository_directory, manifest)
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    runs = typing.cast("JsonObject", state.get("runs", {}))
    for task in selected_tasks(manifest, typing.cast("list[int]", arguments.task)):
        task_identifier = int(task["id"])
        run = runs.get(str(task_identifier), {})
        pid = run.get("pid") if isinstance(run, dict) else None
        alive = isinstance(pid, int) and running_process_exists(pid)
        run_directory = state_directory(repository_directory, manifest) / "runs" / f"{task_identifier:02d}"
        final_message_exists = (run_directory / "worker-final.md").exists()
        worktree_path = resolve_manifest_path(repository_directory, str(task["worktree"]))
        worktree_state = "missing"
        if worktree_path.exists():
            git_status = run_command(["git", "status", "--short"], cwd=worktree_path)
            worktree_state = "dirty" if git_status.stdout.strip() else "clean"
        print(
            f"{task_identifier:02d}  status={runtime_task_status(state, task)}  alive={alive}"
            f"  final={final_message_exists}  worktree={worktree_state}  {task['title']}"
        )
    return 0


def command_review(arguments: argparse.Namespace) -> int:
    """Handle review."""
    repository_directory = repository_root()
    manifest = load_manifest(repository_directory)
    refresh_runtime_statuses(repository_directory, manifest)
    defaults = typing.cast("JsonObject", manifest["defaults"])
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    tasks = selected_tasks(manifest, typing.cast("list[int]", arguments.task))
    if not tasks:
        raise ValueError("Select at least one task to review.")
    ensure_dependencies_ready(
        manifest,
        state,
        tasks,
        {
            TaskStatus.IMPLEMENTED.value,
            TaskStatus.REVIEWED.value,
            TaskStatus.INTEGRATING.value,
            TaskStatus.MERGED.value,
        },
        "review",
    )
    exit_code = 0
    for task in tasks:
        worktree_path = resolve_manifest_path(repository_directory, str(task["worktree"]))
        if not worktree_path.exists():
            raise FileNotFoundError(f"Worktree does not exist: {worktree_path}")
        review_directory = state_directory(repository_directory, manifest) / "reviews"
        review_directory.mkdir(parents=True, exist_ok=True)
        task_identifier = int(task["id"])
        final_message_path = review_directory / f"{task_identifier:02d}.md"
        jsonl_log_path = review_directory / f"{task_identifier:02d}.jsonl"
        command_arguments = build_review_command(
            worktree_path=worktree_path,
            model=str(defaults.get("reviewer_model", "gpt-5.5")),
            reasoning_effort=str(defaults.get("reviewer_reasoning_effort", "xhigh")),
            final_message_path=final_message_path,
        )
        completed_process = subprocess.run(
            command_arguments,
            cwd=repository_directory,
            input=build_review_prompt(task),
            check=False,
            capture_output=True,
            text=True,
        )
        jsonl_log_path.write_text(completed_process.stdout)
        if completed_process.stderr:
            (review_directory / f"{task_identifier:02d}.stderr.log").write_text(completed_process.stderr)
        if completed_process.returncode == 0:
            set_runtime_task_status(state, task, TaskStatus.REVIEWED)
            print(f"Reviewed task {task_identifier:02d}: {final_message_path}")
        else:
            exit_code = completed_process.returncode
            print(f"Review failed for task {task_identifier:02d}; see {jsonl_log_path}.", file=sys.stderr)
        save_state(state_path, state)
    return exit_code


def command_integrate(arguments: argparse.Namespace) -> int:
    """Handle integrate."""
    repository_directory = repository_root()
    manifest = load_manifest(repository_directory)
    refresh_runtime_statuses(repository_directory, manifest)
    defaults = typing.cast("JsonObject", manifest["defaults"])
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    tasks = selected_tasks(manifest, typing.cast("list[int]", arguments.task))
    if not tasks:
        raise ValueError("Select at least one task to integrate.")
    ensure_dependencies_ready(
        manifest,
        state,
        tasks,
        {TaskStatus.MERGED.value},
        "integrate",
    )
    dangerously_bypass_approvals = bool(
        arguments.dangerous or defaults.get("integrator_dangerously_bypass_approvals", False),
    )
    integration_worktree_path = ensure_integration_worktree(repository_directory, defaults)
    integration_branch = str(defaults.get("integration_branch", DEFAULT_INTEGRATION_BRANCH))
    exit_code = 0
    for task in tasks:
        worktree_path = resolve_manifest_path(repository_directory, str(task["worktree"]))
        if not worktree_path.exists():
            raise FileNotFoundError(f"Worktree does not exist: {worktree_path}")
        integration_directory = state_directory(repository_directory, manifest) / "integrations"
        integration_directory.mkdir(parents=True, exist_ok=True)
        task_identifier = int(task["id"])
        final_message_path = integration_directory / f"{task_identifier:02d}-final.md"
        jsonl_log_path = integration_directory / f"{task_identifier:02d}.jsonl"
        review_report_path = state_directory(repository_directory, manifest) / "reviews" / f"{task_identifier:02d}.md"
        command_arguments = build_integration_command(
            integration_worktree_path=integration_worktree_path,
            worktree_path=worktree_path,
            model=str(defaults.get("integrator_model", "gpt-5.5")),
            reasoning_effort=str(defaults.get("integrator_reasoning_effort", "xhigh")),
            final_message_path=final_message_path,
            dangerously_bypass_approvals=dangerously_bypass_approvals,
        )
        set_runtime_task_status(state, task, TaskStatus.INTEGRATING)
        save_state(state_path, state)
        completed_process = subprocess.run(
            command_arguments,
            cwd=integration_worktree_path,
            input=build_integration_prompt(task, review_report_path, integration_branch),
            check=False,
            capture_output=True,
            text=True,
        )
        jsonl_log_path.write_text(completed_process.stdout)
        if completed_process.stderr:
            (integration_directory / f"{task_identifier:02d}.stderr.log").write_text(completed_process.stderr)
        if completed_process.returncode == 0:
            set_runtime_task_status(state, task, TaskStatus.MERGED)
            print(f"Integrated task {task_identifier:02d}: {final_message_path}")
        else:
            set_runtime_task_status(state, task, TaskStatus.BLOCKED)
            exit_code = completed_process.returncode
            print(f"Integration failed for task {task_identifier:02d}; see {jsonl_log_path}.", file=sys.stderr)
            break
        save_state(state_path, state)
    return exit_code


def command_integrate_ready(arguments: argparse.Namespace) -> int:
    """Handle integrate-ready."""
    repository_directory = repository_root()
    manifest = load_manifest(repository_directory)
    refresh_runtime_statuses(repository_directory, manifest)
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    allowed_statuses = {TaskStatus.REVIEWED.value}
    if bool(arguments.allow_unreviewed):
        allowed_statuses.add(TaskStatus.IMPLEMENTED.value)
    ready_identifiers = [
        int(task["id"])
        for task in selected_tasks(manifest, [])
        if runtime_task_status(state, task) in allowed_statuses
    ]
    if not ready_identifiers:
        print("No reviewed tasks are ready to integrate.")
        return 0
    arguments.task = ready_identifiers
    return command_integrate(arguments)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Manage Codex task worktrees generated from docs/code-review.md.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync-manifest", help="Regenerate docs/code-review.tasks.json.")
    sync_parser.set_defaults(handler=command_sync_manifest)

    doctor_parser = subparsers.add_parser("doctor", help="Check task-farm prerequisites.")
    doctor_parser.add_argument("--strict", action="store_true", help="Treat missing optional tools as failures.")
    doctor_parser.set_defaults(handler=command_doctor)

    list_parser = subparsers.add_parser("list", help="List tasks from the manifest.")
    list_parser.add_argument("--task", action="append", type=int, default=[], help="Task id to list.")
    list_parser.set_defaults(handler=command_list)

    run_parser = subparsers.add_parser("run", help="Launch worker agents.")
    run_parser.add_argument("--task", action="append", type=int, default=[], help="Task id to run.")
    run_parser.add_argument("--jobs", type=int, help="Maximum number of tasks to launch.")
    run_parser.add_argument("--wait", action="store_true", help="Wait for launched workers to finish.")
    run_parser.add_argument("--force", action="store_true", help="Launch tasks regardless of recorded status.")
    run_parser.add_argument("--dangerous", action="store_true", help="Allow workers to bypass approvals and sandbox.")
    run_parser.set_defaults(handler=command_run)

    status_parser = subparsers.add_parser("status", help="Show worker and worktree status.")
    status_parser.add_argument("--task", action="append", type=int, default=[], help="Task id to inspect.")
    status_parser.set_defaults(handler=command_status)

    review_parser = subparsers.add_parser("review", help="Run Codex review for task branches.")
    review_parser.add_argument("task", nargs="+", type=int, help="Task id to review.")
    review_parser.set_defaults(handler=command_review)

    integrate_parser = subparsers.add_parser("integrate", help="Run the main integration agent.")
    integrate_parser.add_argument("task", nargs="+", type=int, help="Task id to integrate.")
    integrate_parser.add_argument(
        "--dangerous",
        action="store_true",
        help="Allow integrator to bypass approvals and sandbox.",
    )
    integrate_parser.set_defaults(handler=command_integrate)

    integrate_ready_parser = subparsers.add_parser("integrate-ready", help="Integrate implemented tasks in order.")
    integrate_ready_parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Allow implemented tasks without reviewed status.",
    )
    integrate_ready_parser.add_argument(
        "--dangerous",
        action="store_true",
        help="Allow integrator to bypass approvals and sandbox.",
    )
    integrate_ready_parser.set_defaults(handler=command_integrate_ready)
    return parser


def main() -> int:
    """Run the CLI."""
    parser = build_argument_parser()
    arguments = parser.parse_args()
    handler = typing.cast("typing.Callable[[argparse.Namespace], int]", arguments.handler)
    try:
        return handler(arguments)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return int(signal.SIGINT)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

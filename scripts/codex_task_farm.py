#!/usr/bin/env python3
"""Manage Codex task worktrees generated from review task manifests."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import enum
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import typing
from pathlib import Path

JsonObject = dict[str, typing.Any]

REPO_RELATIVE_SOURCE_PATH = Path("docs/code-review.md")
REPO_RELATIVE_MANIFEST_PATH = Path("docs/code-review.tasks.json")
REPO_RELATIVE_STATE_DIRECTORY = Path(".codex-task-worktrees")
DEFAULT_WORKTREE_ROOT = "../g-worktrees"
DEFAULT_INTEGRATION_WORKTREE = "../g-worktrees/integration-code-review"
DEFAULT_INTEGRATION_BRANCH = "integration/code-review"
DEFAULT_BRANCH_PREFIX = "codex/review-"
DEFAULT_WORKTREE_PREFIX = "../g-worktrees/review-"
REVIEW2_SOURCE_PATH = Path("docs/02.code-review-2-06-26.md")
REVIEW2_MANIFEST_PATH = Path("docs/code-review-2.tasks.json")
REVIEW2_PLAN_PATH = Path("docs/code-review-2-plan.md")
REVIEW2_STATE_DIRECTORY = Path(".codex-task-worktrees/code-review-2")
REVIEW2_BRANCH_PREFIX = "codex/review2-"
REVIEW2_WORKTREE_PREFIX = "../g-worktrees/review2-"
REVIEW2_INTEGRATION_WORKTREE = "../g-worktrees/integration-code-review-2"
REVIEW2_INTEGRATION_BRANCH = "integration/code-review-2"
MANIFEST_VERSION = 2
DEFAULT_LEASE_SECONDS = 4 * 60 * 60

PRESERVED_MANUAL_KEYS = {
    "assignee",
    "dependencies",
    "enabled",
    "logs",
    "manual",
    "manual_expected_paths",
    "notes",
    "runtime",
    "status",
}
PATH_WITH_LINE_SUFFIX_PATTERN = re.compile(r"^(?P<path>.+?):[0-9]+(?:-[0-9]+)?$")
STRING_TASK_IDENTIFIER_PATTERN = re.compile(r"^T(?P<number>[0-9]+)$")


class TaskStatus(enum.StrEnum):
    """Known lifecycle states for task farm records."""

    READY = "ready"
    CLAIMED = "claimed"
    RUNNING = "running"
    IMPLEMENTED = "implemented"
    REVIEWING = "reviewing"
    NEEDS_CHANGES = "needs_changes"
    REVIEWED = "reviewed"
    INTEGRATING = "integrating"
    INTEGRATED = "integrated"
    MERGED = "merged"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"


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


@dataclasses.dataclass(frozen=True)
class StatusRow:
    """One rendered task status row."""

    task_identifier: str
    status: str
    worker: str
    final: str
    exit_code: str
    worktree: str
    lease: str
    title: str


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


def remove_stale_output_files(paths: list[Path]) -> None:
    """Remove output files from an earlier task-farm command attempt."""
    for path in paths:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


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
        normalized_path = normalize_expected_path(candidate_path)
        if normalized_path is None:
            continue
        expected_paths.append(normalized_path)
    return sorted(set(expected_paths))


def normalize_expected_path(candidate_path: str) -> str | None:
    """Normalize an inline-code path reference into a repository path."""
    if " " in candidate_path or any(character in candidate_path for character in "*?[]"):
        return None
    path_match = PATH_WITH_LINE_SUFFIX_PATTERN.match(candidate_path)
    normalized_candidate = path_match.group("path") if path_match is not None else candidate_path
    path = Path(normalized_candidate)
    if path.is_absolute() or ".." in path.parts:
        return None
    if normalized_candidate.startswith(
        ("src/", "tests/", "docs/", "scripts/", "Cargo.", "pyproject.toml", "Justfile"),
    ):
        return normalized_candidate
    return None


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


def default_manifest(
    *,
    source_path: Path = REPO_RELATIVE_SOURCE_PATH,
    plan_path: Path | None = None,
    state_directory_path: Path = REPO_RELATIVE_STATE_DIRECTORY,
    branch_prefix: str = DEFAULT_BRANCH_PREFIX,
    worktree_prefix: str = DEFAULT_WORKTREE_PREFIX,
    integration_branch: str = DEFAULT_INTEGRATION_BRANCH,
    integration_worktree: str = DEFAULT_INTEGRATION_WORKTREE,
    id_style: str = "legacy",
) -> JsonObject:
    """Create a manifest shell.

    Args:
        source_path: Markdown source path relative to the repository.
        plan_path: Optional shared plan path relative to the repository.
        state_directory_path: Runtime state directory relative to the repository.
        branch_prefix: Prefix for generated task branches.
        worktree_prefix: Prefix for generated task worktrees.
        integration_branch: Branch used for serial integration.
        integration_worktree: Worktree used for serial integration.
        id_style: ``legacy`` keeps numeric ids for the original workflow;
            ``string`` generates Review 2 style ids such as ``T001``.

    Returns:
        A manifest JSON object.
    """
    defaults: JsonObject = {
        "base_branch": "main",
        "branch_prefix": branch_prefix,
        "worktree_prefix": worktree_prefix,
        "worktree_root": DEFAULT_WORKTREE_ROOT,
        "state_directory": state_directory_path.as_posix(),
        "worker_model": "gpt-5.5",
        "worker_reasoning_effort": "high",
        "reviewer_model": "gpt-5.5",
        "reviewer_reasoning_effort": "xhigh",
        "integrator_model": "gpt-5.5",
        "integrator_reasoning_effort": "xhigh",
        "worker_dangerously_bypass_approvals": False,
        "integrator_dangerously_bypass_approvals": False,
        "integration_worktree": integration_worktree,
        "integration_branch": integration_branch,
        "jobs": 5,
        "lease_seconds": DEFAULT_LEASE_SECONDS,
        "id_style": id_style,
        "push_integration_branch": False,
    }
    if plan_path is not None:
        defaults["plan_path"] = plan_path.as_posix()
    return {
        "version": MANIFEST_VERSION,
        "source_path": source_path.as_posix(),
        "defaults": defaults,
        "tasks": [],
    }


def task_identifier_from_number(identifier: int, defaults: JsonObject) -> int | str:
    """Return the manifest task id for a parsed numeric source id."""
    if str(defaults.get("id_style", "legacy")) == "string":
        return f"T{identifier:03d}"
    return identifier


def task_identifier_sort_key(task_identifier: object) -> tuple[int, str]:
    """Return a stable sort key for mixed legacy and string task ids."""
    if isinstance(task_identifier, int):
        return (task_identifier, str(task_identifier))
    task_identifier_text = str(task_identifier)
    task_identifier_match = STRING_TASK_IDENTIFIER_PATTERN.match(task_identifier_text)
    if task_identifier_match is not None:
        return (int(task_identifier_match.group("number")), task_identifier_text)
    if task_identifier_text.isdigit():
        return (int(task_identifier_text), task_identifier_text)
    return (sys.maxsize, task_identifier_text)


def task_identifier_number(task_identifier: object) -> int:
    """Return the numeric component of a task id."""
    return task_identifier_sort_key(task_identifier)[0]


def task_identifier_key(task_identifier: object) -> str:
    """Return the state-map key for a task id."""
    return str(task_identifier)


def task_key(task: JsonObject) -> str:
    """Return the state-map key for a task."""
    return task_identifier_key(task["id"])


def task_display_id(task: JsonObject) -> str:
    """Return a human-readable task id."""
    task_identifier = task["id"]
    if isinstance(task_identifier, int):
        return f"{task_identifier:02d}"
    return str(task_identifier)


def task_run_directory_name(task: JsonObject) -> str:
    """Return the runtime run directory name for a task."""
    return task_display_id(task)


def task_generated_suffix(parsed_task: ParsedTask, defaults: JsonObject) -> str:
    """Return the generated branch/worktree suffix for a parsed task."""
    task_identifier = task_identifier_from_number(parsed_task.identifier, defaults)
    if isinstance(task_identifier, int):
        return f"{task_identifier:02d}-{parsed_task.slug}"
    return f"{task_identifier}-{parsed_task.slug}"


def task_branch(parsed_task: ParsedTask, defaults: JsonObject | None = None) -> str:
    """Return the branch name for a task."""
    effective_defaults = defaults or typing.cast("JsonObject", default_manifest()["defaults"])
    branch_prefix = str(effective_defaults.get("branch_prefix", DEFAULT_BRANCH_PREFIX))
    return f"{branch_prefix}{task_generated_suffix(parsed_task, effective_defaults)}"


def task_worktree(defaults: JsonObject, parsed_task: ParsedTask) -> str:
    """Return the manifest worktree path for a task."""
    worktree_prefix = str(defaults.get("worktree_prefix", DEFAULT_WORKTREE_PREFIX))
    return f"{worktree_prefix}{task_generated_suffix(parsed_task, defaults)}"


def infer_conflict_group(expected_paths: list[str]) -> str:
    """Infer a conflict group from expected task paths."""
    groups: set[str] = set()
    for expected_path in expected_paths:
        if "regenie2_binary" in expected_path or "binary" in expected_path:
            groups.add("binary-jax")
        elif "regenie2_linear" in expected_path or "linear" in expected_path:
            groups.add("linear-jax")
        elif expected_path.startswith("src/genotype/bgen") or "bgen" in expected_path:
            groups.add("rust-bgen")
        elif expected_path.startswith("src/sample") or "io/sample" in expected_path:
            groups.add("rust-sample")
        elif expected_path.startswith("src/output") or "io/output" in expected_path:
            groups.add("rust-output")
        elif "interface" in expected_path or expected_path.endswith("cli.py"):
            groups.add("interface")
        elif "codex_task_farm" in expected_path:
            groups.add("task-farm")
        elif expected_path.startswith("docs/"):
            groups.add("docs")
    if not groups:
        return "broad" if not expected_paths else "misc"
    if len(groups) > 1:
        return "broad"
    return next(iter(groups))


def build_log_paths(defaults: JsonObject, task_identifier: object) -> JsonObject:
    """Build default runtime log paths for a task."""
    state_directory_text = str(defaults.get("state_directory", REPO_RELATIVE_STATE_DIRECTORY.as_posix()))
    run_directory_name = f"{task_identifier:02d}" if isinstance(task_identifier, int) else str(task_identifier)
    run_directory = Path(state_directory_text) / "runs" / run_directory_name
    return {
        "run_directory": run_directory.as_posix(),
        "worker_final": (run_directory / "worker-final.md").as_posix(),
        "worker_jsonl": (run_directory / "worker.jsonl").as_posix(),
        "worker_stderr": (run_directory / "worker.stderr.log").as_posix(),
        "worker_prompt": (run_directory / "worker-prompt.md").as_posix(),
        "review_final": (run_directory / "review.md").as_posix(),
        "review_jsonl": (run_directory / "review.jsonl").as_posix(),
        "review_stderr": (run_directory / "review.stderr.log").as_posix(),
        "integration_final": (run_directory / "integration-final.md").as_posix(),
        "integration_jsonl": (run_directory / "integration.jsonl").as_posix(),
        "integration_stderr": (run_directory / "integration.stderr.log").as_posix(),
    }


def normalized_manual_metadata(existing_task: JsonObject | None) -> JsonObject:
    """Return manual metadata preserved across manifest syncs."""
    if existing_task is None:
        return {}
    manual_metadata = existing_task.get("manual", {})
    if not isinstance(manual_metadata, dict):
        manual_metadata = {}
    normalized_manual = dict(typing.cast("JsonObject", manual_metadata))
    if "notes" in existing_task and "notes" not in normalized_manual:
        normalized_manual["notes"] = existing_task["notes"]
    if "manual_expected_paths" in existing_task and "expected_paths" not in normalized_manual:
        normalized_manual["expected_paths"] = existing_task["manual_expected_paths"]
    if "conflict_group" in existing_task and str(existing_task.get("conflict_group_source", "manual")) == "manual":
        normalized_manual["conflict_group"] = existing_task["conflict_group"]
    return normalized_manual


def build_task_record(defaults: JsonObject, parsed_task: ParsedTask, existing_task: JsonObject | None) -> JsonObject:
    """Build one task manifest record while preserving manual metadata."""
    task_identifier = task_identifier_from_number(parsed_task.identifier, defaults)
    manual_metadata = normalized_manual_metadata(existing_task)
    generated_branch = task_branch(parsed_task, defaults)
    generated_worktree = task_worktree(defaults, parsed_task)
    branch = str(manual_metadata.get("branch", generated_branch))
    worktree = str(manual_metadata.get("worktree", generated_worktree))
    conflict_group_source = "manual" if "conflict_group" in manual_metadata else "inferred"
    conflict_group = str(manual_metadata.get("conflict_group", infer_conflict_group(parsed_task.expected_paths)))
    task_record: JsonObject = {
        "id": task_identifier,
        "source_id": parsed_task.identifier,
        "slug": parsed_task.slug,
        "title": parsed_task.title,
        "category": parsed_task.category,
        "source_start_line": parsed_task.source_start_line,
        "source_end_line": parsed_task.source_end_line,
        "body_markdown": parsed_task.body_markdown,
        "guidance_markdown": parsed_task.guidance_markdown,
        "kind": "implementation",
        "priority": parsed_task.category,
        "problem_markdown": parsed_task.body_markdown,
        "expected_paths": parsed_task.expected_paths,
        "dependencies": [],
        "status": TaskStatus.READY.value,
        "enabled": True,
        "branch": branch,
        "worktree": worktree,
        "logs": build_log_paths(defaults, task_identifier),
        "manual": manual_metadata,
        "runtime": {},
        "conflict_group": conflict_group,
        "conflict_group_source": conflict_group_source,
    }
    if existing_task is None:
        return task_record
    for key in PRESERVED_MANUAL_KEYS:
        if key in existing_task:
            task_record[key] = existing_task[key]
    task_record["manual"] = manual_metadata
    if "conflict_group" in manual_metadata:
        task_record["conflict_group"] = manual_metadata["conflict_group"]
        task_record["conflict_group_source"] = "manual"
    elif str(existing_task.get("conflict_group_source", "inferred")) == "manual" and "conflict_group" in existing_task:
        task_record["conflict_group"] = existing_task["conflict_group"]
        task_record["conflict_group_source"] = "manual"
    if "logs" not in task_record:
        task_record["logs"] = build_log_paths(defaults, task_identifier)
    if "runtime" not in task_record:
        task_record["runtime"] = {}
    return task_record


def existing_tasks_by_source_identifier(existing_manifest: JsonObject) -> dict[int, JsonObject]:
    """Return existing manifest tasks keyed by source task number."""
    existing_tasks: dict[int, JsonObject] = {}
    for existing_task in existing_manifest.get("tasks", []):
        if not isinstance(existing_task, dict):
            continue
        typed_task = typing.cast("JsonObject", existing_task)
        source_identifier = typed_task.get("source_id", typed_task.get("id"))
        if isinstance(source_identifier, str):
            source_identifier_match = STRING_TASK_IDENTIFIER_PATTERN.match(source_identifier)
            if source_identifier_match is not None:
                source_identifier = int(source_identifier_match.group("number"))
            elif source_identifier.isdigit():
                source_identifier = int(source_identifier)
        if isinstance(source_identifier, int):
            existing_tasks[source_identifier] = typed_task
    return existing_tasks


def default_id_style_for_manifest(manifest_path: Path) -> str:
    """Return the default id style for a manifest path."""
    if manifest_path == REPO_RELATIVE_MANIFEST_PATH:
        return "legacy"
    return "string"


def sync_plan_file(repository_directory: Path, manifest: JsonObject) -> None:
    """Create a shared plan file when a manifest declares one and it is missing."""
    defaults = typing.cast("JsonObject", manifest.get("defaults", {}))
    plan_path_text = defaults.get("plan_path")
    if not isinstance(plan_path_text, str) or not plan_path_text:
        return
    plan_path = repository_directory / plan_path_text
    if plan_path.exists():
        return
    lines = [
        "# Code Review 2 Task Plan",
        "",
        "This plan is updated serially by the task integrator. Workers write runtime logs only.",
        "",
    ]
    for task in selected_tasks(manifest, []):
        lines.append(f"## {task_display_id(task)}. {task['title']}")
        lines.append("")
        lines.append(f"- Status: {task.get('status', TaskStatus.READY.value)}")
        lines.append(f"- Branch: `{task['branch']}`")
        lines.append(f"- Runtime log: `{typing.cast('JsonObject', task['logs'])['run_directory']}`")
        lines.append("")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("\n".join(lines).rstrip() + "\n")


def sync_manifest(
    repository_directory: Path,
    *,
    manifest_relative_path: Path = REPO_RELATIVE_MANIFEST_PATH,
    source_relative_path: Path | None = None,
    plan_relative_path: Path | None = None,
    state_directory_path: Path | None = None,
    branch_prefix: str | None = None,
    worktree_prefix: str | None = None,
    integration_branch: str | None = None,
    integration_worktree: str | None = None,
) -> JsonObject:
    """Synchronize the task manifest from the markdown source."""
    source_path_from_arguments = source_relative_path
    manifest_path = repository_directory / manifest_relative_path
    existing_manifest = read_json_object(manifest_path) if manifest_path.exists() else {}
    id_style = str(
        typing.cast("JsonObject", existing_manifest.get("defaults", {})).get(
            "id_style",
            default_id_style_for_manifest(manifest_relative_path),
        )
    )
    existing_source_path = existing_manifest.get("source_path")
    default_source_path = (
        Path(str(existing_source_path))
        if isinstance(existing_source_path, str)
        else REPO_RELATIVE_SOURCE_PATH
    )
    manifest = default_manifest(
        source_path=source_path_from_arguments or default_source_path,
        plan_path=plan_relative_path,
        state_directory_path=state_directory_path or REPO_RELATIVE_STATE_DIRECTORY,
        branch_prefix=branch_prefix or DEFAULT_BRANCH_PREFIX,
        worktree_prefix=worktree_prefix or DEFAULT_WORKTREE_PREFIX,
        integration_branch=integration_branch or DEFAULT_INTEGRATION_BRANCH,
        integration_worktree=integration_worktree or DEFAULT_INTEGRATION_WORKTREE,
        id_style=id_style,
    )
    if isinstance(existing_manifest.get("defaults"), dict):
        manifest["defaults"].update(typing.cast("JsonObject", existing_manifest["defaults"]))
    existing_defaults = typing.cast("JsonObject", existing_manifest.get("defaults", {}))
    if "worktree_root" in existing_defaults and "worktree_prefix" not in existing_defaults:
        typing.cast("JsonObject", manifest["defaults"])["worktree_prefix"] = (
            f"{str(existing_defaults['worktree_root']).rstrip('/')}/review-"
        )
    if source_relative_path is not None:
        manifest["source_path"] = source_relative_path.as_posix()
    if plan_relative_path is not None:
        typing.cast("JsonObject", manifest["defaults"])["plan_path"] = plan_relative_path.as_posix()
    if state_directory_path is not None:
        typing.cast("JsonObject", manifest["defaults"])["state_directory"] = state_directory_path.as_posix()
    if branch_prefix is not None:
        typing.cast("JsonObject", manifest["defaults"])["branch_prefix"] = branch_prefix
    if worktree_prefix is not None:
        typing.cast("JsonObject", manifest["defaults"])["worktree_prefix"] = worktree_prefix
    if integration_branch is not None:
        typing.cast("JsonObject", manifest["defaults"])["integration_branch"] = integration_branch
    if integration_worktree is not None:
        typing.cast("JsonObject", manifest["defaults"])["integration_worktree"] = integration_worktree
    if manifest_relative_path != REPO_RELATIVE_MANIFEST_PATH or source_relative_path is not None:
        typing.cast("JsonObject", manifest["defaults"])["id_style"] = "string"
        typing.cast("JsonObject", manifest["defaults"])["push_integration_branch"] = True
    defaults = typing.cast("JsonObject", manifest["defaults"])
    source_path = repository_directory / str(manifest["source_path"])
    existing_tasks = existing_tasks_by_source_identifier(existing_manifest)

    parsed_tasks = parse_review_tasks(source_path.read_text())
    manifest["tasks"] = [
        build_task_record(defaults, parsed_task, existing_tasks.get(parsed_task.identifier))
        for parsed_task in parsed_tasks
    ]
    write_json_object(manifest_path, manifest)
    sync_plan_file(repository_directory, manifest)
    return manifest


def task_runtime_identity(task: JsonObject) -> JsonObject:
    """Build the runtime identity fields that make a task status reusable."""
    return {
        "id": task["id"],
        "slug": str(task.get("slug", "")),
        "branch": str(task.get("branch", "")),
        "worktree": str(task.get("worktree", "")),
        "source_start_line": int(task.get("source_start_line", 0)),
        "source_end_line": int(task.get("source_end_line", 0)),
    }


def task_identity_matches(task: JsonObject, identity: JsonObject | None) -> bool:
    """Return whether a stored task identity belongs to the current manifest task."""
    if identity is None:
        return False
    return task_runtime_identity(task) == {
        "id": identity.get("id"),
        "slug": identity.get("slug"),
        "branch": identity.get("branch"),
        "worktree": identity.get("worktree"),
        "source_start_line": identity.get("source_start_line"),
        "source_end_line": identity.get("source_end_line"),
    }


def manifest_task_identity(parsed_task: ParsedTask, defaults: JsonObject) -> JsonObject:
    """Build the identity a parsed markdown task should have in the manifest."""
    task_identifier = task_identifier_from_number(parsed_task.identifier, defaults)
    return {
        "id": task_identifier,
        "slug": parsed_task.slug,
        "branch": task_branch(parsed_task, defaults),
        "worktree": task_worktree(defaults, parsed_task),
        "source_start_line": parsed_task.source_start_line,
        "source_end_line": parsed_task.source_end_line,
    }


def validate_manifest_is_current(repository_directory: Path, manifest: JsonObject, manifest_path: Path) -> None:
    """Raise when the manifest does not match the current markdown source."""
    source_path = repository_directory / str(manifest.get("source_path", REPO_RELATIVE_SOURCE_PATH.as_posix()))
    defaults = typing.cast("JsonObject", manifest.get("defaults", {}))
    parsed_tasks = parse_review_tasks(source_path.read_text())
    existing_tasks = existing_tasks_by_source_identifier(manifest)
    expected_identities = [
        task_runtime_identity(build_task_record(defaults, parsed_task, existing_tasks.get(parsed_task.identifier)))
        for parsed_task in parsed_tasks
    ]
    task_records = selected_tasks(manifest, [])
    actual_identities = [task_runtime_identity(task) for task in task_records]
    if actual_identities != expected_identities:
        message = f"{manifest_path.as_posix()} is missing or stale. Run sync-manifest first."
        raise ValueError(message)


def load_manifest(
    repository_directory: Path,
    *,
    manifest_relative_path: Path = REPO_RELATIVE_MANIFEST_PATH,
    validate_current: bool = True,
) -> JsonObject:
    """Load the task manifest."""
    manifest_path = repository_directory / manifest_relative_path
    if not manifest_path.exists():
        message = f"{manifest_relative_path.as_posix()} is missing. Run sync-manifest first."
        raise ValueError(message)
    manifest = read_json_object(manifest_path)
    if validate_current:
        validate_manifest_is_current(repository_directory, manifest, manifest_relative_path)
    return manifest


def save_manifest(
    repository_directory: Path,
    manifest: JsonObject,
    *,
    manifest_relative_path: Path = REPO_RELATIVE_MANIFEST_PATH,
) -> None:
    """Save the task manifest."""
    write_json_object(repository_directory / manifest_relative_path, manifest)


def state_directory(repository_directory: Path, manifest: JsonObject) -> Path:
    """Resolve the state directory for runtime logs."""
    defaults = typing.cast("JsonObject", manifest.get("defaults", {}))
    return repository_directory / str(defaults.get("state_directory", REPO_RELATIVE_STATE_DIRECTORY.as_posix()))


def task_logs(task: JsonObject) -> JsonObject:
    """Return task log path metadata."""
    logs = task.get("logs", {})
    if isinstance(logs, dict):
        return typing.cast("JsonObject", logs)
    return {}


def task_run_directory(repository_directory: Path, manifest: JsonObject, task: JsonObject) -> Path:
    """Resolve the runtime run directory for a task."""
    run_directory_text = task_logs(task).get("run_directory")
    if isinstance(run_directory_text, str):
        return resolve_manifest_path(repository_directory, run_directory_text)
    return state_directory(repository_directory, manifest) / "runs" / task_run_directory_name(task)


def task_log_path(
    repository_directory: Path,
    manifest: JsonObject,
    task: JsonObject,
    log_key: str,
    fallback_filename: str,
) -> Path:
    """Resolve one task log path with fallback for legacy manifests."""
    log_path_text = task_logs(task).get(log_key)
    if isinstance(log_path_text, str):
        return resolve_manifest_path(repository_directory, log_path_text)
    return task_run_directory(repository_directory, manifest, task) / fallback_filename


def resolve_manifest_path(repository_directory: Path, manifest_path: str) -> Path:
    """Resolve a path stored in the manifest."""
    path = Path(manifest_path)
    if path.is_absolute():
        return path
    return (repository_directory / path).resolve()


def normalized_selector(identifier: object) -> str:
    """Normalize a command-line task selector."""
    if isinstance(identifier, int):
        return str(identifier)
    identifier_text = str(identifier)
    if identifier_text.isdigit():
        return str(int(identifier_text))
    return identifier_text


def task_selector_values(task: JsonObject) -> set[str]:
    """Return all accepted selector spellings for a task."""
    task_identifier = task["id"]
    values = {str(task_identifier)}
    source_identifier = task.get("source_id")
    if isinstance(source_identifier, int):
        values.add(str(source_identifier))
        values.add(f"{source_identifier:02d}")
    if isinstance(task_identifier, int):
        values.add(str(task_identifier))
        values.add(f"{task_identifier:02d}")
    task_identifier_match = STRING_TASK_IDENTIFIER_PATTERN.match(str(task_identifier))
    if task_identifier_match is not None:
        number_text = task_identifier_match.group("number")
        values.add(str(int(number_text)))
        values.add(f"{int(number_text):02d}")
    return values


def selected_tasks(manifest: JsonObject, identifiers: list[object]) -> list[JsonObject]:
    """Return selected manifest task records."""
    tasks = [task for task in manifest.get("tasks", []) if isinstance(task, dict)]
    typed_tasks = [typing.cast("JsonObject", task) for task in tasks]
    typed_tasks.sort(key=lambda task: task_identifier_sort_key(task.get("id")))
    if not identifiers:
        return typed_tasks
    selected_identifiers = {normalized_selector(identifier) for identifier in identifiers}
    return [task for task in typed_tasks if task_selector_values(task) & selected_identifiers]


def subprocess_environment(cwd: Path) -> dict[str, str]:
    """Return a subprocess environment consistent with an explicit cwd."""
    environment = dict(os.environ)
    environment["PWD"] = str(cwd)
    return environment


def run_command(command_arguments: list[str], *, cwd: Path) -> CommandResult:
    """Run a command and capture output."""
    completed_process = subprocess.run(
        command_arguments,
        cwd=cwd,
        env=subprocess_environment(cwd),
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
    run_directory = task_run_directory(repository_directory, manifest, task)
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


def task_expected_paths(task: JsonObject) -> list[str]:
    """Return generated and manual expected paths without duplicates."""
    expected_paths = [
        str(path)
        for path in typing.cast("list[object]", task.get("expected_paths", []))
        if isinstance(path, str)
    ]
    manual_expected_paths = [
        str(path)
        for path in typing.cast("list[object]", task.get("manual_expected_paths", []))
        if isinstance(path, str)
    ]
    return sorted(set(expected_paths + manual_expected_paths))


def build_worker_prompt(task: JsonObject) -> str:
    """Build the implementation prompt for a worker agent."""
    expected_paths = task_expected_paths(task)
    logs = task_logs(task)
    run_directory = str(logs.get("run_directory", "the task runtime run directory"))
    plan_instruction = ""
    if logs:
        plan_instruction = (
            "Do not edit shared task plans or manifests such as docs/code-review-2-plan.md or "
            "docs/code-review-2.tasks.json. Write runtime notes only in the log paths below."
        )
    return f"""You are a Codex implementation worker in a dedicated git worktree.

Read AGENTS.md, docs/STYLEGUIDE.md, and Justfile before editing code.
If the worktree already contains edits, inspect them and continue from them unless they are clearly wrong.
Implement exactly this task and keep the change narrow.
Commit logical intermediate steps and leave a clean worktree when done.
Run relevant tests through `nix develop --command just ...` when feasible.
Never commit files under data/ or .codex-task-worktrees/.
{plan_instruction}

Task {task["id"]}: {task["title"]}
Category: {task["category"]}
Expected paths: {", ".join(expected_paths) or "not precomputed"}
Runtime log directory: {run_directory}

{task["body_markdown"]}

Guidance:
{task.get("guidance_markdown", "")}

Final response must include changed files, commits created, tests run, benchmarks if relevant, failed hypotheses,
remaining blockers, and remaining risks.
"""


def build_review_prompt(task: JsonObject) -> str:
    """Build the review prompt for a task branch."""
    logs = task_logs(task)
    review_path = str(logs.get("review_final", "the task runtime review log"))
    return f"""Review task branch {task["branch"]} against main.

Task {task["id"]}: {task["title"]}

Review for correctness, behavioral regressions, styleguide compliance, missing tests, and whether the implementation
actually satisfies the source review task.
Run read-only. Do not edit files in the task worktree or shared plan.
Write the review result to {review_path}.
Start the final response with exactly one decision line: `Decision: accept`, `Decision: needs_changes`, or
`Decision: reject`.
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


def classify_review_decision(review_text: str) -> TaskStatus:
    """Classify a review final message into a lifecycle status."""
    normalized_text = review_text.lower()
    decision_match = re.search(r"decision:\s*(accept|needs[_ -]changes|reject)", normalized_text)
    decision = decision_match.group(1).replace("-", "_").replace(" ", "_") if decision_match is not None else "accept"
    if decision == "needs_changes":
        return TaskStatus.NEEDS_CHANGES
    if decision == "reject":
        return TaskStatus.BLOCKED
    return TaskStatus.REVIEWED


def build_worker_command(
    worktree_path: Path,
    git_metadata_path: Path | None,
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
    ]
    if git_metadata_path is not None:
        command_arguments.extend(["--add-dir", str(git_metadata_path)])
    command_arguments.extend(
        [
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
    )
    if dangerously_bypass_approvals:
        command_arguments.insert(command_arguments.index("exec"), "--dangerously-bypass-approvals-and-sandbox")
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
    git_metadata_path: Path | None,
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
    ]
    if git_metadata_path is not None:
        command_arguments.extend(["--add-dir", str(git_metadata_path)])
    command_arguments.extend(
        [
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
    )
    if dangerously_bypass_approvals:
        command_arguments.insert(command_arguments.index("exec"), "--dangerously-bypass-approvals-and-sandbox")
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
        return {"leases": {}, "runs": {}, "statuses": {}, "task_identities": {}}
    return read_json_object(state_path)


def save_state(state_path: Path, state: JsonObject) -> None:
    """Save runtime state."""
    write_json_object(state_path, state)


@contextlib.contextmanager
def manifest_lock(repository_directory: Path, manifest: JsonObject, lock_name: str) -> typing.Iterator[None]:
    """Acquire a manifest-scoped advisory lock."""
    lock_directory = state_directory(repository_directory, manifest)
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / lock_name
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def lease_owner() -> str:
    """Return a default lease owner string."""
    username = os.environ.get("USER", "unknown")
    hostname = os.uname().nodename
    return f"{username}@{hostname}:{os.getpid()}"


def parse_utc_timestamp(timestamp_text: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp produced by this module."""
    normalized_timestamp = timestamp_text.replace("Z", "+00:00")
    parsed_timestamp = datetime.datetime.fromisoformat(normalized_timestamp)
    if parsed_timestamp.tzinfo is None:
        return parsed_timestamp.replace(tzinfo=datetime.UTC)
    return parsed_timestamp.astimezone(datetime.UTC)


def lease_is_expired(lease: object, *, now: datetime.datetime | None = None) -> bool:
    """Return whether a lease object is expired."""
    if not isinstance(lease, dict):
        return True
    typed_lease = typing.cast("JsonObject", lease)
    expires_at = typed_lease.get("expires_at")
    if not isinstance(expires_at, str):
        return True
    try:
        expires_at_timestamp = parse_utc_timestamp(expires_at)
    except ValueError:
        return True
    effective_now = now or datetime.datetime.now(datetime.UTC)
    return expires_at_timestamp <= effective_now


def set_task_lease(
    state: JsonObject,
    task: JsonObject,
    *,
    owner: str,
    status: TaskStatus,
    lease_seconds: int,
) -> None:
    """Set a task lease and status."""
    now = datetime.datetime.now(datetime.UTC)
    expires_at = now + datetime.timedelta(seconds=lease_seconds)
    leases = typing.cast("JsonObject", state.setdefault("leases", {}))
    leases[task_key(task)] = {
        "owner": owner,
        "status": status.value,
        "acquired_at": now.isoformat(timespec="seconds"),
        "expires_at": expires_at.isoformat(timespec="seconds"),
    }
    set_runtime_task_status(state, task, status)


def clear_task_lease(state: JsonObject, task: JsonObject) -> None:
    """Clear a task lease if present."""
    leases = typing.cast("JsonObject", state.setdefault("leases", {}))
    leases.pop(task_key(task), None)


def reset_stale_task_leases(state: JsonObject, manifest: JsonObject, *, force: bool = False) -> list[str]:
    """Reset expired claimed/running task leases to ready."""
    reset_identifiers: list[str] = []
    leases = typing.cast("JsonObject", state.setdefault("leases", {}))
    for task in selected_tasks(manifest, []):
        task_identifier = task_key(task)
        lease = leases.get(task_identifier)
        status = runtime_task_status(state, task)
        if status not in {TaskStatus.CLAIMED.value, TaskStatus.RUNNING.value}:
            continue
        if not force and not lease_is_expired(lease):
            continue
        leases.pop(task_identifier, None)
        set_runtime_task_status(state, task, TaskStatus.READY)
        reset_identifiers.append(task_identifier)
    return reset_identifiers


def active_conflict_groups(state: JsonObject, manifest: JsonObject) -> set[str]:
    """Return conflict groups currently claimed, running, or integrating."""
    groups: set[str] = set()
    for task in selected_tasks(manifest, []):
        status = runtime_task_status(state, task)
        if status not in {TaskStatus.CLAIMED.value, TaskStatus.RUNNING.value, TaskStatus.INTEGRATING.value}:
            continue
        if "conflict_group" in task:
            groups.add(str(task["conflict_group"]))
    return groups


def conflict_group_available(
    task: JsonObject,
    *,
    active_groups: set[str],
    selected_groups: set[str],
) -> bool:
    """Return whether a task can be scheduled without conflict-group overlap."""
    if "conflict_group" not in task:
        return True
    conflict_group = str(task["conflict_group"])
    unavailable_groups = active_groups | selected_groups
    if conflict_group == "broad":
        return not unavailable_groups
    if "broad" in unavailable_groups:
        return False
    return conflict_group not in unavailable_groups


def runtime_task_status(state: JsonObject, task: JsonObject) -> str:
    """Return the current runtime status for a task."""
    statuses = typing.cast("JsonObject", state.get("statuses", {}))
    identities = typing.cast("JsonObject", state.get("task_identities", {}))
    task_identifier = task_key(task)
    stored_identity = identities.get(task_identifier)
    if (
        stored_identity is not None
        and isinstance(stored_identity, dict)
        and not task_identity_matches(task, typing.cast("JsonObject", stored_identity))
    ):
        return str(task.get("status", TaskStatus.READY.value))
    status = statuses.get(task_identifier, task.get("status", TaskStatus.READY.value))
    return str(status)


def set_runtime_task_status(state: JsonObject, task: JsonObject, status: TaskStatus) -> None:
    """Set a task status in ignored runtime state."""
    statuses = typing.cast("JsonObject", state.setdefault("statuses", {}))
    task_identifier = task_key(task)
    statuses[task_identifier] = status.value
    identities = typing.cast("JsonObject", state.setdefault("task_identities", {}))
    identities[task_identifier] = task_runtime_identity(task)
    status_updates = typing.cast("JsonObject", state.setdefault("status_updated_at", {}))
    status_updates[task_identifier] = utc_timestamp()


def record_worker_completion(state: JsonObject, task: JsonObject, completion: WorkerCompletion) -> None:
    """Record worker completion details in runtime state."""
    set_runtime_task_status(state, task, completion.status)
    worker_results = typing.cast("JsonObject", state.setdefault("worker_results", {}))
    worker_results[task_key(task)] = {
        "branch_ahead": completion.branch_ahead,
        "exit_code": completion.exit_code,
        "final_message_exists": completion.final_message_exists,
        "reason": completion.reason,
        "verification": completion.verification,
        "worktree_clean": completion.worktree_clean,
    }


def state_run_matches_task(run: object, task: JsonObject) -> bool:
    """Return whether a runtime run record belongs to the current manifest task."""
    if not isinstance(run, dict):
        return False
    typed_run = typing.cast("JsonObject", run)
    return typed_run.get("branch") == task.get("branch") and typed_run.get("worktree") == task.get("worktree")


def archive_runtime_state_entry(
    state: JsonObject,
    *,
    task_identifier: str,
    reason: str,
    task: JsonObject | None,
) -> None:
    """Move stale runtime state for one task identifier out of active scheduling maps."""
    archived_entries = typing.cast("list[JsonObject]", state.setdefault("archived_stale_entries", []))
    runs = typing.cast("JsonObject", state.setdefault("runs", {}))
    statuses = typing.cast("JsonObject", state.setdefault("statuses", {}))
    status_updates = typing.cast("JsonObject", state.setdefault("status_updated_at", {}))
    worker_results = typing.cast("JsonObject", state.setdefault("worker_results", {}))
    identities = typing.cast("JsonObject", state.setdefault("task_identities", {}))
    leases = typing.cast("JsonObject", state.setdefault("leases", {}))
    entry: JsonObject = {
        "archived_at": utc_timestamp(),
        "task_identifier": task_identifier,
        "reason": reason,
    }
    if task is not None:
        entry["current_task_identity"] = task_runtime_identity(task)
    for key, mapping in (
        ("run", runs),
        ("status", statuses),
        ("status_updated_at", status_updates),
        ("worker_result", worker_results),
        ("task_identity", identities),
        ("lease", leases),
    ):
        if task_identifier in mapping:
            entry[key] = mapping.pop(task_identifier)
    archived_entries.append(entry)


def prune_stale_runtime_state(state: JsonObject, manifest: JsonObject) -> bool:
    """Remove runtime state that does not match the current task manifest."""
    tasks_by_runtime_identifier = {task_key(task): task for task in selected_tasks(manifest, [])}
    active_identifiers = set(tasks_by_runtime_identifier)
    candidate_identifiers: set[str] = set()
    for mapping_name in ("leases", "runs", "statuses", "status_updated_at", "worker_results", "task_identities"):
        mapping = state.setdefault(mapping_name, {})
        if isinstance(mapping, dict):
            candidate_identifiers.update(str(identifier) for identifier in mapping)
    changed = False
    for task_identifier in sorted(candidate_identifiers):
        task = tasks_by_runtime_identifier.get(task_identifier)
        if task is None:
            archive_runtime_state_entry(
                state,
                task_identifier=task_identifier,
                reason="task-not-in-current-manifest",
                task=None,
            )
            changed = True
            continue
        runs = typing.cast("JsonObject", state.setdefault("runs", {}))
        identities = typing.cast("JsonObject", state.setdefault("task_identities", {}))
        stored_identity = identities.get(task_identifier)
        run = runs.get(task_identifier)
        if task_identifier not in active_identifiers:
            continue
        if isinstance(stored_identity, dict) and not task_identity_matches(
            task,
            typing.cast("JsonObject", stored_identity),
        ):
            archive_runtime_state_entry(
                state,
                task_identifier=task_identifier,
                reason="task-identity-mismatch",
                task=task,
            )
            changed = True
            continue
        if run is not None and not state_run_matches_task(run, task):
            archive_runtime_state_entry(
                state,
                task_identifier=task_identifier,
                reason="run-branch-or-worktree-mismatch",
                task=task,
            )
            changed = True
    return changed


def tasks_by_identifier(manifest: JsonObject) -> dict[str, JsonObject]:
    """Return manifest tasks keyed by task id."""
    indexed_tasks: dict[str, JsonObject] = {}
    for task in selected_tasks(manifest, []):
        for selector in task_selector_values(task):
            indexed_tasks[normalized_selector(selector)] = task
    return indexed_tasks


def dependency_identifiers(task: JsonObject) -> list[str]:
    """Return normalized dependency task identifiers."""
    dependencies = task.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ValueError(f"Task {task['id']} dependencies must be a list.")
    identifiers: list[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, (int, str)):
            raise ValueError(f"Task {task['id']} has invalid dependency {dependency!r}.")
        identifiers.append(normalized_selector(dependency))
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


def ensure_tasks_have_status(
    state: JsonObject,
    tasks: list[JsonObject],
    allowed_statuses: set[str],
    operation: str,
) -> None:
    """Raise if selected tasks are not in an allowed state."""
    for task in tasks:
        task_status = runtime_task_status(state, task)
        if task_status not in allowed_statuses:
            raise ValueError(
                f"Task {task['id']} cannot {operation}: status is "
                f"{task_status}, expected one of {sorted(allowed_statuses)}.",
            )


def git_current_branch(worktree_path: Path) -> str:
    """Return the current branch name for a worktree."""
    command_result = run_command(["git", "branch", "--show-current"], cwd=worktree_path)
    ensure_success(command_result)
    return command_result.stdout.strip()


def git_head_commit(worktree_path: Path) -> str:
    """Return the current HEAD commit for a worktree."""
    command_result = run_command(["git", "rev-parse", "HEAD"], cwd=worktree_path)
    ensure_success(command_result)
    return command_result.stdout.strip()


def ensure_integration_worktree_ready(worktree_path: Path, integration_branch: str) -> None:
    """Raise unless the integration worktree is clean and on the integration branch."""
    current_branch = git_current_branch(worktree_path)
    if current_branch != integration_branch:
        message = (
            f"Integration worktree is on {current_branch or '<detached>'}, "
            f"expected {integration_branch}."
        )
        raise ValueError(message)
    if not git_worktree_is_clean(worktree_path):
        message = "Integration worktree has uncommitted changes."
        raise ValueError(message)


def push_integration_branch(integration_worktree_path: Path, integration_branch: str) -> None:
    """Push the integration branch to origin."""
    command_result = run_command(["git", "push", "origin", integration_branch], cwd=integration_worktree_path)
    ensure_success(command_result)


def refresh_runtime_statuses(repository_directory: Path, manifest: JsonObject) -> None:
    """Refresh task statuses from detached worker state."""
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    changed = prune_stale_runtime_state(state, manifest)
    runs = typing.cast("JsonObject", state.get("runs", {}))
    for task in selected_tasks(manifest, []):
        task_identifier = task_key(task)
        run = runs.get(task_identifier, {})
        if not isinstance(run, dict):
            continue
        pid = run.get("pid")
        if not isinstance(pid, int):
            continue
        run_directory = task_run_directory(repository_directory, manifest, task)
        if running_process_exists(pid):
            if runtime_task_status(state, task) != TaskStatus.RUNNING.value:
                set_runtime_task_status(state, task, TaskStatus.RUNNING)
                changed = True
            continue
        if not run_directory.exists():
            set_runtime_task_status(state, task, TaskStatus.BLOCKED)
            changed = True
            continue
        final_message_path = run_directory / "worker-final.md"
        exit_code_path = run_directory / "exit-code.txt"
        if not final_message_path.exists() and not exit_code_path.exists():
            if runtime_task_status(state, task) != TaskStatus.RUNNING.value:
                set_runtime_task_status(state, task, TaskStatus.RUNNING)
                changed = True
            continue
        if runtime_task_status(state, task) not in {TaskStatus.RUNNING.value, TaskStatus.BLOCKED.value}:
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
    run_directory = task_run_directory(repository_directory, manifest, task)
    run_directory.mkdir(parents=True, exist_ok=True)
    final_message_path = task_log_path(repository_directory, manifest, task, "worker_final", "worker-final.md")
    jsonl_log_path = task_log_path(repository_directory, manifest, task, "worker_jsonl", "worker.jsonl")
    stderr_log_path = task_log_path(repository_directory, manifest, task, "worker_stderr", "worker.stderr.log")
    prompt_path = task_log_path(repository_directory, manifest, task, "worker_prompt", "worker-prompt.md")
    wrapper_path = run_directory / "worker-wrapper.sh"
    exit_code_path = run_directory / "exit-code.txt"
    for log_path in [final_message_path, jsonl_log_path, stderr_log_path, prompt_path, wrapper_path, exit_code_path]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    remove_stale_output_files(
        [final_message_path, jsonl_log_path, stderr_log_path, prompt_path, wrapper_path, exit_code_path]
    )
    prompt = build_worker_prompt(task)
    prompt_path.write_text(prompt)
    command_arguments = build_worker_command(
        worktree_path=worktree_path,
        git_metadata_path=repository_directory / ".git",
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
            env=subprocess_environment(repository_directory),
            stdout=stdout_log,
            stderr=stderr_log,
            text=True,
            start_new_session=True,
        )
        return WorkerLaunch(
            task_identifier=task_identifier_number(task["id"]),
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
    source_path = Path(str(manifest.get("source_path", REPO_RELATIVE_SOURCE_PATH.as_posix())))
    checks = [
        doctor_command_check("git"),
        doctor_command_check("codex"),
        doctor_command_check("nix", warning=not strict),
        make_doctor_check(
            "repo root",
            passed=(repository_directory / ".git").exists(),
            message=f"repo root: {repository_directory}",
        ),
        doctor_file_check(repository_directory, source_path),
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


def argument_manifest_path(arguments: argparse.Namespace) -> Path:
    """Return the repository-relative manifest path selected by CLI arguments."""
    manifest_path = getattr(arguments, "manifest", None)
    if manifest_path is None:
        return REPO_RELATIVE_MANIFEST_PATH
    return Path(str(manifest_path))


def load_manifest_for_arguments(repository_directory: Path, arguments: argparse.Namespace) -> JsonObject:
    """Load the manifest selected by CLI arguments."""
    manifest_relative_path = argument_manifest_path(arguments)
    if manifest_relative_path == REPO_RELATIVE_MANIFEST_PATH:
        return load_manifest(repository_directory)
    return load_manifest(repository_directory, manifest_relative_path=manifest_relative_path)


def command_doctor(arguments: argparse.Namespace) -> int:
    """Handle doctor."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
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
    manifest_relative_path = argument_manifest_path(arguments)
    source_relative_path = Path(arguments.source) if getattr(arguments, "source", None) else None
    plan_relative_path = Path(arguments.plan) if getattr(arguments, "plan", None) else None
    state_directory_path = Path(arguments.state_dir) if getattr(arguments, "state_dir", None) else None
    manifest = sync_manifest(
        repository_directory,
        manifest_relative_path=manifest_relative_path,
        source_relative_path=source_relative_path,
        plan_relative_path=plan_relative_path,
        state_directory_path=state_directory_path,
        branch_prefix=getattr(arguments, "branch_prefix", None),
        worktree_prefix=getattr(arguments, "worktree_prefix", None),
        integration_branch=getattr(arguments, "integration_branch", None),
        integration_worktree=getattr(arguments, "integration_worktree", None),
    )
    print(f"Synced {len(manifest.get('tasks', []))} tasks into {manifest_relative_path.as_posix()}.")
    return 0


def command_list(arguments: argparse.Namespace) -> int:
    """Handle list."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    refresh_runtime_statuses(repository_directory, manifest)
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    for task in selected_tasks(manifest, typing.cast("list[object]", arguments.task)):
        print(f"{task_display_id(task):6}  {runtime_task_status(state, task):14}  {task['branch']}  {task['title']}")
    return 0


def command_run(arguments: argparse.Namespace) -> int:
    """Handle run."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    refresh_runtime_statuses(repository_directory, manifest)
    defaults = typing.cast("JsonObject", manifest["defaults"])
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    jobs = int(arguments.jobs if arguments.jobs is not None else defaults.get("jobs", 5))
    wait_for_completion = bool(arguments.wait)
    force = bool(arguments.force)
    owner = str(getattr(arguments, "owner", None) or lease_owner())
    lease_seconds = int(
        getattr(arguments, "lease_seconds", None) or defaults.get("lease_seconds", DEFAULT_LEASE_SECONDS),
    )
    dangerously_bypass_approvals = bool(
        arguments.dangerous or defaults.get("worker_dangerously_bypass_approvals", False),
    )
    with manifest_lock(repository_directory, manifest, "claim.lock"):
        state = load_state(state_path)
        reset_stale_task_leases(state, manifest)
        candidates = selected_tasks(manifest, typing.cast("list[object]", arguments.task))
        runnable_tasks: list[JsonObject] = []
        active_groups = active_conflict_groups(state, manifest)
        selected_groups: set[str] = set()
        for task in candidates:
            if len(runnable_tasks) >= jobs:
                break
            if not bool(task.get("enabled", True)):
                continue
            current_status = runtime_task_status(state, task)
            if not force and current_status != TaskStatus.READY.value:
                continue
            if not conflict_group_available(task, active_groups=active_groups, selected_groups=selected_groups):
                continue
            runnable_tasks.append(task)
            if "conflict_group" in task:
                selected_groups.add(str(task["conflict_group"]))
        if not runnable_tasks:
            print("No runnable tasks selected.")
            save_state(state_path, state)
            return 0
        ensure_dependencies_ready(
            manifest,
            state,
            runnable_tasks,
            {TaskStatus.INTEGRATED.value, TaskStatus.MERGED.value},
            "run",
        )
        for task in runnable_tasks:
            set_task_lease(
                state,
                task,
                owner=owner,
                status=TaskStatus.CLAIMED,
                lease_seconds=lease_seconds,
            )
        save_state(state_path, state)

    exit_code = 0
    launched_workers: list[WorkerLaunch] = []
    for task in runnable_tasks:
        launched_worker = launch_worker(
            repository_directory=repository_directory,
            manifest=manifest,
            task=task,
            dangerously_bypass_approvals=dangerously_bypass_approvals,
        )
        state = load_state(state_path)
        runs = typing.cast("JsonObject", state.setdefault("runs", {}))
        task_identifier = task_key(task)
        runs[task_identifier] = {
            "pid": launched_worker.process_identifier,
            "returncode": None,
            "started_at": utc_timestamp(),
            "branch": task["branch"],
            "worktree": task["worktree"],
        }
        set_task_lease(
            state,
            task,
            owner=owner,
            status=TaskStatus.RUNNING,
            lease_seconds=lease_seconds,
        )
        launched_workers.append(launched_worker)
        print(f"Launched task {task_display_id(task)}: {task['branch']}")
        save_state(state_path, state)
    if wait_for_completion:
        for launched_worker in launched_workers:
            returncode = launched_worker.process.wait()
            task = next(
                task
                for task in runnable_tasks
                if task_identifier_number(task["id"]) == launched_worker.task_identifier
            )
            state = load_state(state_path)
            runs = typing.cast("JsonObject", state.setdefault("runs", {}))
            run = runs.get(task_key(task), {})
            if isinstance(run, dict):
                run["returncode"] = returncode
                run["finished_at"] = utc_timestamp()
            completion = classify_worker_completion(repository_directory, manifest, task)
            record_worker_completion(state, task, completion)
            clear_task_lease(state, task)
            if completion.status == TaskStatus.BLOCKED and exit_code == 0:
                exit_code = returncode or 1
            save_state(state_path, state)
    save_state(state_path, state)
    return exit_code


def status_count_text(rows: list[StatusRow], attribute_name: str) -> str:
    """Return compact counts for one StatusRow attribute."""
    counts: dict[str, int] = {}
    for row in rows:
        value = str(getattr(row, attribute_name))
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return "-"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def truncate_status_cell(value: str, width: int) -> str:
    """Return a display cell no wider than width."""
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return f"{value[: width - 3]}..."


def read_exit_code_text(exit_code_path: Path, run: object) -> str:
    """Return an exit-code display value from logs or runtime state."""
    if exit_code_path.exists():
        exit_code_text = exit_code_path.read_text().strip()
        return exit_code_text or "-"
    if isinstance(run, dict):
        typed_run = typing.cast("JsonObject", run)
        returncode = typed_run.get("returncode")
        if isinstance(returncode, int):
            return str(returncode)
    return "-"


def worker_log_recent(run_directory: Path, *, recent_seconds: float = 120.0) -> bool:
    """Return whether a worker log was updated recently."""
    current_time = time.time()
    for log_file_name in ("worker.jsonl", "worker.stderr.log"):
        log_path = run_directory / log_file_name
        if log_path.exists() and current_time - log_path.stat().st_mtime <= recent_seconds:
            return True
    return False


def worker_status_label(
    *,
    run: object,
    alive: bool,
    final_message_exists: bool,
    exit_code: str,
    log_recent: bool,
) -> str:
    """Return a concise worker process state label."""
    if alive:
        return "running"
    if final_message_exists:
        return "finished"
    if exit_code != "-":
        return "exited"
    if log_recent:
        return "active"
    if isinstance(run, dict) and isinstance(typing.cast("JsonObject", run).get("pid"), int):
        return "stale"
    return "-"


def lease_status_label(lease: object) -> str:
    """Return a concise lease display value."""
    if not isinstance(lease, dict) or not lease:
        return "-"
    typed_lease = typing.cast("JsonObject", lease)
    owner = str(typed_lease.get("owner", "-"))
    if lease_is_expired(lease):
        return f"expired:{owner}"
    return owner


def worktree_status_label(repository_directory: Path, task: JsonObject, *, check_worktree: bool) -> str:
    """Return a concise worktree state label."""
    if not check_worktree:
        return "skipped"
    worktree_path = resolve_manifest_path(repository_directory, str(task["worktree"]))
    if not worktree_path.exists():
        return "missing"
    git_status = run_command(["git", "status", "--short"], cwd=worktree_path)
    if git_status.returncode != 0:
        return "unknown"
    return "dirty" if git_status.stdout.strip() else "clean"


def collect_status_rows(
    repository_directory: Path,
    manifest: JsonObject,
    state: JsonObject,
    tasks: list[JsonObject],
    *,
    check_worktrees: bool,
) -> list[StatusRow]:
    """Collect status rows for selected tasks."""
    runs = typing.cast("JsonObject", state.get("runs", {}))
    leases = typing.cast("JsonObject", state.get("leases", {}))
    rows: list[StatusRow] = []
    for task in tasks:
        task_identifier = task_key(task)
        run = runs.get(task_identifier, {})
        pid = run.get("pid") if isinstance(run, dict) else None
        alive = isinstance(pid, int) and running_process_exists(pid)
        run_directory = task_run_directory(repository_directory, manifest, task)
        final_message_exists = (run_directory / "worker-final.md").exists()
        exit_code = read_exit_code_text(run_directory / "exit-code.txt", run)
        lease = leases.get(task_identifier, {})
        rows.append(
            StatusRow(
                task_identifier=task_display_id(task),
                status=runtime_task_status(state, task),
                worker=worker_status_label(
                    run=run,
                    alive=alive,
                    final_message_exists=final_message_exists,
                    exit_code=exit_code,
                    log_recent=worker_log_recent(run_directory),
                ),
                final="yes" if final_message_exists else "no",
                exit_code=exit_code,
                worktree=worktree_status_label(repository_directory, task, check_worktree=check_worktrees),
                lease=lease_status_label(lease),
                title=str(task["title"]),
            )
        )
    return rows


def format_status_snapshot(rows: list[StatusRow], *, updated_at: str | None = None) -> str:
    """Return a formatted status snapshot."""
    lines = [
        f"updated={updated_at or utc_timestamp()}",
        f"statuses: {status_count_text(rows, 'status')}",
        f"workers: {status_count_text(rows, 'worker')}",
        f"{'task':6}  {'status':14}  {'worker':8}  {'final':5}  {'exit':4}  {'worktree':8}  {'lease':24}  title",
    ]
    for row in rows:
        lines.append(
            f"{row.task_identifier:6}  "
            f"{truncate_status_cell(row.status, 14):14}  "
            f"{truncate_status_cell(row.worker, 8):8}  "
            f"{row.final:5}  "
            f"{truncate_status_cell(row.exit_code, 4):4}  "
            f"{truncate_status_cell(row.worktree, 8):8}  "
            f"{truncate_status_cell(row.lease, 24):24}  "
            f"{row.title}"
        )
    return "\n".join(lines)


def should_check_status_worktrees(arguments: argparse.Namespace) -> bool:
    """Return whether status should probe worktree dirtiness."""
    if bool(getattr(arguments, "no_worktree_check", False)):
        return False
    return not (bool(getattr(arguments, "watch", False)) and not bool(getattr(arguments, "check_worktrees", False)))


def print_status_snapshot(arguments: argparse.Namespace) -> None:
    """Print one task-farm status snapshot."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    refresh_runtime_statuses(repository_directory, manifest)
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    rows = collect_status_rows(
        repository_directory,
        manifest,
        state,
        selected_tasks(manifest, typing.cast("list[object]", arguments.task)),
        check_worktrees=should_check_status_worktrees(arguments),
    )
    print(format_status_snapshot(rows))


def command_status(arguments: argparse.Namespace) -> int:
    """Handle status."""
    if not bool(getattr(arguments, "watch", False)):
        print_status_snapshot(arguments)
        return 0
    interval_seconds = float(getattr(arguments, "interval", 10.0))
    try:
        while True:
            if sys.stdout.isatty():
                print("\x1b[2J\x1b[H", end="")
            print_status_snapshot(arguments)
            sys.stdout.flush()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        return 130
    return 0


def command_review(arguments: argparse.Namespace) -> int:
    """Handle review."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    refresh_runtime_statuses(repository_directory, manifest)
    defaults = typing.cast("JsonObject", manifest["defaults"])
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    tasks = selected_tasks(manifest, typing.cast("list[object]", arguments.task))
    if not tasks:
        raise ValueError("Select at least one task to review.")
    ensure_tasks_have_status(
        state,
        tasks,
        {TaskStatus.IMPLEMENTED.value, TaskStatus.REVIEWED.value},
        "review",
    )
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
        run_directory = task_run_directory(repository_directory, manifest, task)
        run_directory.mkdir(parents=True, exist_ok=True)
        final_message_path = task_log_path(repository_directory, manifest, task, "review_final", "review.md")
        jsonl_log_path = task_log_path(repository_directory, manifest, task, "review_jsonl", "review.jsonl")
        stderr_log_path = task_log_path(repository_directory, manifest, task, "review_stderr", "review.stderr.log")
        for log_path in [final_message_path, jsonl_log_path, stderr_log_path]:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        remove_stale_output_files([final_message_path, jsonl_log_path, stderr_log_path])
        command_arguments = build_review_command(
            worktree_path=worktree_path,
            model=str(defaults.get("reviewer_model", "gpt-5.5")),
            reasoning_effort=str(defaults.get("reviewer_reasoning_effort", "xhigh")),
            final_message_path=final_message_path,
        )
        completed_process = subprocess.run(
            command_arguments,
            cwd=repository_directory,
            env=subprocess_environment(repository_directory),
            input=build_review_prompt(task),
            check=False,
            capture_output=True,
            text=True,
        )
        jsonl_log_path.write_text(completed_process.stdout)
        if completed_process.stderr:
            stderr_log_path.write_text(completed_process.stderr)
        if completed_process.returncode == 0:
            review_text = final_message_path.read_text() if final_message_path.exists() else ""
            review_status = classify_review_decision(review_text)
            set_runtime_task_status(state, task, review_status)
            review_results = typing.cast("JsonObject", state.setdefault("review_results", {}))
            review_results[task_key(task)] = {
                "status": review_status.value,
                "reviewed_at": utc_timestamp(),
                "final_message_path": str(final_message_path),
            }
            print(f"Reviewed task {task_display_id(task)}: {review_status.value}: {final_message_path}")
        else:
            exit_code = completed_process.returncode
            print(f"Review failed for task {task_display_id(task)}; see {jsonl_log_path}.", file=sys.stderr)
        save_state(state_path, state)
    return exit_code


def command_integrate(arguments: argparse.Namespace) -> int:
    """Handle integrate."""
    repository_directory = repository_root()
    manifest_relative_path = argument_manifest_path(arguments)
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    refresh_runtime_statuses(repository_directory, manifest)
    defaults = typing.cast("JsonObject", manifest["defaults"])
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    tasks = selected_tasks(manifest, typing.cast("list[object]", arguments.task))
    if not tasks:
        raise ValueError("Select at least one task to integrate.")
    allowed_statuses = {TaskStatus.REVIEWED.value}
    if bool(getattr(arguments, "allow_unreviewed", False)):
        allowed_statuses.add(TaskStatus.IMPLEMENTED.value)
    ensure_tasks_have_status(state, tasks, allowed_statuses, "integrate")
    ensure_dependencies_ready(
        manifest,
        state,
        tasks,
        {TaskStatus.INTEGRATED.value, TaskStatus.MERGED.value},
        "integrate",
    )
    dangerously_bypass_approvals = bool(
        arguments.dangerous or defaults.get("integrator_dangerously_bypass_approvals", False),
    )
    exit_code = 0
    with manifest_lock(repository_directory, manifest, "integration.lock"):
        integration_worktree_path = ensure_integration_worktree(repository_directory, defaults)
        integration_branch = str(defaults.get("integration_branch", DEFAULT_INTEGRATION_BRANCH))
        ensure_integration_worktree_ready(integration_worktree_path, integration_branch)
        for task in tasks:
            worktree_path = resolve_manifest_path(repository_directory, str(task["worktree"]))
            if not worktree_path.exists():
                raise FileNotFoundError(f"Worktree does not exist: {worktree_path}")
            ensure_integration_worktree_ready(integration_worktree_path, integration_branch)
            head_before_integration = git_head_commit(integration_worktree_path)
            run_directory = task_run_directory(repository_directory, manifest, task)
            run_directory.mkdir(parents=True, exist_ok=True)
            final_message_path = task_log_path(
                repository_directory,
                manifest,
                task,
                "integration_final",
                "integration-final.md",
            )
            jsonl_log_path = task_log_path(
                repository_directory,
                manifest,
                task,
                "integration_jsonl",
                "integration.jsonl",
            )
            stderr_log_path = task_log_path(
                repository_directory,
                manifest,
                task,
                "integration_stderr",
                "integration.stderr.log",
            )
            review_report_path = task_log_path(repository_directory, manifest, task, "review_final", "review.md")
            for log_path in [final_message_path, jsonl_log_path, stderr_log_path]:
                log_path.parent.mkdir(parents=True, exist_ok=True)
            remove_stale_output_files([final_message_path, jsonl_log_path, stderr_log_path])
            command_arguments = build_integration_command(
                integration_worktree_path=integration_worktree_path,
                worktree_path=worktree_path,
                git_metadata_path=repository_directory / ".git",
                model=str(defaults.get("integrator_model", "gpt-5.5")),
                reasoning_effort=str(defaults.get("integrator_reasoning_effort", "xhigh")),
                final_message_path=final_message_path,
                dangerously_bypass_approvals=dangerously_bypass_approvals,
            )
            set_runtime_task_status(state, task, TaskStatus.INTEGRATING)
            task["status"] = TaskStatus.INTEGRATING.value
            save_state(state_path, state)
            save_manifest(repository_directory, manifest, manifest_relative_path=manifest_relative_path)
            completed_process = subprocess.run(
                command_arguments,
                cwd=integration_worktree_path,
                env=subprocess_environment(integration_worktree_path),
                input=build_integration_prompt(task, review_report_path, integration_branch),
                check=False,
                capture_output=True,
                text=True,
            )
            jsonl_log_path.write_text(completed_process.stdout)
            if completed_process.stderr:
                stderr_log_path.write_text(completed_process.stderr)
            if completed_process.returncode == 0:
                try:
                    ensure_integration_worktree_ready(integration_worktree_path, integration_branch)
                    head_after_integration = git_head_commit(integration_worktree_path)
                    if bool(defaults.get("push_integration_branch", False)):
                        push_integration_branch(integration_worktree_path, integration_branch)
                except (RuntimeError, ValueError) as error:
                    set_runtime_task_status(state, task, TaskStatus.BLOCKED)
                    task["status"] = TaskStatus.BLOCKED.value
                    exit_code = 1
                    print(
                        f"Integration produced an invalid worktree for task {task_display_id(task)}: {error}",
                        file=sys.stderr,
                    )
                    save_state(state_path, state)
                    save_manifest(repository_directory, manifest, manifest_relative_path=manifest_relative_path)
                    break
                if head_after_integration == head_before_integration:
                    set_runtime_task_status(state, task, TaskStatus.BLOCKED)
                    task["status"] = TaskStatus.BLOCKED.value
                    exit_code = 1
                    print(
                        f"Integration for task {task_display_id(task)} returned success but did not advance HEAD.",
                        file=sys.stderr,
                    )
                    save_state(state_path, state)
                    save_manifest(repository_directory, manifest, manifest_relative_path=manifest_relative_path)
                    break
                set_runtime_task_status(state, task, TaskStatus.INTEGRATED)
                clear_task_lease(state, task)
                task["status"] = TaskStatus.INTEGRATED.value
                runtime_metadata = typing.cast("JsonObject", task.setdefault("runtime", {}))
                runtime_metadata["integrated_at"] = utc_timestamp()
                runtime_metadata["integration_head"] = head_after_integration
                print(f"Integrated task {task_display_id(task)}: {final_message_path}")
            else:
                set_runtime_task_status(state, task, TaskStatus.BLOCKED)
                task["status"] = TaskStatus.BLOCKED.value
                exit_code = completed_process.returncode
                print(f"Integration failed for task {task_display_id(task)}; see {jsonl_log_path}.", file=sys.stderr)
                break
            save_state(state_path, state)
            save_manifest(repository_directory, manifest, manifest_relative_path=manifest_relative_path)
    return exit_code


def command_integrate_ready(arguments: argparse.Namespace) -> int:
    """Handle integrate-ready."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    refresh_runtime_statuses(repository_directory, manifest)
    state_path = state_directory(repository_directory, manifest) / "state.json"
    state = load_state(state_path)
    allowed_statuses = {TaskStatus.REVIEWED.value}
    if bool(arguments.allow_unreviewed):
        allowed_statuses.add(TaskStatus.IMPLEMENTED.value)
    ready_identifiers = [
        task["id"]
        for task in selected_tasks(manifest, [])
        if runtime_task_status(state, task) in allowed_statuses
    ]
    if not ready_identifiers:
        print("No reviewed tasks are ready to integrate.")
        return 0
    arguments.task = ready_identifiers
    return command_integrate(arguments)


def command_claim(arguments: argparse.Namespace) -> int:
    """Handle claim."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    refresh_runtime_statuses(repository_directory, manifest)
    defaults = typing.cast("JsonObject", manifest["defaults"])
    state_path = state_directory(repository_directory, manifest) / "state.json"
    owner = str(getattr(arguments, "owner", None) or lease_owner())
    lease_seconds = int(
        getattr(arguments, "lease_seconds", None) or defaults.get("lease_seconds", DEFAULT_LEASE_SECONDS),
    )
    jobs = int(getattr(arguments, "jobs", None) or 1)
    claimed_tasks: list[JsonObject] = []
    with manifest_lock(repository_directory, manifest, "claim.lock"):
        state = load_state(state_path)
        reset_stale_task_leases(state, manifest)
        candidates = selected_tasks(manifest, typing.cast("list[object]", arguments.task))
        active_groups = active_conflict_groups(state, manifest)
        selected_groups: set[str] = set()
        for task in candidates:
            if len(claimed_tasks) >= jobs:
                break
            if not bool(task.get("enabled", True)):
                continue
            if not bool(arguments.force) and runtime_task_status(state, task) != TaskStatus.READY.value:
                continue
            if not conflict_group_available(task, active_groups=active_groups, selected_groups=selected_groups):
                continue
            claimed_tasks.append(task)
            if "conflict_group" in task:
                selected_groups.add(str(task["conflict_group"]))
        if claimed_tasks:
            ensure_dependencies_ready(
                manifest,
                state,
                claimed_tasks,
                {TaskStatus.INTEGRATED.value, TaskStatus.MERGED.value},
                "claim",
            )
        for task in claimed_tasks:
            set_task_lease(
                state,
                task,
                owner=owner,
                status=TaskStatus.CLAIMED,
                lease_seconds=lease_seconds,
            )
            print(f"Claimed task {task_display_id(task)}: {task['branch']}")
        save_state(state_path, state)
    if not claimed_tasks:
        print("No claimable tasks selected.")
    return 0


def set_tasks_status_command(arguments: argparse.Namespace, status: TaskStatus) -> int:
    """Set selected tasks to a terminal manual status."""
    repository_directory = repository_root()
    manifest_relative_path = argument_manifest_path(arguments)
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    state_path = state_directory(repository_directory, manifest) / "state.json"
    tasks = selected_tasks(manifest, typing.cast("list[object]", arguments.task))
    if not tasks:
        raise ValueError("Select at least one task.")
    with manifest_lock(repository_directory, manifest, "claim.lock"):
        state = load_state(state_path)
        for task in tasks:
            set_runtime_task_status(state, task, status)
            clear_task_lease(state, task)
            task["status"] = status.value
            runtime_metadata = typing.cast("JsonObject", task.setdefault("runtime", {}))
            runtime_metadata[f"{status.value}_at"] = utc_timestamp()
            reason = getattr(arguments, "reason", None)
            if reason:
                manual_metadata = typing.cast("JsonObject", task.setdefault("manual", {}))
                manual_metadata["notes"] = str(reason)
            print(f"{status.value}: {task_display_id(task)} {task['title']}")
        save_state(state_path, state)
        save_manifest(repository_directory, manifest, manifest_relative_path=manifest_relative_path)
    return 0


def command_block(arguments: argparse.Namespace) -> int:
    """Handle block."""
    return set_tasks_status_command(arguments, TaskStatus.BLOCKED)


def command_abandon(arguments: argparse.Namespace) -> int:
    """Handle abandon."""
    return set_tasks_status_command(arguments, TaskStatus.ABANDONED)


def command_reset_claim(arguments: argparse.Namespace) -> int:
    """Handle reset-claim."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    state_path = state_directory(repository_directory, manifest) / "state.json"
    with manifest_lock(repository_directory, manifest, "claim.lock"):
        state = load_state(state_path)
        tasks = selected_tasks(manifest, typing.cast("list[object]", arguments.task))
        if tasks:
            reset_identifiers: list[str] = []
            for task in tasks:
                status = runtime_task_status(state, task)
                lease = typing.cast("JsonObject", state.setdefault("leases", {})).get(task_key(task))
                if not bool(arguments.force) and status not in {TaskStatus.CLAIMED.value, TaskStatus.RUNNING.value}:
                    continue
                if not bool(arguments.force) and not lease_is_expired(lease):
                    continue
                clear_task_lease(state, task)
                set_runtime_task_status(state, task, TaskStatus.READY)
                reset_identifiers.append(task_key(task))
        else:
            reset_identifiers = reset_stale_task_leases(state, manifest, force=bool(arguments.force))
        save_state(state_path, state)
    for task_identifier in reset_identifiers:
        print(f"Reset claim: {task_identifier}")
    if not reset_identifiers:
        print("No stale claims reset.")
    return 0


def command_diff(arguments: argparse.Namespace) -> int:
    """Handle diff."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    defaults = typing.cast("JsonObject", manifest["defaults"])
    base_branch = str(defaults.get("base_branch", "main"))
    tasks = selected_tasks(manifest, typing.cast("list[object]", arguments.task))
    if not tasks:
        raise ValueError("Select at least one task to diff.")
    exit_code = 0
    for task in tasks:
        branch = str(task["branch"])
        command_arguments = ["git", "diff", "--stat", f"{base_branch}..{branch}"]
        if bool(arguments.patch):
            command_arguments = ["git", "diff", f"{base_branch}..{branch}"]
        command_result = run_command(command_arguments, cwd=repository_directory)
        if command_result.returncode != 0:
            exit_code = command_result.returncode
            print(command_result.stderr.strip(), file=sys.stderr)
            continue
        print(f"# {task_display_id(task)} {branch}")
        print(command_result.stdout.rstrip())
    return exit_code


def command_log(arguments: argparse.Namespace) -> int:
    """Handle log."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    tasks = selected_tasks(manifest, typing.cast("list[object]", arguments.task))
    if not tasks:
        raise ValueError("Select at least one task to inspect logs.")
    for task in tasks:
        print(f"# {task_display_id(task)} {task['title']}")
        for name, log_path_text in sorted(task_logs(task).items()):
            log_path = resolve_manifest_path(repository_directory, str(log_path_text))
            exists = "exists" if log_path.exists() else "missing"
            print(f"{name}: {log_path} ({exists})")
            if bool(arguments.cat) and log_path.exists() and log_path.is_file():
                print(log_path.read_text())
    return 0


def integration_branch_is_pushed(integration_worktree_path: Path, integration_branch: str) -> bool:
    """Return whether the local integration branch matches origin."""
    local_result = run_command(["git", "rev-parse", integration_branch], cwd=integration_worktree_path)
    remote_result = run_command(["git", "rev-parse", f"origin/{integration_branch}"], cwd=integration_worktree_path)
    return (
        local_result.returncode == 0
        and remote_result.returncode == 0
        and local_result.stdout.strip() == remote_result.stdout.strip()
    )


def command_clean_integrated(arguments: argparse.Namespace) -> int:
    """Handle clean-integrated."""
    repository_directory = repository_root()
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    defaults = typing.cast("JsonObject", manifest["defaults"])
    integration_branch = str(defaults.get("integration_branch", DEFAULT_INTEGRATION_BRANCH))
    integration_worktree_path = ensure_integration_worktree(repository_directory, defaults)
    if not bool(arguments.skip_push_check) and not integration_branch_is_pushed(
        integration_worktree_path,
        integration_branch,
    ):
        print(f"Integration branch {integration_branch} is not confirmed pushed; cleanup skipped.")
        return 1
    state = load_state(state_directory(repository_directory, manifest) / "state.json")
    tasks = [
        task
        for task in selected_tasks(manifest, typing.cast("list[object]", arguments.task))
        if runtime_task_status(state, task) == TaskStatus.INTEGRATED.value
    ]
    for task in tasks:
        worktree_path = resolve_manifest_path(repository_directory, str(task["worktree"]))
        if not worktree_path.exists():
            continue
        if bool(arguments.dry_run):
            print(f"Would remove worktree {worktree_path}")
            continue
        ensure_success(run_command(["git", "worktree", "remove", str(worktree_path)], cwd=repository_directory))
        print(f"Removed worktree {worktree_path}")
    return 0


def command_promote_to_main(arguments: argparse.Namespace) -> int:
    """Handle promote-to-main."""
    repository_directory = repository_root()
    manifest_relative_path = argument_manifest_path(arguments)
    manifest = load_manifest_for_arguments(repository_directory, arguments)
    defaults = typing.cast("JsonObject", manifest["defaults"])
    base_branch = str(defaults.get("base_branch", "main"))
    integration_branch = str(defaults.get("integration_branch", DEFAULT_INTEGRATION_BRANCH))
    with manifest_lock(repository_directory, manifest, "integration.lock"):
        if git_current_branch(repository_directory) != base_branch:
            raise ValueError(f"Main checkout must be on {base_branch} before promotion.")
        if not git_worktree_is_clean(repository_directory):
            raise ValueError("Main checkout has uncommitted changes.")
        if bool(arguments.dry_run):
            command_result = run_command(
                ["git", "merge-base", "--is-ancestor", base_branch, integration_branch],
                cwd=repository_directory,
            )
        else:
            command_result = run_command(["git", "merge", "--ff-only", integration_branch], cwd=repository_directory)
        ensure_success(command_result)
        if bool(arguments.push) and not bool(arguments.dry_run):
            ensure_success(run_command(["git", "push", "origin", base_branch], cwd=repository_directory))
        if not bool(arguments.dry_run):
            state_path = state_directory(repository_directory, manifest) / "state.json"
            state = load_state(state_path)
            for task in selected_tasks(manifest, []):
                if runtime_task_status(state, task) == TaskStatus.INTEGRATED.value:
                    set_runtime_task_status(state, task, TaskStatus.MERGED)
                    task["status"] = TaskStatus.MERGED.value
            save_state(state_path, state)
            save_manifest(repository_directory, manifest, manifest_relative_path=manifest_relative_path)
    print(f"Promoted {integration_branch} to {base_branch}.")
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Manage Codex task worktrees generated from review manifests.")
    parser.add_argument(
        "--manifest",
        default=REPO_RELATIVE_MANIFEST_PATH.as_posix(),
        help="Repository-relative manifest path.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync-manifest", help="Regenerate docs/code-review.tasks.json.")
    sync_parser.add_argument(
        "--manifest",
        default=argparse.SUPPRESS,
        help="Repository-relative manifest path to write.",
    )
    sync_parser.add_argument("--source", help="Repository-relative markdown source path.")
    sync_parser.add_argument("--plan", help="Repository-relative shared plan path.")
    sync_parser.add_argument("--state-dir", help="Repository-relative runtime state directory.")
    sync_parser.add_argument("--branch-prefix", help="Generated task branch prefix.")
    sync_parser.add_argument("--worktree-prefix", help="Generated task worktree prefix.")
    sync_parser.add_argument("--integration-branch", help="Integration branch.")
    sync_parser.add_argument("--integration-worktree", help="Integration worktree path.")
    sync_parser.set_defaults(handler=command_sync_manifest)

    doctor_parser = subparsers.add_parser("doctor", help="Check task-farm prerequisites.")
    doctor_parser.add_argument("--strict", action="store_true", help="Treat missing optional tools as failures.")
    doctor_parser.set_defaults(handler=command_doctor)

    list_parser = subparsers.add_parser("list", help="List tasks from the manifest.")
    list_parser.add_argument("--task", action="append", default=[], help="Task id to list.")
    list_parser.set_defaults(handler=command_list)

    claim_parser = subparsers.add_parser("claim", help="Claim ready tasks without launching workers.")
    claim_parser.add_argument("--task", action="append", default=[], help="Task id to claim.")
    claim_parser.add_argument("--jobs", type=int, default=1, help="Maximum number of tasks to claim.")
    claim_parser.add_argument("--owner", help="Lease owner.")
    claim_parser.add_argument("--lease-seconds", type=int, help="Lease duration in seconds.")
    claim_parser.add_argument("--force", action="store_true", help="Claim regardless of recorded status.")
    claim_parser.set_defaults(handler=command_claim)

    run_parser = subparsers.add_parser("run", help="Launch worker agents.")
    run_parser.add_argument("--task", action="append", default=[], help="Task id to run.")
    run_parser.add_argument("--jobs", type=int, help="Maximum number of tasks to launch.")
    run_parser.add_argument("--wait", action="store_true", help="Wait for launched workers to finish.")
    run_parser.add_argument("--force", action="store_true", help="Launch tasks regardless of recorded status.")
    run_parser.add_argument("--owner", help="Lease owner.")
    run_parser.add_argument("--lease-seconds", type=int, help="Lease duration in seconds.")
    run_parser.add_argument("--dangerous", action="store_true", help="Allow workers to bypass approvals and sandbox.")
    run_parser.set_defaults(handler=command_run)

    status_parser = subparsers.add_parser("status", help="Show worker and worktree status.")
    status_parser.add_argument("--task", action="append", default=[], help="Task id to inspect.")
    status_parser.add_argument("--watch", action="store_true", help="Refresh status until interrupted.")
    status_parser.add_argument("--interval", type=float, default=10.0, help="Watch refresh interval in seconds.")
    status_parser.add_argument(
        "--no-worktree-check",
        action="store_true",
        help="Skip git status checks for task worktrees.",
    )
    status_parser.add_argument(
        "--check-worktrees",
        action="store_true",
        help="Probe worktree dirtiness while watching.",
    )
    status_parser.set_defaults(handler=command_status)

    review_parser = subparsers.add_parser("review", help="Run Codex review for task branches.")
    review_parser.add_argument("task", nargs="+", help="Task id to review.")
    review_parser.set_defaults(handler=command_review)

    integrate_parser = subparsers.add_parser("integrate", help="Run the main integration agent.")
    integrate_parser.add_argument("task", nargs="+", help="Task id to integrate.")
    integrate_parser.add_argument(
        "--dangerous",
        action="store_true",
        help="Allow integrator to bypass approvals and sandbox.",
    )
    integrate_parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Allow implemented tasks without reviewed status.",
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

    diff_parser = subparsers.add_parser("diff", help="Show task branch diffs.")
    diff_parser.add_argument("task", nargs="+", help="Task id to diff.")
    diff_parser.add_argument("--patch", action="store_true", help="Show full patch instead of stat.")
    diff_parser.set_defaults(handler=command_diff)

    log_parser = subparsers.add_parser("log", help="Show task runtime log paths.")
    log_parser.add_argument("task", nargs="+", help="Task id to inspect.")
    log_parser.add_argument("--cat", action="store_true", help="Print existing log file contents.")
    log_parser.set_defaults(handler=command_log)

    block_parser = subparsers.add_parser("block", help="Mark tasks blocked.")
    block_parser.add_argument("task", nargs="+", help="Task id to block.")
    block_parser.add_argument("--reason", help="Manual note explaining the blocker.")
    block_parser.set_defaults(handler=command_block)

    abandon_parser = subparsers.add_parser("abandon", help="Mark tasks abandoned.")
    abandon_parser.add_argument("task", nargs="+", help="Task id to abandon.")
    abandon_parser.add_argument("--reason", help="Manual note explaining abandonment.")
    abandon_parser.set_defaults(handler=command_abandon)

    reset_claim_parser = subparsers.add_parser("reset-claim", help="Reset stale task claims.")
    reset_claim_parser.add_argument("task", nargs="*", help="Task id to reset.")
    reset_claim_parser.add_argument("--force", action="store_true", help="Reset selected claims even if not stale.")
    reset_claim_parser.set_defaults(handler=command_reset_claim)

    clean_parser = subparsers.add_parser("clean-integrated", help="Remove worktrees for integrated tasks.")
    clean_parser.add_argument("task", nargs="*", help="Task id to clean.")
    clean_parser.add_argument("--dry-run", action="store_true", help="Print worktrees without removing them.")
    clean_parser.add_argument("--skip-push-check", action="store_true", help="Skip origin push verification.")
    clean_parser.set_defaults(handler=command_clean_integrated)

    promote_parser = subparsers.add_parser("promote-to-main", help="Fast-forward main to the integration branch.")
    promote_parser.add_argument("--dry-run", action="store_true", help="Verify fast-forward eligibility only.")
    promote_parser.add_argument("--push", action="store_true", help="Push main after promotion.")
    promote_parser.set_defaults(handler=command_promote_to_main)
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

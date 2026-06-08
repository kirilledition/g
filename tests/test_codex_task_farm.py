from __future__ import annotations

import argparse
import importlib
import sys
import typing
from pathlib import Path

import pytest

if typing.TYPE_CHECKING:
    import subprocess

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIRECTORY))
codex_task_farm = importlib.import_module("scripts.codex_task_farm")


class FakeWorkerProcess:
    def __init__(self, process_identifier: int, returncode: int, launch_log: list[int]) -> None:
        self.pid = process_identifier
        self.returncode = returncode
        self.launch_log = launch_log

    def wait(self) -> int:
        assert len(self.launch_log) == 2
        return self.returncode


def test_parse_review_tasks_ignores_headings_inside_fenced_code() -> None:
    markdown = """# Category

## 7. First task

```python
# this is not a category
```

**Guidance**

Use `src/g/example.py`.

---

## 8. Second task

No guidance.
"""

    tasks = codex_task_farm.parse_review_tasks(markdown)

    assert [task.identifier for task in tasks] == [7, 8]
    assert tasks[0].category == "Category"
    assert tasks[0].guidance_markdown == "Use `src/g/example.py`."
    assert tasks[0].expected_paths == ["src/g/example.py"]
    assert tasks[1].guidance_markdown == ""


def test_expected_paths_strip_line_suffixes_and_skip_globs() -> None:
    markdown = """# Category

## 1. Paths

Use `src/g/current.py:12-20`, `tests/test_current.py:7`, `src/g/compute/*`, and `/tmp/outside`.
"""

    tasks = codex_task_farm.parse_review_tasks(markdown)

    assert tasks[0].expected_paths == ["src/g/current.py", "tests/test_current.py"]


def test_sync_manifest_preserves_only_manual_metadata(tmp_path: Path) -> None:
    docs_directory = tmp_path / "docs"
    scratchpad_directory = docs_directory / "scratchpad"
    scratchpad_directory.mkdir(parents=True)
    (scratchpad_directory / "code-review.md").write_text(
        """# Category

## 7. First task

Body with `src/g/current.py`.
"""
    )
    (scratchpad_directory / "code-review.tasks.json").write_text(
        """{
  "defaults": {
    "jobs": 2,
    "worktree_root": "../custom-worktrees"
  },
  "source_path": "docs/scratchpad/code-review.md",
  "tasks": [
    {
      "id": 7,
      "status": "blocked",
      "dependencies": [3],
      "enabled": false,
      "assignee": "agent",
      "notes": "manual",
      "manual_expected_paths": ["src/g/manual.py"],
      "expected_paths": ["src/g/stale.py"],
      "priority": "stale",
      "unexpected": "drop"
    }
  ],
  "version": 1
}
"""
    )

    manifest = codex_task_farm.sync_manifest(tmp_path)
    task = manifest["tasks"][0]

    assert manifest["defaults"]["jobs"] == 2
    assert task["status"] == "blocked"
    assert task["dependencies"] == [3]
    assert task["enabled"] is False
    assert task["assignee"] == "agent"
    assert task["notes"] == "manual"
    assert task["manual_expected_paths"] == ["src/g/manual.py"]
    assert task["expected_paths"] == ["src/g/current.py"]
    assert task["priority"] == "Category"
    assert "unexpected" not in task
    assert task["branch"] == "codex/review-07-first-task"
    assert task["worktree"] == "../custom-worktrees/review-07-first-task"


def test_review2_sync_uses_isolated_v2_manifest_and_prefixes(tmp_path: Path) -> None:
    docs_directory = tmp_path / "docs"
    scratchpad_directory = docs_directory / "scratchpad"
    scratchpad_directory.mkdir(parents=True)
    (scratchpad_directory / "02.code-review-2-06-26.md").write_text(
        """# Category

## 1. First task

Body with `src/g/compute/regenie2_binary/api.py`.
"""
    )

    manifest = codex_task_farm.sync_manifest(
        tmp_path,
        manifest_relative_path=Path("docs/scratchpad/code-review-2.tasks.json"),
        source_relative_path=Path("docs/scratchpad/02.code-review-2-06-26.md"),
        plan_relative_path=Path("docs/scratchpad/code-review-2-plan.md"),
        state_directory_path=Path(".codex-task-worktrees/code-review-2"),
        branch_prefix="codex/review2-",
        worktree_prefix="../g-worktrees/review2-",
        integration_branch="integration/code-review-2",
        integration_worktree="../g-worktrees/integration-code-review-2",
    )

    task = manifest["tasks"][0]
    assert manifest["version"] == 2
    assert manifest["source_path"] == "docs/scratchpad/02.code-review-2-06-26.md"
    assert manifest["defaults"]["plan_path"] == "docs/scratchpad/code-review-2-plan.md"
    assert manifest["defaults"]["state_directory"] == ".codex-task-worktrees/code-review-2"
    assert manifest["defaults"]["integration_branch"] == "integration/code-review-2"
    assert manifest["defaults"]["push_integration_branch"] is True
    assert task["id"] == "T001"
    assert task["branch"] == "codex/review2-T001-first-task"
    assert task["worktree"] == "../g-worktrees/review2-T001-first-task"
    assert task["conflict_group"] == "binary-jax"
    assert task["logs"]["run_directory"] == ".codex-task-worktrees/code-review-2/runs/T001"
    assert (scratchpad_directory / "code-review-2.tasks.json").exists()
    assert (scratchpad_directory / "code-review-2-plan.md").read_text().startswith("# Code Review 2 Task Plan")
    assert not (scratchpad_directory / "code-review.tasks.json").exists()


def test_review2_resync_preserves_manual_and_runtime_metadata(tmp_path: Path) -> None:
    docs_directory = tmp_path / "docs"
    scratchpad_directory = docs_directory / "scratchpad"
    scratchpad_directory.mkdir(parents=True)
    source_path = scratchpad_directory / "02.code-review-2-06-26.md"
    source_path.write_text(
        """# New Category

## 1. New title

Body with `src/g/compute/regenie2_linear/api.py`.
"""
    )
    (scratchpad_directory / "code-review-2.tasks.json").write_text(
        """{
  "defaults": {
    "branch_prefix": "codex/review2-",
    "id_style": "string",
    "integration_branch": "integration/code-review-2",
    "integration_worktree": "../g-worktrees/integration-code-review-2",
    "state_directory": ".codex-task-worktrees/code-review-2",
    "worktree_prefix": "../g-worktrees/review2-"
  },
  "source_path": "docs/scratchpad/02.code-review-2-06-26.md",
  "tasks": [
    {
      "id": "T001",
      "source_id": 1,
      "status": "blocked",
      "enabled": false,
      "dependencies": ["T000"],
      "assignee": "agent",
      "branch": "stale-branch",
      "worktree": "stale-worktree",
      "conflict_group": "manual-group",
      "conflict_group_source": "manual",
      "logs": {"run_directory": "custom/logs/T001"},
      "manual": {"notes": "keep", "conflict_group": "manual-group"},
      "runtime": {"started_at": "2026-06-02T00:00:00+00:00"}
    }
  ],
  "version": 2
}
"""
    )

    manifest = codex_task_farm.sync_manifest(
        tmp_path,
        manifest_relative_path=Path("docs/scratchpad/code-review-2.tasks.json"),
        source_relative_path=Path("docs/scratchpad/02.code-review-2-06-26.md"),
        branch_prefix="codex/review2-",
        worktree_prefix="../g-worktrees/review2-",
    )

    task = manifest["tasks"][0]
    assert task["status"] == "blocked"
    assert task["enabled"] is False
    assert task["dependencies"] == ["T000"]
    assert task["assignee"] == "agent"
    assert task["manual"]["notes"] == "keep"
    assert task["conflict_group"] == "manual-group"
    assert task["runtime"] == {"started_at": "2026-06-02T00:00:00+00:00"}
    assert task["logs"] == {"run_directory": "custom/logs/T001"}
    assert task["title"] == "New title"
    assert task["priority"] == "New Category"
    assert task["expected_paths"] == ["src/g/compute/regenie2_linear/api.py"]
    assert task["branch"] == "codex/review2-T001-new-title"
    assert task["worktree"] == "../g-worktrees/review2-T001-new-title"


def test_load_manifest_requires_explicit_sync(tmp_path: Path) -> None:
    (tmp_path / "docs" / "scratchpad").mkdir(parents=True)
    (tmp_path / "docs" / "scratchpad" / "code-review.md").write_text("")

    with pytest.raises(ValueError, match="sync-manifest"):
        codex_task_farm.load_manifest(tmp_path)


def test_codex_command_builders_default_to_safe_execution() -> None:
    worker_command = codex_task_farm.build_worker_command(
        worktree_path=Path("/repo/worktree"),
        git_metadata_path=Path("/repo/.git"),
        model="gpt-5.5",
        reasoning_effort="high",
        final_message_path=Path("/state/final.md"),
        dangerously_bypass_approvals=False,
    )
    review_command = codex_task_farm.build_review_command(
        worktree_path=Path("/repo/worktree"),
        model="gpt-5.5",
        reasoning_effort="xhigh",
        final_message_path=Path("/state/review.md"),
    )
    integration_command = codex_task_farm.build_integration_command(
        integration_worktree_path=Path("/repo/integration"),
        worktree_path=Path("/repo/worktree"),
        git_metadata_path=Path("/repo/.git"),
        model="gpt-5.5",
        reasoning_effort="xhigh",
        final_message_path=Path("/state/integration.md"),
        dangerously_bypass_approvals=False,
    )

    assert "--dangerously-bypass-approvals-and-sandbox" not in worker_command
    assert worker_command[:5] == ["codex", "--cd", "/repo/worktree", "--add-dir", "/repo/.git"]
    assert worker_command[-1] == "-"
    assert review_command == [
        "codex",
        "--cd",
        "/repo/worktree",
        "-m",
        "gpt-5.5",
        "-c",
        'model_reasoning_effort="xhigh"',
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "never",
        "exec",
        "--json",
        "-o",
        "/state/review.md",
        "-",
    ]
    assert "--dangerously-bypass-approvals-and-sandbox" not in review_command
    assert integration_command[:7] == [
        "codex",
        "--cd",
        "/repo/integration",
        "--add-dir",
        "/repo/worktree",
        "--add-dir",
        "/repo/.git",
    ]
    assert "--dangerously-bypass-approvals-and-sandbox" not in integration_command


def test_dangerous_flag_adds_bypass_only_for_worker_and_integrator() -> None:
    worker_command = codex_task_farm.build_worker_command(
        worktree_path=Path("/repo/worktree"),
        git_metadata_path=Path("/repo/.git"),
        model="gpt-5.5",
        reasoning_effort="high",
        final_message_path=Path("/state/final.md"),
        dangerously_bypass_approvals=True,
    )
    integration_command = codex_task_farm.build_integration_command(
        integration_worktree_path=Path("/repo/integration"),
        worktree_path=Path("/repo/worktree"),
        git_metadata_path=Path("/repo/.git"),
        model="gpt-5.5",
        reasoning_effort="xhigh",
        final_message_path=Path("/state/integration.md"),
        dangerously_bypass_approvals=True,
    )

    assert "--dangerously-bypass-approvals-and-sandbox" in worker_command
    assert "--dangerously-bypass-approvals-and-sandbox" in integration_command


def test_review_clears_stale_outputs_before_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["defaults"]["state_directory"] = ".state"
    manifest["tasks"] = [
        {
            "id": 1,
            "status": "implemented",
            "branch": "codex/one",
            "worktree": "worktree",
            "title": "Review task",
            "body_markdown": "Body",
            "guidance_markdown": "Guidance",
            "dependencies": [],
        }
    ]
    (tmp_path / "worktree").mkdir()
    review_directory = tmp_path / ".state" / "runs" / "01"
    review_directory.mkdir(parents=True)
    final_message_path = review_directory / "review.md"
    jsonl_log_path = review_directory / "review.jsonl"
    stderr_log_path = review_directory / "review.stderr.log"
    final_message_path.write_text("old final")
    jsonl_log_path.write_text("old jsonl")
    stderr_log_path.write_text("old stderr")

    class FakeCompletedProcess:
        returncode: int = 0
        stdout: str = '{"type": "ok"}\n'
        stderr: str = ""

    def fake_subprocess_run(
        command_arguments: list[str],
        **subprocess_arguments: object,
    ) -> FakeCompletedProcess:
        assert subprocess_arguments["cwd"] == tmp_path
        assert subprocess_arguments["input"] == codex_task_farm.build_review_prompt(manifest["tasks"][0])
        assert subprocess_arguments["check"] is False
        assert subprocess_arguments["capture_output"] is True
        assert subprocess_arguments["text"] is True
        assert not final_message_path.exists()
        assert not jsonl_log_path.exists()
        assert not stderr_log_path.exists()
        output_flag_index = command_arguments.index("-o")
        Path(command_arguments[output_flag_index + 1]).write_text("new final")
        return FakeCompletedProcess()

    monkeypatch.setattr(codex_task_farm, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(codex_task_farm, "load_manifest", lambda repository_directory: manifest)
    monkeypatch.setattr(codex_task_farm, "refresh_runtime_statuses", lambda repository_directory, manifest: None)
    monkeypatch.setattr(codex_task_farm.subprocess, "run", fake_subprocess_run)

    exit_code = codex_task_farm.command_review(argparse.Namespace(task=[1]))

    state = codex_task_farm.read_json_object(tmp_path / ".state" / "state.json")
    assert exit_code == 0
    assert state["statuses"] == {"1": "reviewed"}
    assert final_message_path.read_text() == "new final"
    assert jsonl_log_path.read_text() == '{"type": "ok"}\n'
    assert not stderr_log_path.exists()


def test_worker_launch_clears_stale_outputs_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["defaults"]["state_directory"] = ".state"
    task = {
        "id": 1,
        "status": "ready",
        "branch": "codex/one",
        "worktree": "worktree",
        "title": "Run task",
        "category": "Category",
        "body_markdown": "Body",
        "guidance_markdown": "Guidance",
        "dependencies": [],
    }
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    run_directory = tmp_path / ".state" / "runs" / "01"
    run_directory.mkdir(parents=True)
    stale_paths = [
        run_directory / "worker-final.md",
        run_directory / "worker.jsonl",
        run_directory / "worker.stderr.log",
        run_directory / "worker-prompt.md",
        run_directory / "worker-wrapper.sh",
        run_directory / "exit-code.txt",
    ]
    for stale_path in stale_paths:
        stale_path.write_text("stale")

    class FakePopen:
        pid = 123

    def fake_popen(*args: object, **kwargs: object) -> FakePopen:
        del args, kwargs
        return FakePopen()

    monkeypatch.setattr(
        codex_task_farm,
        "ensure_task_worktree",
        lambda repository_directory, task, base_branch: worktree_path,
    )
    monkeypatch.setattr(codex_task_farm.subprocess, "Popen", fake_popen)

    launched_worker = codex_task_farm.launch_worker(
        repository_directory=tmp_path,
        manifest=manifest,
        task=task,
        dangerously_bypass_approvals=False,
    )

    assert launched_worker.process_identifier == 123
    assert not (run_directory / "worker-final.md").exists()
    assert (run_directory / "worker-prompt.md").read_text() != "stale"
    assert (run_directory / "worker-wrapper.sh").read_text() != "stale"
    assert not (run_directory / "exit-code.txt").exists()


def test_worker_prompt_uses_runtime_logs_and_forbids_shared_plan_edits() -> None:
    task = {
        "id": "T001",
        "title": "Runtime prompt",
        "category": "Category",
        "branch": "codex/review2-T001-runtime-prompt",
        "body_markdown": "Body",
        "guidance_markdown": "Guidance",
        "expected_paths": ["src/g/example.py"],
        "logs": {"run_directory": ".codex-task-worktrees/code-review-2/runs/T001"},
    }

    prompt = codex_task_farm.build_worker_prompt(task)

    assert ".codex-task-worktrees/code-review-2/runs/T001" in prompt
    assert "Do not edit shared task plans or manifests" in prompt
    assert "docs/scratchpad/code-review-2-plan.md" in prompt
    assert "Final response must include changed files" in prompt


def test_wrapper_exit_code_classification_accepts_verified_and_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["defaults"]["state_directory"] = ".state"
    task = {"id": 7, "worktree": "../worktrees/task", "branch": "codex/task"}
    run_directory = tmp_path / ".state" / "runs" / "07"
    run_directory.mkdir(parents=True)
    (run_directory / "worker-final.md").write_text("done")
    monkeypatch.setattr(codex_task_farm, "git_worktree_is_clean", lambda worktree_path: True)
    monkeypatch.setattr(codex_task_farm, "git_branch_ahead_count", lambda worktree_path, base_branch: 1)

    (run_directory / "exit-code.txt").write_text("0\n")
    verified = codex_task_farm.classify_worker_completion(tmp_path, manifest, task)
    (run_directory / "exit-code.txt").unlink()
    legacy = codex_task_farm.classify_worker_completion(tmp_path, manifest, task)

    assert verified.status == codex_task_farm.TaskStatus.IMPLEMENTED
    assert verified.verification == "verified"
    assert legacy.status == codex_task_farm.TaskStatus.IMPLEMENTED
    assert legacy.verification == "legacy-unverified"


def test_worker_classification_blocks_incomplete_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["defaults"]["state_directory"] = ".state"
    task = {"id": 7, "worktree": "../worktrees/task", "branch": "codex/task"}
    run_directory = tmp_path / ".state" / "runs" / "07"
    run_directory.mkdir(parents=True)
    monkeypatch.setattr(codex_task_farm, "git_worktree_is_clean", lambda worktree_path: True)
    monkeypatch.setattr(codex_task_farm, "git_branch_ahead_count", lambda worktree_path, base_branch: 1)

    missing_final = codex_task_farm.classify_worker_completion(tmp_path, manifest, task)
    (run_directory / "worker-final.md").write_text("done")
    (run_directory / "exit-code.txt").write_text("1\n")
    failed_exit = codex_task_farm.classify_worker_completion(tmp_path, manifest, task)
    (run_directory / "exit-code.txt").write_text("0\n")
    monkeypatch.setattr(codex_task_farm, "git_worktree_is_clean", lambda worktree_path: False)
    dirty = codex_task_farm.classify_worker_completion(tmp_path, manifest, task)
    monkeypatch.setattr(codex_task_farm, "git_worktree_is_clean", lambda worktree_path: True)
    monkeypatch.setattr(codex_task_farm, "git_branch_ahead_count", lambda worktree_path, base_branch: 0)
    no_commit = codex_task_farm.classify_worker_completion(tmp_path, manifest, task)

    assert missing_final.status == codex_task_farm.TaskStatus.BLOCKED
    assert failed_exit.status == codex_task_farm.TaskStatus.BLOCKED
    assert dirty.verification == "dirty-worktree"
    assert no_commit.verification == "no-task-commit"


def test_prune_stale_runtime_state_archives_mismatched_task_identity() -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["tasks"] = [
        {
            "id": 7,
            "slug": "new-task",
            "branch": "codex/review-07-new-task",
            "worktree": "../g-worktrees/review-07-new-task",
            "source_start_line": 10,
            "source_end_line": 20,
        }
    ]
    state: codex_task_farm.JsonObject = {
        "runs": {
            "7": {
                "branch": "codex/review-07-old-task",
                "worktree": "../g-worktrees/review-07-old-task",
            },
            "11": {
                "branch": "codex/review-11-old-task",
                "worktree": "../g-worktrees/review-11-old-task",
            },
        },
        "statuses": {"7": "merged", "11": "merged"},
        "worker_results": {"7": {"verification": "legacy"}, "11": {"verification": "legacy"}},
    }

    changed = codex_task_farm.prune_stale_runtime_state(state, manifest)

    assert changed is True
    assert state["runs"] == {}
    assert state["statuses"] == {}
    assert len(state["archived_stale_entries"]) == 2


def test_refresh_keeps_unobservable_incomplete_worker_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["defaults"]["state_directory"] = ".state"
    manifest["tasks"] = [
        {
            "id": 1,
            "slug": "task",
            "branch": "codex/task",
            "worktree": "../worktrees/task",
            "source_start_line": 1,
            "source_end_line": 10,
            "status": "ready",
        }
    ]
    state_path = tmp_path / ".state" / "state.json"
    run_directory = tmp_path / ".state" / "runs" / "01"
    run_directory.mkdir(parents=True)
    codex_task_farm.write_json_object(
        state_path,
        {
            "runs": {
                "1": {
                    "pid": 123,
                    "branch": "codex/task",
                    "worktree": "../worktrees/task",
                }
            },
            "statuses": {"1": "blocked"},
            "task_identities": {"1": codex_task_farm.task_runtime_identity(manifest["tasks"][0])},
        },
    )
    monkeypatch.setattr(codex_task_farm, "running_process_exists", lambda process_identifier: False)

    codex_task_farm.refresh_runtime_statuses(tmp_path, manifest)

    state = codex_task_farm.read_json_object(state_path)
    assert state["statuses"] == {"1": "running"}
    assert state["worker_results"] == {}


def test_run_wait_launches_all_workers_before_waiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["defaults"]["state_directory"] = ".state"
    manifest["tasks"] = [
        {"id": 1, "status": "ready", "branch": "codex/one", "worktree": "../one", "dependencies": []},
        {"id": 2, "status": "ready", "branch": "codex/two", "worktree": "../two", "dependencies": []},
    ]
    launch_log: list[int] = []

    def fake_launch_worker(
        *,
        repository_directory: Path,
        manifest: codex_task_farm.JsonObject,
        task: codex_task_farm.JsonObject,
        dangerously_bypass_approvals: bool,
    ) -> codex_task_farm.WorkerLaunch:
        task_identifier = int(task["id"])
        launch_log.append(task_identifier)
        fake_process = FakeWorkerProcess(task_identifier + 100, 0, launch_log)
        return codex_task_farm.WorkerLaunch(
            task_identifier=task_identifier,
            process_identifier=fake_process.pid,
            process=typing.cast("subprocess.Popen[str]", fake_process),
        )

    monkeypatch.setattr(codex_task_farm, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(codex_task_farm, "load_manifest", lambda repository_directory: manifest)
    monkeypatch.setattr(codex_task_farm, "refresh_runtime_statuses", lambda repository_directory, manifest: None)
    monkeypatch.setattr(codex_task_farm, "launch_worker", fake_launch_worker)
    monkeypatch.setattr(
        codex_task_farm,
        "classify_worker_completion",
        lambda repository_directory, manifest, task: codex_task_farm.WorkerCompletion(
            status=codex_task_farm.TaskStatus.IMPLEMENTED,
            final_message_exists=True,
            worktree_clean=True,
            branch_ahead=True,
            exit_code=0,
            verification="verified",
            reason="worker completed cleanly",
        ),
    )
    arguments = argparse.Namespace(task=[], jobs=2, wait=True, force=False, dangerous=False)

    exit_code = codex_task_farm.command_run(arguments)

    assert exit_code == 0
    assert launch_log == [1, 2]


def test_claim_skips_conflicting_active_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = codex_task_farm.default_manifest(state_directory_path=Path(".state"), id_style="string")
    manifest["tasks"] = [
        {
            "id": "T001",
            "status": "ready",
            "branch": "codex/review2-T001-one",
            "worktree": "../one",
            "dependencies": [],
            "conflict_group": "binary-jax",
        },
        {
            "id": "T002",
            "status": "ready",
            "branch": "codex/review2-T002-two",
            "worktree": "../two",
            "dependencies": [],
            "conflict_group": "binary-jax",
        },
    ]

    monkeypatch.setattr(codex_task_farm, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(codex_task_farm, "load_manifest", lambda repository_directory: manifest)
    monkeypatch.setattr(codex_task_farm, "refresh_runtime_statuses", lambda repository_directory, manifest: None)

    exit_code = codex_task_farm.command_claim(
        argparse.Namespace(task=[], jobs=2, owner="agent", lease_seconds=60, force=False),
    )

    state = codex_task_farm.read_json_object(tmp_path / ".state" / "state.json")
    assert exit_code == 0
    assert state["statuses"] == {"T001": "claimed"}
    assert state["leases"]["T001"]["owner"] == "agent"


def test_reset_stale_task_leases_returns_running_tasks_to_ready() -> None:
    manifest = codex_task_farm.default_manifest(id_style="string")
    task = {"id": "T001", "status": "ready", "branch": "codex/one", "worktree": "../one"}
    manifest["tasks"] = [task]
    state: codex_task_farm.JsonObject = {
        "statuses": {"T001": "running"},
        "leases": {
            "T001": {
                "owner": "agent",
                "status": "running",
                "acquired_at": "2026-06-02T00:00:00+00:00",
                "expires_at": "2000-01-01T00:00:00+00:00",
            }
        },
    }

    reset_identifiers = codex_task_farm.reset_stale_task_leases(state, manifest)

    assert reset_identifiers == ["T001"]
    assert state["statuses"] == {"T001": "ready"}
    assert state["leases"] == {}


def test_review_decision_can_request_changes_or_reject() -> None:
    assert (
        codex_task_farm.classify_review_decision("Decision: needs_changes\nFinding")
        == codex_task_farm.TaskStatus.NEEDS_CHANGES
    )
    assert codex_task_farm.classify_review_decision("Decision: reject") == codex_task_farm.TaskStatus.BLOCKED
    assert codex_task_farm.classify_review_decision("No findings") == codex_task_farm.TaskStatus.REVIEWED


def test_review_rejects_tasks_that_are_not_implemented(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["defaults"]["state_directory"] = ".state"
    manifest["tasks"] = [
        {"id": 1, "status": "ready", "branch": "codex/one", "worktree": "worktree", "dependencies": []}
    ]

    monkeypatch.setattr(codex_task_farm, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(codex_task_farm, "load_manifest", lambda repository_directory: manifest)
    monkeypatch.setattr(codex_task_farm, "refresh_runtime_statuses", lambda repository_directory, manifest: None)

    with pytest.raises(ValueError, match="cannot review"):
        codex_task_farm.command_review(argparse.Namespace(task=[1]))


def test_integrate_rejects_unreviewed_tasks_without_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["defaults"]["state_directory"] = ".state"
    manifest["tasks"] = [
        {"id": 1, "status": "implemented", "branch": "codex/one", "worktree": "worktree", "dependencies": []}
    ]

    monkeypatch.setattr(codex_task_farm, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(codex_task_farm, "load_manifest", lambda repository_directory: manifest)
    monkeypatch.setattr(codex_task_farm, "refresh_runtime_statuses", lambda repository_directory, manifest: None)

    with pytest.raises(ValueError, match="cannot integrate"):
        codex_task_farm.command_integrate(argparse.Namespace(task=[1], dangerous=False, allow_unreviewed=False))


def test_integrate_marks_task_integrated_not_merged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "docs").mkdir()
    manifest = codex_task_farm.default_manifest(
        state_directory_path=Path(".state"),
        integration_branch="integration/code-review-2",
        integration_worktree="../integration",
        id_style="string",
    )
    manifest["tasks"] = [
        {
            "id": "T001",
            "status": "reviewed",
            "branch": "codex/review2-T001-one",
            "worktree": "worktree",
            "title": "Integrate task",
            "dependencies": [],
            "logs": {
                "run_directory": ".state/runs/T001",
                "integration_final": ".state/runs/T001/integration-final.md",
                "integration_jsonl": ".state/runs/T001/integration.jsonl",
                "integration_stderr": ".state/runs/T001/integration.stderr.log",
                "review_final": ".state/runs/T001/review.md",
            },
        }
    ]
    (tmp_path / "worktree").mkdir()
    integration_worktree = tmp_path / "integration"
    integration_worktree.mkdir()
    head_commits = iter(["before", "after"])

    class FakeCompletedProcess:
        returncode: int = 0
        stdout: str = '{"type": "ok"}\n'
        stderr: str = ""

    def fake_subprocess_run(
        command_arguments: list[str],
        **subprocess_arguments: object,
    ) -> FakeCompletedProcess:
        del subprocess_arguments
        output_flag_index = command_arguments.index("-o")
        Path(command_arguments[output_flag_index + 1]).write_text("integrated")
        return FakeCompletedProcess()

    monkeypatch.setattr(codex_task_farm, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(codex_task_farm, "load_manifest", lambda repository_directory: manifest)
    monkeypatch.setattr(codex_task_farm, "refresh_runtime_statuses", lambda repository_directory, manifest: None)
    monkeypatch.setattr(
        codex_task_farm,
        "ensure_integration_worktree",
        lambda repository_directory, defaults: integration_worktree,
    )
    monkeypatch.setattr(
        codex_task_farm,
        "ensure_integration_worktree_ready",
        lambda worktree_path, integration_branch: None,
    )
    monkeypatch.setattr(codex_task_farm, "git_head_commit", lambda worktree_path: next(head_commits))
    monkeypatch.setattr(codex_task_farm.subprocess, "run", fake_subprocess_run)

    exit_code = codex_task_farm.command_integrate(
        argparse.Namespace(task=["T001"], dangerous=False, allow_unreviewed=False),
    )

    state = codex_task_farm.read_json_object(tmp_path / ".state" / "state.json")
    assert exit_code == 0
    assert state["statuses"] == {"T001": "integrated"}
    assert manifest["tasks"][0]["status"] == "integrated"
    assert manifest["tasks"][0]["runtime"]["integration_head"] == "after"


def test_dependency_gating_blocks_missing_or_unready_dependencies() -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["tasks"] = [
        {"id": 1, "status": "implemented", "enabled": True, "dependencies": []},
        {"id": 2, "status": "ready", "enabled": True, "dependencies": [1]},
        {"id": 3, "status": "ready", "enabled": True, "dependencies": [99]},
        {"id": 4, "status": "ready", "enabled": False, "dependencies": []},
        {"id": 5, "status": "ready", "enabled": True, "dependencies": [4]},
    ]
    state: codex_task_farm.JsonObject = {"statuses": {"1": "implemented"}}

    with pytest.raises(ValueError, match="dependency 1 is implemented"):
        codex_task_farm.ensure_dependencies_ready(
            manifest,
            state,
            [manifest["tasks"][1]],
            {codex_task_farm.TaskStatus.MERGED.value},
            "run",
        )
    with pytest.raises(ValueError, match="dependency 99 is missing"):
        codex_task_farm.ensure_dependencies_ready(
            manifest,
            state,
            [manifest["tasks"][2]],
            {codex_task_farm.TaskStatus.MERGED.value},
            "run",
        )
    codex_task_farm.ensure_dependencies_ready(
        manifest,
        state,
        [manifest["tasks"][4]],
        {codex_task_farm.TaskStatus.MERGED.value},
        "run",
    )


def test_integrate_ready_requires_reviewed_unless_explicitly_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["defaults"]["state_directory"] = ".state"
    manifest["tasks"] = [
        {"id": 1, "status": "implemented", "branch": "codex/one", "worktree": "../one"},
        {"id": 2, "status": "reviewed", "branch": "codex/two", "worktree": "../two"},
    ]
    captured_tasks: list[list[int]] = []

    def fake_command_integrate(arguments: argparse.Namespace) -> int:
        captured_tasks.append(list(arguments.task))
        return 0

    monkeypatch.setattr(codex_task_farm, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(codex_task_farm, "load_manifest", lambda repository_directory: manifest)
    monkeypatch.setattr(codex_task_farm, "refresh_runtime_statuses", lambda repository_directory, manifest: None)
    monkeypatch.setattr(codex_task_farm, "command_integrate", fake_command_integrate)

    codex_task_farm.command_integrate_ready(
        argparse.Namespace(allow_unreviewed=False, dangerous=False),
    )
    codex_task_farm.command_integrate_ready(
        argparse.Namespace(allow_unreviewed=True, dangerous=False),
    )

    assert captured_tasks == [[2], [1, 2]]


def test_doctor_reports_missing_nix_as_warning_unless_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "docs" / "scratchpad").mkdir(parents=True)
    (tmp_path / "docs" / "development").mkdir()
    (tmp_path / "docs" / "scratchpad" / "code-review.md").write_text("")
    (tmp_path / "docs" / "development" / "STYLEGUIDE.md").write_text("")
    (tmp_path / "Justfile").write_text("")
    (tmp_path / "AGENTS.md").write_text("")
    manifest = codex_task_farm.default_manifest()

    def fake_which(command_name: str) -> str | None:
        if command_name in {"git", "codex"}:
            return f"/usr/bin/{command_name}"
        return None

    def fake_run_command(command_arguments: list[str], *, cwd: Path) -> codex_task_farm.CommandResult:
        if command_arguments == ["git", "branch", "--show-current"]:
            return codex_task_farm.CommandResult(command_arguments, 0, "main\n", "")
        if command_arguments == ["git", "status", "--short"]:
            return codex_task_farm.CommandResult(command_arguments, 0, "", "")
        return codex_task_farm.CommandResult(command_arguments, 1, "", "unexpected")

    monkeypatch.setattr(codex_task_farm.shutil, "which", fake_which)
    monkeypatch.setattr(codex_task_farm, "git_branch_exists", lambda repository_directory, branch: True)
    monkeypatch.setattr(codex_task_farm, "run_command", fake_run_command)

    relaxed_checks = codex_task_farm.collect_doctor_checks(tmp_path, manifest, strict=False)
    strict_checks = codex_task_farm.collect_doctor_checks(tmp_path, manifest, strict=True)

    relaxed_nix = next(check for check in relaxed_checks if check.name == "nix")
    strict_nix = next(check for check in strict_checks if check.name == "nix")
    assert relaxed_nix.warning is True
    assert relaxed_nix.passed is False
    assert strict_nix.warning is False
    assert strict_nix.passed is False


def test_collect_status_rows_marks_finished_and_stale_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest: codex_task_farm.JsonObject = {
        "defaults": {"state_directory": ".state"},
        "tasks": [
            {
                "id": "T001",
                "status": "ready",
                "branch": "codex/one",
                "worktree": "../one",
                "title": "Finished task",
            },
            {
                "id": "T002",
                "status": "ready",
                "branch": "codex/two",
                "worktree": "../two",
                "title": "Stale task",
            },
            {
                "id": "T003",
                "status": "ready",
                "branch": "codex/three",
                "worktree": "../three",
                "title": "Active task",
            },
        ],
    }
    state: codex_task_farm.JsonObject = {
        "leases": {
            "T002": {
                "owner": "agent",
                "expires_at": "2999-01-01T00:00:00+00:00",
            },
        },
        "runs": {
            "T001": {"pid": 11, "returncode": 0},
            "T002": {"pid": 22, "returncode": None},
            "T003": {"pid": 33, "returncode": None},
        },
        "statuses": {
            "T001": "implemented",
            "T002": "running",
            "T003": "running",
        },
        "task_identities": {},
    }
    run_directory = tmp_path / ".state" / "runs" / "T001"
    run_directory.mkdir(parents=True)
    (run_directory / "worker-final.md").write_text("done")
    (run_directory / "exit-code.txt").write_text("0\n")
    active_run_directory = tmp_path / ".state" / "runs" / "T003"
    active_run_directory.mkdir(parents=True)
    (active_run_directory / "worker.jsonl").write_text("{}\n")
    monkeypatch.setattr(codex_task_farm, "running_process_exists", lambda process_identifier: False)

    rows = codex_task_farm.collect_status_rows(
        tmp_path,
        manifest,
        state,
        codex_task_farm.selected_tasks(manifest, []),
        check_worktrees=False,
    )

    assert [row.worker for row in rows] == ["finished", "stale", "active"]
    assert [row.status for row in rows] == ["implemented", "running", "running"]
    assert [row.worktree for row in rows] == ["skipped", "skipped", "skipped"]
    assert [row.lease for row in rows] == ["-", "agent", "-"]


def test_format_status_snapshot_includes_summary_counts() -> None:
    rows = [
        codex_task_farm.StatusRow("T001", "implemented", "finished", "yes", "0", "clean", "-", "Done"),
        codex_task_farm.StatusRow("T002", "running", "stale", "no", "-", "skipped", "agent", "Stale"),
    ]

    snapshot = codex_task_farm.format_status_snapshot(rows, updated_at="now")

    assert "updated=now" in snapshot
    assert "statuses: implemented=1, running=1" in snapshot
    assert "workers: finished=1, stale=1" in snapshot
    assert "T001" in snapshot
    assert "T002" in snapshot


def test_status_watch_skips_worktree_checks_by_default() -> None:
    one_shot_arguments = argparse.Namespace(
        check_worktrees=False,
        no_worktree_check=False,
        watch=False,
    )
    watch_arguments = argparse.Namespace(
        check_worktrees=False,
        no_worktree_check=False,
        watch=True,
    )
    watch_with_worktrees_arguments = argparse.Namespace(
        check_worktrees=True,
        no_worktree_check=False,
        watch=True,
    )

    assert codex_task_farm.should_check_status_worktrees(one_shot_arguments) is True
    assert codex_task_farm.should_check_status_worktrees(watch_arguments) is False
    assert codex_task_farm.should_check_status_worktrees(watch_with_worktrees_arguments) is True

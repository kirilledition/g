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


def test_sync_manifest_preserves_only_manual_metadata(tmp_path: Path) -> None:
    docs_directory = tmp_path / "docs"
    docs_directory.mkdir()
    (docs_directory / "code-review.md").write_text(
        """# Category

## 7. First task

Body with `src/g/current.py`.
"""
    )
    (docs_directory / "code-review.tasks.json").write_text(
        """{
  "defaults": {
    "jobs": 2,
    "worktree_root": "../custom-worktrees"
  },
  "source_path": "docs/code-review.md",
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


def test_codex_command_builders_default_to_safe_execution() -> None:
    worker_command = codex_task_farm.build_worker_command(
        worktree_path=Path("/repo/worktree"),
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
        model="gpt-5.5",
        reasoning_effort="xhigh",
        final_message_path=Path("/state/integration.md"),
        dangerously_bypass_approvals=False,
    )

    assert "--dangerously-bypass-approvals-and-sandbox" not in worker_command
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
    assert integration_command[:5] == ["codex", "--cd", "/repo/integration", "--add-dir", "/repo/worktree"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in integration_command


def test_dangerous_flag_adds_bypass_only_for_worker_and_integrator() -> None:
    worker_command = codex_task_farm.build_worker_command(
        worktree_path=Path("/repo/worktree"),
        model="gpt-5.5",
        reasoning_effort="high",
        final_message_path=Path("/state/final.md"),
        dangerously_bypass_approvals=True,
    )
    integration_command = codex_task_farm.build_integration_command(
        integration_worktree_path=Path("/repo/integration"),
        worktree_path=Path("/repo/worktree"),
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
    review_directory = tmp_path / ".state" / "reviews"
    review_directory.mkdir(parents=True)
    final_message_path = review_directory / "01.md"
    jsonl_log_path = review_directory / "01.jsonl"
    stderr_log_path = review_directory / "01.stderr.log"
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
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "code-review.md").write_text("")
    (tmp_path / "docs" / "STYLEGUIDE.md").write_text("")
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

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIRECTORY))
codex_task_farm = importlib.import_module("scripts.codex_task_farm")


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


def test_sync_manifest_preserves_manual_metadata(tmp_path: Path) -> None:
    docs_directory = tmp_path / "docs"
    docs_directory.mkdir()
    (docs_directory / "code-review.md").write_text(
        """# Category

## 7. First task

Body.
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
      "notes": "manual"
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
    assert task["notes"] == "manual"
    assert task["branch"] == "codex/review-07-first-task"
    assert task["worktree"] == "../custom-worktrees/review-07-first-task"


def test_codex_command_builders_pin_models_and_reasoning() -> None:
    worker_command = codex_task_farm.build_worker_command(
        worktree_path=Path("/repo/worktree"),
        model="gpt-5.5",
        reasoning_effort="high",
        final_message_path=Path("/state/final.md"),
    )
    review_command = codex_task_farm.build_review_command(
        worktree_path=Path("/repo/worktree"),
        model="gpt-5.5",
        reasoning_effort="xhigh",
        base_branch="main",
        final_message_path=Path("/state/review.md"),
    )
    integration_command = codex_task_farm.build_integration_command(
        repository_directory=Path("/repo/main"),
        worktree_path=Path("/repo/worktree"),
        model="gpt-5.5",
        reasoning_effort="xhigh",
        final_message_path=Path("/state/integration.md"),
    )

    assert worker_command == [
        "codex",
        "--cd",
        "/repo/worktree",
        "-m",
        "gpt-5.5",
        "-c",
        'model_reasoning_effort="high"',
        "exec",
        "--json",
        "-o",
        "/state/final.md",
        "-",
    ]
    assert "review" in review_command
    assert 'model_reasoning_effort="xhigh"' in review_command
    assert "--add-dir" in integration_command
    assert "/repo/worktree" in integration_command


def test_refresh_runtime_statuses_marks_finished_detached_worker(tmp_path: Path) -> None:
    manifest = codex_task_farm.default_manifest()
    manifest["defaults"]["state_directory"] = ".state"
    manifest["tasks"] = [
        {
            "id": 7,
            "status": "running",
            "branch": "codex/review-07-first-task",
            "worktree": "../worktrees/review-07-first-task",
        }
    ]
    run_directory = tmp_path / ".state" / "runs" / "07"
    run_directory.mkdir(parents=True)
    (run_directory / "worker-final.md").write_text("done")
    codex_task_farm.write_json_object(
        tmp_path / ".state" / "state.json",
        {
            "runs": {
                "7": {
                    "pid": 999999999,
                }
            }
        },
    )

    changed = codex_task_farm.refresh_runtime_statuses(tmp_path, manifest)

    assert changed is True
    assert manifest["tasks"][0]["status"] == "implemented"

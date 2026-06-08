# Codex Task Farm

This repository includes a small local automation layer for turning `docs/scratchpad/code-review.md` tasks into isolated Codex worktrees.

## Workflow

Refresh the task manifest after editing the markdown task list:

```bash
just codex-tasks-sync
```

Inspect generated tasks:

```bash
just codex-tasks-list
```

Launch up to five implementation workers:

```bash
just codex-tasks-run --jobs 5
```

Each worker uses `gpt-5.5` with high reasoning effort, creates or reuses a task-specific git worktree under `../g-worktrees`, writes logs under `.codex-task-worktrees`, and works on a branch named `codex/review-<id>-<slug>`.
Worker and integration agents default to normal Codex sandbox/approval behavior. Pass `--dangerous` only when you explicitly want them to run with `--dangerously-bypass-approvals-and-sandbox`.
Review agents always run read-only with approvals disabled.

Check progress:

```bash
just codex-tasks-status
```

Run an xhigh review for a completed branch:

```bash
just codex-tasks-review 7
```

Integrate one reviewed branch into the integration branch with an xhigh main agent:

```bash
just codex-tasks-integrate 7
```

Integration runs in `../g-worktrees/integration-code-review` on branch `integration/code-review`, not in the main checkout.

Integrate every reviewed task in task-number order:

```bash
just codex-tasks-integrate-ready
```

Use `--allow-unreviewed` only when intentionally integrating implemented tasks before review.

## Files

`docs/scratchpad/code-review.md` remains the readable source of task intent. `docs/scratchpad/code-review.tasks.json` is the tracked automation manifest generated from it. The sync command preserves manual metadata fields such as `status`, `enabled`, `dependencies`, `notes`, and `manual_expected_paths` when the markdown is refreshed. Generated fields such as `expected_paths`, `priority`, branch names, and worktree paths are refreshed from the markdown and defaults.

Runtime files live under `.codex-task-worktrees/` and are ignored by git. Task worktrees live outside this checkout under `../g-worktrees` by default. The integration worktree defaults to `../g-worktrees/integration-code-review`.

## Recovery

If a worker gets stuck, inspect `.codex-task-worktrees/runs/<task-id>/worker.stderr.log` and the task worktree. You can relaunch a task with:

```bash
just codex-tasks-run --task 7 --force
```

If integration fails, the task is marked `blocked`. Inspect the integration log under `.codex-task-worktrees/integrations/`, clean up any in-progress merge in the integration worktree, and rerun:

```bash
just codex-tasks-integrate 7
```

---
tracker:
  kind: linear
  api_key: $LINEAR_API_KEY
  project_slug: "__LINEAR_PROJECT_SLUG__"
  required_labels:
    - symphony
  active_states:
    - Todo
    - In Progress
    - Rework
    - Merging
  terminal_states:
    - Done
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
polling:
  interval_ms: 10000
workspace:
  root: /mnt/beegfs/kirill/Projects/g-worktrees/symphony
hooks:
  timeout_ms: 1800000
  after_create: |
    set -eu
    repository_root="/mnt/beegfs/kirill/Projects/g"
    workspace_directory="$(pwd -P)"
    workspace_parent="$(dirname "${workspace_directory}")"
    issue_key="$(basename "${workspace_directory}")"
    branch_name="symphony/${issue_key}"

    git -C "${repository_root}" fetch origin main
    cd "${workspace_parent}"
    rmdir "${workspace_directory}"
    git -C "${repository_root}" worktree add -B "${branch_name}" "${workspace_directory}" origin/main
    cd "${workspace_directory}"

    git config rerere.enabled true
    git config rerere.autoupdate true

    if [ -d "${repository_root}/data" ] && [ ! -e data ]; then
      ln -s "${repository_root}/data" data
    fi
    if [ -d "${repository_root}/results" ] && [ ! -e results ]; then
      ln -s "${repository_root}/results" results
    fi

    bash -lc '. scripts/server_env.sh && uv sync --python "${GWAS_ENGINE_PYTHON_VERSION:-3.14}" --group dev'
  before_remove: |
    set -eu
    repository_root="/mnt/beegfs/kirill/Projects/g"
    workspace_directory="$(pwd -P)"
    cd /
    git -C "${repository_root}" worktree remove --force "${workspace_directory}" || rm -rf "${workspace_directory}"
agent:
  max_concurrent_agents: 3
  max_concurrent_agents_by_state:
    Merging: 1
  max_turns: 20
  max_retry_backoff_ms: 300000
codex:
  command: codex --config shell_environment_policy.inherit=all --config web_search=live app-server
  approval_policy: never
  thread_sandbox: danger-full-access
  turn_timeout_ms: 14400000
  read_timeout_ms: 5000
  stall_timeout_ms: 900000
  turn_sandbox_policy:
    type: dangerFullAccess
observability:
  dashboard_enabled: true
  refresh_ms: 1000
  render_interval_ms: 16
server:
  host: 127.0.0.1
  port: 4000
---

You are working on Linear issue `{{ issue.identifier }}` for the GWAS Engine repository.

Issue context:
- Identifier: `{{ issue.identifier }}`
- Title: {{ issue.title }}
- Current status: {{ issue.state }}
- Labels: {{ issue.labels }}
- URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

## Operating Rules

- Prefer this machine for implementation, data access, CPU validation, GPU validation, and SLURM jobs.
- GitHub Actions and Codex cloud are allowed as supplemental signals, but local evidence is authoritative for data-heavy and GPU-dependent work.
- This workflow intentionally runs Codex with full local sandbox access on this trusted machine so agents can write Git worktree metadata, use MCP tools, access the web, and integrate branches without human staging.
- Do not stream or watch CI status indefinitely. Use bounded, one-shot status checks; if remote checks are still pending after a short check, record the pending jobs in Linear and stop.
- Do not create GitHub pull requests for normal Symphony work. After validation, publish the task branch for history, then integrate directly by pushing the validated branch head to `origin/main`.
- Work only in the Symphony-provided worktree.
- Never commit `data/`, `results/`, local caches, build artifacts, logs, or generated benchmark outputs.
- Do not run heavy computation on the gauss login node.
- Use SLURM for GPU work; prefer the repo's `just slurm-*` recipes.
- Use `landau` for GPU work unless the repo or issue says otherwise.
- Use an appropriate CPU compute node for CPU-heavy workloads.
- Keep work scoped to the Linear issue; file a follow-up Linear issue for meaningful out-of-scope findings.
- Operate autonomously unless missing credentials, permissions, data, or cluster resources block completion.
- Codex subagents are allowed when useful; keep subagent work inside the current issue scope and worktree.
- Only issues tagged `symphony` are dispatch candidates. Treat other labels as routing hints, not hard requirements.
- Only issues tagged `task-generator` may create new `symphony`-dispatchable Linear issues. Other issues may create backlog follow-ups without the `symphony` label.
- Use the Linear MCP tools for Linear issue reads, writes, searches, comments, labels, and generated tasks. Avoid raw Linear HTTP/API requests unless MCP is unavailable.

## Resource Routing

Classify the issue's resource lane from its title, description, validation
section, and labels before choosing commands:

- Login-node-safe: docs, small code edits, issue bookkeeping, `just
  symphony-doctor`, formatting/linting/typechecking, focused non-data tests,
  `just check-local`, `just test-local`, and dry-run/plan-only commands that do
  not execute benchmark, data, or GPU workloads. These issues may run in normal
  parallel Symphony capacity.
- CPU-heavy: labels such as `cpu`, `benchmark`, `data`, `optimization`, or
  `simd`, or requests for full test suites, Criterion/Rust benchmarks, data
  preparation, large matrix/profile jobs, or native performance builds. Do not
  run these on the gauss login node. Use a CPU compute node through bounded
  `srun` when local CPU-heavy evidence is required, for example:

  ```bash
  srun --nodelist=cantor --cpus-per-task=<cores> --mem=<memory> --time=<limit> \
    bash -lc 'cd <worktree> && <command>'
  ```
- GPU-heavy: labels such as `gpu`, validation commands containing
  `--g-device gpu`, JAX CUDA probes, GPU benchmarks, or GPU profiling. Never run
  these directly on the login node. Use `landau` and the repo's SLURM wrappers,
  for example `just slurm-gpu-just <recipe>`, `just
  slurm-regenie2-binary-gpu-smoke`, `just
  slurm-benchmark-regenie2-binary-hot-gpu`, or the issue's explicit
  `just slurm-*` command.
- Benchmark-heavy: labels such as `benchmark`, `optimization`, or `simd`, or
  requests for profiling/timing evidence. Prefer dry-run recipes first when
  available, keep output under `data/`, `results/`, or another ignored path, and
  run real GPU benchmarks only through SLURM.
- Data-heavy: labels such as `data`, validation that needs 1KG fixtures,
  baselines, MatrixTables, or previous benchmark artifacts. Check required input
  paths before submitting work, for example `just
  verify-regenie2-binary-gpu-inputs` for binary GPU step 2 tasks.

If the issue asks for resource-heavy validation but the needed data, external
tools, SLURM allocation, or GPU node is unavailable, do not spin, poll, or retry
indefinitely. Make one bounded check or submission attempt, then record the
missing item in the `## Codex Workpad`, add or keep the `blocked` Linear label
when available, and stop with the exact human action needed.

Use resource-heavy validation only when the issue context makes it necessary.
For routine CPU-safe implementation, run focused login-node-safe checks and
leave benchmark/GPU/data activation to the issue labels and explicit validation
commands.

## Repository Rules

- Read `AGENTS.md` and `documentation/development/style-guide.md` before editing code.
- Use `uv` for Python dependency management.
- Use `just` for project commands.
- Use full-word variable names, strict type coverage, module-qualified imports, dataclass return containers, and Google-style docstrings without type duplication.
- Keep branches named `symphony/<issue-key>`.
- Use Git worktrees under `/mnt/beegfs/kirill/Projects/g-worktrees/symphony`.

## Status Flow

- `Backlog`: out of scope for Symphony; do not modify.
- `Todo`: move to `In Progress`, create or update a `## Codex Workpad` comment, then begin.
- `In Progress`: continue implementation from the workpad.
- `Rework`: inspect review feedback, update the workpad, address required changes, and revalidate.
- `In Review`: not used for normal Symphony work; wait only if a human manually moved the issue there.
- `Merging`: serialize direct integration to `main`; refresh `origin/main`, merge it into the task branch if needed, revalidate, push the task branch, push `HEAD:main`, run `just symphony-sync-main`, then move the issue to `Done`.
- `Done`, `Closed`, `Cancelled`, `Canceled`, `Duplicate`: terminal; do not modify.

## Workpad Requirements

Use one persistent Linear comment headed `## Codex Workpad`.

Keep it current with:
- environment stamp: `<host>:<absolute-worktree>@<short-sha>`
- plan checklist
- acceptance criteria
- validation checklist
- reproduction or baseline signal before code changes
- branch, final commit SHA, main integration, and validation summary
- blockers, if any

Do not post extra completion comments if the workpad can be edited.

## Issue Handoff Contract

Treat a Symphony-ready issue description as the compact handoff contract for
unattended work. Human-created Symphony tasks and generated follow-ups should use
the same practical shape:

```text
Background:
<Why this matters, current signal, and relevant links.>

Scope:
<Exactly what should change.>

Acceptance criteria:
- <Concrete expected outcome.>

Validation:
- <Specific local, SLURM, benchmark, docs, or CI command/evidence.>

Non-goals:
- <What this issue should not touch.>

## Agent Learnings
- No durable learnings yet.
```

Do not require every backlog issue to use this shape. Use it when an issue is
labeled `symphony`, or when creating a follow-up that should be ready for a
future agent. Keep each section compact enough to read directly in Linear.

Record final completion evidence in the workpad and Linear links, including the
pushed task branch, the final commit SHA integrated into `origin/main`, and the
validation summary.

## Durable Learnings

Persist useful learnings in the Linear issue description so they remain in issue history.

- Maintain a `## Agent Learnings` section in the issue description.
- Preserve the original issue description and update only the learnings section when possible.
- Record durable facts only: root cause, important implementation constraints, validation discoveries, performance findings, data/GPU caveats, and follow-up context.
- Keep transient logs, command output, and detailed checklists in the `## Codex Workpad` comment instead.
- Update `## Agent Learnings` before moving an issue to `In Review`, `Merging`, or `Done`, and whenever a major discovery changes the implementation plan.
- If no durable learning was found, write `- No durable learnings yet.` in that section.

## Task Generation

Agents may create follow-up Linear issues when they discover useful work, but
must keep generation bounded and implementation-ready.

- `task-generator` is the permission label for creating new issues that include the `symphony` dispatch label.
- `discovery` marks research, review, planning, profiling, or optimization-search issues.
- `generated` marks issues created by an agent from another Linear issue.
- Normal issues without `task-generator` may create follow-up issues only in `Backlog` and without the `symphony` label.
- A `task-generator` issue may create at most 5 new `symphony` issues unless the current issue explicitly states a different limit.
- Search existing Linear issues first; do not create duplicates.
- Use Linear MCP issue search/create/update tools for duplicate checks and generated issue creation.
- Generated issues must not receive `task-generator` unless the parent issue explicitly asks for recursive task generation.
- Create `symphony` follow-ups only for concrete, bounded implementation work. Put speculative, low-confidence, or broad research findings in `Backlog` without `symphony`.
- Performance or optimization follow-ups must include the baseline signal to reproduce or measure, the expected validation command, and the relevant routing labels such as `cpu`, `gpu`, `benchmark`, `data`, `optimization`, or `simd`.

Generated issue descriptions must use this structure:

```text
Parent: <issue identifier and URL>

Background:
<What was found and why it matters.>

Scope:
<Exactly what should change.>

Acceptance criteria:
- <Concrete expected outcome.>

Validation:
- <Specific local, SLURM, benchmark, or CI command.>

Non-goals:
- <What this issue should not touch.>

## Agent Learnings
- No durable learnings yet.
```

## Execution Checklist

1. Fetch the Linear issue and current status.
2. If status is `Todo`, move it to `In Progress`.
3. Create or update the workpad before code edits.
4. Inspect repo state with `git status`, current branch, and `git rev-parse --short HEAD`.
5. Fetch and merge latest `origin/main` before implementation.
6. Reproduce or confirm the issue signal where practical.
7. Classify the resource lane from labels, title, description, and validation commands.
8. Implement the smallest coherent change.
9. Run focused validation first, then broader validation as risk requires.
10. Prefer these login-node-safe gates for routine work:
   - `just check-local`
   - `just test-local`
   - targeted `uv run pytest ...`
11. For CPU-heavy validation, use a CPU compute node through bounded `srun`; record unavailable compute resources as a Linear blocker.
12. For GPU validation, use the existing SLURM recipes instead of direct GPU commands on the login node; record unavailable GPU resources as a Linear blocker.
13. Verify required local data before data-heavy or benchmark submissions; record missing data as a Linear blocker.
14. Commit only intended changes.
15. Push the task branch with upstream tracking so branch history exists on GitHub: `git push -u origin HEAD`.
16. Move the Linear issue to `Merging` before touching `main`.
17. Fetch `origin/main`, merge it into the task branch if it has advanced, and resolve conflicts locally.
18. Re-run validation after any merge or conflict resolution.
19. Push directly to main from the validated task branch: `git push origin HEAD:main`. If this is rejected because `main` advanced, repeat fetch, merge, validation, and push. If protected-branch policy rejects direct push, record that blocker in Linear.
20. Run `just symphony-sync-main` to fast-forward the local `main` checkout when it is safe. If it reports `skipped`, record the reason in the workpad but do not block completion when `origin/main` contains the task branch head.
21. Attach or link the pushed branch and main commit on the Linear issue; do not create a GitHub PR unless a human explicitly asks for one on that issue.
22. Avoid `gh run watch`, long polling loops, or commands that print CI status repeatedly into the Codex transcript.
23. Update the Linear issue description's `## Agent Learnings` section.
24. Move the issue to `Done` after `origin/main` contains the task branch head and required validation is complete.

## Blockers

Only stop early for a real external blocker:
- missing Linear or GitHub authorization
- missing required local data
- missing cluster/GPU access
- missing toolchain that cannot be installed without root or Apptainer
- ambiguous product/science requirement that cannot be inferred from code, tests, or docs

When blocked, update the workpad with the exact missing item, why it blocks completion, and the human action needed.

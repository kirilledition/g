# Symphony

This repo can be driven by the experimental
[openai/symphony](https://github.com/openai/symphony) orchestrator with Linear as
the work queue.

The intended posture is local-first hybrid:

- Prefer this machine for implementation, data access, CPU validation, GPU
  validation, and SLURM jobs.
- GitHub Actions and Codex cloud are allowed as supplemental signals.
- Local validation is authoritative for data-heavy and GPU-dependent work.
- Symphony runs Codex with full local sandbox access for this trusted repo so
  agents can stage, commit, push branches, update Git worktree metadata, use MCP
  tools, and use live web access without approval prompts. Do not reuse this
  workflow unchanged for untrusted repositories.
- Symphony runs up to three issue agents concurrently by default, with `Merging`
  capped at one concurrent agent.
- Resource labels do not dispatch work by themselves; they tell agents which
  validation lane to use and when to require SLURM, data checks, or bounded
  blocker reporting.
- Normal Symphony tasks do not require GitHub pull requests. Agents push their
  task branch for history, then push the validated branch head directly to
  `origin/main`.
- Discovery issues can create more Linear tasks when explicitly labeled
  `task-generator`; generated implementation tasks can then be picked up by
  Symphony.

## Human Setup

Create a Linear project for this repository, for example `GWAS Engine
Symphony`, and configure these statuses:

- `Backlog`
- `Todo`
- `In Progress`
- `In Review`
- `Rework`
- `Merging`
- `Done`
- `Closed`
- `Cancelled`
- `Duplicate`

Configure these team issue labels:

- `symphony`: required dispatch label; Symphony ignores issues without it.
- `local-first`: prefer local machine validation as the primary evidence.
- `gpu`: GPU-sensitive work; use SLURM GPU recipes where relevant.
- `cpu`: CPU-focused work; avoid heavy login-node runs.
- `data`: touches local data or data-dependent validation.
- `benchmark`: requires profiling or performance evidence.
- `blocked`: blocked by missing data, auth, cluster resources, or external input.
- `discovery`: research, review, profiling, planning, or optimization-search
  work that may identify follow-up tasks.
- `task-generator`: allows an agent to create new `symphony`-dispatchable
  implementation issues.
- `generated`: marks issues created by an agent from another Linear issue.
- `optimization`: performance, throughput, memory, or efficiency optimization
  work.
- `simd`: SIMD or vectorization-specific optimization work.

## Resource-Heavy Activation

Only `symphony` is required for dispatch. Humans activate resource-heavy work by
adding the relevant routing labels and writing the expected validation command
or evidence in the issue description:

- Add `gpu` when correctness or performance depends on CUDA/JAX GPU behavior.
  Include a `just slurm-*` recipe when possible. Agents must use `landau` and
  the repo's SLURM wrappers instead of running GPU commands on the login node.
- Add `benchmark` for profiling, timing, Criterion, or comparison evidence.
  State the benchmark recipe, output location under an ignored path such as
  `data/` or `results/`, and whether a smoke run or full run is expected.
- Add `data` when the task requires 1KG fixtures, baselines, MatrixTables, or
  previous benchmark artifacts. State the expected `GWAS_ENGINE_DATA_DIR` when
  it is not the default `data`.
- Add `cpu` when validation is too large for the login node but does not require
  a GPU. Agents should use a CPU compute node through bounded `srun`, for
  example `cantor` when available.

Keep ordinary docs, small edits, lint/typecheck, focused tests, and
`just symphony-doctor` unlabeled beyond `symphony` unless resource-heavy
validation is actually needed. This keeps normal CPU-safe work parallel while
making costly work opt in.

When activating resource-heavy work, include enough context for unattended
execution:

```text
Labels: symphony, gpu, benchmark, data
Validation:
- just verify-regenie2-binary-gpu-inputs
- GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data just slurm-regenie2-binary-gpu-smoke
Expected output:
- data/regenie2_binary_chr22_gpu_smoke.regenie2_binary.run/parts/*.parquet
```

If required data, external tools, SLURM capacity, or the GPU node is unavailable,
agents should make one bounded check or submission attempt, update the
`## Codex Workpad` with the missing item and human action needed, add or keep
the `blocked` label, and stop instead of polling indefinitely.

## Symphony-Ready Issues

Not every backlog issue needs to be ready for unattended Symphony work. Add the
`symphony` label only when the issue is bounded enough for an agent to start
without another planning pass.

Use this compact Linear description shape for human-created Symphony tasks:

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

Keep each section short: one paragraph or a few bullets is usually enough.
Put exact commands, required data paths, expected artifacts, and any
resource-heavy routing labels in `Validation`. Use `Non-goals` to exclude
neighboring refactors, labels, data preparation, or performance work that should
not be bundled into the task.

Agents preserve the original description and update only `## Agent Learnings`
when they discover durable facts such as root causes, important implementation
constraints, validation discoveries, performance findings, data/GPU caveats, or
follow-up context. Detailed logs, transient command output, and checklists stay
in the `## Codex Workpad` comment.

When finishing an issue, agents record completion evidence in the workpad and
Linear links: the pushed task branch, the final commit SHA integrated into
`origin/main`, and a concise validation summary.

Example draft issue:

```text
Background:
The rendered Symphony runtime workflow should match the documented issue
contract so generated and human-created tasks carry the same handoff fields.

Scope:
Document the issue handoff fields and update the runtime workflow prompt to use
the same structure for generated follow-ups.

Acceptance criteria:
- Human-created Symphony tasks have a compact recommended description shape.
- Generated follow-ups use the same handoff fields.
- Agents preserve and update `## Agent Learnings`.

Validation:
- Review docs/symphony.md and WORKFLOW.md for matching section names.
- just docs-build

Non-goals:
- Do not require every backlog issue to be Symphony-ready.
- Do not add labels automatically.

## Agent Learnings
- No durable learnings yet.
```

Create a Linear personal API key in Linear settings. Store secrets and local
project values outside git:

```bash
mkdir -p ~/.config/g-symphony
chmod 700 ~/.config/g-symphony
cat > ~/.config/g-symphony/env <<'EOF'
export LINEAR_API_KEY='replace-with-your-token'
export LINEAR_PROJECT_SLUG='replace-with-your-project-slug'
EOF
chmod 600 ~/.config/g-symphony/env
```

The project slug is the slug from the Linear project URL. `WORKFLOW.md` contains
`__LINEAR_PROJECT_SLUG__`; `just symphony-run` renders a temporary workflow with
the private slug before starting Symphony.

## Generated Tasks

Agents may create follow-up Linear issues when they discover useful work. Only
issues labeled `task-generator` may create new issues with the `symphony`
dispatch label. Other issues may create follow-ups in `Backlog`, but those
follow-ups are not automatically dispatched.

Agents should use the Linear MCP tools for searching existing issues, creating
generated issues, applying labels, and updating comments/descriptions. Raw
Linear API requests are fallback-only.

Use this pattern for discovery work:

- Create a Linear issue with labels `symphony`, `discovery`, and
  `task-generator`.
- Ask for a bounded review, for example: "Find SIMD optimization opportunities
  in the genotype preprocessing path. Create up to 5 concrete Symphony tasks for
  high-confidence implementation opportunities."
- The discovery agent searches for existing Linear issues first, records findings
  in `## Agent Learnings`, and creates at most 5 generated implementation issues
  unless the parent issue states another limit.

For optimization-search issues, use the
[Performance Discovery Playbook](performance-discovery.md). It requires a
baseline signal, suspected bottleneck, proposed change, validation command,
non-goals, duplicate Linear search, and an explicit dispatch decision before a
finding can become an implementation issue.

Generated issues must include:

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

Generated implementation issues are labeled `generated` and, when they are ready
for unattended implementation, `symphony`. Speculative or broad findings stay in
`Backlog` without `symphony`.

## Runtime Install

Install the runtime under the user account:

```bash
curl https://mise.run | sh
~/.local/bin/mise use -g erlang elixir
```

Clone and build Symphony:

```bash
cd /mnt/beegfs/kirill/Projects
git clone https://github.com/openai/symphony
cd symphony/elixir
mise exec -- mix setup
mise exec -- mix build
```

## Run

Check prerequisites:

```bash
mkdir -p /tmp/g-runtime-$(id -u)
JUST_TEMPDIR=/tmp/g-runtime-$(id -u) just symphony-doctor
```

The doctor prints a redacted pass/fail report and exits non-zero if any local
runtime prerequisite is missing. It checks Linear API auth, the configured
Linear project slug, Codex Linear MCP config/auth, Git and GitHub remote
reachability, worktree root writability, `uv`, SLURM client commands, the
Symphony checkout, and the Elixir toolchain. Failure rows include remediation
steps and token-like values are redacted from all output. The command is safe to
run on the login node; it does not start Symphony agents, submit SLURM jobs, or
run benchmarks.

Start the daemon:

```bash
mkdir -p /tmp/g-runtime-$(id -u)
JUST_TEMPDIR=/tmp/g-runtime-$(id -u) just symphony-run
```

The run recipe passes Symphony's required engineering-preview acknowledgement
flag. The daemon launches unattended Codex sessions; reduce
`agent.max_concurrent_agents` in `WORKFLOW.md` if local resources become
contended.

To review the rendered runtime workflow without starting the daemon, point
`SYMPHONY_ELIXIR_DIR` at a path that does not exist and set an explicit
`SYMPHONY_RUNTIME_WORKFLOW`. The recipe writes the temporary workflow before it
attempts to enter the Symphony checkout:

```bash
SYMPHONY_RUNTIME_WORKFLOW=/tmp/g-symphony-review.WORKFLOW.md \
SYMPHONY_ELIXIR_DIR=/tmp/g-symphony-no-server \
  just symphony-run
sed -n '1,260p' /tmp/g-symphony-review.WORKFLOW.md
```

Restart the daemon after changing `WORKFLOW.md`; running agents keep the sandbox
policy they were launched with.

The dashboard is served at:

```text
http://127.0.0.1:4000
```

For a durable shell session, run it inside `tmux`:

```bash
tmux new -s g-symphony
mkdir -p /tmp/g-runtime-$(id -u)
JUST_TEMPDIR=/tmp/g-runtime-$(id -u) just symphony-run
```

## Workspace Model

Symphony creates worktrees under:

```text
/mnt/beegfs/kirill/Projects/g-worktrees/symphony/<linear-issue-key>
```

Each issue branch is named:

```text
symphony/<linear-issue-key>
```

When a task is complete, the agent keeps that branch on GitHub for audit
history and integrates directly with:

```bash
git push -u origin HEAD
git push origin HEAD:main
just symphony-sync-main
```

If `origin/main` has advanced, the agent fetches and merges `origin/main` into
the task branch, resolves conflicts in the task worktree, reruns validation, and
then retries the `HEAD:main` push. If GitHub branch protection blocks direct
pushes to `main`, the agent records that blocker in Linear.

`just symphony-sync-main` safely fast-forwards the local `main` checkout after a
direct merge. It can be run from a Symphony issue worktree because it locates the
worktree that has `main` checked out. It updates only when local `main` is clean
and is an ancestor of `origin/main`; otherwise it reports `skipped` and leaves
the checkout untouched.

The workflow grants Codex write access to the Symphony worktree root and the
main checkout's `.git` metadata so Git worktree commits and branch updates work
inside the sandbox.

If the main checkout has local `data/` or `results/` directories, the
`after_create` hook symlinks them into each Symphony worktree. These paths are
ignored by git and must not be committed.

## Validation Policy

Agents should classify each issue before running validation:

- Login-node-safe work: docs, small code edits, issue bookkeeping, formatting,
  linting, typechecking, focused non-data tests, `just symphony-doctor`, and
  dry-run/plan-only commands. These can run on `gauss` and may proceed in normal
  parallel Symphony capacity.
- CPU-heavy work: full test suites, data preparation, Criterion/Rust
  benchmarks, large matrix runs, native performance builds, and CPU profiles.
  Use a CPU compute node through bounded `srun`; do not run these on the login
  node.
- GPU-heavy work: JAX CUDA probes, commands with `--g-device gpu`, GPU smoke
  tests, GPU benchmarks, and GPU profiles. Use `landau` through `just slurm-*`
  wrappers.
- Data-heavy work: first check required data with an existing verification or
  dry-run recipe. Missing data is a Linear blocker, not a reason to spin.

For routine login-node-safe work, prefer:

```bash
just check-local
just test-local
uv run pytest <focused-tests>
```

GPU work should use existing SLURM recipes such as:

```bash
just slurm-regenie2-binary-gpu-smoke
just slurm-benchmark-regenie2-binary-hot-gpu
```

Benchmark and profiling tasks should prefer smoke or dry-run recipes first, keep
artifacts under ignored paths, and record pending or unavailable SLURM resources
in Linear rather than retrying indefinitely.

Merge integration remains serialized by `agent.max_concurrent_agents_by_state`
for `Merging: 1`. If `origin/main` advances, the merging agent refreshes the
task branch, reruns the appropriate validation lane, pushes the task branch for
history, and then pushes the validated head to `origin/main`.

## Recovery

List active worktrees:

```bash
git worktree list
```

Review stale Symphony worktrees and branches without deleting anything:

```bash
just symphony-cleanup
```

The cleanup dry-run only selects direct child git worktrees under
`SYMPHONY_WORKTREE_ROOT` and never selects `data/`, `results/`, cache paths,
nested paths, or paths outside that configured root. When Linear credentials are
available, the report classifies issue state as active, completed, canceled, or
unknown. Active and unknown issues are retained by default; completed and
canceled issues are stale candidates.

After reviewing the dry-run, apply worktree removal with:

```bash
just symphony-cleanup-apply
```

The apply command uses non-forced `git worktree remove`, so dirty worktrees or
worktrees containing real protected `data/`, `results/`, or cache directories are
not removed. Branch cleanup is separate from worktree cleanup:

```bash
just symphony-cleanup-apply --delete-local-branches
just symphony-cleanup-apply --delete-remote-branches
```

Local branch cleanup uses `git branch -d`; it will refuse unmerged branches.
Remote branch cleanup is intentionally a separate opt-in because it deletes
published branch refs. It does not rewrite branch history.

If Linear is unavailable, issue states are reported as unknown. Unknown states
are not selected unless explicitly requested:

```bash
just symphony-cleanup --include-unknown
```

If the daemon cannot talk to Linear, verify `~/.config/g-symphony/env` and run:

```bash
JUST_TEMPDIR=/tmp/g-runtime-$(id -u) just symphony-doctor
```

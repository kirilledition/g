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

Start the daemon:

```bash
mkdir -p /tmp/g-runtime-$(id -u)
JUST_TEMPDIR=/tmp/g-runtime-$(id -u) just symphony-run
```

The run recipe passes Symphony's required engineering-preview acknowledgement
flag. The daemon launches unattended Codex sessions; reduce
`agent.max_concurrent_agents` in `WORKFLOW.md` if local resources become
contended.

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
```

If `origin/main` has advanced, the agent fetches and merges `origin/main` into
the task branch, resolves conflicts in the task worktree, reruns validation, and
then retries the `HEAD:main` push. If GitHub branch protection blocks direct
pushes to `main`, the agent records that blocker in Linear.

The workflow grants Codex write access to the Symphony worktree root and the
main checkout's `.git` metadata so Git worktree commits and branch updates work
inside the sandbox.

If the main checkout has local `data/` or `results/` directories, the
`after_create` hook symlinks them into each Symphony worktree. These paths are
ignored by git and must not be committed.

## Validation Policy

Agents should prefer:

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

Do not run heavy workloads on the login node.

## Recovery

List active worktrees:

```bash
git worktree list
```

Remove a stale Symphony worktree:

```bash
git -C /mnt/beegfs/kirill/Projects/g worktree remove --force /mnt/beegfs/kirill/Projects/g-worktrees/symphony/<issue-key>
```

If the daemon cannot talk to Linear, verify `~/.config/g-symphony/env` and run:

```bash
JUST_TEMPDIR=/tmp/g-runtime-$(id -u) just symphony-doctor
```

# Performance Discovery Playbook

Use this playbook when a Linear issue asks an agent to search for optimization
opportunities instead of implementing one specific change. Discovery work must
produce evidence, ranking, and bounded implementation tasks. It must not turn a
broad performance prompt into unbounded profiling or speculative refactoring.

## Scope

Performance discovery is allowed to:

- Identify one target path, workload, or benchmark family.
- Run lightweight planning commands on the login node.
- Submit smoke or full profiling through SLURM when the issue explicitly asks
  for profiling evidence.
- Rank findings by measured signal, implementation confidence, and validation
  cost.
- Create follow-up Linear issues only within the task-generation rules below.

Performance discovery is not allowed to:

- Perform the optimization itself.
- Run heavy computation, data preparation, large test suites, GPU commands, or
  native performance builds on the gauss login node.
- Create dispatchable follow-ups from issues that are not labeled
  `task-generator`.
- Promote speculative findings to Symphony dispatch without a human decision.

## Login-Node-Safe Preparation

On the gauss login node, keep work limited to repository inspection, issue
bookkeeping, small documentation/code reads, and dry-run or plan-only commands. The
following command is safe and is the first check when choosing an existing
recipe:

```bash
just help
```

Use `documentation/development/justfile.md`, `documentation/public/gpu-and-clusters.md`, and the issue validation
section to choose a profiling recipe. Do not run a workload merely because a
recipe exists.

For CPU-heavy work, use a bounded CPU compute allocation, for example `cantor`
when available:

```bash
srun --nodelist=cantor --cpus-per-task=<cores> --mem=<memory> --time=<limit> \
  bash -lc 'cd <worktree> && <command>'
```

For GPU-heavy work, use `landau` through the repo's SLURM helpers. Prefer a
smoke run before a full profiling run:

```bash
just slurm-gpu-just profile-app-full-smoke tool.output_dir=data/profiles/<issue-key>-smoke
just profile-app-full-landau tool.output_dir=data/profiles/<issue-key>-full
```

Keep profile, benchmark, and trace artifacts under ignored paths such as
`data/profiles/` or `results/`. Never commit those artifacts.

## Discovery Steps

1. Define the target.
   State the exact path, command, workload, or user-facing scenario being
   investigated. Include the Linear issue identifier and current commit SHA.

2. Record the baseline before proposing changes.
   Capture the command, host or SLURM node, date, data path, workload size,
   relevant environment variables, and artifact location. A baseline signal may
   be a profiler trace, stage timing, cProfile entry, Criterion median, memory
   profile, output-size measurement, or a reproducible dry-run plan when the
   issue is planning-only.

3. Isolate the suspected bottleneck.
   Connect the baseline to a specific hot function, stage, kernel, I/O phase,
   allocation site, synchronization point, or benchmark group. If the signal
   only says "the command is slow", keep the finding in discovery notes.

   For startup findings, separate first-process costs from amortizable costs.
   Python imports, JAX plugin discovery, backend initialization, and dynamic
   library loading are unavoidable for a fresh Python process. Before proposing
   import or backend changes, include either a same-process hot measurement or a
   multi-phenotype measurement that shows the cost still matters after users can
   batch work in one process. Use
   `-m tooling.cli.benchmark tool.name=linear_startup tool.same_process_trials=3`
   for quantitative Step 2 startup questions.

   GLA-43 measured this on 2026-06-09: fresh CPU median was 15.77s versus
   6.79s for hot same-process CPU, and fresh GPU median was 22.51s versus
   2.14s for hot same-process GPU. The production decision was to keep backend
   validation intact and prefer batching/repeated API calls in one process over
   import-boundary cleanup unless a future profile remains slow after this
   amortization.

4. Propose the smallest plausible change.
   Describe one implementation direction, the files likely to change, expected
   benefit, and known risks. Do not bundle adjacent cleanup or broader rewrites.

5. Name the validation command.
   Give the exact local, SLURM, benchmark, or test command a future
   implementation issue should run. Use smoke validation first when available,
   then name any full-run evidence required before merging.

6. State non-goals.
   Exclude statistical-mode changes, data preparation, output contract changes,
   unrelated refactors, and further profiling that should not be part of the
   implementation issue.

7. Rank findings.
   Rank only findings with a baseline signal. Use impact, confidence, and
   implementation risk:

   | Rank | Meaning |
   | --- | --- |
   | P1 | Strong measured signal, narrow change, clear validation command. |
   | P2 | Measured signal, plausible narrow change, moderate validation cost or risk. |
   | P3 | Useful but small, risky, or blocked on extra confirmation. |
   | Backlog | Speculative, duplicate-prone, broad, or missing a baseline signal. |

## Required Discovery Output

Every discovery finding must use this shape in the workpad or discovery
document:

```text
Finding: <short title>
Rank: <P1/P2/P3/Backlog>
Baseline signal:
<Command, host/node, commit, workload, artifact path, and measured signal.>
Suspected bottleneck:
<Specific stage, function, kernel, I/O phase, allocation, or synchronization.>
Proposed change:
<Small implementation direction and likely files.>
Validation command:
<Exact command a follow-up should run.>
Non-goals:
- <Explicit exclusion.>
Duplicate check:
<Linear search terms used and matching issues reviewed.>
Dispatch decision:
<Backlog only, or generated+symphony when allowed and ready.>
```

If any required field is missing, the finding is not implementation-ready.

## Linear Duplicate Checks

Before creating any follow-up issue, search existing Linear issues with the
Linear MCP tools. Search by at least:

- The main file, module, recipe, or CLI option.
- The suspected bottleneck phrase.
- The proposed change or optimization technique.
- Any existing planning-document finding identifier.

Record the search terms and reviewed matches in the parent issue's
`## Codex Workpad`. If an existing issue covers the same work, update the
workpad with the duplicate and do not create another issue.

## Follow-Up Issue Limits And Labels

Agents may create follow-up issues only when the parent issue allows task
generation.

- Parent without `task-generator`: may create only `Backlog` follow-ups and
  must not add `symphony`.
- Parent with `task-generator`: may create at most 5 generated issues with
  `symphony` unless the parent issue states a different limit.
- All agent-created follow-ups receive `generated`.
- Add `symphony` only for concrete, bounded implementation work with all
  required discovery fields, a clean duplicate check, and an exact validation
  command.
- Add routing labels such as `cpu`, `gpu`, `benchmark`, `data`,
  `optimization`, or `simd` only when they describe the expected validation or
  implementation lane.
- Do not add `task-generator` to generated issues unless the parent explicitly
  asks for recursive task generation.
- Speculative, broad, low-confidence, or missing-baseline findings remain in
  `Backlog` without `symphony` until a human promotes them.

Generated performance issues must include the parent identifier and URL, the
baseline signal, suspected bottleneck, proposed change, validation command, and
non-goals. Use the generated issue template in `documentation/development/symphony.md`, adding the
performance-specific fields under `Background` when necessary.

## Completion Checklist

- Baseline signal recorded before any proposed implementation.
- Heavy work kept off the login node.
- Findings ranked and non-goals stated.
- Duplicate Linear searches recorded.
- Generated issues stayed within the allowed count and label rules.
- Speculative findings left in `Backlog` without `symphony`.
- Workpad updated with validation evidence and any durable learnings.

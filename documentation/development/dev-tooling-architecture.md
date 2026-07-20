# Development Tooling Architecture

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | current tooling architecture | Development tooling maintainers |

The top-level `tooling/` package is development-only. It is excluded from the
installed Python package and cannot define public application behavior.
Production REGENIE options and execution remain in `src/g`, `crates/`, and the
TOML-backed `g regenie --config` flow.

## Boundaries

The maintained tooling layers are:

- `tooling.cli`: Hydra entrypoints and campaign orchestration;
- `tooling.benchmark`: benchmark implementations and the shared native
  lifecycle harness;
- `tooling.profile_deep`: typed deep-profile models, budget/config policy,
  profiler wrappers, diagnostics, and reports;
- `tooling.common`: production command/config rendering, subprocess handling,
  cache snapshots, downloads, registry dispatch, and artifact contracts;
- `tooling.configs`: saved dataset, machine, telemetry, workload, and complete
  workflow configurations;
- `tooling.data`, `tooling.debug`, `tooling.performance`, and `tooling.server`:
  narrow domain implementations.

The canonical shared contracts are:

- `tooling.common.g_regenie` for generated production TOML and
  `g regenie --config` command vectors;
- `tooling.benchmark.native_lifecycle` for fresh, discarded-warm, hot,
  telemetry, cache, and completed-output evidence;
- `tooling.common.commands` for shell-free subprocesses, logs, timeouts, and
  redacted environment capture;
- `tooling.common.jax_cache` for CPU feature-aware cache paths and
  fingerprints;
- `tooling.common.reports` and `tooling.common.artifact_format` for versioned
  machine-readable evidence;
- `tooling.common.registry` for grouped `tool.name` dispatch.

Benchmark and profiler modules should reuse these boundaries instead of
building production argument vectors, output discovery, cache logic, or report
dictionaries independently.

## Configuration Ownership

Every maintained Hydra entrypoint selects an explicit saved config. Workflow
identity and defaults live in `tooling/configs/`; Justfile recipes only select
those configs and forward intentional overrides. Hydra must keep
`hydra.job.chdir: false` so repository-relative inputs remain stable.

There is no generic context object or generic sweep framework. An entrypoint
parses only the fields it owns into a typed arguments dataclass. Reusable
machine and dataset values come from Hydra config groups, while each campaign
owns its valid tuning dimensions and validation.

## Production Invocation

Development tools may invoke production in two supported ways:

1. a fresh subprocess running `g regenie --config <generated.toml>`;
2. a same-process call to `g._core.cli.run(["regenie", "--config", ...])` when
   measuring an already-loaded native lifecycle.

Both use `tooling.common.g_regenie` to generate the same application config.
Legacy Python orchestration APIs, native PyClasses created only for benchmarks,
alternate output formats, and post-run finalization are not compatibility
surfaces.

## Evidence Ownership

Headline timing and diagnostics are different lifecycles. Production headline
runs use telemetry off. Profile runs use a distinct output root and may collect
native stage summaries or profiler artifacts; they never contribute to the
headline elapsed-time distribution.

The shared lifecycle helper owns exact cache-tree snapshots and validates
successful manifests, exact committed chunk coverage, safe and complete
Parquet part sets, row counts, schemas, metadata, content hashes, and telemetry
evidence. Persistent JAX caches are campaign-owned, must start in the state
declared by the workflow, and are snapshotted before and after every lifecycle
whose cache semantics matter.

Tooling Artifact Format v1 is an internal development evidence format. Its
version and the benchmark-, matrix-, and deep-profile-specific evidence schema
versions are intentionally independent from public pre-release product
contracts. Public option, output-manifest, telemetry, and profile-summary
contracts remain at version 0 until release; do not reset internal tooling
evidence versions when changing those product contracts.

The canonical bundle may include `artifact_manifest.json`, `report.json`,
`events.jsonl`, `metrics.jsonl`, `commands/commands.jsonl`, resolved Hydra/tool
configuration, and `summary.md`. Validate a completed bundle with:

```bash
uv run --no-sync python -m tooling.cli.schema_check \
  --config-name schema_check tool.path=<artifact-directory>
```

## Execution Policy

Dry plans and small report checks may run on the login node. CPU compilation,
large tests, and Criterion belong on an exclusive CPU allocation. GPU benchmark
and profiler runs must be serialized on `landau` through the Justfile SLURM
wrappers.

The matrix workflows replace handwritten application command recipes. Bounded
execution requires a prepared bounded BGEN fixture because the production CLI
does not expose a variant-limit option. Criterion targets run directly by
package and bench name; they are not embedded in the GPU profiler orchestrator.

See [Development Tooling Guide](tooling.md) for commands and evidence details.

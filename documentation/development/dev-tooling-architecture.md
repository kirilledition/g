# Development Tooling Architecture

Development-only benchmark and profiling entrypoints live in the top-level `tooling/` package. This package is intentionally not listed in `tool.maturin.python-packages` and is not exposed through `[project.scripts]`, so packaged consumers continue to receive only `src/g` and the existing `g` entrypoints.

Hydra is used for development tool parameters, saved development configurations, benchmark campaigns, machine profiles, telemetry defaults, and sweep profiles. Production REGENIE configuration remains in `src/g` and continues to use the TOML-backed `RegenieConfig` and `ExecutionPlan` flow.

The tooling package now has a small internal framework:

- `tooling.common.context` resolves repository, data, output, machine, telemetry, cwd, and Hydra path policy at execution time.
- `tooling.common.registry` owns grouped CLI dispatch tables.
- `tooling.common.commands` owns shell-free subprocess execution, logs, timeouts, and redacted environment reporting.
- `tooling.common.reports` owns versioned JSON report validation.
- `tooling.common.artifact_format` owns Tooling Artifact Format v1 report, manifest, metric, event, command, failure, finding, and comparison models.
- `tooling.common.g_regenie` owns shared `g regenie` CLI rendering and Python API payload rendering.
- `tooling.common.downloads` owns Pooch-backed retrieval, cache reuse, SHA-256 validation, archive processors, and download manifests for data and server-tool assets.
- `tooling.profile_deep.models` owns the deep-profile enums and dataclasses so the profiler package can be split without circular imports through the CLI module.

Benchmark and profiler code should use these contracts instead of rebuilding command vectors, report dictionaries, or grouped dispatch chains by hand.

Artifact-producing tools should write Tooling Artifact Format v1 beside any legacy compatibility summaries. The canonical machine-readable files are `artifact_manifest.json`, `report.json`, `events.jsonl`, `metrics.jsonl`, `commands/commands.jsonl`, `config/resolved_hydra.yaml`, `config/resolved_tool.json`, and `summary.md`. Use `uv run --no-sync python -m tooling.cli.schema_check tool.path=<output_dir>` to validate a completed artifact directory.

The base tooling config is `tooling/configs/config.yaml`. It sets:

```yaml
hydra:
  job:
    chdir: false
```

This preserves the current repository-relative behavior of benchmark commands. Dataset paths still honor `GWAS_ENGINE_DATA_DIR`.

Data and server-tool retrieval should go through the shared Pooch wrapper rather
than calling `urllib`, `zipfile`, or `tarfile` directly. Tool-specific code may
still own installation steps after retrieval, such as linking executables or
extracting `.deb` payloads.

The long-form usage and extension guide is `documentation/development/tooling.md`.

The migrated benchmark and profiling commands are invoked through module execution. There is no compatibility-wrapper layer under `scripts/` for these entrypoints. Justfile recipes call the modules directly, for example:

```bash
uv run --no-sync python -m tooling.cli.benchmark_bgen_reader
```

Native extension build-profile measurements use the same pattern:

```bash
uv run --no-sync python -m tooling.cli.rust_build_profiles tool.labels=[dev-fast]
```

The build-profile harness writes timestamped JSON/Markdown summaries under
`results/perf/rust-build-profiles/`, stores command logs beside each summary,
and keeps per-profile Cargo artifacts isolated under `target/rust-build-profiles/`.

Optional GPU smoke validation should run through SLURM rather than on the head node:

```bash
just slurm-gpu-run 'uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot machine=landau_gpu tool.variant_limit=1000 tool.include_cold_process=false tool.include_finalized_hot=false'
```

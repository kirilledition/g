# Review of `tooling/`

## Executive summary

The move from loose scripts to a real `tooling/` Python package is absolutely the right direction. It gives
the project a single development-only namespace, reusable Hydra configuration, shared path/report/command
helpers, and a clearer boundary between production `g` and benchmark/profiling/data-prep workflows.

The current package is already better than a `scripts/` folder. The docs explicitly define `tooling/` as
development-only, not packaged through Maturin, and not exposed as public project scripts. That is the right
safety boundary: packaged users get only `src/g` and the `g` entrypoint, while development workflows live in
`tooling/`.

But the implementation is only halfway to the intended design. It is currently a package of migrated scripts,
not yet a clean tooling framework. The largest risks are configuration drift, duplicated command construction,
overly large monolithic tools, and weak schema/version contracts for benchmark artifacts.

Overall verdict:

```text
Direction: excellent
Current architecture: useful but still script-shaped
Main refactor needed: introduce a shared tool framework + typed command/report contracts
Urgency: medium-high, because tooling now drives profiling, benchmarking, and migration validation
```

## Highest-risk findings

### P1 - `tooling.configuration` is not the real source of truth for most tools

The package has a central `tooling.configuration` module with structured dataclasses for `dataset`, `machine`,
`workload`, `telemetry`, and `sweep`. It registers those schemas with Hydra and exposes `compose_config()` and
`instantiate_config()`.

But most real tools are driven by large, tool-specific `tool:` nodes. `benchmark_regenie2_binary_hot.yaml`
defines many fields under `tool`, including input paths, output settings, storage modes, Firth settings, stage
timing, process mode flags, and JAX cache settings. `profile_regenie2_deep.yaml` has an even larger schema with
profiler toggles, budgets, sweep grids, timeouts, REGENIE baseline settings, and more.

`instantiate_config()` does not instantiate these real tool schemas. Individual tools manually pull values out
of `config.tool` with string keys and casts. This weakens the unified configuration goal because Hydra composes
the config, but actual schema validation is mostly manual and scattered.

Suggested fix: create an internal tooling framework with `ToolContext`, `ToolSpec`, typed tool argument builders,
and tool-specific structured schemas registered with Hydra.

### P1 - `profile_regenie2_deep.py` is too large to be reliable as one module

`tooling.cli.profile_regenie2_deep` is doing too much. It defines profiler enums, workload selectors,
campaign-budget accounting, large argument dataclasses, trial/result dataclasses, baseline scoping, artifact
manifests, command building, child Python code generation, JAX cache diagnostics, profiler detection,
execution, summary aggregation, diagnostics extraction, report writing, and campaign orchestration.

The profiler is important enough to deserve a package, not one mega-module. Suggested split:

```text
tooling/profile_deep/
  __init__.py
  models.py
  config.py
  budget.py
  commands.py
  profilers.py
  jax_cache.py
  diagnostics.py
  artifacts.py
  reports.py
  runner.py
```

### P1 - Command construction is duplicated and too easy to drift from production `g`

Several tools construct `g regenie` commands or Python API payloads by hand. The matrix runner builds CLI
argument vectors directly. The deep profiler generates inline `python -c` code that manually constructs
`api.regenie.from_options(...)` dictionaries. The binary-hot benchmark constructs Python API options directly.

Suggested fix: add `tooling.common.g_regenie` with `RegenieRunSpec`, compute/output/diagnostic/binary options,
`render_g_regenie_cli()`, `render_python_api_options()`, and `expected_output_run_directory()`.

Tests should assert required `--out`, binary versus quantitative flags, no stale `--g-*` aliases, and that
Python options round-trip through `g.RegenieConfig.from_options`.

### P1 - Report and artifact schemas need explicit versioned contracts

The tooling writes many JSON reports and manifests, but most are ad hoc dataclass/dict serializations. The
shared report helper is convenient but does not enforce schemas.

Suggested fix: define versioned report models and validate write/read for durable artifacts, especially binary
hot summaries, profile campaign summaries, matrix manifests, artifact manifests, performance smoke summaries,
and parity reports.

#### Addendum - Tooling Artifact Format v1

Standardize durable tooling output around a three-layer artifact format:

```text
1. Machine source of truth: JSON / JSONL
2. Human + agent reading layer: Markdown
3. Large tabular optional layer: Parquet or CSV only when needed
```

Markdown must not be the source of truth, and arbitrary nested JSON blobs should not be the only output either.
The standard artifact directory should be:

```text
artifact_manifest.json      # index and provenance for the whole run
report.json                 # normalized summary and metrics
events.jsonl                # structured event log
metrics.jsonl               # optional long-form metrics table
summary.md                  # readable human/agent summary
logs/*.stdout.log           # raw child stdout
logs/*.stderr.log           # raw child stderr
config/*.yaml/json          # resolved config snapshots
```

For full benchmark and profiling campaigns, use this layout:

```text
<output_dir>/
|-- artifact_manifest.json
|-- report.json
|-- summary.md
|-- events.jsonl
|-- metrics.jsonl
|-- comparisons.json
|-- config/
|   |-- resolved_hydra.yaml
|   |-- resolved_tool.json
|   `-- effective_g_config.toml
|-- commands/
|   |-- commands.jsonl
|   `-- child_commands.md
|-- logs/
|   |-- <run_id>.stdout.log
|   |-- <run_id>.stderr.log
|   `-- events.jsonl
|-- profiles/
|-- outputs/
`-- data/
```

Small tools can use the minimal layout:

```text
<output_dir>/
|-- artifact_manifest.json
|-- report.json
`-- summary.md
```

Every machine-readable artifact should use a common envelope:

```json
{
  "schema_name": "g.tooling.report",
  "schema_version": 1,
  "producer": {
    "tool_name": "benchmark_regenie2_binary_hot",
    "tool_version": 1,
    "repository": "kirilledition/g",
    "git_head": "25ec7701b8b4a9339d424d8d08b1bc4527ddb2c8",
    "dirty": false,
    "dirty_diff_sha256": null
  },
  "run": {
    "run_id": "binary-hot-20260626T120102Z-3f42d9",
    "created_at": "2026-06-26T12:01:02Z",
    "status": "success",
    "status_reason": null,
    "output_directory": "results/perf/gpu/binary-hot-..."
  },
  "summary": {},
  "metrics": [],
  "artifacts": [],
  "findings": [],
  "recommended_actions": []
}
```

Use this envelope for `report.json`, `artifact_manifest.json`, `comparisons.json`,
`dataset_manifest.json`, and `release_gate_report.json`. Use `snake_case` field names everywhere; do not
mix names such as `returncode` and `return_code`.

`artifact_manifest.json` should answer what was produced, where it is, what generated it, whether an agent can
trust it, and how an agent should read it. It should include producer metadata, run identity, context
snapshot, primary artifact paths, artifact records with size/hash/media type, external input records,
configuration snapshot paths, and notes. The existing deep-profiler manifest is the right direction, but it
should become a universal shared contract rather than a one-off dictionary.

`report.json` should be the single source of truth for results. Recommended top-level fields:

```json
{
  "schema_name": "g.tooling.report",
  "schema_version": 1,
  "producer": {},
  "run": {},
  "context": {},
  "configuration": {},
  "summary": {
    "title": "Binary REGENIE Step 2 GPU hot benchmark",
    "status": "success",
    "headline": "Hot no-final median improved by 3.2%; finalized output unchanged.",
    "primary_metric": {
      "metric_name": "wall_time_seconds",
      "value": 4.07,
      "unit": "s",
      "aggregation": "median",
      "case_id": "binary_gpu_packed8_default"
    }
  },
  "cases": [],
  "trials": [],
  "metrics": [],
  "comparisons": [],
  "diagnostics": {},
  "failures": [],
  "findings": [],
  "recommended_actions": []
}
```

Metrics should be long-form, not only nested in tool-specific structures:

```json
{
  "metric_name": "wall_time_seconds",
  "value": 4.070816,
  "unit": "s",
  "aggregation": "median",
  "higher_is_better": false,
  "case_id": "binary_gpu_packed8_default",
  "trial_id": null,
  "phase": "hot_same_process_no_final",
  "dimensions": {
    "trait_type": "binary",
    "device": "gpu",
    "output_format": "parquet",
    "finalize_parquet": false,
    "gpu_genotype_format": "packed8",
    "variant_limit": null,
    "chunk_size": 8192
  },
  "source": {
    "artifact_path": "report.json",
    "json_pointer": "/cases/0/results/hot_same_process_no_final/median_wall_time_seconds"
  }
}
```

`events.jsonl` should be append-only JSON Lines, one event per line. Every event should have
`timestamp`, `level`, `tool_name`, `run_id`, `phase`, `event`, `message`, and JSON-serializable `fields`.
Large text should live in logs and be referenced by path.

`metrics.jsonl` should contain one normalized benchmark metric per line. It is the first file agents and
comparison tools should read for performance analysis. Large runs can additionally write `metrics.parquet`,
but JSONL remains the lowest-common-denominator exchange format.

`summary.md` should be optimized for human and agent reading. It should include status, run ID, commit,
machine, dataset, output directory, executive summary, headline metrics, findings, failures/skipped work,
recommended actions, and an artifact map. Every claim should point to structured files.

Use the same statuses everywhere:

```text
success
partial
failed
skipped
unsupported
dry_run
interrupted
timed_out
invalid
```

Definitions:

```text
success      all required work completed
partial      core run completed but optional sections failed/skipped
failed       required work failed
skipped      intentionally not run
unsupported  not applicable on this machine/config
dry_run      planned but did not execute workload
interrupted  stopped by signal/user/system
timed_out    stopped by timeout
invalid      inputs/config/artifacts malformed
```

Do not invent tool-local alternatives like `ok`, `error`, `done`, `missing`, or `not_available`; put details in
`status_reason`.

Metric names should be stable and unambiguous:

```text
wall_time_seconds
cpu_time_seconds
gpu_time_seconds
peak_rss_bytes
peak_gpu_memory_bytes
output_row_count
output_file_count
output_total_bytes
final_parquet_bytes
chunk_count
variant_count
sample_count
phenotype_count
candidate_count
throughput_rows_per_second
throughput_variants_per_second
jax_compile_count
jax_cache_hit_count
jax_cache_miss_count
stage.<stage_name>.seconds
```

Units should use stable tokens such as `s`, `ms`, `us`, `bytes`, `count`, `row`, `variant`, `sample`,
`phenotype`, `ratio`, and `percent`.

Comparisons should be first-class in `comparisons.json`, with baseline/current report references,
thresholds, long-form comparison rows, and judgements from this set:

```text
improvement
neutral
regression
inconclusive
not_comparable
```

Every `report.json` should include an agent-readable summary with one sentence, key observations, risks, and
next actions. Agents need concise status, explicit failures, exact artifact paths, exact commands, exact git
commit, stable metric names, units, `null` rather than missing values when unavailable, and no huge raw logs
embedded in JSON.

Child commands should be recorded in `commands/commands.jsonl` with command ID, tool name, phase, shell-free
args, display command, cwd, redacted environment overrides, stdout/stderr log paths, status, return code,
timestamps, and wall time. Inline `python -c` commands should be written to
`commands/scripts/<command_id>.py`, and command records should point to the script file instead of embedding
large code strings.

Raw logs should be kept but never required as the primary source for agents. Structured failures should include
bounded excerpts plus stdout/stderr log paths and a `command_id`.

Every artifact-producing tool should snapshot configuration:

```text
config/resolved_hydra.yaml
config/resolved_tool.json
config/g_effective/<command_id>.toml
```

Paths inside artifact JSON should be relative to `output_directory` unless they are external inputs. Use
explicit `path_type` values such as `artifact_relative` and `absolute_input`.

Schema versions are simple integers. Adding optional fields does not require a bump; removing or renaming
fields, changing units, or changing metric semantics does require a bump or a new metric name. Readers should
reject unknown future major versions.

Recommended implementation order:

1. Add `tooling/common/artifact_format.py` with `ToolProducer`, `ToolRunIdentity`, `ToolContextSnapshot`,
   `ArtifactRecord`, `InputFileRecord`, `CommandRecord`, `MetricRecord`, `FailureRecord`, `FindingRecord`,
   `RecommendedAction`, `ReportEnvelope`, and `ArtifactManifest`.
2. Extend `tooling.common.reports` with `write_report_envelope()`, `write_jsonl()`, and
   `validate_report_envelope()`.
3. Add `schema_name` and `schema_version` envelopes to current high-value reports:
   `benchmark_regenie2_binary_hot`, `run_regenie2_matrix`, `profile_regenie2_deep`,
   `benchmark_bgen_reader`, and performance smoke/compare.
4. Add `commands.jsonl` to tools that launch child processes.
5. Add `metrics.jsonl` to benchmark tools, starting with binary-hot, output stages, callback overhead, and
   the REGENIE step 2 matrix runner.
6. Add a schema checker CLI and wire it into a tooling-specific check.

## P2 / Medium findings

### P2 - The grouped CLIs still use hand-written dispatch chains

Grouped entrypoints define enums and dispatch through `if` statements. Replace this with a registry so dispatch,
documentation, and tests share the same source of truth.

### P2 - Several parsing helpers silently drop empty comma-list entries

`tooling.common.sweeps` silently drops empty list entries. Use strict parsing by default and require explicit
sentinels such as `null`, `default`, or `none`.

### P2 - Boolean conversion helper is unsafe for string values

`boolean_value("false")` returns `True`. Make it strict, or support explicit true/false string parsing.

### P2 - Path and environment behavior is still inconsistent

Some tools resolve environment-dependent defaults at import time. Resolve these values in a tool context instead
and include final path policy in reports.

### P2 - Subprocess execution helpers are fragmented

Create a command runner abstraction that supports captured output, streaming output, timeout, structured result,
environment diff recording, redaction, `cwd=REPOSITORY_ROOT` defaults, and shell-free argument vectors.

### P2 - Optional profiler/tool installation needs stricter operational controls

Use Pooch for development-only download and fixture registries: `tooling.data.fetch`, server tool downloads where
practical, test fixture downloads, benchmark dataset registries, prediction-list fixture registries, and external
baseline data registries. Pooch should own retrieval, cache reuse, SHA-256 validation, and archive processors such
as ZIP/tar extraction. Tool-specific code can still own install logic after retrieval, such as linking executables
or extracting `.deb` payloads.

Server downloads should have timeouts, stream to disk, record URL/hash/file size in an install manifest, and clean
up partial downloads.

### P2 - Documentation should be generated or checked against registries

Once the tool registry exists, generate or verify tool names, config names, Justfile recipe references, workload
keys, report artifact names, and required external tools.

## Recommended next actions

1. Keep the package and current Justfile workflows.
2. Add the shared `g_regenie` command-spec layer first.
3. Split `profile_regenie2_deep.py` into a package.
4. Introduce report schema models.
5. Create a tool registry.
6. Add guardrail tests:

```text
Every rendered g command has --out.
No rendered command uses --g-* flags.
Every tool default config composes.
Every tool can produce a dry-run artifact.
Every JSON summary has a schema_version.
```

## Bottom line

The idea is good and the current module is a meaningful improvement over scripts. But to fully realize the goal,
the next step is a small internal tooling framework:

```text
typed config
+ typed command specs
+ typed reports
+ one command runner
+ tool registry
+ split profiler package
```

That will make `tooling/` reliable enough to guide the Rust migration instead of becoming another source of drift
while the production architecture is moving.

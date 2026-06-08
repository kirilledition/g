# Development Tooling Guide

This guide documents the repository-local `tooling/` package: what it contains,
how to run the tools, which Hydra profiles are saved, and how to extend the
tooling system.

The short architecture note is in `docs/development/dev-tooling-architecture.md`. This file
is the operational reference.

## Scope

`tooling/` is development-only code. It is for benchmark campaigns, profiling
campaigns, machine profiles, workload profiles, telemetry defaults, report
writing, and reusable benchmark adapters.

`tooling/` is intentionally not listed in `tool.maturin.python-packages` and is
not exposed through `[project.scripts]`. Packaged users continue to receive only
`src/g` and the public `g` entrypoints.

Do not put production REGENIE behavior, public API types, or packaged CLI
behavior in `tooling/`. Production configuration remains in `src/g` and uses the
TOML-backed `RegenieConfig`, `ExecutionPlan`, and existing `g` command flow.

## Directory Map

```text
tooling/
  cli/
    benchmark_bgen_reader.py
    benchmark_output_stages.py
    benchmark_regenie2_binary_hot.py
    profile_regenie2_deep.py
    run_regenie2_matrix.py
    tune_regenie2_gpu.py
  common/
    commands.py
    hydra_arguments.py
    hydra_compat.py
    logging.py
    paths.py
    reports.py
    sweeps.py
  configs/
    benchmark_bgen_reader.yaml
    benchmark_output_stages.yaml
    benchmark_regenie2_binary_hot.yaml
    config.yaml
    profile_regenie2_deep.yaml
    run_regenie2_chr10_matrix.yaml
    run_regenie2_chr22_matrix.yaml
    tune_regenie2_gpu.yaml
    dataset/
    machine/
    sweep/
    telemetry/
    workload/
  regenie/
    bgen_reader.py
  configuration.py
```

## Hydra Interface

All migrated `tooling.cli` entrypoints are Hydra-driven. Run them with module
execution and pass Hydra overrides:

```bash
uv run --no-sync python -m tooling.cli.benchmark_bgen_reader
uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot machine=landau_gpu tool.variant_limit=1000
```

Use `--cfg job` to inspect the composed tool config:

```bash
uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot --cfg job
```

Use `--info defaults` to inspect the selected config groups:

```bash
uv run --no-sync python -m tooling.cli.benchmark_bgen_reader --info defaults
```

Hydra override rules are the public parameter interface:

- Group override: `machine=landau_gpu`
- Scalar override: `tool.variant_limit=1000`
- Boolean override: `tool.include_cold_process=false`
- List override: `sweep.chunk_sizes=[1024,2048,4096]`
- Path override: `telemetry.json_summary_path=data/profiles/run.json`

Every tool config sets:

```yaml
hydra:
  job:
    chdir: false
```

Keep this setting. It preserves repository-relative paths for benchmark inputs,
outputs, and reports.

## Entry Points

`tooling.cli.benchmark_bgen_reader`

Benchmarks native BGEN chunk delivery paths, including variant-major dosage and
packed8 probability-pair paths. It supports chunk-size sweeps, sample-selection
modes, trusted no-missing diploid mode, decode tile-size sweeps, Rayon thread
sweeps, JSON summaries, and Markdown summaries.

`tooling.cli.benchmark_regenie2_binary_hot`

Benchmarks binary REGENIE step 2 while separating cold process, same-process
hot, no-final output, and finalized Parquet timings. It supports storage mode
sweeps, fallback-density scenarios, binary trait-count sweeps, Firth batch-size
sweeps, Firth candidate-capacity sweeps, stage timing mode, and explicit JAX
cache paths.

The JSON summary includes `binary_diagnostics_by_case`, keyed by benchmark case
and mode. Exact stage-timing runs persist candidate counts, Firth outcome
counts, failure-code counts, correction branch and attempt counts, sparse/dense
correction counts, Firth iteration summaries, code label values, and the
available stage timing metadata used to interpret them. Throughput runs with
`telemetry.stage_timing_mode=off` still emit the same diagnostic object, but
mark it unavailable and set diagnostic counts to `null`.

`tooling.cli.benchmark_output_stages`

Benchmarks output-stage behavior across bsize, phenotype count, writer threads,
queue depth, chunks per Arrow file, Arrow compression, finalization, and optional
JAX tracing.

`tooling.cli.profile_regenie2_deep`

Runs the deep landau profiling campaign for original REGENIE and `g` REGENIE
step 2. It includes BGEN pre-sweeps, candidate tuning, headline trials,
optional perf/py-spy/cProfile/JAX trace runs, and a smoke mode.

`scripts/benchmark_regenie2_linear_fresh_process.py`

Benchmarks quantitative REGENIE step 2 startup behavior. By default it measures
fresh Python child-process wall time, including interpreter startup, imports,
JAX backend setup, and the run itself. Pass `--same-process-trials` to append a
hot same-process section to the JSON report, `--multi-phenotype-count` to
generate cloned quantitative traits for amortization measurements, and
`--emit-stage-timings` to write per-trial stage timing JSON. Same-process trials
disable telemetry output so repeated runs can share one process-global logging
configuration.

`tooling.cli.run_regenie2_matrix`

Runs the standard chromosome binary and linear REGENIE step 2 comparison matrix:
CPU, GPU, and GPU with the JAX persistent cache already populated. Saved configs
make this directly available for chr10 and chr22. It executes production
`g regenie` commands in isolated subprocesses, writes a manifest and Markdown
report, keeps per-run telemetry logs, and compares against the previous matrix
manifest for the same chromosome prefix.

`tooling.cli.tune_regenie2_gpu`

Runs the sequential GPU tuning workflow for REGENIE step 2 and BGEN reader
knobs. It ranks BGEN candidates, compute-stage candidates, writer-stage
candidates, and finalists for quantitative and binary trait modes.

## Justfile Entry Points

Prefer Justfile recipes for common workflows:

```bash
just benchmark-bgen-reader
just benchmark-regenie2-binary-hot-gpu tool.variant_limit=1000
just benchmark-output-stages-gpu tool.trials=1
just benchmark-regenie2-binary-hot-gpu-smoke telemetry.stage_timing_mode=exact
just perf-smoke
just perf-cpu sweep.chunk_sizes=[4096,8192]
just perf-gpu tool.variant_limit=1000
just perf-compare results/perf/baseline.json results/perf/new.json
just regenie2-chr10-matrix-dry-run tool.output_dir=data/benchmarks/regenie2_chr10_matrix_plan
just slurm-regenie2-chr10-matrix tool.output_dir=data/benchmarks/regenie2_chr10_matrix_current
just regenie2-chr22-matrix-dry-run tool.output_dir=data/benchmarks/regenie2_chr22_matrix_plan
just slurm-regenie2-chr22-matrix tool.output_dir=data/benchmarks/regenie2_chr22_matrix_current
just tune-regenie2-gpu tool.trials=1
just profile-app-full-dry-run tool.output_dir=data/profiles/app_profile_plan
just profile-app-full-landau tool.output_dir=data/profiles/app_profile_current
```

The full Justfile command reference is `docs/development/justfile.md`. It covers recipe
inputs, outputs, and when to use each command.

Hydra-backed Justfile recipes accept trailing Hydra overrides. Use this form for
routine agent work instead of adding shell flags:

```bash
just regenie2-chr10-matrix-dry-run tool.variant_limit=1000
just slurm-regenie2-chr10-matrix tool.variant_limit=1000 tool.output_dir=data/benchmarks/regenie2_chr10_matrix_smoke
just regenie2-chr22-matrix-dry-run tool.variant_limit=1000
just slurm-regenie2-chr22-matrix tool.variant_limit=1000 tool.output_dir=data/benchmarks/regenie2_chr22_matrix_smoke
```

GPU recipes should run through SLURM on `landau`:

```bash
just slurm-benchmark-regenie2-binary-hot-gpu
just slurm-gpu-just benchmark-regenie2-binary-hot-gpu-smoke
just profile-app-full-landau
```

Do not run GPU workloads, heavy benchmark sweeps, large test suites, or
compilation-heavy work on the `gauss` head node.

## Performance Harness

Use the `perf-*` recipes as the stable command surface for optimization tasks:

- `just perf-smoke` is login-node-safe. It runs a tiny deterministic workload,
  writes `performance_smoke_summary.json` under `results/perf/smoke/`, and
  validates that JSON summary generation works.
- `just perf-cpu` requires SLURM. It submits the BGEN reader benchmark through
  `slurm-cpu-just` and writes JSON/Markdown summaries under `results/perf/cpu/`.
- `just perf-gpu` requires SLURM GPU access. It wraps
  `slurm-benchmark-regenie2-binary-hot-gpu` and writes binary-hot artifacts under
  `results/perf/gpu/`.
- `just perf-compare BASE.json NEW.json` is login-node-safe. It compares common
  speed, memory, and numerical metrics from smoke summaries, BGEN reader
  summaries, binary-hot summaries, and matrix manifests. Malformed JSON,
  nonnumeric metric values, or summaries with no common metrics fail with a
  nonzero exit status.

All default `perf-*` outputs live under `results/perf`, which is gitignored.
Set `GWAS_ENGINE_PERF_RESULTS_DIR` to route local artifacts elsewhere.

## Common Tasks

### Full App Profiling

This is the standard profiling task:

> Run full documented profiling of the app with JAX tracing, JAX memory
> profiles, Python cProfile, py-spy sampling, optional Scalene/Memray/Nsight
> passes, Linux perf native stack samples, Rust Criterion benches, stage
> timings, logs, telemetry perturbation runs, bottleneck summaries, and a
> Markdown report.

Use the full app profiling recipe. It is backed by
`tooling.cli.profile_regenie2_deep` and the saved Hydra config
`profile_regenie2_deep.yaml`. The `profile-app-full-*` Justfile recipes set
`tool.include_regenie_baseline=false` so the app profiling workflow does not
require the external `regenie` executable. Existing binary and quantitative
step 1 prediction lists must be present.

Start with a dry run. This writes `profile_plan.json` and `profile_plan.md`
without running workloads. It also writes `artifact_manifest.json`, including
optional profiler availability and skipped-tool reasons:

```bash
just profile-app-full-dry-run tool.output_dir=data/profiles/app_profile_plan
```

Use the REGENIE-focused dry run when planning paired original or patched
REGENIE comparisons:

```bash
just profile-regenie2-deep-dry-run \
  tool.include_regenie_baseline=true \
  tool.output_dir=data/profiles/regenie_pair_plan
```

Install optional user-local profiler tools before a deep campaign when the host
does not already provide them:

```bash
just install-profiling-tools
```

Nsight Systems (`nsys`) and Nsight Compute (`ncu`) are not installed by this
recipe; use a local NVIDIA install or module when available on the GPU node.
Scalene and Memray are Python profilers, so the harness runs them through
`uv run --no-sync --with ...` when they are not importable in the project
environment. This keeps JAX, Polars, and the installed `g` package visible to
the profiled child process.
The harness records missing or permission-blocked profilers as skipped results
instead of failing the campaign.

Run a small end-to-end smoke profile before spending a full SLURM allocation:

```bash
just slurm-gpu-just profile-app-full-smoke tool.output_dir=data/profiles/app_profile_smoke
```

The smoke recipe sets `tool.enable_rust_criterion=false` so it validates the
JAX/Python/native-profiler workflow without spending time in Criterion. The
full `profile-app-full-landau` recipe keeps Criterion enabled.

Run the full profile bundle on `landau`:

```bash
just profile-app-full-landau tool.output_dir=data/profiles/app_profile_current
```

The full run writes:

- `tooling.log`: phase-level progress for long-running jobs.
- `preflight.json`: git, hardware, JAX, Rust, CUDA, REGENIE, and input metadata.
- `summary.json`: structured run results, comparisons, stage totals, and
  profiler metadata.
- `summary.md`: human-readable bottleneck report.
- `artifact_manifest.json`: artifact list, profiler availability, per-profiler
  artifact and application output paths, and skipped profiler reasons.
- `logs/*.stdout.log` and `logs/*.stderr.log`: subprocess logs.
- `bgen_sweep/bgen_sweep.json`: native BGEN reader pre-sweep.
- `tuning_*.json`: candidate tuning grids and finalists.
- `headline_runs/`: winning `g` outputs, plus original REGENIE outputs when
  `tool.include_regenie_baseline=true`.
- `logging_perturbation/logging_perturbation.json`: telemetry/logging
  perturbation trials for representative winners.
- `deep_profiles/*_jax_trace/`: JAX profiler traces for TensorBoard or
  Perfetto-compatible viewers.
- `deep_profiles/*_device_memory.prof`: JAX device memory profiles.
- `deep_profiles/*.cprofile` and `deep_profiles/*.cprofile.txt`: Python
  cProfile data and cumulative-time text summaries.
- `deep_profiles/*.speedscope.json`: py-spy sampling profiles when `py-spy` is
  installed.
- `deep_profiles/*.scalene.json`: Scalene CPU/memory profile output when
  `tool.enable_scalene=true` and either Scalene is importable or `uv` can inject
  it with `--with scalene`.
- `deep_profiles/*.memray.bin`: Memray allocation traces when
  `tool.enable_memray=true` and either Memray is importable or `uv` can inject
  it with `--with memray`.
- `deep_profiles/*_nsys.*`: Nsight Systems reports when
  `tool.enable_nsight_systems=true` and `nsys` is available.
- `deep_profiles/*_ncu.*`: Nsight Compute reports when
  `tool.enable_nsight_compute=true` and `ncu` is available.
- `deep_profiles/*.perf.data`: Linux perf native stack profiles when `perf` is
  available.
- `deep_profiles/profile_*_<profiler>.g/`: isolated application output run
  directories for profiler-wrapped child processes. Each profiler gets its own
  output root and `profile_*_<profiler>.stage_timings.json`, while the primary
  profiler artifacts above keep stable names such as `*.scalene.json` and
  `*.memray.bin`.
- Rust Criterion output for `bgen_read` and `preprocess` when
  `tool.enable_rust_criterion=true`.

The defaults profile chr22 through `dataset=local_1kg`. To profile chr10 with
the same harness, use the chr10 dataset and matching baseline paths:

```bash
just profile-app-full-dry-run \
  dataset=chr10_local \
  tool.chromosome_label=chr10 \
  tool.bed_prefix=1kg_chr10_full \
  tool.baseline_dir=baselines_chr10 \
  tool.linear_prediction_list=baselines_chr10/regenie_step1_qt_pred.list
```

Useful overrides:

- `tool.variant_limit=1000`: cap variants for smoke work.
- `tool.smoke=true`: shrink sweeps, warmups, and trial counts.
- `tool.skip_deep_profiles=true`: run sweeps/headlines but skip profiler
  captures.
- `tool.enable_linux_perf=false`: skip perf when the node disallows it.
- `tool.enable_py_spy=false`: skip py-spy sampling.
- `tool.enable_scalene=true`: run optional Scalene CPU/memory profile passes.
- `tool.enable_memray=true`: run optional Memray allocation profile passes.
- `tool.enable_nsight_systems=true`: run optional Nsight Systems CUDA timeline
  passes for representative winners.
- `tool.enable_nsight_compute=true`: run optional Nsight Compute kernel reports;
  use this only after a hot kernel is identified because it is intrusive.
- `tool.enable_logging_perturbation=false`: skip telemetry/logging perturbation
  trials when reproducing a narrower benchmark.
- `tool.trait_types=[binary]`: tune only binary `g` candidates. Defaults to
  both `quantitative` and `binary`.
- `tool.devices=[cpu,gpu]`: tune a selected device set. Defaults to both CPU
  and GPU.
- `tool.result_in_flight_limits=[default,4]`: include explicit result
  in-flight slot limits in the candidate grid. `default` keeps the runtime
  derived capacity of `staging_depth + 1`.
- `tool.dosage_buffer_limits=[default,4]`: include explicit reusable native
  dosage buffer pool limits in the candidate grid. `default` keeps the runtime
  derived capacity of `staging_depth + 1`.
- `tool.rust_benchmarks=[bgen_read]`: limit Rust Criterion benches.
- `tool.include_regenie_baseline=true`: also run original REGENIE headline
  trials when `regenie` is available.
- `tool.regenie_executable=/path/to/regenie`: use a specific original or
  patched REGENIE binary instead of `REGENIE_BIN`/`regenie`.
- `tool.regenie_baseline_trait_types=[quantitative,binary]`: choose which
  REGENIE traits get paired baseline trials. The default is the faster
  quantitative pair.
- `tool.regenie_baseline_trials=1`: keep paired REGENIE runtime evidence small;
  increase only for dedicated baseline campaigns.
- `tool.regenie_baseline_variant_limit=1000`: override the baseline bound. When
  unset, bounded smoke runs reuse `tool.variant_limit`; the harness writes a
  REGENIE `--extract` list from the first variants in the matching `.pvar` or
  `.bim` file so the original REGENIE run is comparable to `g`'s first-N
variant workload.

Queue timings in `*.stage_timings.json` need direction-aware interpretation:
`result_queue:put` and `result_in_flight_slots:producer_blocking` are producer
backpressure signals. `result_queue:consumer_wait` is normally the writer
thread sleeping while JAX/native work is still upstream, so it is expected idle
time unless paired with blocked producer puts. `dosage_buffer_pool:consumer_wait`
blocks native callback delivery while waiting for a reusable decode buffer; tune
`tool.staging_depths`, `tool.result_in_flight_limits`, and
`tool.dosage_buffer_limits` before treating that wait as lost wall time.
In the GLA-47 binary sweep, larger dosage-buffer capacity removed most
`dosage_buffer_pool:consumer_wait`, but the same runs shifted time into
`dosage_queue:producer_blocking` because the bounded staging queue correctly
throttled native delivery behind JAX compute. With `result_queue:put` still at
zero blocked seconds and mixed CPU/GPU headline results, that pattern did not
justify increasing packaged queue-capacity defaults.

The summary separates successful direct ratios from unsupported comparisons
such as disabled baselines, missing REGENIE binaries, or missing `.pvar`/`.bim`
metadata for bounded pairs. Failed comparisons are reserved for attempted runs
that did not produce measured runtimes. `artifact_manifest.json` records the
baseline commands, resolved binaries, input files, generated extract lists, and
baseline scope used for the run.

### chr10 Binary And Linear Step 2 Matrix

This is the standard agent task:

> Run binary and linear step 2 on chr10 on CPU, GPU, and GPU with JAX cached;
> save results; report back; compare to the previous run.

Use the dedicated Hydra tool. It runs these six production `g regenie`
commands:

- `binary_cpu`
- `binary_gpu`
- `binary_gpu_cached`
- `linear_cpu`
- `linear_gpu`
- `linear_gpu_cached`

The `gpu` and `gpu_cached` runs use separate Python subprocesses and the same
JAX persistent cache directory. With the default timestamped output directory,
the first GPU run starts from a fresh cache and the cached run reuses the cache
populated by that first GPU run.

First inspect the commands without running heavy work:

```bash
just regenie2-chr10-matrix-dry-run tool.output_dir=data/benchmarks/regenie2_chr10_matrix_current
```

Use a stable output directory when you want to tail logs while the job runs:

```bash
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
  just slurm-regenie2-chr10-matrix tool.output_dir=/mnt/beegfs/kirill/Projects/g/data/benchmarks/regenie2_chr10_matrix_current
```

For the normal timestamped run on `landau`, use:

```bash
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
  just slurm-regenie2-chr10-matrix
```

Do not run the real matrix on the head node. The dry-run recipe is safe on the
head node; the real matrix should go through SLURM.

Historical full-chr10 artifacts under
`data/benchmarks/chr10_regenie_g_20260605` used a different profile from the
matrix default: `bsize=8192`, `threads=72`, binary
`g-firth-batch-size=64`, binary `g-firth-candidate-capacity=1024`, writer
threads `4`, writer queue depth `16`, finalized Parquet, and profile telemetry.
Use those settings when the task is to reproduce or compare against those older
full-chromosome data points:

```bash
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
GWAS_ENGINE_SLURM_CPUS_PER_TASK=72 \
  just slurm-regenie2-chr10-matrix \
  tool.chunk_size=8192 \
  tool.cpu_threads=72 \
  tool.binary_firth_batch_size=64 \
  tool.binary_firth_candidate_capacity=1024 \
  tool.output_writer_thread_count=4 \
  tool.output_writer_queue_depth=16 \
  tool.finalize_parquet=true \
  tool.telemetry_mode=profile \
  tool.trusted_no_missing_diploid=false
```

The matrix default intentionally follows current production defaults. The
historical profile is for apples-to-apples performance comparisons against the
June 2026 full-chr10 logs.

The tool writes:

- `data/benchmarks/regenie2_chr10_matrix_<timestamp>/manifest.json`
- `data/benchmarks/regenie2_chr10_matrix_<timestamp>/report.md`
- `data/benchmarks/regenie2_chr10_matrix_<timestamp>/tooling.log`
- `data/benchmarks/regenie2_chr10_matrix_<timestamp>/logs/<run>/events.jsonl`
- `data/benchmarks/regenie2_chr10_matrix_<timestamp>/logs/<run>/stage_timings.json`
- `data/benchmarks/regenie2_chr10_matrix_<timestamp>/runs/<run>.g/...`

Use `tooling.log` for orchestration progress:

```bash
tail -f data/benchmarks/regenie2_chr10_matrix_current/tooling.log
```

Use each run's `events.jsonl` for production chunk progress:

```bash
tail -f data/benchmarks/regenie2_chr10_matrix_current/logs/binary_gpu/events.jsonl
```

Previous-run comparison is automatic. The tool finds the most recent real
(non-dry-run) `data/benchmarks/regenie2_chr10_matrix_*/manifest.json` with the
same `tool.variant_limit` scope other than the current run and compares matching
run names. To force a specific baseline:

```bash
just regenie2-chr10-matrix-dry-run \
  tool.previous_manifest_path=data/benchmarks/regenie2_chr10_matrix_baseline/manifest.json
```

For smoke validation, cap variants:

```bash
just regenie2-chr10-matrix-dry-run \
  tool.variant_limit=1000 \
  tool.output_dir=data/benchmarks/regenie2_chr10_matrix_smoke
```

Then submit the same override through SLURM without `tool.dry_run=true`:

```bash
just slurm-regenie2-chr10-matrix \
  tool.variant_limit=1000 \
  tool.output_dir=data/benchmarks/regenie2_chr10_matrix_smoke
```

### chr22 Binary And Linear Step 2 Matrix

chr22 has the same matrix surface as chr10. It uses the local 1KG chr22 BGEN,
sample file, binary baseline predictions in `data/baselines`, and quantitative
baseline predictions in `data/baselines/regenie_step1_qt_pred.list`.

First inspect the commands:

```bash
just regenie2-chr22-matrix-dry-run tool.output_dir=data/benchmarks/regenie2_chr22_matrix_current
```

Run on `landau` with a stable output directory:

```bash
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
  just slurm-regenie2-chr22-matrix tool.output_dir=/mnt/beegfs/kirill/Projects/g/data/benchmarks/regenie2_chr22_matrix_current
```

For a normal timestamped chr22 run:

```bash
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
  just slurm-regenie2-chr22-matrix
```

chr22 output directories use the `regenie2_chr22_matrix_*` prefix. Previous-run
comparison is automatic within that prefix and therefore does not pick up chr10
matrix manifests.

For smoke validation:

```bash
just regenie2-chr22-matrix-dry-run \
  tool.variant_limit=1000 \
  tool.output_dir=data/benchmarks/regenie2_chr22_matrix_smoke

just slurm-regenie2-chr22-matrix \
  tool.variant_limit=1000 \
  tool.output_dir=data/benchmarks/regenie2_chr22_matrix_smoke
```

### Benchmark Artifact Analysis Recipes

These `jq` recipes are the standard first pass when an agent needs to answer
"was this faster than before?" or inspect where time moved. Set variables first
so commands are reusable:

```bash
MATRIX=data/benchmarks/regenie2_chr10_matrix_current/manifest.json
PREVIOUS=data/benchmarks/regenie2_chr10_matrix_previous/manifest.json
OLD_REPORT=data/benchmarks/chr10_regenie_g_20260605/benchmark_report.json
HOT_GPU=data/benchmarks/chr10_regenie_g_20260605/hot_jax_precompiled_20260606_gpu_a361691_sharedlog/gpu_hot_jax_summary.json
```

Use `regenie2_chr22_matrix_current` and `regenie2_chr22_matrix_previous` for
chr22. Summarize the current matrix manifest:

```bash
jq -r '
  def cell: if . == null then "null" else tostring end;
  .runs[]
  | [
      .name,
      (.wall_time_seconds | cell),
      (.output_row_count | cell),
      (.committed_chunk_count | cell),
      (.stage_seconds.python_api_entry | cell),
      (.stage_seconds.native_engine_delivery | cell),
      (.stage_seconds.jax_compute | cell),
      (.stage_seconds.host_to_device_transfer | cell),
      (.output_total_bytes | cell)
    ]
  | @tsv
' "$MATRIX"
```

Read the matrix tool's built-in previous-run comparison:

```bash
jq -r '
  def cell: if . == null then "null" else tostring end;
  .comparisons[]
  | [
      .run_name,
      .metric,
      (.current_value | cell),
      (.previous_value | cell),
      (.delta | cell),
      (.ratio | cell)
    ]
  | @tsv
' "$MATRIX"
```

Compare wall times from two matrix manifests without relying on the embedded
comparison block:

```bash
jq -n -r --slurpfile current "$MATRIX" --slurpfile previous "$PREVIOUS" '
  ($current[0].runs | map({key: .name, value: .}) | from_entries) as $current_runs
  | ($previous[0].runs | map({key: .name, value: .}) | from_entries) as $previous_runs
  | ($current_runs | keys[]) as $name
  | select($previous_runs[$name] != null)
  | ($current_runs[$name].wall_time_seconds) as $current_wall
  | ($previous_runs[$name].wall_time_seconds) as $previous_wall
  | select($current_wall != null and $previous_wall != null)
  | [
      $name,
      ($current_wall | tostring),
      ($previous_wall | tostring),
      (($current_wall - $previous_wall) | tostring),
      (($current_wall / $previous_wall) | tostring)
    ]
  | @tsv
'
```

Extract historical full-chr10 wall times from the old benchmark report:

```bash
jq -r '
  .runs[]
  | [
      .label,
      (.wall_time_seconds | tostring),
      ((.variants_per_second // "null") | tostring)
    ]
  | @tsv
' "$OLD_REPORT"
```

Extract hot benchmark means from a binary-hot summary:

```bash
jq -r '
  .measured_summary
  | to_entries[]
  | [
      .key,
      (.value.trial_count | tostring),
      (.value.wall_time_seconds_mean | tostring),
      (.value.wall_time_seconds_min | tostring),
      (.value.wall_time_seconds_max | tostring),
      (.value.variants_per_second_mean | tostring)
    ]
  | @tsv
' "$HOT_GPU"
```

Compare profile-stage totals across one or more profile summaries:

```bash
jq -r '
  [
    input_filename,
    (.stage_totals_seconds.python_api_entry | tostring),
    (.stage_totals_seconds.native_engine_delivery | tostring),
    (.stage_totals_seconds.jax_compute | tostring),
    (.stage_totals_seconds.host_to_device_transfer | tostring),
    (.stage_counts.jax_compute | tostring)
  ]
  | @tsv
' data/benchmarks/regenie2_chr10_matrix_current/logs/*/profile_summary.json
```

Inspect binary Firth candidate diagnostics for one binary run:

```bash
jq '.binary_chunk_summary' \
  data/benchmarks/regenie2_chr10_matrix_current/logs/binary_gpu_cached/profile_summary.json
```

Recover the exact generated command for one matrix run:

```bash
jq -r '
  .runs[]
  | select(.name == "binary_gpu_cached")
  | .command_arguments
  | @sh
' "$MATRIX"
```

Watch chunk progress from a JSONL event log:

```bash
jq -r '
  select(.event == "progress_tick")
  | [
      .ts,
      (.processed_chunk_count | tostring),
      (.variant_start_index | tostring),
      (.variant_stop_index | tostring)
    ]
  | @tsv
' data/benchmarks/regenie2_chr10_matrix_current/logs/binary_gpu/events.jsonl | tail
```

Pair `jq` summaries with an effective-config scan when timings look surprising:

```bash
rg --no-ignore --glob '**/effective_config.toml' -n \
  "bsize|threads|firth-batch|firth-candidate|writer|queue|finalize|telemetry" \
  data/benchmarks/regenie2_chr10_matrix_current/runs
```

### BGEN Reader Default

```bash
uv run --no-sync python -m tooling.cli.benchmark_bgen_reader
```

### BGEN Reader Smoke

```bash
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
  uv run --no-sync python -m tooling.cli.benchmark_bgen_reader \
  workload.variant_limit=16 \
  workload.repeat_count=1 \
  sweep.chunk_sizes=[16] \
  sweep.path_modes=[variant_major_buffered] \
  sweep.sample_selection_modes=[full,strided_half] \
  telemetry.json_summary_path=data/profiles/bgen_reader_smoke.json \
  telemetry.markdown_summary_path=data/profiles/bgen_reader_smoke.md
```

### Compare Safe And Trusted BGEN Paths

```bash
uv run --no-sync python -m tooling.cli.benchmark_bgen_reader \
  sweep.trusted_no_missing_diploid_modes=[false,true] \
  sweep.path_modes=[variant_major_buffered,variant_major_packed8_buffered] \
  sweep.sample_selection_modes=[full,contiguous_half,strided_half] \
  workload.variant_limit=16384 \
  workload.repeat_count=7
```

The packed8 path is only valid for trusted no-missing diploid cases. Unsupported
packed8 cases are filtered when trusted mode is disabled.

### Binary-Hot GPU Smoke

```bash
uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot \
  machine=landau_gpu \
  tool.variant_limit=1000 \
  tool.include_cold_process=false \
  tool.include_finalized_hot=false
```

The Justfile recipe is:

```bash
just benchmark-regenie2-binary-hot-gpu-smoke tool.variant_limit=1000
```

### Binary-Hot Packed8 Workload

```bash
uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot \
  machine=landau_gpu \
  sweep.storage_modes=[packed8] \
  sweep.fallback_density_scenarios=[default] \
  telemetry.stage_timing_mode=exact \
  tool.variant_limit=50000 \
  tool.include_cold_process=false \
  tool.include_finalized_hot=false
```

### Output-Stage Profiling

```bash
uv run --no-sync python -m tooling.cli.benchmark_output_stages \
  machine=landau_gpu \
  tool.trials=3 \
  tool.variant_limit=16384 \
  tool.writer_thread_counts=[1,2,4,8,12] \
  tool.writer_queue_depth_multipliers=[1,2,4] \
  tool.chunks_per_arrow_file_values=[4,16,64] \
  tool.arrow_compressions=[zstd,none] \
  telemetry.json_summary_path=data/profiles/output_stages.json \
  telemetry.markdown_summary_path=data/profiles/output_stages.md
```

### Full App Profile Smoke

```bash
just profile-app-full-dry-run tool.output_dir=data/profiles/app_profile_plan
just slurm-gpu-just profile-app-full-smoke tool.output_dir=data/profiles/app_profile_smoke
```

For direct Hydra execution, use:

```bash
uv run --no-sync python -m tooling.cli.profile_regenie2_deep \
  machine=landau_gpu \
  tool.smoke=true \
  tool.variant_limit=1000
```

### Quantitative Startup Amortization

Use the linear fresh-process script when a profile shows quantitative Step 2 is
dominated by one-time Python or JAX backend startup. The deep profiler's headline
trials are isolated subprocesses; this is the right baseline for separate CLI
invocations, but it overstates repeated Python API workflows and batched
multi-phenotype runs.

Run CPU checks on a CPU compute node and GPU checks through `landau`:

```bash
uv run --no-sync python scripts/benchmark_regenie2_linear_fresh_process.py \
  --device cpu \
  --data-dir /mnt/beegfs/kirill/Projects/g/data \
  --output-dir data/benchmarks/linear_startup_cpu \
  --trials 3 \
  --same-process-trials 3 \
  --emit-stage-timings

just slurm-gpu-run 'uv run --no-sync python scripts/benchmark_regenie2_linear_fresh_process.py --device gpu --data-dir /mnt/beegfs/kirill/Projects/g/data --output-dir data/benchmarks/linear_startup_gpu --trials 3 --same-process-trials 3 --emit-stage-timings'
```

Use `--multi-phenotype-count N` when the question is whether one process can do
more useful work per BGEN decode/JAX initialization. The generated phenotype and
prediction-list inputs live under the benchmark output directory and are for
timing only; do not use cloned traits as scientific evidence.

Interpretation:

- Fresh-process wall time includes import, JAX plugin discovery, backend
  initialization, dynamic library loading, BGEN delivery, compute, and output.
- Same-process hot trials reuse Python imports, compatible JAX runtime policy,
  and process-global native runtime setup. Their stage timings should show
  `jax_device_configuration_backend_init` near zero after warmup.
- Multi-phenotype timing is only a valid production recommendation when the
  requested sample mode matches the user's intended statistics. `complete-case`
  can batch traits on one shared sample intersection, but it is not equivalent
  to separate per-phenotype scans when missingness differs.

### GPU Tuning

```bash
uv run --no-sync python -m tooling.cli.tune_regenie2_gpu machine=landau_gpu
```

Limit exploratory runs with:

```bash
uv run --no-sync python -m tooling.cli.tune_regenie2_gpu \
  machine=landau_gpu \
  tool.trait_selection=binary \
  tool.variant_limit=50000 \
  tool.trials=1 \
  tool.finalist_extra_trials=1 \
  tool.top_bgen_candidates=1 \
  tool.top_compute_candidates=1 \
  tool.top_finalists=1
```

### Use Another Dataset

For one run, use environment override:

```bash
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
  uv run --no-sync python -m tooling.cli.benchmark_bgen_reader
```

For repeated use, add a dataset profile under `tooling/configs/dataset/` and
select it with `dataset=my_dataset`.

## Saved Profiles

Saved profiles live under `tooling/configs/`. There are two kinds:

- tool configs at `tooling/configs/*.yaml`;
- reusable config groups under `dataset/`, `machine/`, `workload/`,
  `telemetry/`, and `sweep/`.

### Tool Configs

`benchmark_bgen_reader.yaml`

Default BGEN reader benchmark. Uses `dataset=local_1kg`, `machine=local`,
`workload=bgen_reader`, `telemetry=local`, and `sweep=bgen_reader_default`.

`benchmark_regenie2_binary_hot.yaml`

Default binary-hot GPU benchmark. Uses `dataset=local_1kg`,
`machine=landau_gpu`, `workload=regenie2_binary_hot`, `telemetry=local`, and
`sweep=regenie2_binary_hot_default`.

`benchmark_output_stages.yaml`

Default output-stage benchmark. Uses `machine=landau_gpu` and tool-local output
stage sweep defaults.

`profile_regenie2_deep.yaml`

Default full app profile campaign. Uses `dataset=local_1kg`,
`machine=landau_gpu`, chr22 inputs, BGEN pre-sweeps, candidate tuning, headline
trials, JAX trace capture, JAX memory profiling, Python cProfile, py-spy,
Linux perf, and Rust Criterion benches for `bgen_read` and `preprocess`. Set
`tool.dry_run=true` to write only `profile_plan.json` and `profile_plan.md`.
Set `tool.smoke=true` for the small smoke campaign. The config default includes
original REGENIE headline trials; the `profile-app-full-*` Justfile recipes
override `tool.include_regenie_baseline=false` for app-only profiling.

`run_regenie2_chr10_matrix.yaml`

Default chr10 binary/linear CPU/GPU/cache matrix. Uses `dataset=chr10_local`,
`machine=landau_gpu`, `workload=regenie2_chr10_matrix`, and production
`g regenie` subprocesses.

`run_regenie2_chr22_matrix.yaml`

Default chr22 binary/linear CPU/GPU/cache matrix. Uses `dataset=local_1kg`,
`machine=landau_gpu`, `workload=regenie2_chr22_matrix`, and production
`g regenie` subprocesses.

`tune_regenie2_gpu.yaml`

Default GPU tuning campaign. Uses `machine=landau_gpu` and full tuning sweep
defaults.

`config.yaml`

Generic composition config used by tests and reusable Python composition. It is
not the default entrypoint config for any specific tool.

### Dataset Profiles

`dataset/local_1kg.yaml`

- `data_directory: ${oc.env:GWAS_ENGINE_DATA_DIR,data}`
- `bgen_file: 1kg_chr22_full.bgen`
- `sample_file: 1kg_chr22_full.sample`
- `phenotype_file: pheno_bin.txt`
- `prediction_list: baselines/regenie_step1_pred.list`
- `phenotype_columns: [phenotype_binary]`

Use this for local 1KG chr22 data and binary step 2 baseline predictions.

`dataset/chr10_local.yaml`

- `data_directory: ${oc.env:GWAS_ENGINE_DATA_DIR,data}`
- `bgen_file: 1kg_chr10_full.bgen`
- `sample_file: 1kg_chr10_full.sample`
- `phenotype_file: pheno_bin.txt`
- `prediction_list: baselines_chr10/regenie_step1_pred.list`
- `phenotype_columns: [phenotype_binary]`

Use this for local 1KG chr10 data and binary step 2 baseline predictions.

### Machine Profiles

`machine/local.yaml`

- `name: local`
- `device: cpu`
- no SLURM node, CPU count, or memory hints

Use this for local CPU composition and tests.

`machine/landau_gpu.yaml`

- `name: landau_gpu`
- `device: gpu`
- `slurm_node: landau`
- `cpus_per_task: 8`
- `memory: 64G`

Use this for GPU benchmark planning on `landau`. It does not submit jobs by
itself; use Justfile SLURM recipes for execution.

### Workload Profiles

`workload/bgen_reader.yaml`

- `chunk_size: 8192`
- `variant_limit: 16384`
- `repeat_count: 5`
- `staging_depth: 1`
- `output_writer_thread_count: 8`
- `output_writer_queue_depth: 8`

`workload/regenie2_binary_hot.yaml`

- `chunk_size: 16384`
- `variant_limit: null`
- `repeat_count: 1`
- `staging_depth: 1`
- `output_writer_thread_count: 8`
- `output_writer_queue_depth: 8`

`workload/regenie2_chr10_matrix.yaml`

- `chunk_size: 16384`
- `variant_limit: null`
- `repeat_count: 1`
- `staging_depth: 1`
- `output_writer_thread_count: 8`
- `output_writer_queue_depth: 8`

`workload/regenie2_chr22_matrix.yaml`

- `chunk_size: 16384`
- `variant_limit: null`
- `repeat_count: 1`
- `staging_depth: 1`
- `output_writer_thread_count: 8`
- `output_writer_queue_depth: 8`

### Telemetry Profiles

`telemetry/local.yaml`

- `output_parent: data/profiles`
- `json_summary_path: null`
- `markdown_summary_path: null`
- `stage_timing_mode: exact`

Set explicit summary paths through overrides when a run needs stable artifact
names.

### Sweep Profiles

`sweep/bgen_reader_default.yaml`

- `chunk_sizes: [8192]`
- `path_modes: [variant_major_buffered]`
- `sample_selection_modes: [full]`
- `decode_tile_variant_counts: []`
- `rayon_thread_counts: []`
- `trusted_no_missing_diploid_modes: [false]`
- `storage_modes: [variant_major]`
- `fallback_density_scenarios: [default]`

`sweep/regenie2_binary_hot_default.yaml`

- `chunk_sizes: [1000]`
- `path_modes: [variant_major_buffered]`
- `sample_selection_modes: [full]`
- `decode_tile_variant_counts: []`
- `rayon_thread_counts: []`
- `trusted_no_missing_diploid_modes: [true]`
- `storage_modes: [variant_major]`
- `fallback_density_scenarios: [default]`

The binary-hot tool uses `sweep.storage_modes` and
`sweep.fallback_density_scenarios`. The other fields are kept available for
shared campaign composition.

## Python Composition

Use `tooling.configuration.compose_config()` in tests and helper code:

```python
import tooling.configuration

config = tooling.configuration.compose_config(
    config_name="benchmark_regenie2_binary_hot",
    overrides=[
        "machine=landau_gpu",
        "tool.variant_limit=1000",
        "tool.include_cold_process=false",
    ],
)
```

Use a tool's resolver to materialize internal parameters:

```python
import tooling.cli.benchmark_regenie2_binary_hot as binary_hot

arguments = binary_hot.build_arguments_from_config(config)
```

Use `include_hydra_config=True` when tests need to assert Hydra behavior:

```python
config = tooling.configuration.compose_config(include_hydra_config=True)
assert config.hydra.job.chdir is False
```

## Common Helpers

`tooling.common.paths`

- `find_repository_root(start_path)`
- `configured_data_directory()`
- `resolve_data_directory(repository_root, environment)`
- `resolve_data_path(data_directory, path)`

`tooling.common.reports`

- `write_json_report(path, value, sort_keys=False)`
- `write_markdown_report(path, markdown_text)`
- `to_json_text(value)`

The JSON helpers handle dataclasses, enums, `Path`, dictionaries, lists, and
tuples.

`tooling.common.sweeps`

- optional integer list parsing;
- positive integer list parsing;
- boolean mode parsing;
- queue-depth construction.

`tooling.common.hydra_arguments`

- resolved `tool` node extraction;
- list-to-comma serialization for legacy internal parsers;
- path/integer conversion helpers;
- Hydra override formatting for fresh subprocess cases.

`tooling.common.commands`

Captured subprocess helpers for metadata and diagnostics.

`tooling.regenie.bgen_reader`

Shared BGEN benchmark enums, dataclasses, sample-selection helpers, path-mode
parsing, and path-mode validation.

## Extending Tooling

Use this checklist when adding a new development tool.

1. Decide whether the tool belongs in `tooling/`.

   Put it under `tooling/` if it is GWAS development tooling: benchmarks,
   profiling, campaign orchestration, report generation, or reusable
   config-driven dev workflow.

   Keep it out of `tooling/` if it is production runtime behavior, public CLI
   behavior, or unrelated automation.

2. Add a Hydra config file under `tooling/configs/`.

   Example:

   ```yaml
   defaults:
     - dataset: local_1kg
     - machine: local
     - workload: bgen_reader
     - telemetry: local
     - sweep: bgen_reader_default
     - _self_

   tool:
     output_dir: data/profiles/my_tool
     variant_limit: ${workload.variant_limit}
     repeat_count: ${workload.repeat_count}

   hydra:
     job:
       chdir: false
   ```

3. Create a module under `tooling/cli/`.

   The standard shape is:

   ```python
   import dataclasses
   import typing
   from pathlib import Path

   import hydra
   import omegaconf

   import tooling.configuration as tooling_configuration
   from tooling.common import hydra_arguments as tooling_hydra_arguments


   @dataclasses.dataclass(frozen=True)
   class MyToolArguments:
       output_dir: Path
       variant_limit: int | None


   def build_arguments_from_config(config: omegaconf.DictConfig) -> MyToolArguments:
       tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
       return MyToolArguments(
           output_dir=Path(str(tool_values["output_dir"])),
           variant_limit=tooling_hydra_arguments.integer_or_none(tool_values.get("variant_limit")),
       )


   def build_arguments_from_overrides(
       overrides: typing.Sequence[str] | None = None,
   ) -> MyToolArguments:
       config = tooling_configuration.compose_config(config_name="my_tool", overrides=overrides)
       return build_arguments_from_config(config)


   def run_tool(arguments: MyToolArguments) -> None:
       arguments.output_dir.mkdir(parents=True, exist_ok=True)


   @hydra.main(version_base=None, config_path="../configs", config_name="my_tool")
   def hydra_main(config: omegaconf.DictConfig) -> None:
       run_tool(build_arguments_from_config(config))


   def main() -> None:
       hydra_main()


   if __name__ == "__main__":
       main()
   ```

4. Use shared helpers.

   Use `tooling.common.paths` for repository and data paths, `reports` for JSON
   and Markdown output, `sweeps` for list parsing, and `commands` for captured
   metadata commands.

5. Add tests.

   Cover config composition, `build_arguments_from_overrides()`, report
   serialization, path resolution, and pure expansion logic. Do not run GPU or
   heavy benchmark workloads in unit tests.

6. Add a Justfile recipe when the workflow is common.

   Example:

   ```just
   my-tool:
       {{server_env}} && uv run --no-sync python -m tooling.cli.my_tool machine=local
   ```

7. Document the tool in this guide.

   Add its entrypoint, config file, common overrides, report paths, and any
   SLURM smoke command needed for safe validation.

## Adding Profiles

Add a dataset profile when the input file set changes:

```yaml
data_directory: /mnt/beegfs/kirill/Projects/g/data
bgen_file: 1kg_chr10_full.bgen
sample_file: 1kg_chr10_full.sample
phenotype_file: pheno_bin.txt
prediction_list: baselines_chr10/regenie_step1_pred.list
phenotype_columns:
  - phenotype_binary
```

Run with:

```bash
uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot dataset=chr10_local
```

Add a machine profile when device or scheduler hints change:

```yaml
name: landau_gpu_large
device: gpu
slurm_node: landau
cpus_per_task: 16
memory: 128G
```

Add a workload profile when run size changes:

```yaml
name: binary_hot_smoke
chunk_size: 16384
variant_limit: 1000
repeat_count: 1
staging_depth: 1
output_writer_thread_count: 8
output_writer_queue_depth: 8
```

Add a sweep profile when a matrix is reused:

```yaml
chunk_sizes:
  - 4096
  - 8192
path_modes:
  - variant_major_buffered
  - variant_major_packed8_buffered
sample_selection_modes:
  - full
decode_tile_variant_counts:
  - 64
  - 128
rayon_thread_counts:
  - 4
  - 8
trusted_no_missing_diploid_modes:
  - true
storage_modes:
  - variant_major
  - packed8
fallback_density_scenarios:
  - default
```

## Validation

Lightweight checks for tooling changes:

```bash
uv run --no-sync ruff check tooling tests docs
uv run --no-sync ty check src tests scripts tooling
uv run --no-sync pytest tests/test_regenie_comparison_scripts.py tests/test_tooling_architecture.py
```

Optional GPU smoke on `landau`:

```bash
just slurm-gpu-just benchmark-regenie2-binary-hot-gpu-smoke
just slurm-gpu-just profile-regenie2-deep-smoke
```

Do not run GPU smoke commands on the head node.

## Rules Of Thumb

- Keep `hydra.job.chdir: false`.
- Prefer Hydra groups and `tool.*` overrides over shell flags.
- Put stable defaults in YAML, not in the executable module.
- Keep tool parameters in frozen dataclasses.
- Keep generated data and reports under ignored data/profile directories.
- Keep `tooling/` out of package discovery and `[project.scripts]`.
- Do not add migrated-tool wrappers under `scripts/`.

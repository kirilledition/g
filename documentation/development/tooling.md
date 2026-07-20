# Development Tooling Guide

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | current repository-local tooling | Development tooling maintainers |

This page is the operational reference for the development-only `tooling/`
package. The shorter design summary is in
[Development Tooling Architecture](dev-tooling-architecture.md).

## Scope

`tooling/` owns repeatable development workflows: data preparation, benchmark
campaigns, profiler orchestration, machine profiles, report generation, and
repository guardrails. It is not installed as part of the public Python
package.

Production behavior remains in `src/g` and `crates/`. Tooling invokes the same
supported boundary as users: `g regenie --config <file>` in a subprocess or
`g._core.cli.run(["regenie", "--config", ...])` in a same-process lifecycle.
It must not recreate a second Python orchestration layer.

Generated data and profiler output belong under ignored locations such as
`data/benchmarks/`, `data/profiles/`, and `results/perf/`.

## Layout

```text
tooling/
  benchmark/
    benchmark.py
    linear_startup.py
    native_lifecycle.py
  cli/
    benchmark.py
    benchmark_firth_compute.py
    benchmark_regenie2_binary_hot.py
    benchmark_tensorqtl_chr22.py
    benchmark_torchgwas_chr22.py
    data.py
    debug.py
    performance.py
    profile_regenie2_deep.py
    run_regenie2_matrix.py
    rust_build_profiles.py
    schema_check.py
    server.py
  common/
    artifact_format.py
    commands.py
    downloads.py
    g_regenie.py
    hydra_arguments.py
    jax_cache.py
    paths.py
    registry.py
    reports.py
  configs/
    dataset/
    machine/
    telemetry/
    workload/
    *.yaml
  profile_deep/
  data/
  debug/
  performance/
  server/
```

`tooling.common.g_regenie` is the single tooling-owned renderer for production
TOML and `g regenie --config` commands. `tooling.benchmark.native_lifecycle`
owns fresh-process, discarded-warm, hot same-process, cache, telemetry, and
completed-output evidence shared by the lifecycle benchmarks.

## Configuration

All maintained tooling entrypoints are Hydra-driven. Each Justfile workflow
selects an explicit config with `--config-name`; files under
`tooling/configs/` own datasets, machines, workload geometry, profiler modes,
and output policy.

Inspect a composed configuration without executing it:

```bash
uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot \
  --config-name bench_binary_hot_gpu --cfg job
```

Apply narrow one-off overrides after the config name:

```bash
just slurm-gpu-just bench-binary-hot-gpu tool.hot_run_count=7
just matrix-chr22-dry tool.output_dir=data/benchmarks/chr22-plan
just profile-chr22-binary-gpu-dry tool.output_dir=data/profiles/chr22-plan
```

Every config keeps Hydra's working-directory change disabled:

```yaml
hydra:
  job:
    chdir: false
```

There is no production variant-count cap. A bounded benchmark must use a
physically bounded BGEN fixture with matching metadata and prediction inputs.
The optional `regenie_baseline_variant_limit` field is limited to the external
REGENIE baseline and creates an extract list; it never changes a `g` run.

## Lifecycle Benchmarks

### Binary hot path

Run the production binary approximate-Firth geometry with:

```bash
just slurm-gpu-bench-binary-hot
```

The default chr22 config uses packed8 GPU delivery, 16,384-variant chunks,
Firth batch size 512, candidate capacity 1,024, eight output writers, direct
Parquet parts, and telemetry off.

The harness records:

- an optional fresh-process diagnostic;
- one discarded same-process compile/warm lifecycle;
- the requested number of telemetry-off same-process hot lifecycles, each with
  a distinct output root;
- optional profile-telemetry diagnostics kept out of headline timings.

It rejects pre-populated default cache directories, requires the warm lifecycle
to populate the cache, and requires the cache tree to remain byte-identical
during headline runs. Evidence includes the native-library and dependency-lock
hashes, input fingerprints, runtime and machine details, per-run elapsed time,
cache snapshots, manifest coverage, Parquet row counts, schemas, metadata, and
part hashes.

For a quick harness check on a GPU node:

```bash
just slurm-gpu-just bench-binary-hot-gpu-smoke
```

### Quantitative lifecycle

`bench-linear-startup-gpu` exercises the same current production boundary for
quantitative Step 2. It has the same fresh-process, discarded-warm,
same-process-hot, and optional diagnostic separation as the binary harness.

```bash
just slurm-gpu-just bench-linear-startup-gpu
```

### Focused Firth compute

The focused JAX benchmark compiles one fixed-capacity approximate-Firth
executable and measures synchronized hot calls at representative active
candidate counts:

```bash
just slurm-gpu-bench-firth-compute
```

It records StableHLO and executable fingerprints, compile/cache evidence,
device timing, and correctness hashes. It is a causal compute benchmark, not a
whole-application speed claim.

## CPU, GPU, and Cache Matrices

The matrix tool runs binary and quantitative Step 2 on CPU, GPU, and a repeated
GPU process using the populated persistent cache:

```bash
just matrix-chr10-dry
just slurm-gpu-matrix-chr10
just matrix-chr22-dry
just slurm-gpu-matrix-chr22
```

Dry configs render and validate plans without running the workload. Full runs
write independent production TOML files and output roots. A GPU matrix starts
with an empty campaign-owned cache; its first GPU process populates that cache,
and the repeated process must leave it byte-identical.

Previous manifests are comparable when their input/workload identity, cache
policy, machine/runtime versions, Rust flags, and normalized runner protocol
match. Git, native-library, source, dependency-lock, and raw-runner hashes are
reported separately as implementation provenance. A provenance change is
expected during development and is not itself a workload mismatch.

## Competitor Benchmarks

Use the saved chr22 competitor workflows through SLURM:

```bash
just slurm-gpu-bench-torchgwas-chr22
just slurm-gpu-bench-tensorqtl-chr22
```

These are workflow/runtime comparisons, not statistical-parity claims.

TorchGWAS uses the local PLINK triplet for full runs. That path does not create
a persistent genotype cache, so its cases are named first-process and
repeat-process; the repeat may benefit from the filesystem cache but is not a
declared application-cache hit.

tensorQTL runs dense nominal `trans` association on QTL-shaped phenotype and
covariate matrices. It also reports first-process and repeat-process cases
without claiming a persistent application cache. Its statistical model differs
from REGENIE Step 2 with LOCO predictions.

## Native Boundary Benchmarks

Run focused Rust benchmarks by explicit package and bench name:

```bash
cargo bench --package g-genotype --bench bgen_read
cargo bench --package g-engine --bench scheduler
cargo bench --package g-output --bench writer
```

The repository profiling recipe intentionally runs the genotype target alone
on the CPU node:

```bash
just profile-rust-criterion
```

Criterion is not embedded in the GPU deep-profiler campaign. Add a new direct
recipe only when a real crate bench exists and its node requirements are clear.

## Deep Profiling

Start by rendering a plan:

```bash
just profile-app-full-dry
just profile-chr10-binary-gpu-dry
just profile-chr22-binary-gpu-dry
```

Run retained campaigns with:

```bash
just slurm-gpu-just profile-app-full-smoke
just profile-app-full
just profile-chr10-binary-gpu-full
just profile-chr22-binary-gpu-full
```

The focused full configs can collect JAX trace/device memory, cProfile,
py-spy, Scalene, Memray, Nsight Systems, Nsight Compute, and Linux perf when
the tools and node permissions allow them. Missing or permission-blocked
optional profilers are reported as skipped.

Headline candidate and finalist trials use production telemetry-off execution.
Exact stage timing is a separate diagnostic lifecycle and is excluded from
headline elapsed times. Current native profile telemetry must contain:

- `jax_runtime_configuration`;
- `jax_backend_initialization`;
- `native_run_preparation`;
- `native_run_execution`;
- `runner_total`.

Rust-owned output timing artifacts are preserved independently and may contain
stages such as enqueue, record-batch creation, Parquet writing, manifest
commit, and writer finish. They must not be relabeled as Python callback or JAX
compute attribution.

The main outputs include `preflight.json`, `summary.json`, `summary.md`,
`artifact_manifest.json`, per-run logs, generated trial configs, headline
outputs, and profiler-specific artifacts. Diagnostic output roots are distinct
from headline roots.

Use the public tuning dimensions exposed by the saved configs:

- `tool.workload_keys`;
- `tool.chunk_sizes`;
- `tool.output_writer_thread_counts`;
- `tool.rayon_thread_counts`;
- `tool.firth_batch_sizes`;
- profiler enable/timeout fields;
- trial and campaign budget fields.

Removed callback, staging, decode-tile, queue-depth, alternate-output, and
post-run-finalization controls are not valid tuning dimensions.

## Telemetry and Output Evidence

Headline production runs use `telemetry = "off"`. Profile diagnostics require
native profile summary evidence and Rust output-stage timing evidence. Progress
mode requires progress events. A successful output is accepted only when:

- the manifest reports success and a valid integer variant count;
- committed chunks are ordered, non-overlapping, gap-free, and exactly cover
  `[0, variant_count)`;
- part paths are safe and exactly match observed Parquet files;
- Parquet parts are readable and their row counts sum to the manifest count;
- schema and metadata are consistent across parts.

The tooling records content and manifest hashes in addition to counts so a
timing result remains auditable.

## Artifacts and Comparison

Durable tooling bundles use the repository's shared artifact-format models and
include schema identification, producer and git state, command records, input
fingerprints, metrics, failures, comparisons, and an artifact manifest. Check
a completed bundle with:

```bash
just check-artifact-schema data/profiles/example
```

These internal evidence schemas retain their own existing versions for reader
compatibility. They are separate from public product option, output-manifest,
telemetry, and profile-summary contracts, which remain at version 0 until the
first release.

Do not compare timings solely because two manifests share a filename. Confirm
the compatibility identity first, then retain both implementation provenance
records in the report.

## SLURM and Node Policy

Dry plans, small report transformations, and docs checks are login-node-safe.
Use `cantor` or another exclusive CPU allocation for compilation, full tests,
Criterion, and CPU benchmarks. Serialize GPU work on `landau`:

```bash
just slurm-cpu-just rust-check
just slurm-gpu-just bench-binary-hot-gpu-smoke
just slurm-gpu-matrix-chr22
```

The SLURM wrappers configure the build job count from the allocation and keep
node-specific Cargo artifacts isolated. Do not overlap independent benchmark
or profiler jobs on the single GPU.

## Adding or Changing Tooling

When adding a maintained workflow:

1. Put workflow truth in an entrypoint-specific saved Hydra config.
2. Parse it into a fully typed arguments dataclass.
3. Reuse `g_regenie`, `native_lifecycle`, command, cache, and report helpers
   rather than duplicating production invocation or evidence validation.
4. Add only a thin, named Justfile recipe with `--config-name`.
5. Add focused tests for parsing, generated production TOML/commands, failure
   behavior, cache invariants, and artifact contracts.
6. Run `just check-justfile` and `just docs-build` when the command surface or
   this guide changes.

Avoid compatibility aliases for deleted scripts or configuration fields. A
bounded workflow should get a bounded fixture, and an intrusive profiler should
get a distinct diagnostic run rather than changing headline semantics.

# Justfile Command Reference

This document describes the repository `Justfile`: what each recipe does, what
it consumes, what it produces, and when to use it.

Run `just help` for the compact recipe list. Run `just <recipe>` from the
repository root.

## Environment

The recipes read these environment variables:

| Variable | Default | Used For |
| --- | --- | --- |
| `GWAS_ENGINE_DATA_DIR` | `data` | Local input data and generated benchmark artifacts. |
| `GWAS_ENGINE_PYTHON_VERSION` | `3.14` | Python version requested by bootstrap and checks. |
| `GWAS_ENGINE_TOOLS_DIR` | `.tools` | Server-local tool install location reported by `doctor-server`. |
| `GWAS_ENGINE_GPU_NODE` | `landau` | SLURM GPU node. |
| `GWAS_ENGINE_CPU_NODE` | `cantor` | SLURM CPU node for CPU benchmark recipes. |
| `GWAS_ENGINE_SLURM_PARTITION` | empty | Optional SLURM partition. |
| `GWAS_ENGINE_SLURM_ACCOUNT` | empty | Optional SLURM account. |
| `GWAS_ENGINE_SLURM_TIME` | `04:00:00` | Default SLURM time limit. |
| `GWAS_ENGINE_SLURM_CPUS_PER_TASK` | `8` | Default SLURM CPU allocation. |
| `GWAS_ENGINE_SLURM_MEMORY` | `64G` | Default SLURM memory allocation. |
| `GWAS_ENGINE_SLURM_GPUS_PER_TASK` | `1` | Default SLURM GPU allocation. |
| `GWAS_ENGINE_SLURM_EXTRA_ARGS` | empty | Extra SLURM arguments split by shell words. |
| `GWAS_ENGINE_PERF_RESULTS_DIR` | `results/perf` | Local gitignored result root used by `perf-*` recipes. |
| `SYMPHONY_ELIXIR_DIR` | `/mnt/beegfs/kirill/Projects/symphony/elixir` | Symphony checkout used by `symphony-doctor` and `symphony-run`. |
| `SYMPHONY_PORT` | `4000` | Port passed to the Symphony daemon. |
| `SYMPHONY_WORKTREE_ROOT` | `/mnt/beegfs/kirill/Projects/g-worktrees/symphony` | Worktree root for unattended Symphony task branches. |

Most recipes source `scripts/server_env.sh`, which sets repo-local tool paths
and server cache defaults.

Do not run GPU workloads, large benchmark sweeps, or large test suites on the
`gauss` head node. `just perf-smoke` and `just perf-compare` are safe on the
login node. `just perf-cpu` and `just perf-gpu` submit work through SLURM.

## Hydra Overrides

Hydra-backed tooling recipes accept trailing overrides. These overrides are the
public parameter interface for migrated tooling commands:

```bash
just regenie2-chr10-matrix-dry-run tool.variant_limit=1000
just slurm-regenie2-chr10-matrix tool.output_dir=data/benchmarks/regenie2_chr10_matrix_current
just profile-app-full-dry-run tool.output_dir=data/profiles/app_profile_plan
just benchmark-bgen-reader sweep.chunk_sizes=[4096,8192]
```

Use group overrides for saved profiles, scalar overrides for simple settings,
and list overrides for sweeps. The corresponding `tooling.cli.*` entrypoints
also accept the same overrides when run directly with `uv run --no-sync python
-m ...`.

## Help

### `help`

- Inputs: none.
- Output: compact recipe list and pointer to this document.
- Use when: you need a quick overview of available commands.

### `default`

- Inputs: none.
- Output: same as `help`.
- Use when: running bare `just`.

## Data Preparation

### `setup-data`

- Inputs: network access, baseline tooling, writable `GWAS_ENGINE_DATA_DIR`.
- Output: 1KG chr22 data, simulated phenotypes, covariates, and related local
  data files under the data directory.
- Use when: preparing a fresh checkout for local baselines, comparisons, or
  REGENIE step 2 runs.

### `setup-binary-baseline`

- Inputs: `setup-data`, `regenie` executable on `PATH`, PLINK bed inputs,
  `pheno_bin.txt`, and `covariates.txt`.
- Output: `data/baselines/regenie_step1_pred.list` and associated REGENIE step 1
  files.
- Use when: binary REGENIE step 2 needs baseline predictions.

### `setup-regenie2-binary-gpu-inputs`

- Inputs: same as `setup-binary-baseline`.
- Output: all local inputs needed by the binary GPU step 2 recipes.
- Use when: preparing for `regenie2-binary-gpu` or related GPU smoke runs.

### `verify-regenie2-binary-gpu-inputs`

- Inputs: expected chr22 BGEN/sample, binary phenotype, covariates, and binary
  step 1 prediction list.
- Output: success message or a failing `test -s`.
- Use when: checking that GPU step 2 inputs are present before submitting a job.

## Development Environment

### `setup-server-tools`

- Inputs: internet access and a writable server tool/cache directory.
- Output: repo-local server command-line tools installed by
  `scripts/bootstrap_server_tools.py`.
- Use when: bootstrapping a fresh Ubuntu/SLURM server environment.

### `bootstrap`

- Inputs: `uv`, requested Python version.
- Output: CPU-capable dev environment with the `dev` dependency group.
- Use when: setting up local CPU development.

### `bootstrap-gpu`

- Inputs: `uv`, requested Python version, CUDA-capable host.
- Output: GPU-capable dev environment with `dev` and `gpu` dependency groups.
- Use when: preparing a GPU node for JAX CUDA work.

### `install-gpu-dependencies`

- Inputs: existing Python environment and CUDA-capable host.
- Output: synced `dev` and `gpu` dependencies.
- Use when: refreshing GPU dependencies before GPU benchmarks or profiles.

### `install-perf-extension`

- Inputs: Rust toolchain, `maturin`, current Python environment.
- Output: installed native extension built with `RUSTFLAGS="-C target-cpu=native"`
  and the `perf` Cargo profile.
- Use when: running performance benchmarks or profiling the native paths.

## Diagnostics

### `doctor`

- Inputs: `uv`, `cargo`, `rustc`.
- Output: tool availability checks and resolved Python version.
- Use when: checking local development prerequisites.

### `doctor-server`

- Inputs: server tools including `just`, `uv`, `srun`, `zstd`, Rust tools,
  `plink`, `plink2`, and `regenie`.
- Output: prerequisite checks, host name, tools directory, and uv cache path.
- Use when: validating a server or SLURM login environment.

### `symphony-doctor`

- Inputs: `git`, `gh`, `codex`, `just`, `uv`, SLURM client commands, `mise`,
  Linear credentials from `SYMPHONY_ENV_FILE` or `~/.config/g-symphony/env`, a
  Codex Linear MCP config, a reachable Linear project slug, a reachable GitHub
  `origin`, a writable Symphony worktree root, and a built Symphony checkout.
- Output: a redacted pass/fail report for unattended Symphony/Linear task
  execution. The report validates Linear API auth, Linear MCP auth, Codex MCP
  config readability, Git/GitHub reachability, `uv`, SLURM command availability,
  and the Symphony checkout without printing tokens or starting agents.
- Use when: validating the repo-specific Symphony setup before starting the
  daemon.

### `symphony-run`

- Inputs: same credentials and Symphony checkout as `symphony-doctor`, plus
  `WORKFLOW.md`.
- Output: rendered runtime workflow and a foreground Symphony daemon process.
- Use when: running the Linear-backed unattended agent workflow for this repo.

### `symphony-cleanup *arguments`

- Inputs: git worktree metadata, `SYMPHONY_WORKTREE_ROOT`, and optional Linear
  credentials from `SYMPHONY_ENV_FILE` or `~/.config/g-symphony/env`.
- Output: dry-run report of stale Symphony worktree, local branch, and remote
  branch candidates.
- Use when: reviewing stale completed or canceled Symphony issue worktrees
  before deleting anything.

### `symphony-cleanup-apply *arguments`

- Inputs: same as `symphony-cleanup`, plus optional deletion controls such as
  `--delete-local-branches` or `--delete-remote-branches`.
- Output: cleanup plan and non-forced git deletion command results.
- Use when: applying a reviewed cleanup plan. Worktree cleanup uses
  `git worktree remove` without `--force`; branch cleanup remains opt-in.

### `doctor-baselines`

- Inputs: `plink`, `plink2`, `regenie`.
- Output: baseline tool availability check.
- Use when: preparing to run external baseline or comparison benchmarks.

### `doctor-jax`

- Inputs: Python environment.
- Output: JAX runtime/device probe from `scripts/probe_jax_runtime.py`.
- Use when: checking CPU/GPU visibility for JAX.

### `probe-jax`

- Inputs: same as `doctor-jax`.
- Output: same as `doctor-jax`.
- Use when: you prefer the older probe-oriented recipe name.

## SLURM Helpers

### `slurm-gpu-shell`

- Inputs: SLURM environment and optional `GWAS_ENGINE_SLURM_*` variables.
- Output: interactive shell on the configured GPU node.
- Use when: debugging GPU environment or running manual commands on `landau`.

### `slurm-gpu-run command`

- Inputs: one shell command string and optional `GWAS_ENGINE_SLURM_*` variables.
- Output: command execution through `bash -lc` inside an `srun` allocation on
  the configured GPU node.
- Use when: submitting one GPU command without writing a dedicated recipe.

Example:

```bash
just slurm-gpu-run 'nvidia-smi'
just slurm-gpu-run 'uv run python scripts/probe_jax_runtime.py'
```

### `slurm-gpu-just +just_arguments`

- Inputs: another Just recipe and arguments.
- Output: that recipe executed inside a GPU SLURM allocation.
- Use when: running existing recipes on `landau`.

Example:

```bash
just slurm-gpu-just benchmark-regenie2-binary-hot-gpu-smoke
```

### `slurm-cpu-run command`

- Inputs: one shell command string and optional `GWAS_ENGINE_SLURM_*` variables.
- Output: command execution through `bash -lc` inside an `srun` CPU allocation.
- Use when: running CPU-heavy commands away from the login node. The default
  node is `GWAS_ENGINE_CPU_NODE=cantor`; set it to an empty string or another
  node name when the scheduler should choose a different CPU host.

### `slurm-cpu-just +just_arguments`

- Inputs: another Just recipe and arguments.
- Output: that recipe executed inside a CPU SLURM allocation.
- Use when: wrapping existing CPU benchmark recipes while preserving the Justfile
  interface.

## Direct REGENIE Runs

### `regenie-linear`

- Inputs: continuous phenotype, covariates, BGEN/sample files, and quantitative
  step 1 predictions.
- Output: quantitative REGENIE step 2 output under `data/regenie_linear`.
- Use when: manually running a local quantitative step 2 command through `g`.

### `regenie2-binary-gpu`

- Inputs: binary phenotype, covariates, BGEN/sample files, and binary step 1
  predictions.
- Output: binary REGENIE step 2 GPU output under
  `data/regenie2_binary_chr22_gpu`.
- Use when: running a full chr22 binary GPU step 2 workload.

### `regenie2-binary-gpu-smoke`

- Inputs: same as `regenie2-binary-gpu`.
- Output: 1,000-variant smoke output under
  `data/regenie2_binary_chr22_gpu_smoke`.
- Use when: checking binary GPU step 2 behavior quickly.

### `slurm-regenie2-binary-gpu`

- Inputs: same as `regenie2-binary-gpu`, plus SLURM GPU access.
- Output: full binary GPU step 2 run through SLURM.
- Use when: running the full workload safely on `landau`.

### `slurm-regenie2-binary-gpu-smoke`

- Inputs: same as `regenie2-binary-gpu-smoke`, plus SLURM GPU access.
- Output: smoke binary GPU step 2 run through SLURM.
- Use when: validating GPU step 2 without using the head node.

### `verify-regenie2-binary-gpu-output`

- Inputs: expected full GPU output run directory.
- Output: success message or failing file checks.
- Use when: checking that full binary GPU output contains Parquet parts.

### `verify-regenie2-binary-gpu-smoke-output`

- Inputs: expected smoke GPU output run directory.
- Output: success message or failing file checks.
- Use when: checking that smoke binary GPU output contains Parquet parts.

### `regenie2-chr10-matrix-dry-run *overrides`

- Inputs: Hydra tooling config `run_regenie2_chr10_matrix` through
  `tooling.cli.run_regenie2_matrix`, plus optional trailing Hydra overrides.
- Output: timestamped manifest, Markdown report, and `tooling.log` containing
  the six `g regenie` commands without executing them.
- Use when: inspecting the standard chr10 binary/linear CPU/GPU/cache matrix
  before submitting real work.

Example:

```bash
just regenie2-chr10-matrix-dry-run \
  tool.variant_limit=1000 \
  tool.output_dir=data/benchmarks/regenie2_chr10_matrix_plan
```

### `regenie2-chr10-matrix *overrides`

- Inputs: chr10 BGEN/sample files, continuous and binary phenotypes,
  covariates, chr10 quantitative and binary step 1 prediction lists, installed
  native perf extension, CPU/GPU-capable runtime, and optional trailing Hydra
  overrides.
- Output: `data/benchmarks/regenie2_chr10_matrix_<timestamp>/manifest.json`,
  `report.md`, `tooling.log`, per-run `events.jsonl`, per-run stage timings,
  and six output run directories under `runs/`.
- Use when: running the standard chr10 binary/linear step 2 comparison matrix.
  Prefer the SLURM wrapper for real GPU work.

### `slurm-regenie2-chr10-matrix *overrides`

- Inputs: same as `regenie2-chr10-matrix`, plus SLURM GPU access.
- Output: the standard chr10 matrix executed on the configured GPU node.
- Use when: handling the common agent task "run binary and linear step2 on
  chr10 on CPU/GPU/GPU cached, save results, and compare to previous run."

Example:

```bash
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
  just slurm-regenie2-chr10-matrix \
  tool.output_dir=/mnt/beegfs/kirill/Projects/g/data/benchmarks/regenie2_chr10_matrix_current
```

Historical full-chr10 comparison against the June 2026 logs needs the old
profile knobs:

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

Use the `Benchmark Artifact Analysis Recipes` section in `docs/tooling.md` for
the common `jq` commands that summarize manifests, old benchmark reports, hot
benchmark summaries, profile stage timings, and JSONL progress logs.

### `regenie2-chr22-matrix-dry-run *overrides`

- Inputs: Hydra tooling config `run_regenie2_chr22_matrix` through
  `tooling.cli.run_regenie2_matrix`, plus optional trailing Hydra overrides.
- Output: timestamped manifest, Markdown report, and `tooling.log` containing
  the six chr22 `g regenie` commands without executing them.
- Use when: inspecting the standard chr22 binary/linear CPU/GPU/cache matrix
  before submitting real work.

Example:

```bash
just regenie2-chr22-matrix-dry-run \
  tool.variant_limit=1000 \
  tool.output_dir=data/benchmarks/regenie2_chr22_matrix_plan
```

### `regenie2-chr22-matrix *overrides`

- Inputs: chr22 BGEN/sample files, continuous and binary phenotypes,
  covariates, chr22 quantitative and binary step 1 prediction lists, installed
  native perf extension, CPU/GPU-capable runtime, and optional trailing Hydra
  overrides.
- Output: `data/benchmarks/regenie2_chr22_matrix_<timestamp>/manifest.json`,
  `report.md`, `tooling.log`, per-run `events.jsonl`, per-run stage timings,
  and six output run directories under `runs/`.
- Use when: running the standard chr22 binary/linear step 2 comparison matrix.
  Prefer the SLURM wrapper for real GPU work.

### `slurm-regenie2-chr22-matrix *overrides`

- Inputs: same as `regenie2-chr22-matrix`, plus SLURM GPU access.
- Output: the standard chr22 matrix executed on the configured GPU node.
- Use when: handling the common agent task for chr22: binary and linear step 2
  on CPU/GPU/GPU cached, save results, and compare to the previous chr22 run.

Example:

```bash
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
  just slurm-regenie2-chr22-matrix \
  tool.output_dir=/mnt/beegfs/kirill/Projects/g/data/benchmarks/regenie2_chr22_matrix_current
```

## Baseline And Comparison Benchmarks

### `benchmark-baselines`

- Inputs: prepared data and external baseline tools.
- Output: baseline benchmark report from `scripts/benchmark.py`, excluding slow
  Hail runs by default.
- Use when: refreshing PLINK2/REGENIE baseline timings.

### `benchmark-baselines-full`

- Inputs: same as `benchmark-baselines`, plus cached Hail MatrixTable support.
- Output: full baseline benchmark report including Hail.
- Use when: explicitly refreshing slow baseline evidence.

### `benchmark-regenie-comparison-cpu`

- Inputs: data, baseline tools, installed perf extension.
- Output: original REGENIE versus `g` quantitative step 2 CPU comparison.
- Use when: checking CPU comparison behavior.

### `benchmark-regenie-comparison-gpu`

- Inputs: data, baseline tools, installed perf extension, GPU access.
- Output: original REGENIE versus `g` quantitative step 2 CPU/GPU comparison.
- Use when: checking comparison behavior including GPU. Prefer running through
  `slurm-gpu-just`.

### `benchmark-regenie-comparison`

- Inputs: same as `benchmark-regenie-comparison-cpu`.
- Output: CPU comparison benchmark.
- Use when: you want the default comparison benchmark alias.

## Tooling Benchmarks And Tuning

### `benchmark-bgen-reader *overrides`

- Inputs: installed perf extension and local BGEN data.
- Output: BGEN reader benchmark JSON printed to stdout, with optional report
  paths controlled by trailing Hydra overrides.
- Use when: measuring native BGEN chunk delivery paths.

### `benchmark-regenie2-linear-fresh-gpu`

- Inputs: installed perf extension, GPU access, fresh-process benchmark script
  inputs.
- Output: REGENIE step 2 fresh-process GPU benchmark report.
- Use when: measuring older linear fresh-process behavior. This recipe still
  uses a `scripts/` entrypoint rather than the new Hydra-backed `tooling/`
  interface.

### `benchmark-regenie2-linear-fresh-gpu-parquet`

- Inputs: same as `benchmark-regenie2-linear-fresh-gpu`.
- Output: fresh-process GPU benchmark using Parquet dataset output and
  finalization.
- Use when: comparing fresh-process behavior with finalized Parquet output.

### `benchmark-regenie2-binary-hot-gpu *overrides`

- Inputs: installed perf extension, binary step 2 inputs, GPU access, and
  optional trailing Hydra overrides.
- Output: binary-hot benchmark artifacts under the tool's output directory and a
  summary JSON.
- Use when: measuring cold process, same-process hot, no-final, and finalized
  binary step 2 timings.

### `benchmark-regenie2-binary-hot-gpu-smoke *overrides`

- Inputs: same as `benchmark-regenie2-binary-hot-gpu`.
- Output: smaller binary-hot summary for a 1,000-variant slice.
- Use when: validating the binary-hot harness quickly.

### `slurm-benchmark-regenie2-binary-hot-gpu *overrides`

- Inputs: same as `benchmark-regenie2-binary-hot-gpu`, plus SLURM GPU access.
- Output: binary-hot benchmark run through SLURM.
- Use when: running the benchmark on `landau`.

### `perf-smoke *arguments`

- Inputs: Python environment only; no GWAS data and no GPU are required.
- Output: timestamped smoke summary under
  `results/perf/smoke/smoke_<timestamp>/performance_smoke_summary.json` by
  default.
- Use when: checking the benchmark/report plumbing from the login node. This is
  intentionally small and safe for `gauss`.

Example:

```bash
just perf-smoke
```

### `perf-cpu *overrides`

- Inputs: SLURM CPU access, local benchmark data, and the native perf extension
  build prerequisites.
- Output: BGEN reader JSON and Markdown summaries under
  `results/perf/cpu/bgen_reader_<timestamp>/`.
- Use when: collecting a standard CPU-side benchmark without running heavy work
  on the login node. The recipe submits `benchmark-bgen-reader` through
  `slurm-cpu-just`.

Example:

```bash
just perf-cpu sweep.chunk_sizes=[4096,8192]
```

### `perf-gpu *overrides`

- Inputs: SLURM GPU access, binary step 2 inputs, and GPU-capable dependencies.
- Output: binary-hot benchmark artifacts and
  `regenie2_binary_hot_summary.json` under
  `results/perf/gpu/regenie2_binary_hot_<timestamp>/`.
- Use when: collecting the standard GPU performance benchmark through the
  existing `slurm-benchmark-regenie2-binary-hot-gpu` wrapper.

Example:

```bash
just perf-gpu tool.variant_limit=1000
```

### `perf-compare baseline_json new_json`

- Inputs: two benchmark JSON summaries.
- Output: concise Markdown table comparing common speed, memory, and numerical
  metrics; exits nonzero for malformed JSON or summaries with no common
  metrics.
- Use when: comparing smoke outputs, BGEN reader summaries, binary-hot summaries,
  or matrix manifests.

Example:

```bash
just perf-compare results/perf/baseline.json results/perf/new.json
```

### `benchmark-output-stages-gpu *overrides`

- Inputs: installed perf extension, quantitative step 2 inputs, GPU access.
- Output: output-stage benchmark JSON and Markdown summaries.
- Use when: measuring writer/finalization bottlenecks.

### `tune-regenie2-gpu *overrides`

- Inputs: installed perf extension, data, baseline tools, GPU access.
- Output: GPU tuning artifacts under `data/benchmarks/regenie2_gpu_tuning`.
- Use when: sweeping BGEN, compute, writer, and finalist knobs for step 2.

### `benchmark-rust`

- Inputs: Rust toolchain.
- Output: Rust Criterion benchmark results.
- Use when: measuring Rust-only native components.

## Profiling

### `profile-regenie-comparison-cpu`

- Inputs: data, baseline tools, installed perf extension.
- Output: profile comparison for original REGENIE and `g` quantitative step 2 on
  CPU.
- Use when: profiling the CPU comparison path.

### `profile-regenie-comparison-gpu`

- Inputs: data, baseline tools, installed perf extension, GPU access.
- Output: profile comparison including GPU.
- Use when: profiling the GPU comparison path. Prefer SLURM.

### `profile-regenie-comparison`

- Inputs: same as `profile-regenie-comparison-cpu`.
- Output: CPU profile comparison.
- Use when: you want the default profile-comparison alias.

### `profile-app-full-dry-run *overrides`

- Inputs: Hydra profile config and optional trailing overrides.
- Output: `profile_plan.json`, `profile_plan.md`, and `tooling.log` under the
  configured profile output directory.
- Use when: checking the full app profiling plan before submitting a long run.
- Notes: sets `tool.include_regenie_baseline=false`; existing step 1 prediction
  lists must be present.

Example:

```bash
just profile-app-full-dry-run tool.output_dir=data/profiles/app_profile_plan
```

### `profile-app-full-smoke *overrides`

- Inputs: data, baseline tools, installed perf extension, GPU access, and
  optional trailing Hydra overrides.
- Output: reduced full-profile artifacts under `data/profiles/landau_deep_*` or
  the configured `tool.output_dir`.
- Use when: validating JAX trace, cProfile, py-spy, perf, stage-timing, and
  summary artifact generation on a small workload.
- Notes: sets `tool.include_regenie_baseline=false`; use
  `tool.include_regenie_baseline=true` only when external `regenie` is available.
  Also sets `tool.enable_rust_criterion=false` so smoke checks stay short.

Example:

```bash
just slurm-gpu-just profile-app-full-smoke tool.output_dir=data/profiles/app_profile_smoke
```

### `profile-app-full-landau *overrides`

- Inputs: data, baseline tools, SLURM GPU access, and optional trailing Hydra
  overrides.
- Output: full app profiling bundle on `landau`; defaults to 12 hours, 8 CPUs,
  64G memory, and 1 GPU unless overridden through `GWAS_ENGINE_SLURM_*`.
- Use when: profiling the app end to end for bottleneck analysis.
- Notes: sets `tool.include_regenie_baseline=false`; use
  `tool.include_regenie_baseline=true` only when external `regenie` is available.

The run captures JAX traces, JAX device-memory profiles, Python cProfile,
py-spy speedscope profiles when available, Linux perf data when available, Rust
Criterion benches, stage timings, subprocess logs, `summary.json`, and
`summary.md`.

Example:

```bash
just profile-app-full-landau tool.output_dir=data/profiles/app_profile_current
```

### `profile-regenie2-deep *overrides`

- Inputs: data, baseline tools, installed perf extension, GPU access.
- Output: deep profiling artifacts under `data/profiles/landau_deep_*` unless an
  explicit output directory is configured.
- Use when: running the lower-level full profile harness on the current host.

### `profile-regenie2-deep-smoke *overrides`

- Inputs: same as `profile-regenie2-deep`.
- Output: reduced deep-profile smoke artifacts.
- Use when: validating only sweeps/headlines without deep profiler captures.

### `profile-regenie2-deep-landau *overrides`

- Inputs: data, baseline tools, SLURM GPU access.
- Output: long deep-profile campaign on `landau`; defaults to 12 hours, 8 CPUs,
  64G memory, and 1 GPU unless overridden through `GWAS_ENGINE_SLURM_*`.
- Use when: running the lower-level full profile harness on the GPU node.

## Formatting, Linting, Type Checking, And Tests

### `format`

- Inputs: Python environment and Rust toolchain.
- Output: formatted Python and Rust files.
- Use when: applying repository formatting.

### `lint`

- Inputs: Python environment and Rust toolchain.
- Output: ruff fixes and Cargo clippy diagnostics.
- Use when: applying Python lint fixes and checking Rust lints.

### `typecheck`

- Inputs: Python environment.
- Output: `ty` diagnostics for `src`, `tests`, `scripts`, and `tooling`.
- Use when: checking Python types.

### `check`

- Inputs: same as `format`, `lint`, and `typecheck`.
- Output: full format/lint/typecheck lane.
- Use when: running the default local quality gate with Rust tooling available.

### `format-local-check`

- Inputs: Python environment.
- Output: ruff format check only.
- Use when: checking formatting without Nix or direct Cargo access.

### `lint-local`

- Inputs: Python environment.
- Output: ruff diagnostics without fixes.
- Use when: checking Python lint locally.

### `typecheck-local`

- Inputs: Python environment.
- Output: `ty` diagnostics.
- Use when: type checking without the full Rust/Nix lane.

### `test-local-focused`

- Inputs: Python environment and native extension build support.
- Output: focused pytest results for core and output tests.
- Use when: running a quick no-Nix smoke suite.

### `test-local`

- Inputs: Python environment and native extension build support.
- Output: non-heavy pytest suite excluding `phase0_data` and `phase1_parity`.
- Use when: running broader local tests without data/parity workloads.

### `check-local`

- Inputs: Python environment and native extension build support.
- Output: local format check, lint, typecheck, and focused tests.
- Use when: running a no-Nix verification lane.

### `ci-lint`

- Inputs: frozen lockfile and dev dependencies.
- Output: ruff diagnostics in a no-install-project CI environment.
- Use when: reproducing CI lint behavior.

### `ci-typecheck`

- Inputs: frozen lockfile and dev dependencies.
- Output: `ty` diagnostics in a no-install-project CI environment.
- Use when: reproducing CI typecheck behavior.

### `ci-test`

- Inputs: frozen lockfile and installed project.
- Output: pytest results excluding heavy data/parity suites.
- Use when: reproducing CI test behavior.

### `test`

- Inputs: full Python test environment.
- Output: full pytest run.
- Use when: intentionally running all tests.

### `coverage-python`

- Inputs: full Python test environment.
- Output: Python coverage report with a 90 percent gate.
- Use when: checking Python coverage.

### `coverage-rust`

- Inputs: Rust toolchain with `cargo llvm-cov`.
- Output: Rust line coverage report with a 90 percent gate.
- Use when: checking Rust coverage.

### `coverage`

- Inputs: same as `coverage-python` and `coverage-rust`.
- Output: both coverage gates.
- Use when: running complete coverage validation.

## Codex Task Farm

The Codex task farm recipes wrap `scripts/codex_task_farm.py`. They are
automation tooling, not GWAS benchmark tooling.

### `codex-tasks-sync`

- Inputs: `docs/code-review.md`.
- Output: `docs/code-review.tasks.json`.
- Use when: regenerating the default task manifest.

### `codex-tasks-doctor *arguments`

- Inputs: optional task-farm doctor arguments.
- Output: prerequisite diagnostics.
- Use when: validating task-farm setup.

### `codex-tasks-list *arguments`

- Inputs: optional list filters.
- Output: task list.
- Use when: inspecting default task state.

### `codex-tasks-run *arguments`

- Inputs: optional run controls.
- Output: launched worker agents and task state updates.
- Use when: starting task-farm workers.

### `codex-tasks-status *arguments`

- Inputs: optional status controls.
- Output: task-farm status summary.
- Use when: checking worker/task progress.

### `codex-tasks-review +arguments`

- Inputs: one or more task identifiers or review arguments.
- Output: review results.
- Use when: reviewing task branches.

### `codex-tasks-integrate +arguments`

- Inputs: one or more reviewed task identifiers.
- Output: integrated task branches.
- Use when: integrating selected reviewed work.

### `codex-tasks-integrate-ready *arguments`

- Inputs: optional integration controls.
- Output: all reviewed ready branches integrated in order.
- Use when: advancing the default integration worktree.

## Review 2 Task Farm

The Review 2 recipes use `docs/code-review-2.tasks.json` and the Review 2
state/worktree settings.

### `codex-review2-sync`

- Inputs: `docs/02.code-review-2-06-26.md`.
- Output: Review 2 manifest, plan, state directory, branches, and worktrees.
- Use when: regenerating Review 2 task state.

### `codex-review2-doctor *arguments`

- Inputs: optional doctor arguments.
- Output: Review 2 prerequisite diagnostics.
- Use when: checking Review 2 task-farm setup.

### `codex-review2-list *arguments`

- Inputs: optional list filters.
- Output: Review 2 task list.
- Use when: inspecting Review 2 task state.

### `codex-review2-claim *arguments`

- Inputs: optional claim controls.
- Output: claimed Review 2 tasks.
- Use when: reserving work without launching workers.

### `codex-review2-run *arguments`

- Inputs: optional run controls.
- Output: launched Review 2 workers and task state updates.
- Use when: starting Review 2 task-farm workers.

### `codex-review2-status *arguments`

- Inputs: optional status controls.
- Output: Review 2 status summary.
- Use when: checking Review 2 progress.

### `codex-review2-review +arguments`

- Inputs: one or more Review 2 task identifiers or review arguments.
- Output: review results.
- Use when: reviewing Review 2 branches.

### `codex-review2-integrate +arguments`

- Inputs: one or more reviewed Review 2 task identifiers.
- Output: integrated Review 2 branches.
- Use when: integrating selected Review 2 work.

### `codex-review2-integrate-ready *arguments`

- Inputs: optional integration controls.
- Output: all ready Review 2 branches integrated in order.
- Use when: advancing the Review 2 integration branch.

### `codex-review2-diff +arguments`

- Inputs: one or more Review 2 task identifiers or diff arguments.
- Output: branch diffs.
- Use when: inspecting Review 2 changes.

### `codex-review2-log +arguments`

- Inputs: one or more Review 2 task identifiers or log arguments.
- Output: runtime logs.
- Use when: debugging Review 2 workers.

### `codex-review2-block +arguments`

- Inputs: one or more Review 2 task identifiers and optional reason arguments.
- Output: blocked task state.
- Use when: marking tasks that cannot progress.

### `codex-review2-abandon +arguments`

- Inputs: one or more Review 2 task identifiers and optional reason arguments.
- Output: abandoned task state.
- Use when: retiring Review 2 tasks.

### `codex-review2-reset-claim *arguments`

- Inputs: optional reset filters.
- Output: reset stale Review 2 task claims.
- Use when: clearing old worker claims.

### `codex-review2-clean-integrated *arguments`

- Inputs: optional cleanup filters.
- Output: removed worktrees for integrated Review 2 tasks.
- Use when: cleaning integrated Review 2 worktrees.

### `codex-review2-promote-to-main *arguments`

- Inputs: optional promotion arguments.
- Output: Review 2 integration branch promoted to `main`.
- Use when: explicitly promoting Review 2 integration work. This is a
  repository-history operation; do not run it during ordinary feature work.

## Dependency Upgrades

### `upgrade-python-deps`

- Inputs: network access and Python package indexes.
- Output: updated/synced Python dependencies for dev and GPU groups.
- Use when: intentionally upgrading Python dependencies.

### `upgrade-nix-lock`

- Inputs: network access and Nix.
- Output: updated `flake.lock`.
- Use when: intentionally upgrading Nix inputs.

### `upgrade-deps`

- Inputs: same as `upgrade-python-deps` and `upgrade-nix-lock`.
- Output: upgraded Python dependencies and Nix lockfile.
- Use when: performing a coordinated dependency refresh.

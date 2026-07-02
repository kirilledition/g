# Server Gauss SLURM

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; server-specific development operations | main branch as of 2026-06-30 gauss SLURM workflows | Development maintainers |

This repository was originally developed inside the Nix flake on a personal machine. On the Ubuntu SLURM server, keep `just` as the single entrypoint and split work between:

- login node: dependency sync, formatting, linting, CPU-only tests, lightweight iteration
- GPU node (`landau` by default): JAX CUDA probing, GPU tests, GPU benchmarks, REGENIE profiling
- CPU compute node (`cantor` by default): full CPU validation, Rust builds/tests,
  Python test parallelism, CPU-heavy benchmark runs

## Required Host Tooling

Install or make available on `PATH`:

- `git`
- `uv`
- `srun`
- `zstd`

The repo-local dev-bootstrap installs `just`, `cargo`, `rustc`, `plink`,
`plink2`, and `regenie` into `.tools/`. Python itself does not need to be
preinstalled globally if `uv python install` works in your account.

## Bootstrap

First-run setup before `just` is available:

```bash
UV_CACHE_DIR=/tmp/g-uv-cache uv run --group dev python -m tooling.cli.server tool.name=bootstrap_tools
source tooling/server/server_env.sh
```

CPU-oriented login-node setup after the first-run dev-bootstrap:

```bash
just dev-bootstrap
```

GPU-capable environment:

```bash
just dev-bootstrap-gpu
```

Sanity checks:

```bash
just doctor-server
just doctor
just doctor-baselines
```

## Workflow Tiers

Run quick iteration on the login node:

```bash
just check-local
just test-local
just test-local-focused
just perf-smoke
just perf-compare BASE.json NEW.json
```

`check-local` runs Python format/lint/type checks plus a focused native-extension
smoke. `test-local` runs the non-data Python suite serially. These are the right
commands for small edits and head-node feedback.

Use one CPU SLURM node for full CPU validation:

```bash
just slurm-cpu-check
just slurm-cpu-test
just slurm-cpu-test-full
just slurm-cpu-rust-build
just slurm-cpu-rust-test
just slurm-cpu-coverage
```

`slurm-cpu-check` wraps `just check`. `slurm-cpu-test` runs the non-data Python
suite with large-node pytest parallelism. `slurm-cpu-test-full` and
`just slurm-cpu-just test` run the full Python suite, including data/parity
tests when the required local files are present. Do not run full `just check`,
full `just test`, Rust dependency builds, or Rust test builds directly on the
login node.

`perf-smoke` and `perf-compare` are intentionally login-node-safe. Do not run
`perf-cpu`, `perf-gpu`, full benchmark sweeps, or GPU commands directly on the
login node.

Prepare benchmark data only after `plink2` is available:

```bash
just data-prepare
```

Generate binary REGENIE step 1 predictions required by binary step 2:

```bash
just data-baseline-binary
just data-verify-binary-gpu-inputs
```

## CPU Workflow Through SLURM

The CPU node defaults to `cantor`, which currently reports 40 CPUs and about
192 GB memory. The CPU helpers request one task on one node, use
`--exclusive` by default, and set `CARGO_BUILD_JOBS` from the allocation inside
the job. Override cluster-specific settings independently from GPU settings:

```bash
export GWAS_ENGINE_CPU_NODE=cantor
export GWAS_ENGINE_CPU_CPUS_PER_TASK=40
export GWAS_ENGINE_CPU_MEMORY=128G
export GWAS_ENGINE_CPU_TIME=04:00:00
export GWAS_ENGINE_CPU_PARTITION=compute
export GWAS_ENGINE_CPU_ACCOUNT=my-account
export GWAS_ENGINE_CPU_EXTRA_ARGS='--reservation=my-reservation'
export GWAS_ENGINE_SLURM_EXCLUSIVE=1
```

The older generic `GWAS_ENGINE_SLURM_*` variables still work as fallbacks, but
prefer `GWAS_ENGINE_CPU_*` for CPU jobs and `GWAS_ENGINE_GPU_*` for GPU jobs.
Set `GWAS_ENGINE_CPU_NODE` to an empty string when the scheduler should choose
the CPU host.

Open an interactive CPU shell:

```bash
just slurm-cpu-shell
```

Run one-off commands or existing recipes on the CPU node:

```bash
just slurm-cpu-run 'cargo build --workspace --all-targets'
just slurm-cpu-just check
just slurm-cpu-just test
just slurm-cpu-just rust-test
```

Inside CPU SLURM jobs, `tooling/server/server_env.sh` derives
`GWAS_ENGINE_ALLOCATED_CPU_COUNT` from `SLURM_CPUS_PER_TASK`,
`SLURM_CPUS_ON_NODE`, or `nproc`; sets `CARGO_BUILD_JOBS` to that count unless
already configured; and sets `GWAS_ENGINE_PYTEST_WORKERS` for pytest. Rust
build recipes call `gwas_engine_configure_rust_build_environment` for Cargo job
parallelism only; set linker and rustc-wrapper environment variables explicitly
when a run needs them. The maintained `maturin develop` entrypoints pass the
resolved `CARGO_BUILD_JOBS` value with `-j` so the explicit compile cap reaches
Cargo.
Python tests default to at most 8 xdist workers because the suite imports JAX
and the native extension, so one pytest worker per core can oversubscribe
process-level JAX/native thread pools. Override after measuring:

```bash
GWAS_ENGINE_CPU_PYTEST_WORKERS=16 just slurm-cpu-test
GWAS_ENGINE_CPU_PYTEST_WORKERS=1 just slurm-cpu-test-full
```

When xdist is active, pytest subprocesses get conservative BLAS/OpenMP thread
limits to reduce accidental oversubscription. Cargo builds and Rust tests use
the full allocated CPU count through Cargo's own scheduler.

Build-environment defaults:

- Do not share one global Rust `target/` directory across Symphony worktrees by
  default; that can reduce cold builds but risks cross-worktree contention and
  confusing invalidation during unattended branch work.
- Do not push every focused test through SLURM; local `check-local`,
  `test-local`, and targeted `uv run pytest ...` remain faster for small edits.
- Cargo uses the repo-local `.cargo/config.toml` for Linux Rust builds:
  `target-cpu=native` is always enabled. The repo does not choose a linker or
  rustc wrapper; set `RUSTFLAGS`, `CARGO_TARGET_*_LINKER`, `RUSTC_WRAPPER`, and
  cache variables explicitly for isolated experiments.

## GPU Workflow Through SLURM

The GPU node defaults to `landau`. Override cluster-specific settings with GPU
environment variables when needed:

```bash
export GWAS_ENGINE_GPU_NODE=landau
export GWAS_ENGINE_GPU_PARTITION=gpu
export GWAS_ENGINE_GPU_ACCOUNT=my-account
export GWAS_ENGINE_GPU_CPUS_PER_TASK=8
export GWAS_ENGINE_GPU_MEMORY=64G
export GWAS_ENGINE_GPU_TIME=04:00:00
export GWAS_ENGINE_GPU_GPUS_PER_TASK=1
```

Open an interactive GPU shell:

```bash
just slurm-gpu-shell
```

Run a one-off command on the GPU node:

```bash
just slurm-gpu-run 'nvidia-smi'
just slurm-gpu-run 'uv run python -m tooling.cli.performance tool.name=jax_runtime'
```

Run existing repo recipes on the GPU node while keeping `just` as the top-level interface:

```bash
just slurm-gpu-just doctor-jax
just slurm-gpu-just matrix-chr22-smoke
just matrix-chr22-smoke
just slurm-gpu-just matrix-chr22
just matrix-chr22
just slurm-gpu-just legacy-regenie-comparison-gpu
just slurm-gpu-just legacy-profile-regenie-comparison-gpu
```

`slurm-gpu-run` and `slurm-gpu-shell` start in the repository root and call
`gwas_engine_configure_rust_build_environment` before the requested command.
That keeps GPU-side `maturin`, Cargo, clippy, or profiler-tool builds on the
same `CARGO_BUILD_JOBS` policy used by CPU jobs.

Run standard performance harness entrypoints:

```bash
just perf-cpu
just perf-gpu tool.variant_limit=1000
```

`perf-cpu` submits the BGEN reader benchmark through a CPU SLURM allocation and
writes summaries under `results/perf/cpu/`. Override the CPU host with
`GWAS_ENGINE_CPU_NODE`; set it to an empty string if the scheduler should choose
the node. `perf-gpu` wraps the existing binary-hot GPU SLURM recipe and writes
under `results/perf/gpu/`.

## Timing Notes

Baseline measured on `cantor` before the CPU workflow update, using a 40-CPU,
128 GB, exclusive allocation:

- `just check`: 96.53 seconds with a cold Cargo dependency cache.
- `just test`: failed after 714.91 seconds because `cantor` had no `git`
  executable, causing `tests/test_symphony_sync_main.py` to fail near the end of
  the serial run. Installing user-level `git` with `pixi global install git`
  made `git` available on both `gauss` and `cantor`.

Post-change timings on the same node and allocation:

- `just slurm-cpu-just check`: 17.75 seconds with warm Cargo artifacts.
- `just slurm-cpu-check`: 66.69 seconds after a native Rust source change forced
  a rebuild.
- `just test-local`: 731.03 seconds on the login node for the serial non-data
  suite.
- `just slurm-cpu-test`: 212.81 seconds for the same non-data suite with 8 xdist
  workers.
- `just slurm-cpu-just test`: 224.76 seconds for the full suite with 8 xdist
  workers.
- `just slurm-cpu-test-full`: 230.24 seconds through the explicit full-suite
  alias.
- `just slurm-cpu-rust-build`: 128.03 seconds.
- `just slurm-cpu-rust-test`: 32.45 seconds after the strict-resume diagnostic
  fix.
- `just slurm-cpu-coverage`: 408.87 seconds, with Python coverage at 92.73% and
  Rust line coverage at 91.21%.

Use large-node validation when the command will compile Rust dependencies, build
Rust tests, run the full Python suite, or combine Python and Rust checks. For
single-file Python changes, targeted `uv run pytest tests/<file>.py` or
`just test-local-focused` is usually faster than queueing a SLURM job.

The binary chr22 GPU run uses:

```bash
data/1kg_chr22_full.bgen
data/1kg_chr22_full.sample
data/pheno_bin.txt
data/covariates.txt
data/baselines/regenie_step1_pred.list
```

Outputs are written under:

```bash
data/regenie2_binary_chr22_gpu.g/trait_0001_phenotype_binary.regenie2_binary.run/
```

## Notes

- `just upgrade-deps` is now Python-package only. Nix lockfile updates moved to `just upgrade-nix-lock`.
- `just doctor-jax` should be treated as a host-specific check. On a login node without NVIDIA libraries, CPU fallback is expected.
- JAX persistent compilation cache defaults under the platform temporary directory; use
  `--jax_cache_dir` for an explicit cache path.
- Profiling recipes isolate CPU JAX caches under
  `${G_PROFILE_CPU_JAX_CACHE_PARENT:-/tmp/g-jax-cpu-profile-cache}/host-<hostname>/features-<cpu-fingerprint>/`
  to avoid reusing CPU AOT artifacts across SLURM nodes. GPU profile caches stay
  node-local under `${G_PROFILE_GPU_JAX_CACHE_PARENT:-/tmp/g-jax-profile-cache}`.
- `.tools/` and `data/` are local server state and must not be committed.
- `results/` contains local benchmark output, including `perf-*` summaries, and
  must not be committed.
- `tooling/server/server_env.sh` sets repo-local tools on `PATH`,
  `UV_CACHE_DIR=/tmp/g-uv-cache`, `UV_LINK_MODE=copy`, repo-local Rust homes
  unless those variables are already set, and CPU allocation-derived variables
  when a CPU SLURM wrapper calls `gwas_engine_configure_cpu_parallelism`.

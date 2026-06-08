# Ubuntu + SLURM Development

This repository was originally developed inside the Nix flake on a personal machine. On the Ubuntu SLURM server, keep `just` as the single entrypoint and split work between:

- login node: dependency sync, formatting, linting, CPU-only tests, lightweight iteration
- GPU node (`landau` by default): JAX CUDA probing, GPU tests, GPU benchmarks, REGENIE profiling
- CPU compute node (`cantor` by default for benchmark wrappers): CPU-heavy benchmark runs

## Required Host Tooling

Install or make available on `PATH`:

- `uv`
- `srun`
- `zstd`

The repo-local bootstrap installs `just`, `cargo`, `rustc`, `plink`, `plink2`, and `regenie` into `.tools/`. Python itself does not need to be preinstalled globally if `uv python install` works in your account.

## Bootstrap

First-run setup before `just` is available:

```bash
UV_CACHE_DIR=/tmp/g-uv-cache uv run --no-project python scripts/bootstrap_server_tools.py
source scripts/server_env.sh
```

CPU-oriented login-node setup after the first-run bootstrap:

```bash
just bootstrap
```

GPU-capable environment:

```bash
just bootstrap-gpu
```

Sanity checks:

```bash
just doctor-server
just doctor
just doctor-baselines
```

## Normal Development Flow

Run these on the login node:

```bash
just check
just test
just coverage
just perf-smoke
just perf-compare BASE.json NEW.json
```

`perf-smoke` and `perf-compare` are intentionally login-node-safe. Do not run
`perf-cpu`, `perf-gpu`, full benchmark sweeps, or GPU commands directly on the
login node.

Prepare benchmark data only after `plink2` is available:

```bash
just setup-data
```

Generate binary REGENIE step 1 predictions required by binary step 2:

```bash
just setup-binary-baseline
just verify-regenie2-binary-gpu-inputs
```

## GPU Workflow Through SLURM

The GPU node defaults to `landau`. Override cluster-specific settings with environment variables when needed:

```bash
export GWAS_ENGINE_GPU_NODE=landau
export GWAS_ENGINE_SLURM_PARTITION=gpu
export GWAS_ENGINE_SLURM_ACCOUNT=my-account
export GWAS_ENGINE_SLURM_CPUS_PER_TASK=8
export GWAS_ENGINE_SLURM_MEMORY=64G
export GWAS_ENGINE_SLURM_TIME=04:00:00
```

Open an interactive GPU shell:

```bash
just slurm-gpu-shell
```

Run a one-off command on the GPU node:

```bash
just slurm-gpu-run 'nvidia-smi'
just slurm-gpu-run 'uv run python scripts/probe_jax_runtime.py'
```

Run existing repo recipes on the GPU node while keeping `just` as the top-level interface:

```bash
just slurm-gpu-just doctor-jax
just slurm-regenie2-binary-gpu-smoke
just verify-regenie2-binary-gpu-smoke-output
just slurm-regenie2-binary-gpu
just verify-regenie2-binary-gpu-output
just slurm-gpu-just benchmark-regenie-comparison-gpu
just slurm-gpu-just profile-regenie-comparison-gpu
```

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
data/regenie2_binary_chr22_gpu.regenie2_binary.run/
```

## Notes

- `just upgrade-deps` is now Python-package only. Nix lockfile updates moved to `just upgrade-nix-lock`.
- `just doctor-jax` should be treated as a host-specific check. On a login node without NVIDIA libraries, CPU fallback is expected.
- JAX persistent compilation cache lives under `JAX_COMPILATION_CACHE_DIR` when set, otherwise under the current user cache directory.
- `.tools/` and `data/` are local server state and must not be committed.
- `results/` contains local benchmark output, including `perf-*` summaries, and
  must not be committed.
- `scripts/server_env.sh` sets repo-local tools on `PATH`, `UV_CACHE_DIR=/tmp/g-uv-cache`, `UV_LINK_MODE=copy`, and repo-local Rust homes unless those variables are already set.

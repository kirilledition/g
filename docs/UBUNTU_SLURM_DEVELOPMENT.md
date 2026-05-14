# Ubuntu + SLURM Development

This repository was originally developed inside the Nix flake on a personal machine. On the Ubuntu SLURM server, keep `just` as the single entrypoint and split work between:

- login node: dependency sync, formatting, linting, CPU-only tests, lightweight iteration
- GPU node (`landau` by default): JAX CUDA probing, GPU tests, GPU benchmarks, REGENIE profiling

## Required Host Tooling

Install or make available on `PATH`:

- `just`
- `uv`
- `cargo`
- `rustc`
- `plink`
- `plink2`
- `regenie`

Python itself does not need to be preinstalled globally if `uv python install` works in your account.

## Bootstrap

CPU-oriented login-node setup:

```bash
just bootstrap
```

GPU-capable environment:

```bash
just bootstrap-gpu
```

Sanity checks:

```bash
just doctor
just doctor-baselines
```

## Normal Development Flow

Run these on the login node:

```bash
just check
just test
```

Prepare benchmark data only after `plink2` is available:

```bash
just setup-data
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
just slurm-gpu-run nvidia-smi
just slurm-gpu-run uv run python scripts/probe_jax_runtime.py
```

Run existing repo recipes on the GPU node while keeping `just` as the top-level interface:

```bash
just slurm-gpu-just doctor-jax
just slurm-gpu-just benchmark-regenie-comparison-gpu
just slurm-gpu-just profile-regenie-comparison-gpu
```

## Notes

- `just upgrade-deps` is now Python-package only. Nix lockfile updates moved to `just upgrade-nix-lock`.
- `just doctor-jax` should be treated as a host-specific check. On a login node without NVIDIA libraries, CPU fallback is expected.
- JAX persistent compilation cache lives under `JAX_COMPILATION_CACHE_DIR` when set, otherwise under the current user cache directory.

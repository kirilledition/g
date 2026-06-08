# Installation

`g` is managed with `uv` and `just`. Use the project recipes when possible so local and server workflows stay aligned.

## Requirements

- Python `>=3.14,<3.15`
- `uv`
- `just`
- Rust/Cargo and `maturin`
- Optional CUDA-capable JAX environment for GPU execution
- Optional baseline tools: `plink`, `plink2`, `regenie`, and `zstd`

## Nix Development Shell

If Nix is available:

```bash
nix develop
```

Then use the normal `just` commands for checks, tests, and runs.

## CPU-oriented Bootstrap

```bash
just bootstrap
just doctor
```

This installs the configured Python version and syncs development dependencies through `uv`.

## GPU-capable Bootstrap

```bash
just bootstrap-gpu
just doctor-jax
```

GPU support depends on the JAX CUDA environment resolving correctly on the target machine. Use SLURM recipes for GPU work on the server rather than running heavy GPU commands on the login node.

## Ubuntu / SLURM Server Bootstrap

On the server, bootstrap repo-local tools and source the generated environment:

```bash
UV_CACHE_DIR=/tmp/g-uv-cache uv run --no-project python scripts/bootstrap_server_tools.py
source scripts/server_env.sh
just doctor-server
```

More environment-specific notes are in [No-Nix Development](../development/NO_NIX_DEVELOPMENT.md) and [Ubuntu SLURM Development](../development/UBUNTU_SLURM_DEVELOPMENT.md).

## Documentation Dependencies

Documentation dependencies live in the `docs` dependency group:

```bash
uv sync --group docs
just docs-build
```

The GitHub Pages workflow installs the docs group with `uv` and builds with Zensical.

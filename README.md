# GWAS Engine (`g`)

[![PR CI](https://github.com/kirilledition/g/actions/workflows/pr-ci.yml/badge.svg)](https://github.com/kirilledition/g/actions/workflows/pr-ci.yml)
[![Science Monthly](https://github.com/kirilledition/g/actions/workflows/science-monthly.yml/badge.svg)](https://github.com/kirilledition/g/actions/workflows/science-monthly.yml)

`g` is a GPU-accelerated GWAS engine for BGEN-backed **REGENIE step 2** association scans. The current package is a Python 3.14 API/CLI backed by JAX compute code and a Rust/PyO3 native extension for BGEN parsing and output persistence.

The active public surface is intentionally narrow:

- Python API: `g.regenie(config)` and `g.regenie.from_options({...})`
- CLI: `g regenie ...`, `g-regenie ...`, and `g config init|validate|explain`
- Inputs: BGEN 1.2 genotype data, optional `.sample`, phenotype/covariate tables, and REGENIE step 1 `_pred.list` files
- Outputs: resumable Arrow chunk run directories, with optional compressed `final.parquet` finalization

Legacy direct `linear` and `logistic` entrypoints are no longer public.

## Status

Active development targets biobank-scale REGENIE step 2 workflows.

- Quantitative REGENIE step 2 is the primary supported workflow.
- Binary REGENIE step 2 is public but still partial/evolving.
- REGENIE step 1 is not implemented in `g`; use original `regenie` to generate prediction lists.
- The default REGENIE step 2 chunk size is `8192` variants.

Binary mode currently supports score-test-only output by default and approximate Firth fallback with `--firth --approx`. SPA and exact Firth without `--approx` are exposed as REGENIE-style flags but are not implemented yet.

## Repository Layout

- `src/g/` - Python package, CLI, API, JAX setup, I/O, and compute orchestration
- `src/g/compute/` - REGENIE step 2 quantitative and binary kernels
- `src/g/engine/` - BGEN-backed pipeline orchestration and cache warming
- `src/g/io/` - input source handling and output run management
- `src/*.rs` - Rust native extension modules for BGEN, sample, pipeline, and output paths
- `benches/` - Rust Criterion benchmarks
- `tests/` - pytest coverage for API, CLI, I/O, Rust architecture, and REGENIE pipelines
- `scripts/` - data setup, benchmark, profiling, and server bootstrap utilities
- `docs/` - development notes, roadmaps, style guide, and Ubuntu/SLURM instructions
- `archive/` - archived reference/experimental code, not the active package

Local generated state lives in `data/`, `.tools/`, `.venv/`, and `target/`; these are git-ignored.

## Requirements

The project is managed with `uv`, `just`, Rust/Cargo, and `maturin`.

- Python: `>=3.14,<3.15`
- Python runtime dependencies: JAX, NumPy, Click
- Native extension: Rust 2024, PyO3 ABI3 for Python 3.14
- Benchmark/data tools: `plink`, `plink2`, `regenie`, `zstd`
- Optional GPU workflow: CUDA-capable JAX environment and SLURM access

On systems with Nix, the flake provides the expected development tools:

```bash
nix develop
```

On the Ubuntu/SLURM server, bootstrap repo-local tools first:

```bash
UV_CACHE_DIR=/tmp/g-uv-cache uv run --no-project python scripts/bootstrap_server_tools.py
source scripts/server_env.sh
```

## Setup

CPU-oriented development environment:

```bash
just bootstrap
just doctor
```

GPU-capable development environment:

```bash
just bootstrap-gpu
just doctor-jax
```

Server-specific checks:

```bash
just doctor-server
just doctor-baselines
```

Prepare the local 1000 Genomes chromosome 22 benchmark data and simulated phenotypes:

```bash
just setup-data
```

Generate binary REGENIE step 1 baseline predictions for binary step 2:

```bash
just setup-binary-baseline
```

## CLI Usage

Quantitative REGENIE step 2:

```bash
uv run g \
  regenie \
  --step 2 \
  --qt \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --phenoFile data/pheno_cont.txt \
  --phenoCol phenotype_continuous \
  --covarFile data/covariates.txt \
  --covarColList age,sex \
  --pred data/baselines/regenie_step1_qt_pred.list \
  --out data/example_regenie2
```

Binary traits with approximate Firth fallback:

```bash
uv run g \
  regenie \
  --step 2 \
  --bt \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --phenoFile data/pheno_bin.txt \
  --phenoCol phenotype_binary \
  --covarFile data/covariates.txt \
  --covarColList age,sex \
  --pred data/baselines/regenie_step1_pred.list \
  --firth \
  --approx \
  --pThresh 0.01 \
  --out data/example_regenie2_binary
```

Config files use the same option names under TOML sections:

```bash
uv run g config init --out regenie.toml
uv run g config validate regenie.toml
uv run g regenie --config regenie.toml --g-device gpu
```

Runtime configuration is resolved in this order: packaged defaults from
`src/g/config.default.toml`, values in `--config`, then explicit CLI flags. The
packaged default file sets safe runtime defaults for trait mode, binary fallback
knobs, compute, output, and diagnostics, but it intentionally omits
workload-specific required inputs such as `bgen`, `phenoFile`, `phenoCol`,
`pred`, and `out`.

Useful execution flags include `--g-device cpu|gpu`, `--bsize`, `--g-variant-limit`, `--g-staging-depth`, `--g-resume`, `--g-trusted-no-missing-diploid`, `--g-writer-threads`, `--g-writer-queue-depth`, JAX cache settings, BGEN decode tiling, and binary/Firth solver settings.

## Python API

```python
from pathlib import Path

import g

artifacts = g.regenie.from_options(
    {
        "step": 2,
        "qt": True,
        "bgen": Path("data/1kg_chr22_full.bgen"),
        "sample": Path("data/1kg_chr22_full.sample"),
        "phenoFile": Path("data/pheno_cont.txt"),
        "phenoCol": "phenotype_continuous",
        "covarFile": Path("data/covariates.txt"),
        "covarColList": "age,sex",
        "pred": Path("data/baselines/regenie_step1_qt_pred.list"),
        "out": Path("data/example_regenie2"),
        "g-device": "cpu",
    }
)
```

The API returns `g.RunArtifacts` with the output run directory and finalized Parquet path when Parquet finalization is enabled.

## Output Layout

Given `--out data/example_regenie2`, `g` writes Arrow chunks under a `g` run directory and finalizes Parquet by default:

```text
data/example_regenie2.g/phenotype_continuous.regenie2_linear.run/
  chunks/
    chunk_000000000.arrow
    chunk_000000001.arrow
  effective_config.toml
  run_manifest.json
  final.parquet
```

Binary runs use the `.regenie2_binary.run` suffix. Arrow chunks are written incrementally and can be resumed with `--g-resume`. `--g-output-format` controls `parquet` or `arrow`.

## Development Commands

Common local commands:

```bash
just format
just lint
just typecheck
just check
just test
```

No-Nix or reduced-toolchain lanes:

```bash
just check-local
just test-local
just test-local-focused
```

Native extension and Rust benchmarks:

```bash
just install-perf-extension
just benchmark-rust
just benchmark-bgen-reader
```

REGENIE comparison and profiling:

```bash
just benchmark-regenie-comparison
just benchmark-regenie-comparison-gpu
just profile-regenie-comparison
just profile-regenie-comparison-gpu
just profile-regenie2-deep-smoke
```

Binary GPU smoke and full runs:

```bash
just setup-regenie2-binary-gpu-inputs
just verify-regenie2-binary-gpu-inputs
just regenie2-binary-gpu-smoke
just regenie2-binary-gpu
```

## Ubuntu + SLURM

Use the login node for dependency sync, formatting, linting, tests, data preparation, and baseline generation. Use SLURM recipes for GPU work:

```bash
just slurm-gpu-shell
just slurm-gpu-just doctor-jax
just slurm-regenie2-binary-gpu-smoke
just verify-regenie2-binary-gpu-smoke-output
just slurm-regenie2-binary-gpu
just verify-regenie2-binary-gpu-output
```

The default GPU node is `landau`. Override cluster settings with `GWAS_ENGINE_GPU_NODE`, `GWAS_ENGINE_SLURM_PARTITION`, `GWAS_ENGINE_SLURM_ACCOUNT`, `GWAS_ENGINE_SLURM_CPUS_PER_TASK`, `GWAS_ENGINE_SLURM_MEMORY`, `GWAS_ENGINE_SLURM_TIME`, and `GWAS_ENGINE_SLURM_GPUS_PER_TASK`.

Full server notes live in [`docs/UBUNTU_SLURM_DEVELOPMENT.md`](docs/UBUNTU_SLURM_DEVELOPMENT.md). Reduced-toolchain notes live in [`docs/NO_NIX_DEVELOPMENT.md`](docs/NO_NIX_DEVELOPMENT.md). Coding rules live in [`docs/STYLEGUIDE.md`](docs/STYLEGUIDE.md).

# GWAS Engine (`g`)

`g` is a GPU-accelerated GWAS engine for BGEN-backed **REGENIE step 2** association scans. The current package is a Python 3.14 API/CLI backed by JAX compute code and a Rust/PyO3 native extension for BGEN parsing and output persistence.

The active public surface is intentionally narrow:

- Python API: `g.regenie2(...)`, `g.regenie2_linear(...)`, and `g.api.regenie2_warm_cache(...)`
- CLI: `g regenie2 ...`, `g regenie2-linear ...`, and `g regenie2-warm-cache ...`
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
- Python runtime dependencies: JAX, NumPy, Polars, Typer
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

Quantitative REGENIE step 2 shorthand:

```bash
uv run g \
  regenie2-linear \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --pheno data/pheno_cont.txt \
  --pheno-name phenotype_continuous \
  --covar data/covariates.txt \
  --covar-names age,sex \
  --pred data/baselines/regenie_step1_qt_pred.list \
  --out data/example_regenie2 \
  --finalize-parquet
```

General entrypoint for quantitative traits:

```bash
uv run g \
  regenie2 \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --pheno data/pheno_cont.txt \
  --pheno-name phenotype_continuous \
  --covar data/covariates.txt \
  --covar-names age,sex \
  --pred data/baselines/regenie_step1_qt_pred.list \
  --trait-type quantitative \
  --out data/example_regenie2 \
  --finalize-parquet
```

Binary traits with approximate Firth fallback:

```bash
uv run g \
  regenie2 \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --pheno data/pheno_bin.txt \
  --pheno-name phenotype_binary \
  --covar data/covariates.txt \
  --covar-names age,sex \
  --pred data/baselines/regenie_step1_pred.list \
  --trait-type binary \
  --firth \
  --approx \
  --pThresh 0.01 \
  --out data/example_regenie2_binary \
  --finalize-parquet
```

Warm JAX cache shapes without writing association output:

```bash
uv run g \
  regenie2-warm-cache \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --pheno data/pheno_cont.txt \
  --pheno-name phenotype_continuous \
  --covar data/covariates.txt \
  --covar-names age,sex \
  --pred data/baselines/regenie_step1_qt_pred.list \
  --trait-type quantitative \
  --device cpu
```

Useful execution flags include `--device cpu|gpu`, `--chunk-size`, `--variant-limit`, `--prefetch-chunks`, `--resume`, `--trusted-no-missing-diploid`, `--warm-cache-first`, `--output-writer-thread-count`, and `--output-writer-queue-depth`.

## Python API

```python
from pathlib import Path

import g

artifacts = g.regenie2(
    bgen=Path("data/1kg_chr22_full.bgen"),
    sample=Path("data/1kg_chr22_full.sample"),
    pheno=Path("data/pheno_cont.txt"),
    pheno_name="phenotype_continuous",
    covar=Path("data/covariates.txt"),
    covar_names=["age", "sex"],
    pred=Path("data/baselines/regenie_step1_qt_pred.list"),
    trait_type=g.RegenieTraitType.QUANTITATIVE,
    out=Path("data/example_regenie2"),
    compute=g.ComputeConfig(device=g.Device.CPU),
)
```

The API returns `g.RunArtifacts` with the output run directory and, when finalization is enabled, the final Parquet path.

## Output Layout

Given `--out data/example_regenie2`, `g` writes a mode-specific run directory unless the output path already ends with `.run`:

```text
data/example_regenie2.regenie2_linear.run/
  chunks/
    chunk_000000000.arrow
    chunk_000000001.arrow
  final.parquet
```

Binary runs use the `.regenie2_binary.run` suffix. Arrow chunks are written incrementally and can be resumed with `--resume`. CLI Parquet finalization is opt-in with `--finalize-parquet`; the Python `ComputeConfig` default enables finalization.

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

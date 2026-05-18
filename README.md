# GWAS Engine (`g`)

`g` is a GPU-accelerated GWAS engine focused on **REGENIE step 2**.

The package API and CLI are BGEN-backed REGENIE step 2 workflows only.

## Active Public Surface

The currently supported public interface is:

- Python API:
  - `g.regenie2(...)` (general step 2 entrypoint)
  - `g.regenie2_linear(...)` (quantitative shorthand)
- CLI commands:
  - `g regenie2 ...` (general step 2 entrypoint)
  - `g regenie2-linear ...` (quantitative shorthand)
- Output artifact: resumable Arrow chunk run directory with `final.parquet` when Parquet finalization is enabled

### Trait types and binary behavior

`g.regenie2(...)` and `g regenie2 ...` support:

- `--trait-type quantitative` (default)
- `--trait-type binary`

For binary traits, fallback correction is controlled with REGENIE-style flags:

- default: score-test-only output
- `--firth --approx`: approximate Firth fallback
- `--pThresh FLOAT`: fallback p-value threshold, default `0.05`
- `--firth-se`: use LRT-derived standard errors for successful Firth rows

Current binary-mode status is **partial / evolving**. The binary pipeline is exposed as public, but behavior/performance parity with quantitative workflows is still under active development.

## Quick Start

Bootstrap a CPU-oriented development environment:

```bash
UV_CACHE_DIR=/tmp/g-uv-cache uv run --no-project python scripts/bootstrap_server_tools.py
source scripts/server_env.sh
just bootstrap
```

Bootstrap a GPU-capable environment for CUDA JAX work:

```bash
just bootstrap-gpu
```

Check the local toolchain:

```bash
just doctor-server
just doctor
just doctor-baselines
```

Prepare local data:

```bash
just setup-data
```

Run REGENIE step 2 quantitative (linear shorthand):

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
  --out data/example_regenie2
```

Run REGENIE step 2 via the general entrypoint:

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
  --out data/example_regenie2
```

Binary example (public, partial/evolving):

```bash
uv run g \
  regenie2 \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --pheno data/pheno_binary.txt \
  --pheno-name phenotype_binary \
  --covar data/covariates.txt \
  --covar-names age,sex \
  --pred data/baselines/regenie_step1_bt_pred.list \
  --trait-type binary \
  --firth \
  --approx \
  --pThresh 0.01 \
  --out data/example_regenie2_binary
```

Output paths:

- `<out>/chunks/*.arrow` for committed intermediate chunks
- `<out>/final.parquet` when finalization is enabled

## Common Commands

```bash
just bootstrap
just bootstrap-gpu
just setup-server-tools
just doctor-server
just doctor
just doctor-baselines
just slurm-gpu-shell
just slurm-gpu-just doctor-jax
just check
just test
just regenie2-linear
just profile-regenie2-linear-detailed
just benchmark-regenie-comparison
just benchmark-regenie-comparison-gpu
just profile-regenie-comparison
just profile-regenie-comparison-gpu
just setup-regenie2-binary-gpu-inputs
just verify-regenie2-binary-gpu-inputs
just slurm-regenie2-binary-gpu-smoke
just verify-regenie2-binary-gpu-smoke-output
just slurm-regenie2-binary-gpu
just verify-regenie2-binary-gpu-output
```

## REGENIE Comparison Suite

The comparison suite benchmarks and profiles:

- Original `regenie`:
  - step 1 binary (BED input)
  - step 2 binary (BGEN input)
  - step 1 quantitative (BED input)
  - step 2 quantitative (BGEN input)
- `g`:
  - REGENIE step 2 quantitative on CPU
  - REGENIE step 2 quantitative on GPU (optional)

Explicitly unimplemented in `g` and reported as `not_implemented`:

- binary step 1
- quantitative step 1

## Repository Layout

- `src/g/` - active Python package code
- `src/g/compute/regenie2_linear.py` - active REGENIE step 2 kernel
- `tests/` - active tests for REGENIE and shared I/O infrastructure
- `archive/direct_association/` - archived direct linear/logistic reference code and tests (not CI)
- `scripts/` - active utilities for data setup, baseline benchmarking, and REGENIE profiling

## Ubuntu + SLURM

On the Ubuntu server, keep the login node for `just check`, `just test`, dependency sync, data preparation, and baseline generation. Push GPU work through SLURM with `just slurm-gpu-*`.

The default GPU node name is `landau`. Full server notes live in [docs/UBUNTU_SLURM_DEVELOPMENT.md](/mnt/beegfs/kirill/Projects/g/docs/UBUNTU_SLURM_DEVELOPMENT.md).

## Status

Active development targets biobank-scale REGENIE workflows.

The active REGENIE step 2 default chunk size is `8192`.

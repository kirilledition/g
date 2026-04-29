# GWAS Engine (`g`)

`g` is a GPU-accelerated GWAS engine focused on **REGENIE step 2** for quantitative traits.

Direct PLINK-style linear/logistic regression workflows are not active in the package API or CLI. Legacy implementations are preserved under `archive/direct_association/` as reference material only.

## Active Public Surface

- Python API: `g.regenie2_linear(...)`
- CLI command: `g regenie2-linear ...`
- Output modes: TSV or chunked Arrow + optional finalized Parquet

## Quick Start

Enter the development shell:

```bash
nix develop
```

Install dependencies:

```bash
uv sync -U --group dev --group gpu
```

Prepare local data:

```bash
just setup-data
```

Run REGENIE step 2:

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

Output path:

- `<prefix>.regenie2_linear.tsv` (TSV mode)

## Common Commands

```bash
just check
just test
just regenie2-linear
just profile-regenie2-linear-detailed
just benchmark-regenie-comparison
just benchmark-regenie-comparison-gpu
just profile-regenie-comparison
just profile-regenie-comparison-gpu
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
- binary step 2
- quantitative step 1

Reports are written to:

- Benchmarks: `data/benchmarks/regenie_comparison/`
- Profiles: `data/profiles/regenie_comparison/`

## Repository Layout

- `src/g/` - active Python package code
- `src/g/compute/regenie2_linear.py` - active REGENIE step 2 kernel
- `tests/` - active tests for REGENIE and shared I/O infrastructure
- `archive/direct_association/` - archived direct linear/logistic reference code and tests (not CI)
- `scripts/` - active utilities for data setup, baseline benchmarking, and REGENIE profiling

## Status

Active development targets biobank-scale REGENIE workflows. PLINK-style direct regression will remain archived until explicitly resumed.

The active REGENIE step 2 default chunk size is `8192`.

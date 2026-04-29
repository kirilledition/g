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
uv sync --group dev
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
```

## Repository Layout

- `src/g/` - active Python package code
- `src/g/compute/regenie2_linear.py` - active REGENIE step 2 kernel
- `tests/` - active tests for REGENIE and shared I/O infrastructure
- `archive/direct_association/` - archived direct linear/logistic reference code and tests (not CI)
- `scripts/` - active utilities for data setup, baseline benchmarking, and REGENIE profiling

## Status

Active development targets biobank-scale REGENIE workflows. PLINK-style direct regression will remain archived until explicitly resumed.

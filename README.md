# GWAS Engine (`g`)

`g` is a GWAS engine under active development.

Today the maintained implementation is a Python/JAX pipeline that reads PLINK BED datasets through `bed-reader`, preprocesses chunks in Python/JAX, and runs covariate-adjusted linear or logistic association testing.

## Current State

- Phase 1 correctness work is the stable baseline.
- The maintained ingestion path is `bed-reader` plus Python/JAX preprocessing.
- The maintained numeric path is float32-only.
- GPU work is active through JAX, but CPU remains the default execution path.
- The Rust extension is currently placeholder-only scaffolding for future work, not an active compute path.

What the repository currently supports:

- covariate-adjusted linear regression
- covariate-adjusted logistic regression
- PLINK-style hybrid logistic `firth-fallback` behavior
- PLINK parity/evaluation harnesses for the local chr22 baseline dataset
- JAX benchmarking and profiling scripts for Phase 2 performance work

What it is not yet:

- a production-ready whole-genome pipeline
- a Rust-accelerated implementation end to end
- a custom-kernel GPU implementation

## Architecture

The active path looks like this:

1. `bed-reader` streams genotype chunks from PLINK `.bed/.bim/.fam` inputs.
2. `src/g/io/tabular.py` aligns phenotype and covariate tables to sample order.
3. `src/g/io/plink.py` preprocesses genotype chunks and converts them to JAX arrays.
4. `src/g/compute/linear.py` and `src/g/compute/logistic.py` run the association kernels.
5. `src/g/engine.py` orchestrates chunk iteration, result accumulation, and final tabular output.
6. `src/g/cli.py` exposes the `g` command-line interface.

The current result tables include variant metadata plus association statistics such as beta, standard error, test statistic, p-value, observation count, and validity flags. Logistic output also includes method/error annotations aligned to the current PLINK comparison workflow.

## Development Environment

The repository ships with a Nix dev shell in `flake.nix`.

Included tools:

- `uv`
- `just`
- Python 3.13
- Rust toolchain
- `plink2`
- `regenie`

Enter the shell with:

```bash
nix develop
```

## Install Dependencies

Inside the dev shell, sync the default environment with:

```bash
uv sync --group dev
```

For optional GPU JAX work, install the GPU group too:

```bash
uv sync --group gpu --group dev
```

The default dependency set keeps the project on CPU-backed JAX. Use the GPU group only when you are explicitly testing GPU execution.

## Quick Start

Prepare the local chr22 dataset and simulated phenotypes:

```bash
just setup-data
```

Run baseline PLINK/Regenie commands:

```bash
just benchmark-baselines
```

Run the engine directly:

```bash
uv run g \
  --bfile data/1kg_chr22_full \
  --pheno data/pheno_cont.txt \
  --pheno-name phenotype_continuous \
  --covar data/covariates.txt \
  --covar-names age,sex \
  --glm linear \
  --out data/example_linear
```

Run logistic mode:

```bash
uv run g \
  --bfile data/1kg_chr22_full \
  --pheno data/pheno_bin.txt \
  --pheno-name phenotype_binary \
  --covar data/covariates.txt \
  --covar-names age,sex \
  --glm logistic \
  --out data/example_logistic
```

Output files are written as:

- `<prefix>.linear.tsv`
- `<prefix>.logistic.tsv`

You can also force device selection through the CLI:

```bash
uv run g --device cpu ...
uv run g --device gpu ...
```

## Common Commands

Core development commands:

```bash
just check
just test
just phase1-evaluate
```

Data and baseline commands:

```bash
just setup-data
just benchmark-baselines
just phase1-linear
just phase1-logistic
```

JAX runtime and benchmark commands:

```bash
just probe-jax
just benchmark-jax
just benchmark-jax-chunks
just benchmark-logistic-loop
just benchmark-logistic-fallback
just benchmark-plink-reader
```

Profiling commands:

```bash
just profile-logistic-chr22
just profile-linear-chr22
just profile-logistic-detailed
just profile-linear-detailed
```

## Repository Layout

- `src/g/` - Python package code
- `src/g/io/` - genotype and tabular input handling
- `src/g/compute/` - linear and logistic JAX kernels
- `src/g/engine.py` - high-level orchestration
- `src/lib.rs` - placeholder PyO3 module
- `tests/` - correctness, parity, and behavior tests
- `scripts/` - data prep, evaluation, benchmark, and profiling utilities
- `docs/` - plans, status notes, and project documentation

## Current Priorities

Active work is focused on Phase 2 performance, especially:

- improving the logistic path on GPU using pure JAX first
- keeping the Python `bed-reader` path as the incumbent ingestion baseline
- reducing host/device synchronization and orchestration overhead
- using benchmarks and parity checks as gates for performance changes

## Documentation

- `docs/ROADMAP.md` - long-term architecture and milestones
- `docs/VISION.md` - project intent and product direction
- `docs/PLAN_PHASE_0.md` - baseline/data-preparation plan
- `docs/PLAN_PHASE_1.md` - Phase 1 correctness and parity plan
- `docs/PLAN_PHASE_2.md` - master Phase 2 implementation plan
- `docs/PLAN_PHASE_2_GPU.md` - detailed GPU acceleration plan
- `docs/PLAN_PHASE_2_RUST.md` - detailed Rust integration plan
- `docs/PHASE1_STATUS.md` - tracked parity and benchmark snapshot
- `docs/STYLEGUIDE.md` - coding rules
- `AGENTS.md` - repository-specific agent instructions

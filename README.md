# GWAS Engine (`g`)

`g` is a GWAS engine under active development.

Today, the repository contains a Phase 1 Python/JAX implementation that matches PLINK 2 closely on the project chr22 baseline dataset for:

- covariate-adjusted linear regression
- covariate-adjusted logistic regression
- PLINK-style hybrid logistic `firth-fallback` behavior, including method/error-code parity

The longer-term goal is a hybrid Python/Rust/GPU system that outperforms CPU-bound GWAS tools such as PLINK 2 and Regenie on large datasets.

## Current Status

- Phase 1 correctness milestone is complete
- Phase 1 remains CPU-oriented and correctness-first
- Phase 2 has not started yet; major performance work is still ahead

Tracked Phase 1 snapshot:

- `docs/PHASE1_STATUS.md`

Latest tracked parity snapshot from `docs/PHASE1_STATUS.md`:

- linear max abs beta diff: `4.9968e-06`
- linear max abs p diff: `5.17602e-07`
- logistic hybrid max abs beta diff: `1.99773e-05`
- logistic hybrid max abs p diff: `4.58049e-06`
- logistic method mismatches: `0`
- logistic error-code mismatches: `0`

Latest tracked runtime snapshot on the local Phase 0 dataset:

- linear runtime slowdown vs PLINK: `22.459x`
- logistic runtime slowdown vs PLINK: `16.292x`

That performance is expected for Phase 1. This stage was about locking down data alignment, masking, solver behavior, and PLINK-compatible outputs before moving into kernel and I/O optimization.

## What Is Implemented

- BED-based genotype streaming from PLINK `.bed/.bim/.fam`
- phenotype/covariate loading and alignment
- typed result models and chunk interfaces
- JAX linear regression kernel
- JAX logistic regression kernel
- JAX Firth/hybrid logistic fallback path for binary traits
- PLINK parity/evaluation harnesses
- thin CLI/library entrypoints for running association scans

## Development Environment

The repository ships with a Nix dev shell in `flake.nix`.

Included tools:

- `uv`
- `just`
- Python 3.13
- Rust toolchain
- `plink2`
- `regenie`

The dev shell now uses the official AMD AVX2 PLINK 2 binary on this machine for cleaner benchmark comparisons.

Enter the shell with:

```bash
nix develop
```

## Common Commands

Run tests:

```bash
just test
```

Run formatting and linting manually:

```bash
uv run ruff format .
uv run ruff check . --fix
uv run ty check
```

Generate baseline benchmark outputs:

```bash
uv run python scripts/benchmark.py
```

Run the full Phase 1 evaluation:

```bash
uv run python scripts/evaluate_phase1.py
```

## Documentation

- `docs/ROADMAP.md` - long-term architecture and milestones
- `docs/VISION.md` - project intent and product direction
- `docs/PLAN_PHASE_0.md` - baseline/data-preparation plan
- `docs/PLAN_PHASE_1.md` - Phase 1 correctness and parity plan
- `docs/PHASE1_STATUS.md` - tracked end-of-Phase-1 benchmark/parity snapshot
- `docs/STYLEGUIDE.md` - coding rules
- `AGENTS.md` - repository-specific agent instructions

## Are We Ready For Phase 2?

Mostly yes.

Based on `docs/PLAN_PHASE_1.md`, the project is ready to move into Phase 2 because:

- the engine runs on the Phase 0 chr22 dataset
- continuous-trait parity is within tolerance
- binary-trait hybrid logistic parity is within tolerance
- `ruff`, `ty`, and `pytest` pass
- the current codebase has clear I/O / compute / orchestration boundaries suitable for replacement work

Before serious Phase 2 execution, the practical prerequisites are:

- install a CUDA-enabled `jaxlib` or decide the first GPU execution path
- profile the logistic path to identify the dominant bottlenecks
- decide whether Phase 2 starts with Rust I/O, GPU JAX execution, or custom kernels first

So the answer is: yes, the repository looks ready to move on from Phase 1 correctness work into Phase 2 performance work.

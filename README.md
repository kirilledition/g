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

### Optional GPU JAX Path

The default project dependency set keeps the repository on CPU-backed JAX.

For GPU bring-up, install the optional GPU dependency group:

```bash
uv sync --group gpu --group dev
```

This currently installs `jax[cuda12]`, which keeps CPU parity mode as the default while giving the project a more conservative first GPU bring-up target than jumping straight to the CUDA 13 wheel path.

After installation, verify what JAX can see with:

```bash
uv run python scripts/benchmark_jax_execution.py
```

If GPU bring-up succeeds, the runtime section of that report should show a GPU backend and one or more non-CPU JAX devices.

In the Nix dev shell, the project now exports `/run/opengl-driver/lib` on `LD_LIBRARY_PATH` so CUDA-enabled JAX can see the NVIDIA driver libraries. If you test outside the dev shell, you may need to provide the driver library path yourself.

Current local GPU benchmarking snapshot on the RTX 4080 SUPER:

- from `scripts/benchmark_jax_execution.py` with `chunk_size=256`
  - backend: `gpu`
  - warmed `device_put`: `~0.00039s`
  - warmed linear compute: `~0.01171s`
  - warmed logistic standard compute: `~0.01785s`
  - warmed logistic fallback compute: `~0.01885s`
- comparable CPU-backed snapshot for the same script and chunk size:
  - warmed `device_put`: `~0.00055s`
  - warmed linear compute: `~0.00089s`
  - warmed logistic compute: `~0.04480s`
- current interpretation:
  - GPU already helps the logistic path by roughly `2.3x-2.5x`
  - small linear chunks are still worse on GPU, so chunk sizing and batching remain important

Current local GPU chunk-size sweep snapshot from `scripts/benchmark_jax_chunk_sweep.py`:

- warmed linear compute means:
  - `256`: `~0.01253s`
  - `512`: `~0.01150s`
  - `1024`: `~0.01247s`
  - `2048`: `~0.01243s`
- warmed logistic standard compute means:
  - `256`: `~0.01947s`
  - `512`: `~0.02730s`
  - `1024`: `~0.04862s`
  - `2048`: `~0.07549s`
- warmed logistic fallback compute means:
  - `256`: `~0.01806s`
  - `512`: `~0.02729s`
  - `1024`: `~0.04848s`
  - `2048`: `~0.07750s`

These snapshots are intentionally recorded here so later Phase 2 work can compare against a stable reference without rerunning the whole bring-up sequence first.

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
- `docs/PLAN_PHASE_2.md` - master Phase 2 implementation plan
- `docs/PLAN_PHASE_2_GPU.md` - detailed GPU acceleration plan
- `docs/PLAN_PHASE_2_RUST.md` - detailed Rust integration plan
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

- install and verify the optional CUDA-enabled JAX path
- profile the logistic path to identify the dominant bottlenecks
- decide whether Phase 2 starts with Rust I/O, GPU JAX execution, or custom kernels first

So the answer is: yes, the repository looks ready to move on from Phase 1 correctness work into Phase 2 performance work.

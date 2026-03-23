# GWAS Engine (`g`)

`g` is a GWAS engine under active development.

Today, the repository contains a Phase 1 Python/JAX implementation that matches PLINK 2 closely on the project chr22 baseline dataset for:

- covariate-adjusted linear regression
- covariate-adjusted logistic regression
- PLINK-style hybrid logistic `firth-fallback` behavior, including method/error-code parity

The longer-term goal is a hybrid Python/Rust/GPU system that outperforms CPU-bound GWAS tools such as PLINK 2 and Regenie on large datasets.

## Current Benchmarks

All benchmarks run on AMD Ryzen 7 9800X3D (8 cores/16 threads) with 60GB RAM, comparing against PLINK 2.0.0-a.6.33 AVX2 AMD.

### Full Chromosome 22 (418,943 variants, 2,504 samples) - After Hybrid Optimization

**Optimization Strategy:** 
1. **Linear regression**: Eliminated per-chunk DataFrame churn (single host sync at end) → **3.4x speedup**
2. **Logistic regression**: Reverted to batched Firth to avoid 34x wasted computation → **1.7x (GPU) / 2.0x (CPU) speedup**
3. **GPU backend**: Fixed ROCm → auto-detect for NVIDIA compatibility

| Configuration | Linear (s) | Logistic (s) | Slowdown vs PLINK | Notes |
|---------------|-----------|-------------|-------------------|-------|
| **Baseline (PLINK 2)** | 0.23 | 3.45 | 1.0x | ✓ Reference |
| **g CPU chunk=512** | **3.84** | ~301 (5m) | 16.7x / 87x | Best for linear |
| **g GPU chunk=512** | 13.62 | **~114** (1m54s) | 59.2x / **33x** | Best for logistic |

\* Logistic uses batched Firth fallback (only ~2.9% of variants need Firth)

**Key Findings:**

**✅ Major Wins:**
1. **Linear CPU: 3.4x faster** (13.18s → 3.84s)
   - Single host sync at end eliminates per-chunk overhead
   - Architecture successful for compute-light workloads

2. **Logistic GPU: 2.6x faster than CPU** (301s → 114s)
   - Batched Firth: Only 2.9% of variants actually need Firth
   - Avoids 34x wasted computation from running on all variants
   - Now **33x slower** than PLINK (improved from 85x)

3. **All tests pass** - mathematical parity maintained

**Architecture Decision:**
- **Linear regression**: Keep accumulator-based approach (single host sync at end)
  - Eliminates per-chunk DataFrame creation and host sync
  - 3.4x speedup demonstrates success
  
- **Logistic regression**: Use batched Firth with mid-compute host sync
  - Trade-off: Accept small host syncs to avoid 34x wasted computation
  - Only ~2.9% of variants need Firth, so batching is essential
  - ~24 batches for full chromosome vs 819 variants

### Phase 2 GPU/JAX Microbenchmarks

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
  - GPU already helps the logistic path by roughly `2.3x-2.5x` on pure compute
  - small linear chunks are still worse on GPU, so chunk sizing and batching remain important
  - end-to-end throughput is dominated by I/O, preprocessing, and output formatting, not raw compute

### How to Reproduce

Run the full benchmark suite:

```bash
nix develop
just benchmark-baselines  # PLINK and Regenie baselines
just phase1-evaluate      # g vs PLINK comparison with parity checks
```

Run individual profiling commands:

```bash
just probe-jax            # Check JAX device configuration
just benchmark-jax        # JAX execution boundaries
just profile-logistic-chr22  # Full chromosome JAX trace + memory
just profile-linear-chr22    # Full chromosome JAX trace + memory
```

## Current Status

- Phase 1 correctness milestone is complete
- Phase 1 remains the correctness baseline
- Phase 2 performance work is active in both the Rust and GPU/JAX arms
- Current Phase 2 focus is optimized JAX on GPU before any custom-kernel work

Tracked Phase 1 snapshot:

- `docs/PHASE1_STATUS.md`

Latest tracked parity snapshot from `docs/PHASE1_STATUS.md`:

- linear max abs beta diff: `4.9968e-06`
- linear max abs p diff: `5.17602e-07`
- logistic hybrid max abs beta diff: `1.99773e-05`
- logistic hybrid max abs p diff: `4.58049e-06`
- logistic method mismatches: `0`
- logistic error-code mismatches: `0`

Tracked end-of-Phase-1 runtime snapshot on the local Phase 0 dataset:

- linear runtime slowdown vs PLINK: `22.459x`
- logistic runtime slowdown vs PLINK: `16.292x`

That performance is expected for Phase 1. This stage was about locking down data alignment, masking, solver behavior, and PLINK-compatible outputs before moving into kernel, orchestration, and I/O optimization.

Current Phase 2 reality:

- the handwritten Rust BED reader prototype does not beat `bed-reader`
- the Rust preprocessing path is correct but still slower than the incumbent Python/JAX path on the current local benchmark
- GPU JAX bring-up now works in the dev shell on the local RTX 4080 SUPER
- the logistic path already benefits materially from GPU execution
- the linear path is still dominated by chunk iteration, I/O, preprocessing, and output handling rather than the linear math kernel

## What Is Implemented

- BED-based genotype streaming from PLINK `.bed/.bim/.fam`
- phenotype/covariate loading and alignment
- typed result models and chunk interfaces
- JAX linear regression kernel
- JAX logistic regression kernel
- JAX Firth/hybrid logistic fallback path for binary traits
- GPU JAX benchmark and profiling harnesses
- PyO3/Rust prototype surfaces for native BED reading and preprocessing
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

If the GPU path is unstable, use the dedicated runtime probe first:

```bash
just probe-jax
```

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

Current local full-loop logistic GPU snapshot from `scripts/benchmark_logistic_loop_sweep.py` with `variant_limit=4096`:

- warmed `compute_only` means:
  - `128`: `~0.492s`
  - `256`: `~0.375s`
  - `512`: `~0.283s`
  - `1024`: `~0.237s`
- warmed `compute_and_format` means:
  - `128`: `~0.500s`
  - `256`: `~0.379s`
  - `512`: `~0.286s`
  - `1024`: `~0.240s`
- current interpretation:
  - formatting is negligible relative to logistic compute
  - larger chunk sizes materially improve end-to-end logistic loop throughput on GPU
  - `1024` currently looks better than the old `512` default for GPU logistic work

## Common Commands

Run tests:

```bash
just test
```

Run the main development checks:

```bash
just check
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

Phase 2 benchmarking and profiling commands:

```bash
just probe-jax
just benchmark-jax
just benchmark-jax-chunks
just benchmark-logistic-loop
just benchmark-logistic-fallback
just benchmark-plink-reader
just profile-logistic-chr22
just profile-linear-chr22
```

What these do:

- `just probe-jax` checks whether the current JAX runtime sees CPU/GPU devices cleanly
- `just benchmark-jax` measures host read, transfer, compute, and formatting boundaries
- `just benchmark-jax-chunks` sweeps chunk sizes for isolated JAX compute kernels
- `just benchmark-logistic-loop` measures full logistic loop throughput across chunk sizes
- `just benchmark-logistic-fallback` isolates the hybrid fallback-heavy path
- `just benchmark-plink-reader` compares reader and preprocessing paths
- `just profile-logistic-chr22` captures a full-chromosome logistic JAX trace and memory profile
- `just profile-linear-chr22` captures a full-chromosome linear JAX trace and memory profile

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

## Current Phase 2 Direction

The repository is already in active Phase 2 work.

Current priorities are:

- optimize the logistic association path on GPU using pure JAX first
- keep using `bed-reader` as the baseline ingestion path unless a replacement proves it is faster
- use profiling and benchmark gates for every performance change
- defer custom kernels until the optimized-JAX ceiling is understood

The two key planning documents are:

- `docs/PLAN_PHASE_2_GPU.md`
- `docs/PLAN_PHASE_2_RUST.md`

If you are continuing performance work, start with the benchmark/profiling commands above rather than assuming the old defaults or bottlenecks are still current.

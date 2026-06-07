# Rust Optimization Progress

Last updated: 2026-06-07

## Goal

Implement the optimization opportunities from `docs/rust-optimization-opportunities.md`
and report how much faster the application becomes. The headline metric is
full-app REGENIE step 2 GPU wall time, with Rust decode/output microbenchmarks
used to explain where the speedup came from.

## Operating Rules

- Work happens in dedicated `opt/rust-opt-*` worktrees under
  `/mnt/beegfs/kirill/Projects/g-worktrees`.
- Benchmark outputs go under `data/profiles/` and must not be committed.
- Keep changes that are faster than current `main`.
- If a change is performance-neutral, keep it only when it is cleaner or makes
  future optimization materially easier.
- Python API changes are allowed when they reduce Python orchestration,
  allocations, or Rust-Python data transfer.

## Worktrees

| Branch | Worktree | Scope | Status |
| --- | --- | --- | --- |
| `opt/rust-opt-benchmarks-progress` | `rust-opt-benchmarks-progress` | Benchmark gaps and progress log | Integrated |
| `opt/rust-opt-trusted-packed8` | `rust-opt-trusted-packed8` | Trusted BGEN parser reuse, packed8 fused summary, SIMD | Integrating |
| `opt/rust-opt-output-transfer` | `rust-opt-output-transfer` | Native output streaming, cached arrays, reduced Python transfer | Pending integration |
| `opt/rust-opt-decode-profile` | `rust-opt-decode-profile` | Profiling fast path, rolling bit reader, row-major experiments | Pending integration |
| `opt/rust-opt-setup-reuse` | `rust-opt-setup-reuse` | Setup path parsing, sample alignment, LOCO prediction reuse | Pending integration |

## Baseline Commands

Run these from `main` before integration and from the integration worktree after
all accepted branches land.

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --bench bgen_read -- --save-baseline rust-opt-main --sample-size 10 --measurement-time 3 --warm-up-time 1
RUSTFLAGS="-C target-cpu=native" cargo bench --bench preprocess -- --save-baseline rust-opt-main --sample-size 10 --measurement-time 3 --warm-up-time 1
uv run --no-sync python scripts/benchmark_bgen_reader.py --chunk-sizes 1024,2048,4096,8192,16384 --variant-limit 16384 --repeat-count 7 --trusted-no-missing-diploid-modes false,true --path-modes variant_major_buffered,variant_major_packed8_buffered --sample-selection-modes full,contiguous_half,strided_half --json-summary-path data/profiles/rust_opt_bgen_reader_baseline.json
uv run --no-sync python scripts/benchmark_regenie2_binary_hot.py --device gpu --variant-limit 16384 --json-summary-path data/profiles/rust_opt_binary_hot_baseline.json
uv run --no-sync python scripts/benchmark_output_stages.py --device gpu --trials 3 --variant-limit 16384 --json-summary-path data/profiles/rust_opt_output_stages_baseline.json --markdown-summary-path data/profiles/rust_opt_output_stages_baseline.md
```

On the shared server, run heavy CPU work through a CPU worker node and GPU app
benchmarks through `just slurm-gpu-run`.

## Learnings

- `benches/bgen_read.rs` previously covered trusted dosage identity and
  contiguous selection, but not packed8 probability-pair delivery or
  non-contiguous selection.
- `scripts/benchmark_bgen_reader.py` previously timed only variant-major dosage
  delivery. It needs packed8 and sample-selection sweeps to validate the most
  likely decode optimizations.
- Git worktrees do not contain the git-ignored `data/` directory, so Rust and
  Python benchmark harnesses now honor `GWAS_ENGINE_DATA_DIR` for shared input
  data.
- The trusted f32 dosage path already used raw integer accumulation, so packed8
  stats can share the same summary representation and match the existing f32
  tolerances.
- Identity and contiguous packed8 selections previously copied bytes and then
  scanned the same byte slice again for stats; this is the clearest packed8
  optimization point.
- Non-contiguous packed8 selection cannot use the same vectorized copy path
  without gather-style work, but it can still avoid lookup-table f32 stats.

## Completed So Far

- Added Criterion benchmark groups for trusted packed8 delivery with full,
  contiguous-half, and strided-half sample selections.
- Added strided-half variant-major dosage groups to make non-contiguous sample
  selection regressions visible.
- Updated the Rust BGEN Criterion bench to read
  `GWAS_ENGINE_DATA_DIR/1kg_chr22_full.bgen` when the environment variable is
  set, instead of silently skipping in implementation worktrees.
- Extended `scripts/benchmark_bgen_reader.py` with packed8 path mode,
  sample-selection sweeps, median timing, JSON/Markdown report paths, and
  clearer subprocess error reporting.
- Added a shared trusted unphased 8-bit no-missing diploid probability-block
  parser and reused it from validation, f32 dosage decode, and packed8 decode.
- Added scalar and AVX2 packed8 copy-and-summary helpers that copy probability
  pairs and accumulate raw integer dosage stats in one pass.
- Switched trusted packed8 identity and contiguous selections from
  copy-then-rescan to fused copy+summary.
- Switched trusted packed8 non-contiguous selections from lookup-table f32
  summary to raw integer summary while gathering selected pairs.
- Verified the benchmark/progress slice:
  - `cargo fmt --all --check`
  - `uv run ruff check scripts/benchmark_bgen_reader.py`
  - `uv run ruff format --check scripts/benchmark_bgen_reader.py`
  - smoke: `GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data uv run --no-sync python scripts/benchmark_bgen_reader.py --chunk-sizes 16 --variant-limit 16 --repeat-count 1 --path-modes variant_major_buffered --sample-selection-modes full,strided_half --json-summary-path data/profiles/rust_opt_bgen_reader_smoke.json --markdown-summary-path data/profiles/rust_opt_bgen_reader_smoke.md`
- Verified the trusted packed8 slice:
  - `cargo test --lib genotype::bgen`
  - `cargo clippy --lib -- -D warnings`
  - `cargo test --test rust_native_coverage trusted`

## Baseline Measurements

Rust BGEN Criterion baseline was run on `cantor` from
`opt/rust-opt-benchmarks-progress` rebased on local `main` at `37898670`:

```bash
srun --nodelist=cantor --cpus-per-task=40 --mem=16G --time=00:45:00 bash -lc 'cd /mnt/beegfs/kirill/Projects/g-worktrees/rust-opt-benchmarks-progress && . scripts/server_env.sh && GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=40 cargo bench --bench bgen_read -- --save-baseline rust-opt-main --sample-size 10 --measurement-time 2 --warm-up-time 1'
```

Selected 16,384-variant mean times:

| Group | Mean ms |
| --- | ---: |
| trusted variant-major dosage full sample | 14.020 |
| trusted packed8 full sample | 17.711 |
| trusted packed8 contiguous half sample | 14.570 |
| trusted packed8 strided half sample | 19.032 |
| non-trusted strided half sample dosage | 18.046 |

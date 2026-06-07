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
| `opt/rust-opt-trusted-packed8` | `rust-opt-trusted-packed8` | Trusted BGEN parser reuse, packed8 fused summary, SIMD | Integrated |
| `opt/rust-opt-decode-profile` | `rust-opt-decode-profile` | Profiling fast path, rolling bit reader, row-major experiments | Integrated |
| `opt/rust-opt-output-transfer` | `rust-opt-output-transfer` | Native output streaming, cached arrays, reduced Python transfer | Integrated |
| `opt/rust-opt-setup-reuse` | `rust-opt-setup-reuse` | Setup path parsing, sample alignment, LOCO prediction reuse | Integrated |

## Baseline Commands

Run these from `main` before integration and from the integration worktree after
all accepted branches land.

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --bench bgen_read -- --save-baseline rust-opt-main --sample-size 10 --measurement-time 3 --warm-up-time 1
RUSTFLAGS="-C target-cpu=native" cargo bench --bench preprocess -- --save-baseline rust-opt-main --sample-size 10 --measurement-time 3 --warm-up-time 1
uv run --no-sync python -m tooling.cli.benchmark_bgen_reader sweep.chunk_sizes=[1024,2048,4096,8192,16384] workload.variant_limit=16384 workload.repeat_count=7 sweep.trusted_no_missing_diploid_modes=[false,true] sweep.path_modes=[variant_major_buffered,variant_major_packed8_buffered] sweep.sample_selection_modes=[full,contiguous_half,strided_half] telemetry.json_summary_path=data/profiles/rust_opt_bgen_reader_baseline.json
uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot machine=landau_gpu tool.variant_limit=16384 telemetry.json_summary_path=data/profiles/rust_opt_binary_hot_baseline.json
uv run --no-sync python -m tooling.cli.benchmark_output_stages machine=landau_gpu tool.trials=3 tool.variant_limit=16384 telemetry.json_summary_path=data/profiles/rust_opt_output_stages_baseline.json telemetry.markdown_summary_path=data/profiles/rust_opt_output_stages_baseline.md
```

On the shared server, run heavy CPU work through a CPU worker node and GPU app
benchmarks through `just slurm-gpu-run`.

## Learnings

- `benches/bgen_read.rs` previously covered trusted dosage identity and
  contiguous selection, but not packed8 probability-pair delivery or
  non-contiguous selection.
- `tooling/cli/benchmark_bgen_reader.py` previously timed only variant-major dosage
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
- The existing profiling flag already avoids `Instant::now`, but decode still
  did local counter increments and snapshot merges that were discarded later
  when profiling was disabled.
- Row-major direct write is simple mechanically because the reader already
  passes the final pointer and selected variant window into the decode layer,
  but it carries risk from strided writes and partial caller-buffer writes on
  error. It stays behind an opt-in benchmark flag.
- Output chunk and final REGENIE step 2 schemas already share the same public
  14 columns, so finalization can skip name-based projection when the batch
  fields already match.
- The existing Python result-array path already retains NumPy result arrays as
  Arrow buffers without copying. The higher-value output improvement was
  sharing immutable native chunk arrays and streaming batches instead of
  accumulating them.
- Grouped LOCO loading was doing repeated setup work when prediction-list
  entries referenced the same LOCO file. Keeping parsed predictions in a
  grouped-load cache avoids those duplicate parses without changing the public
  API.
- Sample table parsing still needs owned sample keys for duplicate detection.
  This pass removed the simpler per-record selected-field vector allocation
  instead of changing key representation.

## Completed So Far

- Added Criterion benchmark groups for trusted packed8 delivery with full,
  contiguous-half, and strided-half sample selections.
- Added strided-half variant-major dosage groups to make non-contiguous sample
  selection regressions visible.
- Updated the Rust BGEN Criterion bench to read
  `GWAS_ENGINE_DATA_DIR/1kg_chr22_full.bgen` when the environment variable is
  set, instead of silently skipping in implementation worktrees.
- Extended `tooling/cli/benchmark_bgen_reader.py` with packed8 path mode,
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
- Added profiling-disabled fast paths around BGEN tile profile snapshots,
  per-variant tile decode counts, variant-major profile helpers, and probability
  block byte counters.
- Replaced `PackedProbabilityReader` byte-window rebuilds with a rolling
  little-endian bit buffer.
- Added an opt-in row-major direct-write prototype and Criterion groups for
  `bgen_row_major_tile_copy` and `bgen_row_major_direct_write`.
- Added a lazy writer-only Arrow array cache to `NativeChunkHandle` for
  immutable metadata/stat columns shared by cloned handles and multi-trait
  writer sessions.
- Refactored grouped Arrow/Parquet chunk writing to open the writer first and
  write each chunk `RecordBatch` immediately, removing the per-file
  `Vec<RecordBatch>` peak.
- Added a fast ordered-schema finalization path and restricted chunk discovery
  to writer-produced `chunk_*.arrow` and `part_*.parquet` files.
- Removed per-record selected-value `Vec<&str>` allocation from phenotype and
  covariate table parsing.
- Reused sorted sample array indices across grouped multi-trait alignment.
- Added LOCO prediction and per-target alignment caches for repeated
  prediction-list paths during multi-trait and grouped loading.
- Removed small whitespace-split temporary vectors from prediction-list parsing
  and LOCO header/data parsing.
- Verified the benchmark/progress slice:
  - `cargo fmt --all --check`
  - `uv run ruff check tooling/cli/benchmark_bgen_reader.py`
  - `uv run ruff format --check tooling/cli/benchmark_bgen_reader.py`
  - smoke: `GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data uv run --no-sync python -m tooling.cli.benchmark_bgen_reader sweep.chunk_sizes=[16] workload.variant_limit=16 workload.repeat_count=1 sweep.path_modes=[variant_major_buffered] sweep.sample_selection_modes=[full,strided_half] telemetry.json_summary_path=data/profiles/rust_opt_bgen_reader_smoke.json telemetry.markdown_summary_path=data/profiles/rust_opt_bgen_reader_smoke.md`
- Verified the trusted packed8 slice:
  - `cargo test --lib genotype::bgen`
  - `cargo clippy --lib -- -D warnings`
  - `cargo test --test rust_native_coverage trusted`
- Verified the decode/profile slice:
  - `cargo fmt --all --check`
  - `cargo check`
  - `cargo test genotype::bgen::decode`
  - `cargo clippy --lib -- -D warnings -W clippy::pedantic`
  - `cargo bench --bench bgen_read --no-run`
  - `git diff --name-only main...HEAD -- src/g/compute` produced no output
- Verified the output transfer slice:
  - `cargo test --lib output::`
  - `cargo test --lib output::writer`
  - `cargo test --lib output::finalization`
  - `cargo test --lib output::session`
  - `cargo test --lib python::output`
  - `cargo test --test rust_native_coverage output_session`
  - `cargo test --test rust_python_bindings registered_python_module_exercises_core_bindings`
  - `uv run pytest tests/test_io_output.py -q`
  - `cargo clippy --lib -- -D warnings`
- Verified the setup reuse slice:
  - `cargo fmt --all --check`
  - `cargo test sample::tests`
  - `cargo test regenie::tests`
  - `cargo test sample_alignment_public_apis_cover_table_and_input_errors`
  - `cargo test prediction_sources_cover_file_header_alignment_and_matrix_errors`
  - `cargo clippy --lib -- -D warnings`

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

## Final Measurements

Final comparisons use current-main code at `fb9139e6` as the baseline and
`integration/rust-optimization` at `66e080b2` as the optimized build. `main`
advanced to `21d7029e` after the benchmark with docs-only changes, so the
measured code baseline is still current for runtime behavior. Both worktrees
were installed with:

```bash
uv sync --python 3.14 --group dev --group gpu
RUSTFLAGS="-C target-cpu=native" uv run --no-sync maturin develop --profile perf --uv
```

The GPU app benchmark was run on `landau` with same-process warm/hot trials.
Current `main` includes the extended callback worker join handling, so the
final runs did not need the earlier benchmark-only timeout monkey patch.

Hot same-process GPU timings at 4,096 variants:

| Workload | Baseline s | Optimized s | Change | Speedup |
| --- | ---: | ---: | ---: | ---: |
| variant-major, no final Parquet | 0.393307 | 0.409953 | -4.23% | 0.959x |
| packed8, no final Parquet | 0.370362 | 0.395376 | -6.75% | 0.937x |
| variant-major, finalized Parquet | 0.403648 | 0.411899 | -2.04% | 0.980x |
| packed8, finalized Parquet | 0.385484 | 0.386658 | -0.30% | 0.997x |

Warm same-process GPU timings at 4,096 variants, which include first GPU
compilation and setup in that process:

| Workload | Baseline s | Optimized s | Change | Speedup |
| --- | ---: | ---: | ---: | ---: |
| variant-major, no final Parquet | 81.847150 | 79.782940 | +2.52% | 1.026x |
| packed8, no final Parquet | 0.678127 | 0.695899 | -2.62% | 0.975x |
| variant-major, finalized Parquet | 78.252997 | 79.519404 | -1.62% | 0.984x |
| packed8, finalized Parquet | 0.700245 | 0.705254 | -0.72% | 0.993x |

Because the 4,096-variant hot timings are sub-second and noisy, packed8 was
also measured at 16,384 variants:

| Workload | Baseline s | Optimized s | Change | Speedup |
| --- | ---: | ---: | ---: | ---: |
| packed8, no final Parquet hot | 0.483280 | 0.500616 | -3.59% | 0.965x |
| packed8, finalized Parquet hot | 0.498806 | 0.498847 | -0.01% | 1.000x |
| packed8, no final Parquet warm | 78.458851 | 79.531746 | -1.37% | 0.987x |
| packed8, finalized Parquet warm | 78.922604 | 79.508245 | -0.74% | 0.993x |

The measured application impact is therefore not a speedup on the current
baseline. The implemented Rust changes are functionally correct, but the
current GPU app workloads are dominated by JAX/worker/output orchestration, and
the packed8 Rust decode savings do not overcome the added overhead in the
end-to-end benchmark. The most favorable current measurement is effectively
flat: packed8 finalized hot at 16,384 variants changed from 0.498806 seconds to
0.498847 seconds.

The Python BGEN reader harness was also run on `cantor` with
`sweep.chunk_sizes=[16384]`, `workload.variant_limit=16384`,
`workload.repeat_count=5`, `sweep.trusted_no_missing_diploid_modes=[true]`,
`sweep.path_modes=[variant_major_buffered,variant_major_packed8_buffered]`, and
`sweep.sample_selection_modes=[full,contiguous_half,strided_half]`. It preserved
checksums across dosage and packed8 paths, but the measured medians were all
about 3.6 to 3.8 seconds and moved by only -0.5% to -0.7% in the optimized
build. That harness is dominated by Python/process/reader setup overhead at
this workload size, so it is not a useful micro-signal for the fused Rust
packed8 copy+summary path.

Benchmark artifacts are in `data/profiles/`:

- `rust_opt_binary_hot_baseline_4096_current.json`
- `rust_opt_binary_hot_optimized_4096_current.json`
- `rust_opt_binary_hot_baseline_4096_current_finalized.json`
- `rust_opt_binary_hot_optimized_4096_current_finalized.json`
- `rust_opt_binary_hot_baseline_16384_packed8_current.json`
- `rust_opt_binary_hot_optimized_16384_packed8_current.json`
- `rust_opt_binary_hot_baseline_16384_packed8_current_finalized.json`
- `rust_opt_binary_hot_optimized_16384_packed8_current_finalized.json`
- `rust_opt_bgen_reader_baseline.json`
- `rust_opt_bgen_reader_optimized.json`

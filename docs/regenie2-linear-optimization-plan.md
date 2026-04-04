# REGENIE2 Linear Optimization Plan

## Goal

Reduce end-to-end runtime for `g regenie2-linear`, with emphasis on:

- faster BGEN ingestion
- fewer host-device transfers
- fewer Python-managed per-chunk operations

## Current State

Implemented so far:

1. Cached normalized BGEN metadata arrays in `BgenReader`
2. Added a fast path for already chromosome-homogeneous chunks
3. Replaced the chunk-wide probability-tensor read path with a direct host-side dosage reader
4. Inlined dosage conversion inside the direct BGEN worker loop
5. Added batched payload materialization to reduce per-chunk `jax.device_get(...)` frequency

Validated with focused quality checks:

- `ruff check`
- `ty check`
- focused test suites for BGEN, engine iterators, and output persistence

## Profiling Basis

Primary local profiling artifacts:

- baseline path:
  - `data/profiles/regenie2_linear/full_chr22_gpu_chunk1024_summary.json`
  - `data/profiles/regenie2_linear/full_chr22_gpu_chunk1024_summary.txt`
- optimized direct-reader path:
  - `data/profiles/regenie2_linear/full_chr22_gpu_chunk1024_direct_summary.json`
  - `data/profiles/regenie2_linear/full_chr22_gpu_chunk1024_direct_summary.txt`

These are local-only and remain under `data/`.

## Measured Progress

### Warmed GPU Runtime

Best verified warmed GPU benchmark at `chunk_size=1024`:

- before direct BGEN dosage reader: `8.761s`
- after direct BGEN dosage reader: `7.754s`

Verified improvement:

- `1.007s` faster
- about `11.5%` improvement

### Profiled GPU Runtime

Full-chromosome profiled wall time:

- before read-path optimization: `23.416s`
- after direct BGEN dosage reader: `18.103s`

Verified improvement:

- `5.313s` faster
- about `22.7%` improvement

### Stage-Level Improvement

`bgen_read_host`:

- before: `12.325s`
- after: `9.216s`

Verified improvement:

- `3.109s` faster
- about `25.2%` improvement

`get_variant_table_arrays`:

- before: `0.925s`
- after: `0.662s`

Verified improvement:

- `0.263s` faster
- about `28.4%` improvement

## Current Bottlenecks

The pipeline is still dominated by host-side BGEN work.

Latest profiled stage ranking from `full_chr22_gpu_chunk1024_direct_summary.json`:

| Stage | Total | Share of wall |
|---|---:|---:|
| `bgen_read_host` | 9.216s | 50.91% |
| `write_chunk_to_disk` | 1.264s | 6.98% |
| `get_variant_table_arrays` | 0.662s | 3.66% |
| `device_put_genotypes` | 0.355s | 1.96% |
| `build_chunk_payload` | 0.118s | 0.65% |
| `preprocess_genotypes` | 0.080s | 0.44% |
| `compute_regenie2_linear_chunk` | 0.071s | 0.39% |

Important interpretation:

- `persist_chunked_results_total` is an enclosing lifetime metric and overlaps other stages
- JAX compute remains very small relative to ingestion
- the kernel is still not the place to optimize first

## What We Learned

### Confirmed

1. BGEN ingestion was the right first target
2. Metadata caching helped, but only as a secondary win
3. Direct host-side dosage construction was the first optimization that materially improved end-to-end speed
4. The code is still mostly read-bound, not compute-bound

### Still True After Optimization

Evidence from the optimized profile still points to the same underlying backend costs:

- `cbgen._ffi bgen_genotype_read32`
- `cbgen._ffi bgen_file_open_genotype`
- repeated per-variant probability reads inside the BGEN stack

That means our in-repo optimization headroom is now smaller than it was before the direct reader work.

## Host-Device Transfer Status

### Current transfers

Host to device:

- one dosage chunk per chunk via `jax.device_put(...)`

Device to host:

- result materialization for chunk persistence

### Reduction work done

- batched payload materialization has been implemented so multiple REGENIE2 chunks can share one `device_get(...)`

### Current assessment

- transfer overhead is still not the dominant bottleneck
- the next large win is more likely to come from reducing BGEN backend overhead than from kernel or transfer tuning

## Optimization Status By Item

### Done

#### 1. Cache BGEN metadata arrays

Status: complete

Files:

- `src/g/io/bgen.py`

Outcome:

- reduced repeated string splitting and metadata reconstruction
- measurable but secondary improvement

#### 2. Direct BGEN dosage reader

Status: complete

Files:

- `src/g/io/bgen.py`

Outcome:

- removes chunk-wide probability tensor construction
- constructs dosages directly in host buffers
- produced the first meaningful end-to-end speedup

#### 3. Small chunk-splitting cleanup

Status: complete

Files:

- `src/g/engine.py`

Outcome:

- avoids needless subchunk rebuilding for homogeneous chunks

#### 4. Batched payload materialization

Status: implemented

Files:

- `src/g/engine.py`
- `src/g/io/output.py`

Outcome:

- reduces per-chunk `device_get(...)` frequency in the persistence path
- correctness is covered by tests
- end-to-end gain has not yet been isolated cleanly with a dedicated benchmark/profile snapshot

### Not Done Yet

#### 5. Re-sweep chunk size after the new reader path

Status: incomplete

Notes:

- the old best verified chunk size was `1024`
- the optimal size may shift after the direct reader change

#### 6. Revisit prefetching

Status: not started

Notes:

- profiles so far used `prefetch_chunks=0`
- worth retrying once the benchmarking path is stable again

#### 7. Backend-level BGEN redesign or replacement

Status: not started

This is now the main remaining high-impact path.

## Next Priorities

### Priority 1: Re-benchmark the current batched-payload path cleanly

Need:

- one clean warmed GPU benchmark for the current code
- one updated profile snapshot for the batched materialization path

Reason:

- the batching logic is implemented, but its isolated impact is not yet verified as cleanly as the direct reader path

### Priority 2: Re-sweep chunk size on the optimized reader path

Need:

- fresh warmed GPU sweep for `256`, `512`, `1024`, `2048`, `4096`, `8192`

Reason:

- optimal chunk size likely changed after read-path improvements

### Priority 3: Investigate backend-level BGEN overhead

Need:

- determine whether `cbgen`/`bgen-reader` can expose a lower-overhead chunk dosage path
- if not, evaluate:
  - lower-level direct FFI integration
  - a native Rust/C++ reader
  - a cached intermediate dosage format for repeated development/profiling

Reason:

- the remaining top bottleneck is still per-variant BGEN backend work

### Priority 4: Revisit read prefetching

Need:

- benchmark `prefetch_chunks=1`, `2`, and `4`

Reason:

- once the read path is faster, overlap may become more useful

## Recommended Implementation Order From Here

1. Verify current batched-payload performance with a clean benchmark and profile
2. Re-sweep chunk size on the optimized reader path
3. Investigate lower-level BGEN backend overhead
4. Try prefetching
5. Only then consider larger architectural changes such as cached intermediate genotype storage

## Success Metrics

Continue measuring after each step:

- warmed end-to-end runtime
- `bgen_read_host`
- `get_variant_table_arrays`
- `device_put_genotypes`
- `build_chunk_payload`
- `write_chunk_to_disk`
- variants per second

## Non-Goals For Now

- rewriting the REGENIE2 math kernel
- micro-optimizing JAX linear algebra
- changing statistical outputs or formulas

The current optimized profile still does not justify those.

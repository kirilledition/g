# REGENIE2 Linear Optimization Plan

## Goal

Reduce end-to-end runtime for `g regenie2-linear`, with emphasis on:

- faster BGEN ingestion
- fewer host-device transfers
- fewer Python-managed per-chunk operations

## Profiling Basis

Primary saved profiling artifacts live locally under:

- `data/profiles/regenie2_linear/full_chr22_gpu_chunk1024_summary.json`
- `data/profiles/regenie2_linear/full_chr22_gpu_chunk1024_summary.txt`
- `data/profiles/regenie2_linear/full_chr22_gpu_chunk1024_cprofile.txt`
- `data/profiles/regenie2_linear/full_chr22_gpu_chunk1024_jax_trace/`

These are not committed because `data/` is repository-local and git-ignored.

## Main Findings

The full chr22 GPU profile shows the pipeline is dominated by host-side BGEN work rather than JAX compute.

Top stage timings from the profiled run:

| Stage | Total | Share of wall |
|---|---:|---:|
| `bgen_read_host` | 12.325s | 52.64% |
| `write_chunk_to_disk` | 1.006s | 4.30% |
| `get_variant_table_arrays` | 0.925s | 3.95% |
| `device_put_genotypes` | 0.373s | 1.59% |
| `preprocess_genotypes` | 0.083s | 0.36% |
| `compute_regenie2_linear_chunk` | 0.079s | 0.34% |

Important interpretation:

- `persist_chunked_results_total` encloses the writer lifetime and overlaps other stages. It should not be added to the totals above.
- Absolute profiled runtime is slower than benchmark runtime because `cProfile`, JAX tracing, and explicit synchronization add overhead.
- The stage ranking is still useful.

## Bottlenecks

### 1. BGEN decode and host reads

This is the dominant bottleneck.

Evidence from `cProfile`:

- `cbgen._ffi.bgen_file_open_genotype`
- `cbgen._ffi.bgen_genotype_read32`
- `_bgen_file.py:211(read_probability)`

These appear once per variant at very high frequency, which strongly suggests expensive per-variant backend overhead inside the chunk reads.

### 2. Repeated metadata extraction per chunk

`get_variant_table_arrays` is a meaningful secondary cost. The current path repeatedly rebuilds per-chunk metadata arrays and repeatedly splits allele strings.

### 3. Per-chunk host materialization and Arrow writes

The actual disk write time is modest, but per-chunk result packaging still forces:

- `jax.device_get(...)`
- payload dataclass construction
- Polars `DataFrame` construction
- schema casting
- one Arrow IPC write per chunk

### 4. Host-device transfers exist but are not yet dominant

Current transfer points:

- host to device in `src/g/io/reader.py` after BGEN decode
- device to host in `src/g/engine.py` when building persistence payloads

The transfer costs are real, but the profile says BGEN ingestion should be fixed first.

## Optimization Priorities

### Priority 1: Reduce BGEN metadata overhead

Low risk, immediate.

Plan:

- cache normalized metadata arrays once in `BgenReader`
- make `get_variant_table_arrays()` slice cached arrays rather than rebuilding arrays from backend fields each chunk

Expected benefit:

- saves repeated string splitting and array construction
- should reduce `get_variant_table_arrays` materially

### Priority 2: Investigate a faster BGEN read path

Highest potential impact.

Plan:

- verify whether the current `bgen-reader` stack exposes a chunk-oriented dosage path that avoids repeated per-variant open/read/close behavior
- if not, evaluate either:
  - a different backend path for BGEN dosage reads
  - a native reader path in Rust/C++
  - a cached intermediate dosage format for repeated step 2 development and profiling

Expected benefit:

- this is the only area likely to produce large end-to-end wins

### Priority 3: Batch result materialization

Medium risk, likely worthwhile after ingestion improves.

Plan:

- accumulate several output chunks before `jax.device_get(...)`
- write fewer, larger Arrow files
- reduce per-chunk DataFrame/schema/write overhead

Expected benefit:

- fewer device-to-host synchronizations
- less Python and Polars overhead

### Priority 4: Re-tune chunk size after ingestion changes

The best observed warmed GPU chunk size today is `1024`, but that may shift once read overhead changes.

Plan:

- re-sweep `1024`, `2048`, `4096`, and `8192` after each major I/O optimization

### Priority 5: Revisit read prefetching

Current profiles used `prefetch_chunks=0` for stability.

Plan:

- retry bounded prefetching once the loader path is cleaner
- measure whether host read latency can overlap with downstream work

### Priority 6: Small control-flow cleanup

Low priority.

Plan:

- skip chromosome splitting when a chunk is already chromosome-homogeneous
- keep LOCO fetch frequency minimal

## Host-Device Transfer Reduction Plan

### Current transfers

Host to device:

- decoded BGEN dosage chunk -> `jax.device_put(...)`

Device to host:

- per-chunk `jax.device_get(...)` in `build_regenie2_linear_chunk_payload()`

### Transfer reduction strategy

1. Reduce the number of chunks that cross the boundary.
2. Batch result extraction so device-to-host synchronization happens less often.
3. Avoid new sync points in Python loops.
4. Keep profiling-only `block_until_ready()` logic out of production paths.

## Recommended Implementation Order

1. Cache BGEN metadata arrays in `BgenReader`.
2. Benchmark again.
3. Investigate or redesign the BGEN dosage read path.
4. Batch result materialization and Arrow writes.
5. Re-sweep chunk size.
6. Try read prefetching.
7. Only then consider kernel-level changes.

## Success Metrics

After each optimization step, re-measure:

- end-to-end warmed runtime
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

The profile does not justify those yet.

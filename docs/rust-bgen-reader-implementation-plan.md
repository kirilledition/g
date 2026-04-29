# Rust BGEN Reader Implementation Plan

## Summary

Implement a Rust-backed BGEN reader with Python bindings optimized for REGENIE step-2 ingestion. The primary objective is throughput, not API compatibility. The hot path will be strict `float32` dosage chunk reads with direct JAX interoperability through contiguous host arrays.

## Target Scope (v1)

- Local filesystem BGEN only.
- Diploid biallelic variants only.
- Support both unphased (`ncombinations == 3`) and phased (`ncombinations == 4`) layouts.
- Output precision is always `float32`.
- Keep sample alignment and phenotype/covariate joins in Python.

## Architecture

- Rust extension module: `g._core`.
- Python orchestration: `src/g/io/bgen.py` and `src/g/io/source.py`.
- New strict hot-path API:
  - `BgenReader.read_float32(sample_indices, variant_start, variant_stop) -> np.ndarray`
- Compatibility shim remains during migration:
  - `BgenReader.read(...)` continues to work, but hot-path callers should use `read_float32`.

## Implementation Phases

### Phase 1: API + Pipeline Reshape

- Add strict `read_float32` API in Python BGEN reader.
- Route chunk iterators to `read_float32` for `float32` contiguous reads.
- Add tests for strict API parity against existing `read(...)` path.

### Phase 2: Rust Compute Kernels in `g._core`

- Add dosage conversion kernels for phased/unphased probability layouts.
- Validate output parity against existing NumPy implementation.
- Integrate kernels in Python conversion helpers for `float32` workloads.

### Phase 3: Rust Decode Path

- Parse BGEN headers and metadata in Rust.
- Implement block decode and direct dosage materialization.
- Replace cbgen-backed per-variant read loop in the hot path.

### Phase 4: Performance Hardening

- Add optional pinned-host buffer support with safe fallback.
- Sweep chunk sizes and thread counts.
- Benchmark end-to-end REGENIE throughput versus current cbgen path.

## Benchmark Criteria

Success is measured against current pipeline behavior on local chr22 data:

- Faster `bgen_read_host` stage timing.
- Faster end-to-end REGENIE step-2 wall time.
- No regression in output parity on supported layouts.

## Risks and Mitigations

- Risk: API breakage in callers.
  - Mitigation: keep `read(...)` compatibility while migrating internal hot path.
- Risk: layout edge cases in non-UKB files.
  - Mitigation: explicit early validation and hard errors for unsupported layouts.
- Risk: benchmark noise.
  - Mitigation: repeated runs with warmed caches and fixed chunk sweep.

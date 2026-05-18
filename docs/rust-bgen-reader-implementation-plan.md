# Rust BGEN Reader Implementation Plan

## Summary

Implement a Rust-backed BGEN run engine optimized for REGENIE step-2 ingestion. The primary objective is throughput, not API compatibility. The hot path delivers native preprocessed `float32` dosage chunks directly to the active REGENIE2 callbacks.

## Target Scope (v1)

- Local filesystem BGEN only.
- Diploid biallelic variants only.
- Support both unphased (`ncombinations == 3`) and phased (`ncombinations == 4`) layouts.
- Output precision is always `float32`.
- Keep sample alignment and phenotype/covariate joins in Rust through `_core`.

## Architecture

- Rust extension module: `g._core`.
- Python orchestration: `src/g/engine/regenie2_pipeline.py` and `src/g/io/source.py`.
- Native hot-path API:
  - `Regenie2RunEngine.run_bgen_dosage_buffered_chunks(...)`
  - `Regenie2RunEngine.run_bgen_variant_major_dosage_buffered_chunks(...)`
- No Python reader compatibility shim remains in production.

## Implementation Phases

### Phase 1: API + Pipeline Reshape

- Route REGENIE2 callbacks through native preprocessed buffered chunk delivery.
- Remove production Python reader/chunk compatibility paths.
- Add tests for native delivery and output-writer integration.

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
  - Mitigation: keep the public REGENIE-facing API small and remove unreleased compatibility shims instead of maintaining divergent reader behavior.
- Risk: layout edge cases in non-UKB files.
  - Mitigation: explicit early validation and hard errors for unsupported layouts.
- Risk: benchmark noise.
  - Mitigation: repeated runs with warmed caches and fixed chunk sweep.

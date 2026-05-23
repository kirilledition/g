# Rust SIMD Opportunity Audit

Date: 2026-05-23

## Context

The current production SIMD policy is deliberately narrow: AVX2 is the only active x86 SIMD backend, with scalar
fallbacks for non-AVX2 platforms. The trusted BGEN full-sample, no-missing, diploid, unphased 8-bit
variant-major path already uses raw-integer AVX2 decode. Runtime SIMD switches and benchmark-only helpers were
removed after benchmarking selected the raw AVX2 implementation.

This document records remaining SIMD opportunities in the Rust codebase. The priority is BGEN decode work because it
is the main Rust numeric hot path and already has reader benchmarks and profile counters.

## Prior Plan Status

The earlier trusted BGEN SIMD plan has already been implemented and benchmarked for the trusted identity/full-sample
variant-major path:

- Trusted identity decode now uses raw-integer AVX2 on AVX2-capable x86 CPUs and raw scalar fallback elsewhere.
- The runtime SIMD mode switch, temporary microbenchmark, and temporary native profile example were removed after the
  benchmark decision.
- Raw AVX2 improved trusted reader time by about 25-27% for chunk sizes 4096-16384 on `cantor`.
- Raw scalar was benchmarked and was slower than lookup at larger reader chunk sizes, so it is only a portability
  fallback.
- AVX-512, selected-subset SIMD, and row-major SIMD were not kept for the trusted path.
- The trusted hot decode path no longer rescans ploidy with `all_samples_present_diploid`; trusted validation still
  checks the contract before marking the reader validated.

## BGEN Findings

### High Priority: Non-Trusted No-Missing 8-Bit Variant-Major Identity Decode

The strongest remaining opportunity is the non-trusted unphased 8-bit variant-major identity/full-sample path in
`src/genotype/bgen/decode.rs`. The trusted path now decodes probability bytes directly with raw integer math:

```text
raw = 510 - 2*p0 - p1
dosage = raw / 255
```

The non-trusted path still uses scalar lookup-table decode when all samples are present. When the reader discovers that
the variant is unphased 8-bit, `sample_selection.is_identity` is true, and `all_samples_present_diploid(...)` is true,
the same contiguous variant-major row shape exists as in the trusted fast path. That should be able to reuse a
generalized raw-integer AVX2 decoder from `src/genotype/bgen/simd.rs`.

Expected implementation direction:

- Generalize the trusted raw-integer summary type and AVX2 decode function so it is not trusted-path-specific.
- Add an early branch in the non-trusted variant-major unphased 8-bit decoder for identity plus all-present samples.
- Keep scalar fallback and no runtime switch.
- Keep the path only if `bgen_preprocessed_variant_major_trusted_disabled` improves by at least 5% and trusted-path
  benchmark results do not regress.
- This is intentionally outside the original trusted-only SIMD plan; the old plan said not to change generic/untrusted
  behavior during that pass.

### Medium Priority: AVX2 All-Present Detection Outside Trusted Hot Decode

`all_samples_present_diploid` currently checks 16-byte chunks against `[2; 16]` and then scans the remainder. This is
simple and already fairly efficient, but it can be tested with AVX2 by comparing 32 bytes at a time against byte value
`2` and using a movemask to detect mismatches.

This is not a remaining trusted hot-path decode issue; that scan was already removed from trusted decode after
validation. The remaining uses are trusted validation and non-trusted all-present detection. This should be benchmarked
before keeping it. A microbenchmark win is not enough; it should improve trusted validation time or non-trusted reader
time in a realistic BGEN benchmark. If benchmark results are noise-level, keep the current scalar chunk comparison.

### Benchmark-Only: Selected-Subset Decode

Selected-subset paths use `selected_file_indices` or `file_to_selected_index`, so the natural access pattern is
gather/scatter. SIMD is unlikely to help arbitrary sparse or shuffled subsets. A useful subset SIMD path would require
detecting dense monotonic runs and applying the identity/raw decoder only within those runs.

This was explicitly left out of the trusted SIMD implementation and should remain benchmark-only until workload data
shows dense selected subsets are common enough to matter.

### Low Priority: Row-Major Identity Decode

The row-major full-sample unphased 8-bit path writes one dosage per sample with a stride of `variant_count`. Even if
probability decode is vectorized, output stores are not contiguous for a single variant. The current tile copy already
uses contiguous `copy_nonoverlapping` after decoding the tile, so row-major SIMD should not be prioritized before
variant-major work. Row-major SIMD was also explicitly rejected for the trusted SIMD pass.

### Low Priority: Generic Bit-Packed and Phased Decode

The generic BGEN path supports phased variants and probability bit widths from 1 to 32. It is branchy and uses a
bit-reader abstraction. SIMD here would require specialized kernels per bit width and phased mode. Avoid this unless
profile data shows non-8-bit or phased BGEN input is a real hot workload.

### Low Priority: Missing-Value Imputation

Variant-major imputation scans rows to replace `NaN` with the mean only when missing values are present. The hot trusted
and no-missing paths avoid it. This is not a good first SIMD target.

## Other Rust Findings

### Genotype Preprocessing

`src/genotype/preprocess.rs` has scalar numeric loops for row-major preprocessing and variant-major summarization. The
variant-major summarization loop is a plausible SIMD candidate because each variant row is contiguous and the loop
computes sums, square sums, observation counts, and threshold counts. It is lower priority than BGEN decode because the
variant-major BGEN reader already computes stats during decode.

A useful future experiment would add a benchmark for `summarize_variant_major_dosage_matrix` over dense no-missing
float32 rows and compare scalar versus AVX2 reductions. Keep this separate from BGEN decode work.

### Chunk Stats Post-Processing

`build_chunk_stats_from_summaries` is per-variant scalar post-processing over relatively small summary arrays. It has
branches for empty observations and `Option<f32>` info scores. SIMD is unlikely to beat simpler scalar code here unless
profile data proves it is hot.

### Output Writer and Finalization

The Rust output path spends time cloning metadata, building Arrow arrays, writing Arrow IPC, and finalizing Parquet.
Those operations are dominated by Arrow/Parquet library code, memory allocation, strings, and I/O. There is no obvious
hand-written SIMD target in `src/output`.

### Sample Alignment and LOCO Prediction Loading

`src/sample.rs` and `src/regenie.rs` are dominated by CSV/text parsing, string handling, hash maps, and joins. These are
not good targets for hand-written SIMD in this codebase.

### BGEN Metadata and Index Parsing

BGEN index and metadata parsing is control-flow and string-bound. It is not a useful SIMD target compared with dosage
decode and preprocessing.

## Recommended Order

1. Generalize raw-integer AVX2 decode and apply it to the non-trusted no-missing unphased 8-bit variant-major identity
   path.
2. Benchmark AVX2 `all_samples_present_diploid` only for trusted validation and non-trusted all-present detection.
3. Benchmark AVX2 variant-major summarization in `src/genotype/preprocess.rs`.
4. Prototype dense selected-subset run detection only if workload data shows selected subsets are common and mostly
   contiguous.

Each step should be benchmarked independently on `cantor` with 40-70 CPUs. Keep only paths that improve realistic reader
or preprocessing benchmarks beyond noise, and remove temporary benchmark switches or helpers before committing.

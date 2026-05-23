# Rust SIMD Opportunity Audit

Date: 2026-05-23

## Context

The current production SIMD policy is deliberately narrow: AVX2 is the only active x86 SIMD backend, with scalar
fallbacks for non-AVX2 platforms. The trusted BGEN full-sample, no-missing, diploid, unphased 8-bit variant-major path
already uses raw-integer AVX2 decode. Runtime SIMD switches and benchmark-only helpers were removed after benchmarking
selected the raw AVX2 implementation.

This document records SIMD opportunities in the Rust codebase and the decisions from the 2026-05-23 follow-up pass. The
priority was BGEN decode work because it is the main Rust numeric hot path and already has reader benchmarks and profile
counters.

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

### Implemented: Non-Trusted No-Missing 8-Bit Variant-Major Identity Decode

The strongest remaining opportunity was the non-trusted unphased 8-bit variant-major identity/full-sample path in
`src/genotype/bgen/decode.rs`. The trusted path now decodes probability bytes directly with raw integer math:

```text
raw = 510 - 2*p0 - p1
dosage = raw / 255
```

The non-trusted path previously used scalar lookup-table decode when all samples were present. When the reader discovers
that the variant is unphased 8-bit, `sample_selection.is_identity` is true, and
`all_samples_present_diploid(...)` is true, the same contiguous variant-major row shape exists as in the trusted fast
path. The decoder now reuses a generalized raw-integer AVX2 decoder from `src/genotype/bgen/simd.rs`.

Implementation notes:

- Generalize the trusted raw-integer summary type and AVX2 decode function so it is not trusted-path-specific.
- Add an early branch in the non-trusted variant-major unphased 8-bit decoder for identity plus all-present samples.
- Keep scalar fallback and no runtime switch.
- Keep only the AVX2 implementation selected by benchmark, with scalar fallback for non-AVX2 platforms.

`bgen_preprocessed_variant_major_trusted_disabled` on `cantor`, 40 CPUs, `RUSTFLAGS="-C target-cpu=native"`:

| chunk size | prior lookup median | final median | time change |
| --- | ---: | ---: | ---: |
| 1024 | 5.5348 ms | 3.0523 ms | -44.9% |
| 2048 | 5.5542 ms | 3.0110 ms | -45.8% |
| 4096 | 8.9318 ms | 5.1066 ms | -42.8% |
| 8192 | 16.908 ms | 9.2189 ms | -45.5% |
| 16384 | 30.387 ms | 16.886 ms | -44.4% |

### Implemented: AVX2 All-Present Detection Outside Trusted Hot Decode

`all_samples_present_diploid` used to check 16-byte chunks against `[2; 16]` and then scan the remainder. It now uses
AVX2 to compare 32 bytes at a time against byte value `2` and falls back to the old scalar chunk scan on non-AVX2
platforms.

This is not a trusted hot-path decode issue; that scan was already removed from trusted decode after validation. The
remaining uses are trusted validation and non-trusted all-present detection. In the non-trusted reader benchmark, the
incremental result was small: no statistically significant change for 1024, 2048, or 8192 chunks, and Criterion-reported
improvements at 4096 and 16384 chunks. A final full-sample benchmark after all changes stayed within noise for the main
reader path, so this was kept as a small AVX2-only improvement with scalar fallback.

### Implemented For Dense Runs: Selected-Subset Decode

Selected-subset paths use `selected_file_indices` or `file_to_selected_index`, so the natural access pattern is
gather/scatter. SIMD is unlikely to help arbitrary sparse or shuffled subsets. A useful subset SIMD path would require
detecting dense monotonic runs and applying the identity/raw decoder only within those runs.

The implemented path detects a single contiguous selected file-index run at sample-selection preparation time. For that
shape, selected input probabilities and output dosages are both contiguous, so generic and trusted variant-major decode
reuse the raw AVX2 identity decoder. Arbitrary sparse or shuffled subsets remain on the existing scalar lookup path.

`bgen_preprocessed_variant_major_contiguous_subset_trusted_disabled`:

| chunk size | scalar subset median | AVX2 dense-run median | time change |
| --- | ---: | ---: | ---: |
| 1024 | 3.6366 ms | 2.6441 ms | -27.3% |
| 2048 | 3.7166 ms | 2.5660 ms | -31.0% |
| 4096 | 5.9425 ms | 4.3773 ms | -26.3% |
| 8192 | 11.041 ms | 7.9102 ms | -28.4% |
| 16384 | 19.698 ms | 14.339 ms | -27.2% |

`bgen_preprocessed_variant_major_contiguous_subset_trusted_no_missing_diploid`:

| chunk size | scalar subset median | AVX2 dense-run median | time change |
| --- | ---: | ---: | ---: |
| 1024 | 3.8501 ms | 2.5668 ms | -33.3% |
| 2048 | 3.6096 ms | 2.5277 ms | -30.0% |
| 4096 | 5.9490 ms | 4.3354 ms | -27.1% |
| 8192 | 11.017 ms | 7.8202 ms | -29.0% |
| 16384 | 19.719 ms | 14.235 ms | -27.8% |

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

### Implemented: Genotype Preprocessing

`src/genotype/preprocess.rs` has scalar numeric loops for row-major preprocessing and variant-major summarization. The
variant-major summarization loop was a plausible SIMD candidate because each variant row is contiguous and the loop
computes sums, square sums, observation counts, and threshold counts. The implementation now summarizes contiguous
variant-major rows with AVX2 masks for observed values and dosage thresholds, with scalar fallback for non-AVX2
platforms and scalar handling for row tails.

Focused `preprocess_variant_major_summary` benchmark over dense no-missing `f32` rows:

| sample count | scalar median | AVX2 median | time change |
| --- | ---: | ---: | ---: |
| 1024 | 188.52 us | 35.551 us | -81.1% |
| 2048 | 374.18 us | 70.256 us | -81.2% |
| 4096 | 759.30 us | 142.07 us | -81.3% |
| 8192 | 1.5072 ms | 277.53 us | -81.6% |
| 16384 | 3.0026 ms | 546.37 us | -81.8% |

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

Completed in this pass:

1. Generalized raw-integer AVX2 decode and applied it to the non-trusted no-missing unphased 8-bit variant-major identity
   path.
2. Benchmarked and kept AVX2 `all_samples_present_diploid` for non-trusted all-present detection and trusted validation.
3. Benchmarked and kept AVX2 variant-major summarization in `src/genotype/preprocess.rs`.
4. Benchmarked and kept dense contiguous selected-subset decode for generic and trusted variant-major BGEN reads.

No AVX-512 path, runtime SIMD switch, row-major SIMD path, generic bit-packed/phased SIMD path, missing-value imputation
SIMD path, or arbitrary gather/scatter subset SIMD path was added.

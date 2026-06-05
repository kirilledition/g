# SIMD Optimization Reference

Date: 2026-05-23
Host for benchmarks: `cantor`

This document is the authoritative reference for SIMD work tried in the Rust BGEN and preprocessing paths. It merges the
old planning notes, benchmark results, and follow-up opportunity audit so future work can quickly see what was tested,
what was kept, and what should not be repeated without new profiling evidence.

## Current Policy

- AVX2 is the only production x86 SIMD target.
- Scalar fallbacks remain for non-AVX2 platforms.
- AVX-512 is not a production target. It was tried before this reference was written and was slower than AVX2 for this
  workload.
- Runtime benchmark switches were removed after selecting the winning implementations.
- SIMD code should stay isolated in small Rust modules or helpers using `std::arch` and runtime
  `is_x86_feature_detected!("avx2")` checks.
- Keep only implementations that improve realistic reader or preprocessing benchmarks. Do not keep alternate production
  versions just because they are interesting.

## Decision Matrix

| Area | Optimization | Tried | Result | Production decision |
| --- | --- | --- | --- | --- |
| Trusted BGEN identity decode | Raw integer AVX2 decode for full-sample no-missing unphased 8-bit variant-major rows | Yes | Faster by about 25-27% for 4096-16384 variant chunks versus lookup | Kept |
| Trusted BGEN identity decode | Raw integer scalar decode | Yes | Slightly faster at 1024, slower by about 1.7-2.6% at larger chunks | Kept only as non-AVX2 fallback |
| Trusted BGEN identity decode | Lookup-table scalar decode as production default | Yes | Slower than raw AVX2 | Removed from trusted identity hot path |
| Trusted BGEN identity decode | Runtime mode switch for lookup/raw-scalar/raw-AVX2 | Yes | Useful for benchmarking only | Removed |
| Trusted BGEN identity decode | AVX-512 default path | Yes, before this final pass | Slower than AVX2 | Not kept |
| Non-trusted BGEN identity decode | Reuse raw integer AVX2 for full-sample no-missing unphased 8-bit variant-major rows | Yes | Faster by about 43-46% versus prior lookup path | Kept |
| Ploidy/missingness scan | AVX2 `all_samples_present_diploid` 32-byte comparison | Yes | Small win at some larger chunks, neutral elsewhere | Kept with scalar fallback |
| Dense selected subsets | Raw integer AVX2 when selected file indices are one contiguous run | Yes | Faster by about 26-33% versus scalar subset lookup | Kept |
| Arbitrary selected subsets | SIMD gather/scatter for sparse or shuffled selections | No | Expected poor fit because input/output are not both contiguous | Do not implement without workload proof |
| Variant-major preprocessing | AVX2 row summaries for contiguous `f32` rows | Yes | Faster by about 81% in focused benchmark | Kept |
| Row-major BGEN decode | SIMD identity decode with row-major output | No | Output stores are strided for a single variant; current tile copy is already contiguous | Do not prioritize |
| Generic BGEN decode | SIMD for bit-packed widths 1-32 and phased mode | No | Branchy, many specialized kernels required | Avoid unless profiling shows hot workload |
| Missing-value imputation | SIMD `NaN` replacement pass | No | Hot trusted/no-missing paths avoid it | Avoid unless missing-heavy profiles prove it hot |
| Chunk stats post-processing | SIMD over summary vectors | No | Branchy `Option` and per-variant scalar work | Avoid unless profiling proves hot |
| Output writing/finalization | Hand-written SIMD | No | Dominated by Arrow/Parquet, allocation, strings, and I/O | Not a SIMD target |
| Sample alignment/LOCO loading | Hand-written SIMD | No | Dominated by CSV/text parsing, hash maps, and joins | Not a SIMD target |
| BGEN metadata/index parsing | Hand-written SIMD | No | Control-flow and string-bound | Not a SIMD target |

## Core BGEN Math

For unphased 8-bit BGEN probability pairs:

```text
p0 = homozygous reference probability byte
p1 = heterozygous probability byte

raw = 510 - 2*p0 - p1
dosage = raw / 255
```

The optimized paths compute output dosage values from `raw` and compute summary statistics in integer space:

```text
dosage_sum = raw_sum / 255
dosage_square_sum = raw_square_sum / (255 * 255)
```

Counts are also derived from integer `raw` thresholds:

```text
zero_count = raw == 0
nonzero_count = raw > 0
homozygous_reference_count = raw <= 127
heterozygous_count = 128 <= raw <= 382
homozygous_alternate_count = raw >= 383
```

This removes lookup-table reads, avoids per-sample floating-point accumulation in the hot identity path, and gives AVX2 a
simple contiguous byte-decoding problem.

## Implemented Paths

### Trusted Identity Decode

Path:

```text
src/genotype/bgen/trusted.rs
src/genotype/bgen/simd.rs
```

Shape:

```text
trusted no-missing diploid
unphased
8-bit probabilities
variant-major output
identity/full-sample selection
```

The old trusted identity loop used lookup-table dosage decoding and per-sample selection logic. The retained path
branches once on identity selection, decodes contiguous probability bytes with raw integer AVX2, writes a contiguous
variant-major output row, and returns integer-derived summary stats. Scalar raw integer decode remains only as the
fallback when AVX2 is unavailable.

Median reader times from the trusted benchmark:

| chunk size | lookup | raw scalar | raw AVX2 | final auto |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 4.3336 ms | 4.1988 ms | 3.0116 ms | 2.9622 ms |
| 2048 | 4.0504 ms | 4.1825 ms | 2.9495 ms | 2.9606 ms |
| 4096 | 6.5958 ms | 6.6772 ms | 5.0574 ms | 5.0130 ms |
| 8192 | 12.089 ms | 12.366 ms | 9.0766 ms | 9.0681 ms |
| 16384 | 21.904 ms | 22.138 ms | 16.731 ms | 16.715 ms |

Single 16,384-variant native profile:

| mode | decompression_ns | probability_decode_ns | output_write_ns |
| --- | ---: | ---: | ---: |
| lookup | 414,107,129 | 765,949,602 | 763,091,426 |
| raw_scalar | 426,626,205 | 786,003,517 | 783,146,020 |
| raw_avx2 | 455,462,499 | 578,556,384 | 575,467,948 |
| auto | 450,862,963 | 601,849,412 | 598,829,025 |

### Non-Trusted Full-Sample No-Missing Decode

Path:

```text
src/genotype/bgen/decode.rs
src/genotype/bgen/simd.rs
```

Shape:

```text
trusted flag disabled
unphased
8-bit probabilities
variant-major output
identity/full-sample selection
all samples present and diploid
```

Once the generic decoder proves the row is all-present and identity-selected, it uses the same raw integer AVX2 helper as
the trusted path.

`bgen_preprocessed_variant_major_trusted_disabled`, 40 CPUs, `RUSTFLAGS="-C target-cpu=native"`:

| chunk size | prior lookup median | final median | time change |
| --- | ---: | ---: | ---: |
| 1024 | 5.5348 ms | 3.0523 ms | -44.9% |
| 2048 | 5.5542 ms | 3.0110 ms | -45.8% |
| 4096 | 8.9318 ms | 5.1066 ms | -42.8% |
| 8192 | 16.908 ms | 9.2189 ms | -45.5% |
| 16384 | 30.387 ms | 16.886 ms | -44.4% |

### All-Present Diploid Scan

Path:

```text
src/genotype/bgen/trusted.rs
src/genotype/bgen/simd.rs
```

`all_samples_present_diploid` now uses AVX2 to compare 32 bytes at a time against byte value `2`, falling back to the
old scalar chunk scan on non-AVX2 platforms. This is not part of trusted hot decode after validation, but it is still
used by trusted validation and non-trusted all-present detection.

Result:

- No statistically significant change for some smaller chunks.
- Criterion-reported small improvements at 4096 and 16384 in the non-trusted reader benchmark.
- Final full-sample benchmark after all changes stayed within noise for the main reader path.

Decision: kept as a small AVX2-only improvement with scalar fallback.

### Dense Contiguous Selected Subsets

Path:

```text
src/genotype/bgen/sample_selection.rs
src/genotype/bgen/decode.rs
src/genotype/bgen/trusted.rs
```

The sample-selection builder records `contiguous_file_index_start` when selected file indices form one monotonic
contiguous run. For that shape, selected input probabilities and output dosages are both contiguous, so generic and
trusted variant-major decode reuse the raw AVX2 identity decoder. Sparse or shuffled subsets continue to use the scalar
lookup path.

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

### Variant-Major Preprocessing Summaries

Path:

```text
src/genotype/preprocess.rs
benches/preprocess.rs
```

The variant-major summarizer now uses AVX2 masks over contiguous `f32` rows to compute sums, square sums, observation
counts, and dosage threshold counts. Scalar fallback remains for non-AVX2 platforms and row tails.

Focused `preprocess_variant_major_summary` benchmark over dense no-missing `f32` rows:

| sample count | scalar median | AVX2 median | time change |
| --- | ---: | ---: | ---: |
| 1024 | 188.52 us | 35.551 us | -81.1% |
| 2048 | 374.18 us | 70.256 us | -81.2% |
| 4096 | 759.30 us | 142.07 us | -81.3% |
| 8192 | 1.5072 ms | 277.53 us | -81.6% |
| 16384 | 3.0026 ms | 546.37 us | -81.8% |

## Rejected Or Deferred Work

### AVX-512

AVX-512 was tried before the final AVX2-only policy and was slower than AVX2 for this workload. Do not reintroduce an
AVX-512 production path unless there is new hardware-specific evidence and an isolated benchmark branch.

### Raw Scalar As A Replacement For AVX2

Raw scalar integer decode is useful because it provides the same semantics as the AVX2 path on non-AVX2 platforms. It
was not faster than lookup at larger trusted reader chunk sizes, so it should not be treated as a separate optimized
production mode.

### Arbitrary Selected-Subset SIMD

Sparse or shuffled subsets require gather/scatter-style access through `selected_file_indices` or
`file_to_selected_index`. That defeats the contiguous-load/contiguous-store shape that made the kept AVX2 paths fast.
Only the dense single-run subset case is optimized.

### Row-Major Decode SIMD

Row-major full-sample unphased 8-bit decode writes one dosage per sample with a stride of `variant_count`. SIMD decode
could speed up probability math, but stores are not contiguous for a single variant. The current row-major tile copy
already uses contiguous `copy_nonoverlapping` after decoding the tile. Do not prioritize this before new profile data.

### Generic Bit-Packed Or Phased BGEN SIMD

The generic BGEN path supports phased variants and probability bit widths from 1 to 32. It is branchy and uses a bit
reader abstraction. SIMD here would require specialized kernels per bit width and phased mode. Avoid this unless real
input profiles show non-8-bit or phased BGEN is hot.

### Missing-Value Imputation SIMD

Variant-major imputation scans rows to replace `NaN` with the mean only when missing values are present. The main trusted
and no-missing paths avoid it. This is not a good SIMD target without missing-heavy profiles.

### Non-Numeric Rust Paths

Chunk stats post-processing, output writing/finalization, sample alignment, LOCO prediction loading, and BGEN metadata
or index parsing are not current hand-written SIMD targets. They are dominated by control flow, library code, allocation,
strings, I/O, joins, or small per-variant scalar work.

## Correctness Coverage

The SIMD work added or exercised Rust coverage for:

- sample counts around vector boundaries: `0, 1, 7, 8, 15, 16, 17, 31, 32, 33`
- all dosage 0
- all dosage 2
- alternating raw dosage values
- rare-variant-like values
- deterministic valid probability pairs where `p0 + p1 <= 255`
- scalar versus AVX2 output equality or bounded deltas
- integer-derived sum and square-sum parity
- zero/nonzero and genotype-threshold counts
- dense contiguous selected-subset detection
- trusted and non-trusted variant-major BGEN decode behavior

Validation commands used in the final pass:

```text
cargo test --lib
cargo test bgen --lib
cargo test --test rust_native_coverage bgen
cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic
cargo fmt --all --check
```

## Benchmarking Notes

Use `cantor` or another CPU worker node for CPU benchmarks. Do not run heavy compilation or Criterion benchmarks on the
head node.

Representative command shape:

```text
srun --nodelist=cantor --cpus-per-task=40 --mem=16G --time=00:45:00 bash -lc 'cd /mnt/beegfs/kirill/Projects/g && . scripts/server_env.sh && RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=40 cargo bench --bench bgen_read <benchmark-filter> -- --sample-size 10 --measurement-time 1 --warm-up-time 1'
```

Keep benchmark-only mode switches temporary. Once a faster implementation is selected, remove alternate public runtime
switches and leave a single production dispatch policy.

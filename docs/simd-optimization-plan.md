Yes — after hearing that AVX512 was slower than AVX2, I would narrow the SIMD plan substantially.

My revised recommendation:

```text
Do not pursue AVX512 as the default path.
Do pursue a carefully scoped AVX2 fast path for the trusted BGEN identity/full-sample variant-major decode.
Before AVX2, first rewrite the scalar trusted path around integer dosage arithmetic and no sample-selection branch.
```

The main insight: your current hot loop is not just “missing SIMD.” It also has avoidable branch/lookup/scatter structure. The best optimization is probably:

```text
current scalar lookup loop
    ↓
scalar direct integer raw-dosage loop
    ↓
AVX2 direct integer raw-dosage loop
```

AVX512 being slower than AVX2 is not surprising. AVX/AVX512 can reduce CPU frequency, and the penalty is worse with wider/heavier instructions and many cores; if the loop is memory-bandwidth or decompression-limited, AVX512 often does not earn back that cost. ([The Cloudflare Blog][1])

---

# Best SIMD target in current Rust code

The highest-value target is:

```text
src/genotype/bgen/trusted.rs

decode_trusted_unphased_eight_bit_variant_into_variant_major_matrix(...)
```

The current loop is conceptually:

```rust
for (file_sample_index, probability_pair) in packed_probability_bytes.chunks_exact(2).enumerate() {
    let selected_index = if sample_selection.is_identity {
        file_sample_index
    } else {
        sample_selection.file_to_selected_index[file_sample_index]
    };
    if selected_index == usize::MAX {
        continue;
    }

    let packed_probability_index = usize::from(probability_pair[0]) | (usize::from(probability_pair[1]) << 8);
    let dosage_value = dosage_lookup[packed_probability_index];

    selected_dosage_total += dosage_value;
    selected_dosage_square_total += dosage_value * dosage_value;
    increment_dosage_summary_counts(...);

    output_pointer.add(variant_row_offset + selected_index).write(dosage_value);
}
```

For the common trusted full-sample path, this loop should not do:

```text
sample_selection.is_identity branch per sample
file_to_selected_index lookup per sample
selected_index == usize::MAX branch per sample
65,536-entry dosage lookup per sample
floating-point accumulation per sample
function call / branchy count update per sample
```

The trusted identity/full-sample path can be much tighter.

---

# Core mathematical rewrite

For unphased 8-bit BGEN, the current dosage formula is:

```text
p0 = homozygous reference probability byte
p1 = heterozygous probability byte

dosage = 2 - (2*p0 + p1) / 255
```

Equivalently:

```text
raw = 510 - 2*p0 - p1
dosage = raw / 255
```

where:

```text
raw ∈ [0, 510]
```

This is perfect for SIMD and also better scalar code.

You can compute statistics exactly in integer space:

```text
dosage_sum        = raw_sum / 255
dosage_square_sum = raw_square_sum / (255 * 255)
```

Counts can also be computed from `raw` without floating point:

```text
zero_count                  = raw == 0
nonzero_count               = raw > 0

homozygous_reference_count  = dosage < 0.5  => raw <= 127
heterozygous_count          = 0.5 <= dosage < 1.5 => 128 <= raw <= 382
homozygous_alternate_count  = dosage >= 1.5 => raw >= 383
```

This avoids the lookup table and avoids per-sample float comparisons.

Because your code is pre-release, I would define this as the new trusted-path semantics and test it against REGENIE/parity tolerances. It is mathematically cleaner than accumulating f32 dosage values in an arbitrary order.

---

# Implementation plan for coding agent

## Step 1 — Add a scalar direct trusted identity path

Before AVX2, implement a scalar path that handles only:

```text
trusted no-missing diploid
unphased
8-bit probabilities
variant-major output
identity sample selection / full sample
```

Suggested structure:

```rust
fn decode_trusted_unphased_eight_bit_identity_variant_major_scalar(
    packed_probability_bytes: &[u8],
    output_row: &mut [f32],
) -> TrustedVariantSummary
```

where:

```rust
struct TrustedVariantSummary {
    selected_dosage_total: f32,
    selected_dosage_square_total: f32,
    zero_count: i32,
    nonzero_count: i32,
    homozygous_reference_count: i32,
    heterozygous_count: i32,
    homozygous_alternate_count: i32,
}
```

Scalar reference logic:

```rust
const INV_255: f32 = 1.0 / 255.0;
const INV_255_SQUARED: f32 = 1.0 / (255.0 * 255.0);

let mut raw_sum: u64 = 0;
let mut raw_square_sum: u64 = 0;
let mut zero_count: i32 = 0;
let mut homozygous_reference_count: i32 = 0;
let mut homozygous_alternate_count: i32 = 0;

for (sample_index, pair) in packed_probability_bytes.chunks_exact(2).enumerate() {
    let p0 = u16::from(pair[0]);
    let p1 = u16::from(pair[1]);
    let raw = 510_u16 - (2 * p0) - p1;

    output_row[sample_index] = f32::from(raw) * INV_255;

    raw_sum += u64::from(raw);
    raw_square_sum += u64::from(raw) * u64::from(raw);

    if raw == 0 {
        zero_count += 1;
    }
    if raw <= 127 {
        homozygous_reference_count += 1;
    } else if raw >= 383 {
        homozygous_alternate_count += 1;
    }
}

let sample_count = i32::try_from(output_row.len()).unwrap_or(i32::MAX);
let nonzero_count = sample_count - zero_count;
let heterozygous_count = sample_count - homozygous_reference_count - homozygous_alternate_count;

TrustedVariantSummary {
    selected_dosage_total: raw_sum as f32 * INV_255,
    selected_dosage_square_total: raw_square_sum as f32 * INV_255_SQUARED,
    zero_count,
    nonzero_count,
    homozygous_reference_count,
    heterozygous_count,
    homozygous_alternate_count,
}
```

This scalar direct path is valuable even if AVX2 later gives modest gains.

---

## Step 2 — Branch once on identity sample selection

In `decode_trusted_unphased_eight_bit_variant_into_variant_major_matrix(...)`, add an early fast path:

```rust
if sample_selection.is_identity {
    let output_row = unsafe {
        std::slice::from_raw_parts_mut(
            output_pointer.add(variant_row_offset),
            selected_sample_count,
        )
    };

    let summary = decode_trusted_unphased_eight_bit_identity_variant_major(
        packed_probability_bytes,
        output_row,
    );

    return Ok(VariantDecodeResult { ...summary... });
}
```

The current loop checks identity/sample mapping inside every sample iteration. For full-sample trusted BGEN, that branch should disappear.

Keep the existing selected-subset/scatter path for now. SIMD gathers/scatters are not worth doing first.

---

## Step 3 — Add AVX2 dispatch for the identity path

Use stable Rust `std::arch` intrinsics and runtime dispatch. `std::arch` is Rust’s architecture-specific intrinsics module, and `is_x86_feature_detected!` is the standard runtime feature detection mechanism. ([doc.rust-lang.org][2])

Do not use `std::simd` for production yet; Rust’s `std::simd` remains documented as nightly-only experimental. ([doc.rust-lang.org][3])

Suggested dispatch:

```rust
pub(super) fn decode_trusted_unphased_eight_bit_identity_variant_major(
    packed_probability_bytes: &[u8],
    output_row: &mut [f32],
) -> TrustedVariantSummary {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // Safety: AVX2 was checked at runtime.
            return unsafe {
                decode_trusted_unphased_eight_bit_identity_variant_major_avx2(
                    packed_probability_bytes,
                    output_row,
                )
            };
        }
    }

    decode_trusted_unphased_eight_bit_identity_variant_major_scalar(
        packed_probability_bytes,
        output_row,
    )
}
```

Suggested module layout:

```text
src/genotype/bgen/trusted_identity.rs
src/genotype/bgen/trusted_identity_avx2.rs
```

or:

```text
src/genotype/bgen/simd.rs
```

Keep all `std::arch` code isolated. Do not scatter intrinsics across `trusted.rs`.

---

# AVX2 loop design

Process 16 samples per iteration:

```text
32 input bytes = 16 probability pairs
each pair = p0 byte, p1 byte
interpret as 16 u16 lanes: p0 + (p1 << 8)
p0 = lane & 0x00ff
p1 = lane >> 8
raw = 510 - 2*p0 - p1
```

AVX2 pseudo-code:

```rust
let pairs = _mm256_loadu_si256(input_ptr as *const __m256i);

let p0 = _mm256_and_si256(pairs, _mm256_set1_epi16(0x00ff));
let p1 = _mm256_srli_epi16(pairs, 8);

let raw = _mm256_sub_epi16(
    _mm256_set1_epi16(510),
    _mm256_add_epi16(_mm256_slli_epi16(p0, 1), p1),
);
```

Convert raw u16 to f32 dosage in two halves:

```rust
let raw_low_128 = _mm256_castsi256_si128(raw);
let raw_high_128 = _mm256_extracti128_si256(raw, 1);

let raw_low_i32 = _mm256_cvtepu16_epi32(raw_low_128);
let raw_high_i32 = _mm256_cvtepu16_epi32(raw_high_128);

let dosage_low = _mm256_mul_ps(_mm256_cvtepi32_ps(raw_low_i32), inv_255);
let dosage_high = _mm256_mul_ps(_mm256_cvtepi32_ps(raw_high_i32), inv_255);

_mm256_storeu_ps(output_ptr.add(sample_index), dosage_low);
_mm256_storeu_ps(output_ptr.add(sample_index + 8), dosage_high);
```

Counts:

```rust
zero_count += popcnt(cmpeq_epi16(raw, 0)) / 2
homozygous_reference_count += popcnt(raw <= 127) / 2
homozygous_alternate_count += popcnt(raw >= 383) / 2
```

AVX2 comparison trick:

```rust
raw <= 127:
    cmpgt_epi16(128, raw)

raw >= 383:
    cmpgt_epi16(raw, 382)
```

Because `raw` is always non-negative and at most 510, signed i16 comparisons are safe.

For sums:

```text
raw_sum: accumulate in i32 lanes, flush periodically to u64
raw_square_sum: accumulate raw*raw in i32 lanes, flush periodically to u64
```

Do not accumulate raw-square indefinitely in i32 lanes. It can overflow over large sample counts.

Safe pattern:

```text
every 1024 or 4096 vector iterations:
  horizontally reduce raw_sum vector to u64
  horizontally reduce raw_square_sum vector to u64
  zero the vector accumulators
```

This avoids i32 overflow while still reducing scalar overhead.

---

# Do not optimize these first

## 1. Do not make AVX512 the default

Given your result, keep AVX512 disabled by default. Add it only as a temporary benchmark-only
experiment later, not as a production selector.

but the default dispatch should be:

```text
AVX2 if available, else scalar
```

## 2. Do not SIMD the selected-subset path yet

When `sample_selection.is_identity == false`, the loop has:

```text
file_to_selected_index lookup
skip branch
possibly non-contiguous output
```

That is not the clean first SIMD target. Keep it scalar until the full-sample identity path is optimized and benchmarked.

## 3. Do not SIMD row-major output first

The row-major path writes strided columns inside a tile. SIMD stores are much less natural there. Your trusted performance path should be variant-major, where each variant row is contiguous.

## 4. Do not use Polars/PyArrow/other table libraries here

This is BGEN byte decoding and array filling. SIMD belongs in Rust intrinsics or compiler-friendly scalar code, not a DataFrame layer.

---

# Important additional optimization: skip repeated trusted ploidy scans when validated

In `trusted.rs`, the trusted path still checks:

```rust
all_samples_present_diploid(sample_ploidy_and_missingness)
```

for every variant. That scans `sample_count` bytes per variant before decoding probabilities.

If `trusted_no_missing_diploid` is only enabled after a validation pass or validation cache, then the hot decode path should be allowed to skip this scan.

Recommended design:

```text
trusted_validation_mode:
  verify_each_variant   # safe/debug mode
  assume_validated      # performance mode after validation/cache
```

Then:

```rust
if validation_mode == VerifyEachVariant {
    if !all_samples_present_diploid(sample_ploidy_and_missingness) {
        return Err(...);
    }
}
```

This can be more valuable than SIMD if the ploidy scan is visible in profiles.

If you still need a fast verifier, an AVX2 `all_bytes_equal_2` helper is easy, but the first question should be whether the scan should exist in the hot path at all.

---

# Benchmark plan

Your coding agent should not just implement AVX2 and declare victory. It should benchmark four things:

```text
1. current lookup scalar
2. new scalar raw-integer identity path
3. AVX2 raw-integer identity path
4. AVX512 only if already implemented, but not as default
```

## Microbenchmarks

Add Criterion benches for synthetic decompressed probability bytes:

Cover synthetic decompressed probability bytes at 10k, 100k, and 500k samples.

Use several probability patterns:

```text
all p0=255,p1=0      -> dosage 0
all p0=0,p1=0        -> dosage 2
mixed random bytes normalized enough for raw in range
rare-variant-like
real decompressed block sampled from BGEN if easy
```

For BGEN probabilities, `p0 + p1 <= 255` should hold for valid unphased two-probability representation. Synthetic data should respect that if possible.

Metrics:

```text
ns/sample
GB/s input read
GB/s output write
raw_sum/raw_square correctness
counts correctness
```

## Macrobenchmarks

Use existing BGEN benchmark plus pipeline profile:

```text
cargo bench --bench bgen_read
```

and a full app-level benchmark:

```text
trusted no-missing variant-major linear score-only
trusted no-missing variant-major binary score-only
trusted no-missing binary --firth --approx if relevant
```

Record:

```text
wall time
native_engine_delivery
compressed_block_fetch_ns
decompression_ns
probability_decode_ns
output_write_ns
host_to_device_transfer
jax_compute
writer time
```

The SIMD optimization is only worth keeping if it improves **end-to-end wall time**, not only a synthetic inner loop.

---

# Correctness tests to require

Add Rust unit tests comparing scalar and AVX2.

Cases:

```text
sample count 0/1/7/8/15/16/17/31/32/33
all dosage 0
all dosage 2
alternating raw values
rare-variant-like values
random valid p0/p1 where p0+p1 <= 255
```

Test outputs:

```text
dosage output vector equal or within exact expected f32 formula
raw_sum-derived dosage_sum
raw_square-derived dosage_square_sum
zero/nonzero counts
hom-ref/het/hom-alt counts
```

During development, add an integration/parity test:

```text
trusted scalar forced vs trusted AVX2 forced
same BGEN chunk
same ChunkStats within tight tolerance
same association output within tolerance
```

Use a temporary runtime override only while benchmarking and validating candidate paths. Remove it once
the fastest correct implementation is selected so production code has a single dispatch policy.

---

# Suggested task prompt for coding agent

You can send this almost directly:

```text
Task: Implement and benchmark an AVX2 trusted BGEN identity fast path.

Context:
The app is a Rust/Python/JAX GWAS engine. The current Rust trusted BGEN variant-major path decodes unphased 8-bit BGEN probability pairs using a scalar lookup table and per-sample sample-selection logic. AVX512 was tried and was slower than AVX2, so do not make AVX512 the default.

Goal:
Optimize the trusted no-missing diploid unphased 8-bit variant-major BGEN decode path for identity/full-sample selection.

Primary target:
src/genotype/bgen/trusted.rs
decode_trusted_unphased_eight_bit_variant_into_variant_major_matrix(...)

Requirements:
1. Add an early identity/full-sample fast path when sample_selection.is_identity is true.
2. In that fast path, avoid:
   - per-sample sample_selection.is_identity checks
   - file_to_selected_index lookups
   - selected_index == usize::MAX checks
   - dosage lookup table reads
   - f32 per-sample accumulation for stats
3. Compute dosage using integer raw dosage:
   raw = 510 - 2*p0 - p1
   dosage = raw / 255
4. Compute stats from integer sums:
   dosage_sum = raw_sum / 255
   dosage_square_sum = raw_square_sum / (255 * 255)
5. Compute counts from integer raw:
   zero_count = raw == 0
   nonzero_count = raw > 0
   homozygous_reference_count = raw <= 127
   heterozygous_count = 128 <= raw <= 382
   homozygous_alternate_count = raw >= 383
6. Implement scalar direct-integer path first.
7. Implement AVX2 runtime-dispatched path using std::arch and is_x86_feature_detected!("avx2").
8. Keep scalar fallback on all platforms.
9. Do not enable AVX512 by default. If AVX512 code exists, leave it benchmark-only or behind explicit override.
10. Use a temporary environment override for benchmarking candidate paths, then remove it before finalizing.
11. Keep selected-subset/non-identity path scalar for now.
12. Add Rust tests comparing scalar and AVX2 outputs/stats.
13. Add temporary Criterion microbenchmarks comparing:
    - old lookup scalar if easy to preserve as private benchmark helper
    - new scalar integer path
    - AVX2 integer path
14. Run macro benchmark using existing BGEN benchmark and report:
    - native wall time
    - probability_decode_ns
    - output_write_ns
    - decompression_ns
    - end-to-end pipeline wall time if feasible

Safety:
- Isolate unsafe AVX2 code in a small module.
- Only call AVX2 function after runtime feature detection.
- Use unaligned loads/stores.
- Handle tail samples with scalar code.
- Avoid i32 overflow in vector raw_square accumulators by periodically reducing to u64.
- Do not change generic missing/untrusted BGEN behavior.

Success criteria:
- Scalar integer fast path is at least as fast as current lookup scalar.
- AVX2 is faster than scalar integer path on the target server.
- End-to-end trusted BGEN benchmark improves materially.
- Association output remains within established parity tolerances.
```

---

# Expected outcome

I would expect the biggest gains from:

```text
1. identity fast path with no selection branch
2. removing the lookup table
3. exact integer stats
4. AVX2 vectorized decode/store
5. optionally skipping per-variant ploidy scan after trusted validation
```

I would not expect big gains from:

```text
AVX512
selected-subset SIMD
row-major SIMD
SIMD sample parsing
SIMD text output formatting
```

The most important benchmark question is:

```text
After this optimization, is native time still dominated by zlib decompression?
```

If yes, further SIMD decode work will have diminishing returns, and the next native optimization should move toward decompression/backend benchmarking or bigger architectural wins like multi-phenotype batching.

[1]: https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/?utm_source=chatgpt.com "On the dangers of Intel's frequency scaling"
[2]: https://doc.rust-lang.org/std/arch/index.html?utm_source=chatgpt.com "std::arch"
[3]: https://doc.rust-lang.org/std/simd/index.html?utm_source=chatgpt.com "std::simd - Rust"

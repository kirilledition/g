# Integer Type Audit

| Status | Applies to | Owner |
| --- | --- | --- |
| Initial audit | main branch as of 2026-07-09 Rust native integer surfaces | Native runtime maintainers |

This page records the current native integer surfaces and the intended type for
each category. Keep it in sync when changing native input/output contracts,
reader APIs, manifests, or buffer ownership.

## Current Type Map

| Surface | Current type | Classification | Decision |
| --- | --- | --- | --- |
| `ChunkSpec.variant_start_index`, `ChunkSpec.variant_stop_index` | `usize` | Memory index/range | Keep `usize`. |
| BGEN `sample_count`, `variant_count` | `usize` after header parse | Memory count | Keep `usize`; BGEN header `u32` is checked during parse. |
| BGEN and aligned sample selection indices | `usize` internally | Memory index | Convert Python `int64` arrays at the binding edge; export aligned sample indices back to Python as checked `int64`. |
| BGEN variant metadata positions | `Vec<i64>` | Genomic position/output metadata | Keep fixed-width `i64`. |
| BGEN file offsets and block lengths | parser-local checked integers | File format offsets/lengths | Keep checked before indexing mmap data. |
| `ChunkStats` count columns | `Vec<i32>` | Output statistic schema | Keep `i32` for current compatibility; validate count bounds at production sites. |
| Output run manifest chunk identifiers and variant ranges | `i64` | JSON/manifest schema | Keep fixed-width `i64`; convert from `usize` with checked helpers. |
| Output writer row counts | `usize` internally, manifest JSON fixed-width | Memory count and schema field | Keep `usize` internally; convert when serializing/resuming. |
| Runtime exit codes | `i32` | Process contract | Keep `i32`. |
| Python callback counts and telemetry payload counts | `i64` at PyO3 boundary | Python/JSON boundary | Convert from `usize` through checked helpers. |
| NumPy output buffer address and value count | `OutputBufferAddress`, `OutputValueCount` | Raw pointer/buffer boundary | Keep pointer-sized representation quarantined behind wrappers. |

## Boundary Decisions

Internal genotype reader and input alignment APIs should not accept `i64` for
memory indices merely because Python provides signed integers. The binding
layer converts Python sample-index arrays to `usize` before calling native
sample-selection and alignment helpers. Python-facing getters convert native
aligned sample indices back to checked `int64` arrays.

Caller-owned output buffers are explicit:

- `OutputBufferAddress` identifies the raw writable allocation address.
- `OutputValueCount` identifies the number of elements available at that
  address.

These wrappers do not make raw pointers safe by themselves. They make the
unsafe boundary visible and keep raw `usize` pointer values out of higher-level
reader signatures.

## Remaining Audit Items

The integer cast checker is active through `check_rust_architecture`. Current
audited exceptions live in `tooling/debug/integer_cast_allowlist.txt`.

When a new cast is proposed, classify it as one of:

- raw pointer conversion internal to a buffer wrapper;
- float-to-float or float-to-integer conversion required by numerical code;
- SIMD lane extraction or mask count conversion;
- benchmark/test fixture setup;
- boundary conversion that should move to a helper.

Only the last category should be changed mechanically. Test-only casts may stay
local to `tests.rs`; production and benchmark code should use checked
conversion unless the cast is audited.

## Compact Index Buffers

No compact `u32` index buffer is introduced by this audit. The current large
sample-selection surfaces use `usize`, which matches Rust slice indexing and
avoids per-access conversion. A compact representation can be added only after
a benchmark shows a material decode, preprocess, cache, memory-bandwidth, or
end-to-end throughput benefit.

Candidate surfaces for a future benchmark are:

- large sample-selection arrays;
- grouped union sample indices;
- group position maps.

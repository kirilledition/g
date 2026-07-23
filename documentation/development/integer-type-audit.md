# Integer Type Audit

| Status | Applies to | Owner |
| --- | --- | --- |
| Enforced audit | main branch as of 2026-07-10 Rust, PyO3, and JAX integer surfaces | Native runtime maintainers |

This page records the current native integer surfaces and the intended type for
each category. Keep it in sync when changing native input/output contracts,
reader APIs, manifests, or buffer ownership.

## Current Type Map

| Surface | Current type | Classification | Decision |
| --- | --- | --- | --- |
| `ChunkSpec.variant_start_index`, `ChunkSpec.variant_stop_index` | `usize` | Memory index/range | Keep `usize`. |
| BGEN `sample_count`, `variant_count` | `usize` after header parse | Memory count | Keep `usize`; BGEN header `u32` is checked during parse. |
| BGEN and aligned sample selection indices | `usize` internally | Memory index | Keep native; sample-selection indices do not cross PyO3. |
| BGEN variant metadata positions | `Vec<i64>` | Genomic position/output metadata | Keep fixed-width `i64`. |
| BGEN file offsets and block lengths | parser-local checked integers | File format offsets/lengths | Keep checked before snapshot slicing or positioned reads; owned windows are capped at 8 MiB. |
| `ChunkStats` count columns | `Vec<i32>` | Native/JAX/output count contract | Keep `i32`; run preflight bounds the contributing sample count. |
| Output run manifest chunk identifiers and variant ranges | `i64` | JSON/manifest schema | Keep fixed-width `i64`; convert from `usize` with checked helpers. |
| Output writer row counts | `usize` internally, manifest JSON fixed-width | Memory count and schema field | Keep `usize` internally; convert when serializing/resuming. |
| Runtime exit codes | `i32` | Process contract | Keep `i32`. |
| JAX loop limits and candidate capacities | `u32` in host configuration/`RunPlan`, checked `i32` at PyO3 | Host-to-device integer contract | Preserve the unsigned host domain; validate and convert before Python exposure. |
| JAX indices and count reductions | explicit `jnp.int32` | Device index/count contract | Keep `int32` even with JAX x64 enabled; reject shapes and products above the domain. |
| Materialization trait indices | checked `Vec<i32>` at PyO3 | Python/JAX boundary | Convert from engine `usize` at the binding edge. |
| Telemetry counters | `usize` internally, checked `u64` when emitted | JSON/tracing boundary | Keep host counters internal and serialize nonnegative fixed-width snapshots. |
| BGEN decoded genotype allocation | `OwnedGenotypeBuffer` containing `Vec<f32>` or pooled `Vec<u8>` ownership | Rust ownership boundary | Move typed ownership; do not introduce pointer-sized interchange. |

## Boundary Decisions

Internal genotype reader and input alignment APIs use `usize` for memory
indices. The binding converts engine-owned active trait indices to checked
`i32` values when constructing the materialization request; no pointer-sized
integer crosses PyO3.

Owned BGEN decode returns a `DecodedGenotypeBatch`. Its genotype allocation is
a typed vector owned directly or through `PooledPacked8Buffer`; the decoder
initializes reserved spare capacity and publishes the vector length only after
complete success. The API exposes neither an allocation address nor a
pointer-sized value count.

## Enforcement Notes

Workspace Clippy denies truncating, wrapping, and sign-losing casts. The
integer cast checker is active through `check_rust_architecture`. Current
audited exceptions live in `tooling/debug/integer_cast_allowlist.txt`.

When a new cast is proposed, classify it as one of:

- slice-derived SIMD or raw-pointer work, which must stay inside an audited
  unsafe implementation and must not be converted to an integer;
- float-to-float or float-to-integer conversion required by numerical code;
- SIMD lane extraction or mask count conversion;
- benchmark/test fixture setup;
- boundary conversion that should move to a helper.

Only the last category should be changed mechanically. Test-only casts may stay
local to `tests.rs`; production and benchmark code should use checked
conversion unless the cast is audited.

These cast checks do not detect structural boundary mistakes such as a new
serialized `usize`. Review must still inspect Serde, tracing, PyO3, NumPy, and
JAX integer surfaces against this table.

## Compact Index Buffers

No compact `u32` index buffer is introduced by this audit. The current large
sample-selection surfaces use `usize`, which matches Rust slice indexing and
avoids per-access conversion. A compact representation can be added only after
a benchmark shows a material decode, preprocess, cache, memory-bandwidth, or
end-to-end throughput benefit.

The remaining candidate surface for a future benchmark is large
sample-selection arrays.

# Integer Policy

| Status | Applies to | Owner |
| --- | --- | --- |
| Active development contract | main branch as of 2026-07-10 Rust, PyO3, and JAX integer boundaries | Native runtime maintainers |

Native Rust code uses different integer types for different contracts. The goal
is not one project-wide integer type. The goal is to keep `usize` in Rust
memory indexing code, fixed-width integers at persistence and Python
boundaries, and checked conversions where values cross those boundaries.

## Policy

1. In-memory Rust indexing, chunk lengths, matrix dimensions, queue sizes, and
   slice offsets use `usize`.
2. Persistent schemas do not use `usize`.
3. Python, JSON, TOML, manifest, Arrow, and Parquet numeric fields use
   fixed-width integers.
4. Host configuration capacities are unsigned fixed-width values. Use
   `NonZeroU32` when zero is invalid; convert to the consumer's integer domain
   with an explicit bound check.
5. JAX indices, loop counters, count reductions, and output count arrays use
   signed `i32`. Enabling JAX x64 changes floating-point capability; it does
   not change the index policy.
6. Genomic positions use `i64` unless a file format requires a narrower or
   wider type.
7. File byte offsets use `u64` or parser-local checked `usize` after validating
   the mapped buffer bounds.
8. Large stored index arrays may use `u32` only after validating the maximum
   index and benchmarking the memory-bandwidth benefit.
9. Raw pointer addresses are not public interchange values. Public buffer
   boundaries use owned typed containers or borrowed slices. Internal
   unsafe/SIMD pointers derive from slices and never round-trip through
   `usize`.
10. Narrowing or sign-changing conversions use `TryFrom` or a named boundary
   helper.
11. Unchecked `as` casts are not used for integer-to-integer conversion or
    pointer round trips. Audited float conversions may retain them where the
    approximation is part of the numerical algorithm.

The Rust crates declare:

```rust
#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");
```

This makes the supported target assumption explicit. It does not make `usize`
a stable storage or interchange type.

Integer width follows storage and execution cost, not the smallest value seen
in ordinary runs. A scalar thread count gains no meaningful memory or cache
benefit from `u8`, while it would require widening for Rayon, OS, and slice
APIs and would impose an arbitrary limit. Narrow integers are appropriate for
large arrays or stable schemas only when the range is part of the contract and
the footprint reduction is material.

## Boundary Conversion

Checked conversions live at the boundary that owns the source value. A local
helper is appropriate when one boundary performs the same validated conversion
more than once. Do not create forwarding modules or project-wide conversion
frameworks for a single call site.

Correctness paths use `checked_add`, `checked_sub`, and `checked_mul`; saturation
is reserved for explicitly lossy telemetry counters. Native run preparation
proves configured `u32` loop/capacity values, sample counts, trait counts,
chunk sizes, flattened trait-by-variant lanes, candidate capacities, and
padded Firth batches fit `i32` before calling
Python. Python static shape integers originate from those checked values, and
every JAX index-producing operation has an explicit `int32` result. NumPy
arrays crossing PyO3 have explicit fixed-width element types. Fixed-capacity
mask compaction constructs `int32` indices directly instead of producing a
default `int64` `nonzero` result and narrowing it afterward.

Float-to-integer timestamp conversion is isolated to telemetry formatting and
falls back when Chrono
rejects the resulting timestamp.

## Typed Buffer Boundaries

`BgenReadSession::decode_variant_major_batch` returns a
`DecodedGenotypeBatch` whose `OwnedGenotypeBuffer` owns either a `Vec<f32>` or
`Vec<u8>`. The decoder reserves the final capacity, initializes the vector's
spare capacity, and sets its length only after every requested output position
has been written successfully. No address or value-count wrapper crosses from
`g-genotype` to `g-engine`.

## Review Checklist

- New memory indices and lengths are `usize`.
- New output, manifest, JSON, and Python fields are fixed-width integers.
- New JAX indices and integer reductions explicitly request `int32` even when
  JAX x64 is enabled.
- New JAX shapes and derived products are bounded natively before device work.
- New conversions that can overflow or change sign are checked.
- Raw pointer-sized values do not appear in public genotype, engine, or binding
  APIs; internal pointer use remains slice-derived and locally audited.
- Count vectors stay in the documented output schema type.
- Any retained unchecked cast has a local reason and is not a casual boundary
  conversion.

## Enforcement

The Rust workspace enables these Clippy cast lints:

```toml
cast_possible_truncation = "deny"
cast_possible_wrap = "deny"
cast_sign_loss = "deny"
```

Local exceptions use a narrowly scoped `allow` with an audited reason. The
`check_rust_architecture` development check also scans `crates/` and
`src/binding/` for integer `as` casts outside `tests.rs` files. Retained casts
must either be replaced with checked conversion or listed in
`tooling/debug/integer_cast_allowlist.txt` with an audited reason.

The current architecture-check allowlist is limited to timestamp formatting,
where finite floating-point seconds are intentionally split into whole seconds
and nanoseconds for `chrono`.

Clippy and cast scanning cannot identify a newly serialized or PyO3-exposed
`usize`. The boundary-type and JAX-capacity checklist remains a required part
of review; production boundary types are also made concrete in `_core.pyi`.

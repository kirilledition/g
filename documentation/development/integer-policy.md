# Integer Policy

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; development contract | main branch as of 2026-07-09 Rust native integer boundaries | Native runtime maintainers |

Native Rust code uses different integer types for different contracts. The goal
is not one project-wide integer type. The goal is to keep `usize` in memory
indexing code, fixed-width integers at persistence and Python boundaries, and
checked conversions where values cross those boundaries.

## Policy

1. In-memory Rust indexing, chunk lengths, matrix dimensions, queue sizes, and
   slice offsets use `usize`.
2. Persistent schemas do not use `usize`.
3. Python, JSON, TOML, manifest, Arrow, and Parquet numeric fields use
   fixed-width integers.
4. Output statistic count columns use the output schema type. Current native
   chunk statistics keep count columns as signed `i32` for compatibility with
   downstream output and JAX-facing expectations.
5. Genomic positions use `i64` unless a file format requires a narrower or
   wider type.
6. File byte offsets use `u64` or parser-local checked `usize` after validating
   the mapped buffer bounds.
7. Large stored index arrays may use `u32` only after validating the maximum
   index and benchmarking the memory-bandwidth benefit.
8. Raw pointer addresses use `usize` only behind caller-owned buffer wrappers
   such as `OutputBufferAddress` and `OutputValueCount`.
9. Narrowing or sign-changing conversions use `TryFrom` or a named boundary
   helper.
10. Unchecked `as` casts are limited to audited hot paths, raw pointer
    conversion internals, and float conversions where the approximation is part
    of the numerical algorithm.

The Rust crates declare:

```rust
#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");
```

This makes the supported target assumption explicit. It does not make `usize`
a stable storage or interchange type.

## Boundary Helpers

PyO3-facing conversion helpers live under `src/binding/convert/`. They attach
field names to overflow and sign errors and keep Python integer conversion out
of domain crates.

Use these helpers when converting Python or Python-derived `i64` values into
native indices and counts. Do not add broad conversion frameworks for normal
internal arithmetic.

## Raw Buffers

BGEN delivery writes into caller-owned Python/NumPy buffers. Public genotype
reader methods should take explicit buffer wrappers:

```rust
OutputBufferAddress
OutputValueCount
```

The unsafe pointer-to-slice conversion stays inside genotype buffer/decode
modules. Binding code may create these wrappers from NumPy mutable slices after
shape and contiguity validation.

## Review Checklist

- New memory indices and lengths are `usize`.
- New output, manifest, JSON, and Python fields are fixed-width integers.
- New conversions that can overflow or change sign are checked.
- Raw pointer-sized values do not appear in high-level engine or binding APIs
  except as explicit buffer wrappers.
- Count vectors stay in the documented output schema type.
- Any retained unchecked cast has a local reason and is not a casual boundary
  conversion.

## Enforcement

The Rust workspace enables these Clippy cast lints:

```toml
cast_possible_truncation = "warn"
cast_possible_wrap = "warn"
cast_sign_loss = "warn"
```

The `check_rust_architecture` development check also scans `crates/` and
`src/binding/` for integer `as` casts outside `tests.rs` files. Retained casts
must either be replaced with checked conversion or listed in
`tooling/debug/integer_cast_allowlist.txt` with an audited reason.

The current allowlist is limited to timestamp formatting, where finite
floating-point seconds are intentionally split into whole seconds and
nanoseconds for `chrono`.

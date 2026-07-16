# Public API

## This crate owns

Canonical, allocation-preserving data-plane contracts shared between native
genotype producers, the engine, and output consumers.

## Public contracts

The exact opened BGEN source identity, compact shared variant metadata storage
and slices, output-facing chunk statistics, packed nullable `f32` columns, and
the canonical raw-DEFLATE member alignment used by slab producers and consumers.

## This crate must not expose

BGEN readers or decoding policy, compute-only statistics, Arrow arrays or
Parquet writers, engine scheduling, runtime policy, or Python bindings.

## Performance constraints

Metadata slices retain one immutable dictionary-coded store through
`Arc` plus a range. Nullable columns retain dense values and an Arrow-compatible
packed validity bitmap. Consumers must move or share these allocations; they
must not introduce row-wise mirrors or crate-boundary copies.

## Allowed downstream users

`g-genotype`, `g-genotype-cuda`, `g-engine`, and `g-output`.

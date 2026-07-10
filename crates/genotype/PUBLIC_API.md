# Public API

## This crate owns

BGEN reader implementation, chunk planning, and genotype preprocessing summaries.

## Public types

BGEN reader/error/profile types, `GenotypeError`/`GenotypeResult`, chunk specs/stats,
variant metadata columns, and explicit caller-owned output buffer wrappers.

## Public functions

BGEN reader methods for opening files, decoding batches, binding samples, reading metadata,
profiling, trusted-mode validation, and reader-owned chromosome-homogeneous chunk planning.
Variant-major preprocessing summaries and the BGEN decode tile setter are root
exports for native binding adapters. The setter is a validated
process-global runtime policy: runtime applies it before decoding begins, and
the genotype crate rejects zero-sized tiles.

## This crate must not expose

Sample/phenotype alignment, output schema/writers, runtime/JAX policy, engine
scheduling, callback queues, PyO3 classes, or public `debug`, `ffi`, or
`internal` modules.

## Performance constraints

Batch-oriented APIs only. Preserve caller-owned output buffers behind explicit FFI
address/count wrappers and avoid per-variant public calls or JSON conversion.

## Allowed downstream users

`g-engine`, root PyO3 facade, and genotype benches through the root facade.

# Public API

## This crate owns

BGEN reader implementation, chunk planning, and genotype preprocessing summaries.

## Public types

BGEN reader/error/profile types, `GenotypeError`/`GenotypeResult`, chunk specs,
compute statistics, and explicit caller-owned output buffer wrappers. Shared
metadata and output columns are owned by `g-genotype-contracts` and are not
re-exported here.

## Public functions

BGEN reader methods for opening files, decoding batches, binding samples, reading metadata,
profiling, trusted-mode validation, trusted output-only dosage decoding after validation,
and reader-owned chromosome-homogeneous chunk planning.
Variant-major preprocessing summaries and the BGEN decode tile setter are used
by `g-engine`. The setter is validated process-global state applied by the
engine before decoding begins; the genotype crate rejects zero-sized tiles.

## This crate must not expose

Sample/phenotype alignment, output schema/writers, runtime/JAX policy, engine
scheduling, callback queues, PyO3 classes, or public `debug`, `ffi`, or
`internal` modules.

## Performance constraints

Batch-oriented APIs only. Preserve caller-owned output buffers behind explicit
address/count wrappers, produce immutable contract metadata slices, store variant IDs in a
contiguous UTF-8 arena, dictionary-code repeated chromosome/allele text, and
avoid per-variant heap ownership, public calls, or JSON conversion. The trusted
output-only dosage path must not allocate or reduce chunk statistics. Normal
preprocessing reuses the dosage-sum allocation for the final genotype-mean
vector after INFO and sparse-candidate calculations; output observation counts
remain output-only and are not cloned or transferred to JAX. Nullable INFO
statistics use a contiguous `f32` value column and an Arrow-compatible packed
validity bitmap instead of per-value `Option` storage.

## Allowed downstream users

`g-engine` and genotype benches through the root facade.

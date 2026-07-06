# Public API

## This crate owns

Genotype source contracts, BGEN reader core, chunk planning, and genotype preprocessing summaries.

## Public types

BGEN reader/error/profile types, `GenotypeError`/`GenotypeResult`, chunk specs/stats, variant metadata columns, and genotype reader trait.

## Public functions

BGEN decode knobs, chromosome-homogeneous chunk planning, row/variant-major preprocessing, and chunk-stat summary builders.

## This crate must not expose

Sample/phenotype alignment, output schema/writers, runtime/JAX policy, engine scheduling, callback queues, or PyO3 classes.

## Performance constraints

Batch-oriented APIs only. Preserve caller-owned output buffers and avoid per-variant public calls or JSON conversion.

## Allowed downstream users

`g-engine`, root PyO3 facade, and genotype benches.

# Public API

## This crate owns

Sample, phenotype, covariate, prediction, and phenotype-group alignment.

## Public types

`InputError`/`InputResult`, alignment input/result DTOs, grouped/multi-trait aligned data, sample-key mode, sample alignment
errors, prediction sources, chromosome prediction matrix payloads, and prediction errors.

## Public functions

Single/multi/grouped alignment from in-memory sample identifiers or sample files, phenotype compute-group resolution, grouped union-sample planning, and LOCO path resolution.

## This crate must not expose

BGEN decoding internals, output writing, JAX device handling, engine scheduler state, callback queues, or PyO3 classes.

## Performance constraints

Keep sample matrix ownership explicit. Return aligned vectors/matrices in stable row-major layout without hidden cross-crate JSON.

## Allowed downstream users

`g-engine` and root PyO3 facade.

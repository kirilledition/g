# Public API

## This crate owns

Sample, phenotype, covariate, prediction, and phenotype-group alignment.

## Public types

`InputError`, the opaque sample-identifier payload, phenotype-group load
request, aligned phenotype groups, chromosome prediction matrices, and
prediction errors. Sample-key enums remain owned by `g-plan`.

## Public functions

Load sample identifiers, align phenotype groups, plan grouped union-sample
positions, and resolve LOCO paths. Prediction sources expose post-resume use
planning and move-only chromosome matrices; file indexes and alignment recipes
remain private.

## This crate must not expose

BGEN decoding internals, output writing, JAX device handling, engine scheduler state, callback queues, or PyO3 classes.

## Performance constraints

Keep sample matrix ownership explicit. Index each selected LOCO file once per
canonical path, retaining only its path, source sample count, and chromosome
row offsets and raw-row digests after group alignment is built. Identical
headers share one loader-only identifier index and one alignment recipe per
group; identity-aligned groups do not allocate an index vector. File size,
nanosecond mtime, and row digests guard deferred reads against changes after
indexing. Resume planning checks only chromosome blocks with pending output.
The engine resolves the prediction list once; input consumes that borrowed
catalog rather than reparsing it for each group, and the same catalog drives
output-manifest fingerprints.
The source then reads, parses, finite-validates, and aligns one chromosome
directly into its final trait-major matrix when the engine reaches it. The
final allocation transfers to the backend. Only repeated noncontiguous
chromosome blocks retain a counted matrix for clone fallback until the final
use. Do not restore eager all-chromosome, raw-value, or per-trait aligned
caches. Return stable layouts without cross-crate JSON.

## Allowed downstream users

`g-engine` only.

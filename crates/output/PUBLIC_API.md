# Public API

## This crate owns

Output run preparation, manifest compatibility, resume validation, chunk writing, and finalization.

## Public types

Output writer session, native chunk handle/stats, variant metadata columns, manifest fingerprints,
output run path/preparation/initialization payloads, committed chunk records, output format,
resume mode, and writer error.

## Public functions

Prepare/init output runs, validate/extend/load/write manifests, scan/repair committed chunks, write chunks through sessions, and finalize output chunks.

## This crate must not expose

BGEN internals, sample alignment internals, engine scheduler queues, runtime telemetry sinks, or PyO3 classes.

## Performance constraints

Write chunk batches through handles and array views. Do not serialize Rust crate-to-crate hot-path data through JSON except manifest edges.

## Allowed downstream users

`g-engine` and root PyO3 facade.

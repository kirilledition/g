# Public API

## This crate owns

Output run preparation, manifest compatibility, resume validation, chunk writing, and finalization.

## Public types

Output writer session, native chunk handle/stats, variant metadata columns, output run
path/preparation/initialization payloads, output format, resume mode, and `OutputError`/`OutputResult`.
Raw manifest header, fingerprint, strict resume repair, and statistic-shape
helpers are root exports for debug and native binding adapters.

## Public functions

Prepare/init output runs, validate manifest and resume compatibility, write chunks through sessions,
create/finish session batches, finalize output chunks, extend run manifests,
scan committed chunks, and build manifest headers/fingerprints.

## This crate must not expose

BGEN internals, sample alignment internals, engine scheduler queues, runtime
telemetry sinks, PyO3 classes, or public administrative submodules.

## Performance constraints

Write chunk batches through handles and array views. Do not serialize Rust crate-to-crate hot-path data through JSON except manifest edges.

## Allowed downstream users

`g-engine` and root PyO3 facade.

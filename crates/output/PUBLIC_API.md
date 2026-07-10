# Public API

## This crate owns

Output run preparation, manifest compatibility, resume validation, chunk writing, and finalization.

## Public types

Output writer session, native chunk handle/stats, variant metadata columns, output run
path/preparation payloads, output format, resume mode, and `OutputError`/`OutputResult`.
The current manifest-header input, file fingerprints/cache, and typed statistic
slice bundle are root exports for engine and native binding adapters.

## Public functions

Prepare/init output runs, validate manifest and resume compatibility, write chunks through sessions,
create/finish session batches, extend run manifests, and build cached manifest
headers. Finalization is owned by `OutputWriterSession::finish`.

## This crate must not expose

BGEN internals, sample alignment internals, engine scheduler queues, runtime
telemetry sinks, PyO3 classes, or public administrative submodules.

## Performance constraints

Write chunk batches through handles and array views. Do not serialize Rust crate-to-crate hot-path data through JSON except manifest edges.
Persisted row and chunk counts use checked signed 64-bit arithmetic because the
manifest contract is JSON integer based; overflow must fail before mutation.

## Allowed downstream users

`g-engine` and root PyO3 facade.

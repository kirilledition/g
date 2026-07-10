# Public API

## This crate owns

Output run preparation, manifest compatibility, resume validation, and chunked Parquet writing.

## Public types

`OutputManager`, `OutputDeliveryState`, `CompletedOutputRun`, output writer
sessions, native chunk metadata/statistics, typed manifest preparation inputs
and fingerprints, and `OutputError`. Resume mode, compression, and statistic
dtype use the canonical `g-plan` types directly.

## Public functions

Open and initialize output runs through `OutputManager`, select delivery state,
write trait-major chunks, and complete, interrupt, or abort the owned writer
sessions. Manifest construction and resume validation remain internal details.

## This crate must not expose

BGEN internals, sample alignment internals, engine scheduler queues, runtime
telemetry sinks, PyO3 classes, or public administrative submodules.

## Performance constraints

Write chunk batches through handles and array views. Do not serialize Rust
crate-to-crate data through JSON; JSON exists only inside manifest persistence.
Persisted row and chunk counts use checked signed 64-bit arithmetic because the
manifest contract is JSON integer based; overflow must fail before mutation.

## Allowed downstream users

`g-engine`.

# Public API

## This crate owns

Output run preparation, manifest compatibility, resume validation, and chunked Parquet writing.

## Public types

`OutputManager`, `OutputDeliveryState`, `CompletedOutputRun`, opaque output
writer sessions, native chunk handles, typed manifest inputs/fingerprints,
trait-major statistic batches, and `OutputError`. Strict on-disk reconciliation
is fixed policy owned by this crate.

## Public functions

Open and initialize output runs through `OutputManager`, select delivery state,
write validated trait-major chunks, and complete, interrupt, or abort the owned
run. Individual writer-session lifecycle methods remain crate-private.

## This crate must not expose

BGEN internals, sample alignment internals, engine scheduler queues, runtime
telemetry sinks, PyO3 classes, or public administrative submodules.

## Performance constraints

Write chunk batches through handles and array views. Do not serialize Rust
crate-to-crate data through JSON; JSON exists only inside manifest persistence.
Consume canonical `g-genotype-contracts` columns directly; output must not
depend on the BGEN implementation crate or introduce adapter mirrors.
Fresh multi-trait chunks select all writer lanes without constructing an
identity-index vector; only partially resumed chunks carry explicit indices.
Each delivered chunk constructs one output-owned Arrow metadata handle shared
by its trait writers; only sample-dependent statistics are materialized per
phenotype group.
Metadata-handle construction rejects string columns beyond Arrow's 32-bit
`Utf8` offset limit before lazy writer-side array construction can panic.
Normal writes do not collect detailed timers or traverse Arrow memory; that
instrumentation is enabled only for explicit stage timing/profile modes.
Binary correction codes remain `uint8` through device, host, and Arrow staging;
the writer maps them to the existing method/status dictionaries only when it
builds the final record batch.
Parquet parts use version-2 delta fallbacks, 16,384-row internal write batches,
and byte-stream-split encoding for every floating-point result column. These
physical encodings must not change the logical output schema.
Persisted row and chunk counts use checked signed 64-bit arithmetic because the
manifest contract is JSON integer based; overflow must fail before mutation.
The fixed `(FID, IID)` input invariant is not serialized as a configurable
manifest field; input fingerprints and aligned-sample fingerprints cover the
concrete files and selected cohort. Runtime compute-group fingerprints are
required rather than nullable, including the aligned phenotype matrix digest
that prevents mixed-trait resume.

## Allowed downstream users

`g-engine` and the private root PyO3 `AssociationBackend` adapter. The adapter
constructs the output-owned statistic batch directly without invoking writer
services.

# Public API

## This crate owns

Python-free GWAS orchestration: association backend execution, bounded batch
scheduling, consuming run preparation/execution, BGEN/input ownership,
preflight, output lifecycle, resume, packed8-versus-dosage negotiation, and terminal
writer completion/abort policy. Terminal rendering belongs to `g-runner`.

## Public types

`RunHooks`, `EngineRunError`, `AssociationBackend`, completed phenotype
artifacts, and engine-owned backend envelopes used by the private PyO3 JAX
adapter. Genotype, input, and output payloads remain owned by their domain
crates and are referenced directly rather than mirrored or re-exported. Run
preparation/execution state, upstream error types, and scheduler reports remain
internal implementation details.

## Public functions

Invoke the coarse coordinated run entry point used by `g-runner`.

## This crate must not expose

Scheduler internals, raw BGEN/input/output services, writer sessions, buffer
pools, planning helpers, callback types, PyO3 classes, Python objects, or JSON
compute payloads.

## Performance constraints

Compute boundaries remain chunk-oriented. Matrices have explicit layouts,
decoded and device-result queues are bounded, active traits and output
precision are applied before device-to-host transfer, and each backend-bound
genotype, phenotype, covariate, and single-use LOCO allocation moves into NumPy
ownership without a full-buffer copy. One compute worker, one materialization
worker, and one bounded channel set serve each active delivery. Drained
transitions explicitly release and acknowledge each replaced backend state
before its successor is built. Resume planning drops fully committed groups
before sample selection. Resume-aware prediction use counts drop unused
chromosome matrices and transfer the final remaining allocation. Avoid
per-variant dynamic dispatch,
hidden serialization, repeated prediction-list parsing, and clone-heavy
adapters. Shared metadata and output columns come directly from
`g-genotype-contracts`, with no engine-owned mirror. Device batches receive the
native genotype mean directly; output observation counts are not duplicated
into the compute payload. Binary
correction codes use their natural one-byte domain until output maps them to
dictionary labels.

## Allowed downstream users

`g-runner` and the root native JAX backend adapter.

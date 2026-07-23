# Public API

## This crate owns

Python-free GWAS orchestration: association backend execution, bounded batch
scheduling, consuming run preparation/execution, BGEN/input ownership,
preflight, output lifecycle, resume, packed8-versus-dosage negotiation, and
terminal writer completion/abort policy. Terminal rendering belongs to
`g-runner`.

## Public types

`RunHooks`, `EngineRunError`, `AssociationBackend`, completed phenotype
artifacts, and the engine-owned group and materialization envelopes used by the
private PyO3 JAX adapter. Backend capability and transfer-preparation enums are
owned here with the backend lifecycle. The validated association implementation
state is also engine-owned: its constructors admit only valid requested,
effective, FFI-target, and fallback combinations, and its accessors separate
stable implementation and fallback-reason names from diagnostic-only free text.
Every production JAX backend retains its exact observed JAX and JAXlib versions.
Genotype batches, compressed transfer descriptors, and raw statistics remain
owned by `g-genotype`. Input and output payloads remain owned by their domain
crates and are referenced directly rather than mirrored or re-exported. Run preparation/execution state, upstream error
types, and scheduler reports remain internal implementation details.

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
per-variant dynamic dispatch, hidden serialization, repeated prediction-list
parsing, and clone-heavy adapters. Shared metadata and output columns come
directly from `g-genotype-contracts`, with no engine-owned mirror. Device
batches receive the native genotype mean directly; output observation counts
are not duplicated into the compute payload. Binary correction codes use their
natural one-byte domain until output maps them to dictionary labels.

The backend advertises compressed-delivery capability once. The engine selects
raw-DEFLATE only for a requested packed8 run on a compatible zlib source and
derives its fixed slab from the actual resume-aware chunk plan. The pipeline
borrows group state so sample selection is prepared once and reused by every
transfer without another allocation or reference count. Host-decoded statistics
travel through the backend's opaque
batch lifecycle without cloning; compressed batches materialize exact integer
summaries, which the genotype crate validates and converts on the
materialization worker before any writer sees the batch.

The root JAX adapter captures exact observed JAX/JAXlib versions when it builds
the association implementation state. Effective raw CUDA carries the exact FFI
target supplied by `g-compute-cuda`; JAX and fallback states cannot carry a
target. Run/output integration must read this state before opening output and
project only requested implementation, effective implementation, typed
fallback reason, exact JAX/JAXlib versions, and the FFI target ABI name when raw
CUDA is effective into compatibility state.
Host-specific diagnostic detail, device ordinal, UUID, and description remain
non-hashed. The pre-release contract version remains `0`; execution-plan hashing
alone carries this compatibility state.

## Allowed downstream users

`g-runner` and the root native JAX backend adapter.

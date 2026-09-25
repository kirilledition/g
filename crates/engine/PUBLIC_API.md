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
owned here with the backend lifecycle; genotype batches, compressed transfer
descriptors, and raw statistics remain owned by `g-genotype`. Input and output
payloads remain owned by their domain crates and are referenced directly rather
than mirrored or re-exported. Run preparation/execution state, upstream error
types, and scheduler reports remain internal implementation details.

`RunHooks` associates the backend error type and can recover a typed host
interruption from it. The engine performs this classification before choosing
graceful output flushing or ordinary failure abort; it never identifies signals
from error-message text.

## Public functions

Invoke the coarse coordinated run entry point used by `g-runner`. Its backend
initializer receives the canonical plan only after native preparation and strict
resume reconciliation succeed. The runner supplies an already initialized backend
for fresh analyses and a deferred initializer for explicit resumes. Fully committed
runs retain normal delivery,
progress, interruption, and output completion handling without invoking the
initializer. Initialization failures abort prepared output; typed interruptions
flush it and record interruption metadata.

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

Linear GPU backends can additionally advertise immutable shared-source batches.
This is a stable capability promise: source preparation or selection returning
`None` after admission is a typed delivery failure, not a fallback after output
may have been accepted. The engine pairs at most two compressed packed8 groups,
walks the ordered union of pending chunk ranges, and shares source decode only
when both groups need the same range. Each group keeps its own sample selection,
statistics, chromosome transitions, LOCO use counts, progress, and writer lanes.
Each active group submits one private selection before either pipeline is
drained. Both pipelines drain before source release and the next chunk, with an
interruption check between drains that preserves already accepted output.
At most two existing pipelines are resident; output writer-pool bounds do not
change. On failure, pipelines abort and join before the source release hook,
then group release hooks run.

Admission uses checked arithmetic and bounds retained full-source pairs plus
statuses to 128 MiB and combined linear group/chromosome/selection array payloads
to 64 MiB. These are retained-array limits, not total GPU-memory limits. The two
private selected input/result sets can overlap in addition to the source and
state payloads; compiled workspaces, preparation temporaries, and allocator
reservation remain separate.
Unsupported modes, singleton groups, disjoint pending plans, incompatible input,
and oversized geometries retain ordinary delivery. Fully committed resumes
still initialize no backend or source state.

## Allowed downstream users

`g-runner` and the root native JAX backend adapter.

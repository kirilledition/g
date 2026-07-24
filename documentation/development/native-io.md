# Native I/O

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; development contract | Rust I/O and output runtime as of 2026-07-22 | Native runtime maintainers |

Native I/O owns the parts of the hot path that should not depend on Python
DataFrame libraries: BGEN decode, sample/covariate/phenotype alignment, chunk
delivery, output writing, manifest handling, and resume.

Integer boundary decisions for native I/O are maintained in
[Integer Policy](integer-policy.md) and [Integer Type Audit](integer-type-audit.md).

## Source Map

| Path | Responsibility |
| --- | --- |
| `crates/genotype/src/` | BGEN mmap/index/decode/preprocess and genotype source planning. |
| `crates/input/src/` | Sample, phenotype, covariate, prediction-list, and LOCO prediction alignment. |
| `crates/output/src/` | Parquet dataset parts, manifests, resume, and bounded writer sessions. |
| `crates/engine/src/` | Production coordination of input, genotype delivery, output, telemetry facts, and cleanup. |
| `src/binding/` | NumPy/Python adaptation at the JAX backend boundary; no I/O policy. |

## BGEN Contract

The supported public genotype input is Layout 2 BGEN, including uncompressed,
zlib, and BGEN v1.3 Zstandard variant blocks. Native code owns:

- index/open path;
- sample-count validation against identities loaded from the required Oxford
  `.sample` file;
- variant metadata;
- dosage decode;
- missing-value representation;
- cached no-missing diploid packed8 compatibility validation and fast paths;
- chunk delivery in variant-major shape.

Python/JAX kernels should receive already aligned dosage or validated packed8
chunks and metadata, not parse file formats.

The BGEN index publishes variant metadata only through the validated
`g-genotype-contracts` constructors. Store construction performs one-time,
linear, always-on validation of parallel columns, identifier offsets, UTF-8
boundaries, and dictionary codes. Chunk slicing then performs only constant-time range
validation. A parser-created invariant failure is surfaced as contextual
`BgenError::InvalidFormat`; there is no public unchecked metadata constructor.

Optional `.bgi` input is deliberately deferred. The current runtime must remain
fully functional without an external index; future support is tracked in the
[Roadmap](roadmap.md) and must retain exact BGEN/index identity validation.

## Alignment Contract

Native alignment resolves:

- identities loaded from the required Oxford `.sample` file;
- phenotype and covariate rows;
- Step 1 prediction rows;
- fixed `(FID, IID)` identity matching;
- complete-case rows for phenotype/covariate data;
- per-phenotype or complete-case multi-phenotype grouping.

Changing alignment behavior is a public input contract change. Update
[Input Files](../public/input-files.md), tests, and parity expectations.

## Output Contract

Native output writes one result representation:

| Dataset | Directory | File names |
| --- | --- | --- |
| Parquet | `parts/` | `part_<first>[_<last>].parquet` |

Each attempt stores phenotype directories under
`attempts/<attempt>/<phenotype-output-name>/`. Those directories contain
`parts/`, `commits/`, `run_manifest.json`, `effective_config.toml`, and an
optional `output_stage_timings.json`. The parts directory is the completed
dataset; output does not require a consolidation pass.

Output planning is read-only. The engine resolves every phenotype run path and
inspects the immutable lineage before input preparation without creating the
output root. Initialization publishes the durable no-replace owner claim,
repeats validation, publishes genesis or successor authority, and only then
creates attempt-specific state.

## Manifest And Resume Contract

The append-only `.g-output` lineage is the resume authority.
`run_manifest.json` is attempt state whose SHA-256 is bound by a terminal
claim. Pre-release manifest schema version `0` stores prepared-run fields,
input fingerprints, and Parquet writer settings in one canonical
`execution_plan` object plus `execution_plan_hash`. Top-level lifecycle state
binds the attempt, status, receipts, and committed chunks. The pre-release
Parquet output schema is version `0`; `INFO` is nullable when its
expected-variance denominator is undefined.

Each compute group records a required fingerprint of its trait-major aligned
phenotype matrix, including phenotype names, shape, and float32 values. Resume
therefore rejects phenotype-value changes even if a source path changes between
input loading and manifest preparation.

Manifest construction records `association_backend.kind` from the
engine-resolved delivery format so resume and review tooling can distinguish
`jax_dosage` and `jax_packed8`; it is not a `RunPlan` field.

Resume validates each immutable receipt against its part's embedded transaction
footer, canonical output schema, raw byte size, and freshly computed SHA-256.
It checks every exact chunk binding against the canonical chromosome-aware BGEN
chunk plan and rejects duplicate or missing coverage. Parts without current
metadata are rejected rather than reconstructed from result columns. A
verifiable final part that exists before its receipt because of a process crash
can be reconciled only for a nonterminal attempt or pending terminal claim.
BGEN compatibility is bound to the exact opened file's device, inode, size,
modification time, and change time.

Compatibility validation must fail loudly on mismatched result-affecting inputs
or output schema assumptions.

The run root carries an append-only `.g-output` lineage with one immutable
genesis record. Each attempt has one immutable outcome slot. A terminal claim
and an exact nonterminal recovery claim are tagged alternatives published to
that same slot, so an old owner cannot publish a terminal after a takeover
claim wins. A terminal claim becomes durable terminal authority only after its
named manifests are materialized and a separate immutable finalization record
binds the claim SHA-256. Interrupted and failed terminals may later have one
immutable normal-resume successor bound to the finalized terminal outcome's
SHA-256. A completed terminal remains the immutable leaf and is fully verified
read-only on resume.

Records are written and synchronized under unique temporary names, then
published with a same-directory hard-link no-replace operation and a directory
synchronization. There is no authoritative mutable `HEAD`; readers traverse
exact-recovery claims or hash-bound normal successors from genesis and reject
incompatible dual-state artifacts. After owner acquisition, the manager first
reserves a stable attempt identifier and creates only that claim's private
diagnostics directory. Genesis or successor publication later makes the same
attempt identifier authoritative; failed pre-activation claims durably remove
their unreferenced staging. The coordinator uses the deferred activation API:
failures before authority publication return a typed rollback capability
instead of releasing ownership immediately. The runner closes claim-scoped
timing, telemetry, and logging before consuming that capability, so no
contender can sweep diagnostics while the prior session is still open. Dropping
the capability fails closed by leaving ownership Active until an exact external
fence; the ordinary no-session activation API rolls it back immediately.
Completed read-only resumes use the same deferred boundary: their
claim-specific diagnostics are removed before their owner release, while the
completed attempt payload remains unchanged.

The first owner claim is a permanent immutable record at
`.g-output/session.claim.json`. Its current authority is found by traversing
immutable predecessor slots in `.g-output/owner-transitions/`. Each slot can
contain exactly one graceful release, fenced takeover, or reacquisition
transition, so release and takeover contend on the same compare-and-set point.
A second owner is rejected before diagnostics or attempt authority mutation.
Normal terminal paths append a Released leaf only after terminal durability.
Process death leaves an Active leaf and automatic recovery fails closed. There
is deliberately no timeout, PID/liveness guess, authority unlink, or
unlink-on-open behavior. An externally fenced restart may provide the exact
current Active identifier through `fenced_owner_claim_id`; a mismatch, an
absent root, or a historical predecessor fails without publishing a
transition. A match authorizes a no-replace fenced-takeover transition from
that exact predecessor. The root and every historical transition remain
immutable evidence.

A process death before an authority-record hard link may leave a uniquely named
`.session.claim.json.*.tmp` candidate. Those files are preserved for
inspection but are not authority, are never promoted or removed
automatically, and do not make an otherwise fresh root occupied. A death after
the root or transition hard link leaves both a typed Active authority and its
candidate link; the Active leaf continues to block until externally fenced.

The deployed BeeGFS mount failed both Rust `File::try_lock` and POSIX `fcntl`
cross-node exclusion, so neither is a production correctness mechanism.
External recovery must fence the recorded host/process before publishing a
takeover from its exact Active identifier. Manual deletion is unsupported.
Hard-link no-replace visibility and owner-transition contention must be
qualified cross-node.
The current Gauss qualification evidence is recorded in
[Output Transaction BeeGFS Qualification](output-transaction-beegfs-qualification.md).

The hard-link no-replace operation is the durable authority linearization
point. Its same-filesystem, cross-node visibility and `AlreadyExists` behavior
must be qualified on the deployed BeeGFS mount before production use. A failed
qualification is a release blocker; the current Gauss mount passed.

The canonical chunk plan is ordered, contiguous from zero, and SHA-256 bound.
Part footer metadata repeats run-set, producing-attempt, phenotype,
execution-plan, chunk-plan, part, receipt-name, and exact chunk geometry.
Because a raw file digest cannot be embedded in the same bytes that it hashes,
the immutable receipt repeats that footer identity and adds the final raw byte
size and SHA-256. Resume requires footer/receipt equality and a fresh hash of
the raw Parquet bytes.

## Testing

Native I/O changes usually need tests in:

- Rust unit tests under `crates/genotype/src/`, `crates/input/src/`, or `crates/output/src/`;
- integration or pipeline coverage in the owning Rust crate when backend delivery or writer sessions change.

Output contract changes also require [Output Files](../public/output-files.md)
and [Resume and Manifest](../public/resume-and-manifest.md) updates.

Output writer closure uses one session-state mutex to order chunk admission
against complete, interrupt, and abort. The coordinator reserves an RAII
completion ticket under that mutex before detaching a full or tail batch, and a
terminal operation waits for all such tickets before it returns. Queue-send
failure releases its ticket only after recording the failure, so terminal
completion cannot hang or silently pass a detached batch.

The run-scoped `OutputManager`, not an individual writer session, owns the
bounded queue lifetime and worker `JoinHandle`s. Opaque
`OutputDeliveryToken`s contain only clients of one mutex-linearized admission
gate. Acquiring a sender permit is the gate's linearization point. The control
mutex is released before a bounded queue send, so close does not wait for space
in a full queue. A pre-close permit remains admitted and its completion ticket
is drained; a post-close permit is rejected. Explicit finish, interrupted
finish, and abort attempt every session, then close that gate and join all
`g-output-writer-*` threads even when an earlier session failed. Spawn failure
follows the same cleanup order for the workers already created. Join panic is
reported only when no earlier lifecycle error exists. Destruction is a
best-effort safety net that closes admission and never waits for a worker that
is still running; production correctness relies on the explicit terminal
paths. Dropping an initialized manager first aborts all sessions, drains queued
batches, discards pending tails, and leaves retained delivery tokens unable to
admit more work.

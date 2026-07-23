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

Run directories also contain `run_manifest.json` and `effective_config.toml`.
The parts directory is the completed dataset; output does not require a
consolidation pass.

Output planning is read-only. The engine resolves every phenotype run path and
inspects any resume manifest before input preparation without creating output
directories. Directory and initial-manifest creation begins only during output
initialization, after the complete path plan, prepared inputs, and manifest
headers validate.

## Manifest And Resume Contract

`run_manifest.json` is the resume authority. Pre-release manifest schema version `0`
stores prepared-run fields, input fingerprints, and Parquet writer settings in
one canonical `execution_plan` object plus `execution_plan_hash`. Top-level
state is limited to schema and mutable lifecycle fields such as committed
chunks. The pre-release Parquet output schema is version `0`; `INFO` is nullable
when its expected-variance denominator is undefined.

Each compute group records a required fingerprint of its trait-major aligned
phenotype matrix, including phenotype names, shape, and float32 values. Resume
therefore rejects phenotype-value changes even if a source path changes between
input loading and manifest preparation.

Manifest construction records `association_backend.kind` from the
engine-resolved delivery format so resume and review tooling can distinguish
`jax_dosage` and `jax_packed8`; it is not a `RunPlan` field.

Resume always reconciles manifest chunk commits with files on disk. It reads
chunk commit metadata from Parquet footer metadata, compares every part to the
canonical output schema, and checks every commit against the current
chromosome-aware BGEN chunk plan. Parts without the current metadata are
rejected rather than reconstructed from result columns. BGEN compatibility is
bound to the exact opened file's device, inode, size, modification time, and
change time.

Compatibility validation must fail loudly on mismatched result-affecting inputs
or output schema assumptions.

Output ownership does not use advisory file locks. The run root carries an
append-only `.g-output` lineage with one immutable genesis record. Each attempt
has one immutable outcome slot. A durable terminal and an exact nonterminal
recovery claim are tagged alternatives published to that same slot, so an old
owner cannot publish a terminal after a takeover claim wins. A terminal outcome
may later have one immutable normal-resume successor bound to the terminal
outcome's SHA-256 only when its status is `interrupted` or `failed`. A
`completed` outcome remains the immutable leaf and is verified read-only on
resume. Records are written and synchronized under unique temporary
names, then published with a same-directory hard-link no-replace operation and
a directory synchronization. There is no authoritative mutable `HEAD`; readers
traverse exact-recovery claims or hash-bound normal successors from genesis and
reject incompatible dual-state artifacts.

The hard-link no-replace operation is the cross-process linearization point.
Its same-filesystem, cross-node visibility and `AlreadyExists` behavior must be
qualified on the deployed BeeGFS mount before production use. A failed
qualification is a release blocker; advisory locks are not an acceptable
fallback.

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
bounded queue lifetime and worker `JoinHandle`s. Sessions shared through
`OutputDeliveryState` contain only clients of one mutex-linearized admission
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
batches, discards pending tails, and leaves retained session handles closed.

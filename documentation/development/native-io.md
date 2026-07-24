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
| `crates/genotype/src/` | BGEN source ownership/index/decode/preprocess and genotype source planning. |
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
- exact quantized dosage sums and integer threshold counts when binary Firth
  sparse-candidate classification is requested;
- cached no-missing diploid packed8 compatibility validation and fast paths;
- chunk delivery in variant-major shape.

Python/JAX kernels should receive already aligned dosage or validated packed8
chunks and metadata, not parse file formats.

Host-only dosage and packed8 readers carry the exact sum as an integer
numerator with the variant's nonzero BGEN probability denominator. Missing calls
are excluded. Allele orientation, strict `MAC < 50`, and half-zero-density
comparisons use widened integer products; a requested sparse mask must never
fall back to the rounded `f32` dosage sum. Output-facing dosage summaries remain
in their existing `f32` representation.

The reader copies BGEN files up to 256 MiB into one immutable owned snapshot.
Larger files retain the exact opened descriptor and use positioned reads into
one reusable owned window per active batch, never larger than 8 MiB. Byte
windows are independent of the 32-variant compute tile: each positioned window
is read once, then its borrowed bytes are decoded by parallel compute tiles.
Compatibility validation reuses the same byte-window policy. Decode and
raw-DEFLATE packing borrow directly from the snapshot when present; no safe API
is backed by a mutable file mapping. Snapshot indexing uses a concrete
bounds-checked slice cursor; positioned indexing streams bounded 8 MiB windows
instead of issuing one metadata read per variant. The opened descriptor and
configured path are initially identified by device, inode, size, modification
time, and change time. After an owned capture, only the opened descriptor
identity is rechecked: deleting or retargeting the configured locator is
provenance-only because parsing proceeds from bytes that are already immutable.
A published snapshot isolates its readers from later in-place mutation or
configured-path replacement, so snapshot delivery and session finish do not
restat the source. Positioned batches recheck both the descriptor and configured
path after every delivered batch and at session finish. Concurrent truncation,
mutation, or path replacement of a positioned source therefore returns an error
without exposing mapped memory.

`BgenReaderCore::open` remains an unselected compatibility call: every call
acquires, captures or positions, hashes owned bytes, and parses its locator
independently. It never consults or publishes the process snapshot registry.
Callers with an authoritative SHA-256 use `BgenReaderCore::open_request` and
`BgenContentSelector`; the optional expected byte count adds a second assertion.
The engine obtains that selector either from the configured digest or from the
content fingerprint in existing output agreement. When both exist, their
digests must match; persisted authority also supplies the expected byte count.
The agreement read and reconciliation precede locator access, so an existing
null digest or a configured/persisted mismatch fails before any BGEN open or
output ownership mutation.

One private process-wide registry entry strongly owns the latest completely
parsed content-selected small-file payload under a revision-0 key containing
its full SHA-256 and byte count. A selected cache hit performs no operation on
its supplied locator—not even current-directory resolution, metadata lookup,
canonicalization, or open—and records the new locator separately from the
original capture identity in its provenance. Digest-only lookup inherits the
stored byte count. A mismatched explicit byte count is rejected without locator
access. Selected misses capture and hash exact bytes, parse outside the registry
lock, and publish only after content and open-time validation succeed.
Concurrent matching misses canonicalize at publication. A different selected
fingerprint replaces the entry atomically; failed candidates and unselected
opens never evict it. Live readers may keep a replaced payload alive.

Content selection is supported only for owned snapshots. A selected locator
above the 256 MiB ceiling is rejected as typed
`ContentSelectionRequiresOwnedSnapshot`; larger unselected inputs remain
positioned and explicitly unattested.

The registry therefore retains up to 256 MiB of source bytes plus parsed index
and metadata allocations until a replacement passes open/index validation or
the process exits. The parsed allocations are outside the source-byte ceiling
and can be substantially larger for adversarial high-cardinality metadata.
Capturing and parsing a different identity overlaps the candidate with the
still-retained old entry. Each concurrent cold miss owns a separate candidate
outside the registry lock, so transient candidate memory scales with in-flight
opens. After publication, readers of the previous identity can extend the
overlap between old and new payloads.

Header length and first-variant offsets, zero-variant end-of-file placement,
reserved flag bits, embedded sample-block framing/counts, and variant-count
capacity are validated before the index allocates. Variant identifiers, RSIDs,
chromosome labels, and alleles must be valid UTF-8. Probability-pair validation
covers every stored sample, including samples omitted from a delivery
selection.

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
reads one whole-plan existing-output agreement before BGEN input preparation,
without creating the output root. It opens through the reconciled request,
resolves the GPU representation, and constructs every authority-complete
manifest header input from the reader's actual content evidence. Output claim
then independently reinspects BGEN agreement, validates that evidence and
format, and constructs and binds every phenotype manifest header before
publishing the durable no-replace owner claim. It repeats complete
execution-plan validation under that authority. Activation then consumes only
those stored headers, publishes genesis or successor authority, and creates
attempt-specific state.

## Manifest And Resume Contract

The append-only `.g-output` lineage is the resume authority.
`run_manifest.json` is attempt state whose SHA-256 is bound by a terminal
claim. Pre-release manifest schema version `0` stores prepared-run fields,
input fingerprints, and Parquet writer settings in one canonical
`execution_plan` object plus `execution_plan_hash`. Top-level lifecycle state
binds the attempt, status, receipts, and committed chunks. The pre-release
Parquet output schema is version `0`; `INFO` is nullable when its
expected-variance denominator is undefined.
Persisted `schema_version`, `output_schema_version`, and
`attempt_manifest_schema_version` values remain JSON integers with value `0`,
not strings.

Each compute group records a required fingerprint of its trait-major aligned
phenotype matrix, including phenotype names, shape, and float32 values. Resume
therefore rejects phenotype-value changes even if a source path changes between
input loading and manifest preparation.

Manifest construction records `association_backend.kind` from the
engine-resolved delivery format so resume and review tooling can distinguish
`jax_dosage` and `jax_packed8`; it is not a `RunPlan` field.
The exact schema-zero `execution_plan.bgen` value contains only required
`content_sha256` and `byte_count` fields. Owned snapshots contribute the
canonical `BgenContentFingerprint`; positioned unattested sources contribute
an explicit null digest and their byte count. No request locator, selector,
filesystem identity, or redundant hash-algorithm field crosses into output
authority.
Construction and parsing share one exact typed schema for the complete nested
execution-plan graph. Unknown or missing nested fields and inconsistent
backend, device, association-mode, or correction-policy combinations are
rejected. Attempt runtime device and writer-thread metadata must also agree
with the corresponding execution-plan fields, and the nested execution-plan
phenotype must equal the attempt and genesis phenotype. Schema zero preserves
its flat, status-dependent top-level wire layout because canonical manifest
bytes are terminal authority; a tagged or nested status model requires a
schema revision.

Every recovery path reads `run_manifest.json` through one descriptor-backed
reader. It rejects symbolic links and other non-regular files, checks the
opened descriptor, and reads at most 1 GiB plus one byte used only to detect
growth or oversize input. Manifest construction enforces the same 1 GiB
schema-zero ceiling before publication. Terminal materialization verifies its
published SHA-256 from the exact bytes returned by that bounded reader rather
than reopening the path through a generic hash helper. The measured bound
includes the known chromosome-22 worst case with one variant per chunk and
every legal receipt grouping, including one receipt per repeated interrupted
flush. It also uses maximum-length accepted lineage identifiers, a 255-byte
phenotype name, and 32 MiB of variable-header reserve, with at least 20% of the
ceiling retained. Larger datasets must increase chunk size or move to a future
control-plane schema rather than increasing recovery memory without review.

`OutputManager::existing_output_resume_agreement` reads all materialized
phenotype manifests under one fresh lineage snapshot and returns one
`ExistingOutputResumeAgreement` containing `bgen_content_fingerprint` and
`gpu_genotype_format`. Each manifest must bind to the genesis phenotype and
chunk-plan contracts, current leaf attempt, and any finalized terminal
manifest digest. Digest, byte count, and format must agree across the plan.
Terminal authority requires all manifests; a missing nonterminal manifest is
skipped. A pending terminal still reads its fully typed, genesis-bound running
manifests because recovery has not materialized the claimed terminal bytes yet.
Exactly one final lineage-snapshot equality check follows all reads. A null
BGEN digest returns the dedicated unattested-content error rather than resume
authority.

Resume validates each immutable receipt against its part's embedded transaction
footer, canonical output schema, raw byte size, and freshly computed SHA-256.
It checks every exact chunk binding against the canonical chromosome-aware BGEN
chunk plan and rejects duplicate or missing coverage. Parts without current
metadata are rejected rather than reconstructed from result columns. A
verifiable final part that exists before its receipt because of a process crash
can be reconciled only for a nonterminal attempt or pending terminal claim.
Downstream integration
must bind BGEN compatibility to authoritative content evidence rather than
reconstructing identity from a request locator.

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
reserves a fresh staging-attempt identifier and creates only that claim's private
diagnostics directory. Genesis or successor publication later makes the same
attempt identifier authoritative; failed pre-activation claims durably remove
their unreferenced staging. The coordinator uses the deferred activation API:
failures before authority publication return a typed rollback capability
instead of releasing ownership immediately. The runner closes claim-scoped
timing, telemetry, and logging before consuming that capability, so no
contender can sweep diagnostics while the prior session is still open. Dropping
the capability fails closed by leaving ownership Active until an exact external
fence; the ordinary no-session activation API rolls it back immediately.
Completed read-only resumes never make their fresh staging attempt
authoritative, and their owner-staging intent remains until terminal
finalization hands off cleanup. Successful finalization returns a non-cloneable,
idempotent, retryable post-session cleanup capability. Terminal failures can
return the same capability beside their primary error. The runner closes
claim-scoped timing, telemetry, and logging before calling it; cleanup removes
the staging attempt before exact owner release while the completed attempt
payload remains unchanged. It retires the owner-staging intent between those
steps; if exact owner release then fails, the same capability retries from the
already-retired intent. A dropped capability fails closed, and an exactly
fenced successor sweeps the obsolete unreferenced staging. Referenced writable
diagnostics survive fencing.

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

For BGEN performance work, the process-registry-empty and
process-registry-primed open contracts are deliberately separate:

```bash
GWAS_ENGINE_BGEN_BENCHMARK_PATH=/path/to/input.bgen \
  cargo bench -p g-genotype --features benchmark-internals --bench bgen_open_once

GWAS_ENGINE_BGEN_BENCHMARK_PATH=/path/to/input.bgen \
  cargo bench -p g-genotype --bench bgen_read -- \
  bgen_warmed_open_and_index/sequential_index

GWAS_ENGINE_BGEN_BENCHMARK_PATH=/path/to/input.bgen \
  cargo bench -p g-genotype --features benchmark-internals \
  --bench bgen_lifecycle_once
```

`bgen_open_once` starts with an empty process registry and reports the selected
content digest, first-open capture/hash/index time, same-process reopen/index
time, a second reopen after both readers have dropped, structural operation
counts, strong canonical hit status, retained source bytes, and process
resident memory before open and after reader drop. Both cache-hit requests
deliberately use a missing locator, proving that resolution is content-only.
The retained byte counter covers source bytes; the accompanying flag records
that the canonical payload also retains its parsed index and metadata. The
normal Criterion group computes its selector outside measurement, keeps one
selected priming reader alive, and measures process-registry-primed reopen/index
work. The selected benchmark requires an input at or below the owned-snapshot
ceiling; the explicit positioned benchmark remains unselected.
`bgen_lifecycle_once` reports open/index, preparation, aggregate time for all
full 16,384-variant packed8 batches, the final logical tail using the same fixed
production compute shape, policy-specific session finish, and their end-to-end
sum. Snapshot finish is a no-op because the payload is already immutable;
positioned finish verifies the source identity. Use the lifecycle total—not an
isolated open or batch result—when comparing source policies. Set
`GWAS_ENGINE_BGEN_BENCHMARK_PRIMED_REOPEN=1` to open and drop one priming
reader before timing the focused same-process reopen lifecycle. This leaves
only process-persistent state—not an extra live reader or index—during the
measurement.

The lifecycle input must be packed8-compatible: biallelic, diploid, unphased,
eight-bit, and without missing samples. The benchmark asserts that contract
instead of switching to dosage delivery. Persistent packed8 markers exist only
for authoritative owned content. Their revision-0 fingerprint binds SHA-256,
content byte count, sample count, and variant count and contains no path or
filesystem metadata. Positioned unattested readers neither read nor publish a
marker. A paired selected-snapshot comparison may set `XDG_CACHE_HOME` to one
fresh campaign-local directory, run one unmeasured prewarm lifecycle, verify
that it wrote a `compatible` marker, and reuse that state for every measured
process. A selected-versus-positioned source-policy campaign must instead give
every process an empty cache directory so both policies perform compatibility
validation. Never let one measured design create a marker consumed only by
later selected runs.

These process-registry and compatibility-cache labels say nothing about the
kernel filesystem page cache. A same-input source-policy campaign must give
both designs equivalent filesystem-cache state. The paired protocol performs
an untimed full read of the source immediately before every individual
baseline or candidate process, verifies the source size and digest on that
read, and alternates process order across blocks. Record that conditioning
method with the results. Without explicit cache eviction and supporting
evidence, describe these measurements as page-cache-primed rather than
filesystem-cold.

For an exact same-input comparison against positioned I/O, build `bgen_read`
with `--features benchmark-positioned-source`. This benchmark-only feature
exposes and selects an explicit zero-snapshot-limit benchmark constructor and
labels its open group `bgen_positioned_open_and_index`. Enabling the feature
does not change the production `BgenReaderCore::open` policy. Add
`benchmark-positioned-source` to the `bgen_lifecycle_once` feature list for the
corresponding end-to-end positioned measurement.

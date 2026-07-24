# Public API

## This crate owns

Append-only output transactions, strict resume validation, and chunked Parquet
writing.

## Public types

`OutputManager<Planned>`, `OutputManager<Claimed>`, `OutputManager<Active>`,
`OutputManager<Covered>`, `OutputDeliveryToken`, `OutputCompletion`,
`CompletedOutputRun`, `OutputPostSessionCleanup`, `OutputClaimRollback`,
`OutputActivationError`, `OutputActivationFailureParts`, native chunk handles,
`OutputTerminalError`, `OutputTerminalFailureParts`, typed manifest
inputs/fingerprints, `ExistingOutputResumeAgreement`, trait-major statistic
batches, `AssociationImplementationCompatibility`,
`FirthComponentsCompatibility`,
`FirthComponentsImplementationCompatibility`,
`FirthComponentsFallbackReasonCompatibility`,
`RawCudaFirthCapabilityRequirementsCompatibility`,
`RawCudaFirthArtifactCompatibility`, `GenotypeDeliveryExecution`,
`GenotypeDeliveryEffectivePath`, `RawDeflatePacked8Artifact`,
`RawDeflatePacked8CapabilityRequirements`, and `OutputError`. Writer sessions
are private implementation details.
Strict on-disk reconciliation is fixed policy owned by this crate.

## Public functions

Plan and initialize output runs through consuming `OutputManager` typestate
transitions, select an opaque delivery token, write validated trait-major
chunks, and complete, interrupt, or abort the owned attempt.
`OutputManager::open` only inspects paths and immutable lineage hints.
`existing_output_resume_agreement` reads all materialized phenotype manifests
under one lineage snapshot and returns one plan-wide
`ExistingOutputResumeAgreement` containing `bgen_content_fingerprint`,
`gpu_genotype_format`, and `association_implementation`. `claim` validates
current BGEN agreement, constructs and binds every manifest header, and rejects
incompatible existing authority before owner acquisition. It then acquires
durable owner authority, repeats lineage and complete execution-plan validation
under that authority, reserves a fresh attempt identity, and creates
ownership-private diagnostic staging without publishing attempt authority.
Writable `activate` takes the required current
`AssociationImplementationCompatibility`, freshly rereads materialized
manifest agreement under the held claim, rejects any exact mismatch, stores the
validated state, then publishes genesis or a successor from the headers already
bound by `claim` and starts writers. Completed read-only activation leaves the
fresh staging attempt unreferenced. `initialize` is the convenience composition
of `claim` and `activate` and likewise requires current compatibility.
`reject_activation` converts a backend compatibility-construction failure into
an unpublished `OutputActivationError` with rollback authority, allowing
claim-scoped diagnostics to close before ownership is released.
`close_completed` consumes exactly one `GenotypeDeliveryExecution` per planned
phenotype compute group, validates its identity and resume-aware processed
chunk count, then drains writers and returns `Covered` only after exact
canonical chunk coverage is proven. Raw-DEFLATE nvCOMP evidence is accepted
only for a GPU packed8 execution plan. Individual writer-session lifecycle
methods remain crate-private.

Ordinary `activate` resolves unpublished failures and completed-no-op cleanup
within the output lifecycle. External diagnostics owners use
`activate_with_deferred_completed_noop_cleanup`: an unpublished failure returns
an idempotent, retryable `OutputClaimRollback` that must be consumed only after
diagnostics close. A transient cleanup or durability error retains that same
rollback authority. Dropping it deliberately leaves the claim active for exact
external fencing. A successful deferred completed-noop `finish` returns an
`OutputCompletion` containing a non-cloneable, idempotent, retryable
`OutputPostSessionCleanup`. Every consuming terminal method reports failures as
`OutputTerminalError`; callers use `into_parts` to retain the primary
`OutputError`. Only completed-noop terminal failures can additionally carry a
cleanup token. Callers close timing, telemetry, and logging sessions before
calling `OutputPostSessionCleanup::cleanup`, and retry the same token if cleanup
reports a durability error. No cleanup or terminal-release authority is
available from `Claimed`, so a live claim cannot be released while its manager
can still activate. Writable terminal failures carry no post-session cleanup
token.

The output root contains an immutable `.g-output` lineage and attempt
directories under `attempts/<attempt>/<phenotype-output-name>`. Genesis,
successor, terminal-claim, and terminal-finalization records are append-only;
there is no mutable `HEAD`. A completed leaf is fully reverified and returns
read-only delivery tokens without modifying attempt, terminal, manifest, part,
or receipt data; the completed invocation still appends its owner acquisition
and release transitions. Interrupted and failed leaves may create a hash-bound
successor. A nonterminal leaf requires both resume and its exact
`recover_attempt` identifier.
Completed-noop diagnostics use a fresh unreferenced staging attempt. Successful
post-session cleanup removes that attempt and releases only its exact current
owner. Its owner-staging intent remains present through completed read-only
finalization, then cleanup retires it after staging removal and before the
potentially failing owner release. Cleanup retries tolerate an already-absent
intent. If the process dies or drops the token, an exactly fenced successor
removes the obsolete unreferenced staging attempt. Once writable genesis or
successor authority references a staging attempt, takeover preserves its
diagnostics and retires only obsolete staging metadata.
The first owner claim is a permanent immutable record at
`.g-output/session.claim.json`. Release, reacquisition, and fenced takeover use
one hard-link no-replace transition slot per predecessor under
`.g-output/owner-transitions/`; authority records are never removed or
replaced. A second process fails before it creates diagnostics or attempt
authority. Graceful terminal completion publishes a Released leaf only after
terminal durability. Abrupt process death leaves an Active leaf and recovery
fails closed with that leaf's exact identity and external-fencing guidance.
After external fencing, `fenced_owner_claim_id` may authorize one immutable
takeover transition from that exact predecessor. An absent, historical, or
different identifier never changes authority. This crate never guesses that a
claim is stale from age, PID state, or elapsed time.

Every Parquet part embeds its complete transaction footer. A separate immutable
receipt repeats that footer and adds the raw byte size and SHA-256. Resume
accepts a part only after schema, footer, receipt, raw bytes, chunk geometry,
producer ancestry, execution plan, and chunk plan all verify. Verified reuse
prefers a hard link and falls back to a synchronized copy plus rehash.
Before any verified receipt is reused into a successor, output durably writes
that successor's running manifest with the selected association
implementation. It refreshes the manifest after reuse with the complete
receipt subset. A nonterminal attempt with durable receipts but no
implementation-bearing manifest is rejected, so crash recovery cannot combine
parts produced by one compute implementation with new parts from another.
Attempt-manifest schema version `0` is exact: unknown, missing, duplicate, or
wrong-typed fields are rejected. Running and completed manifests omit terminal
detail fields; interrupted manifests contain only a non-empty
`interrupted_signal`; failed manifests contain only a non-empty
`failure_reason`. Command and runtime metadata use their exact typed schema.
Runtime device and writer-thread metadata must agree with the corresponding
typed execution-plan fields, and the execution-plan phenotype must equal the
attempt phenotype. Schema zero intentionally retains this flat,
status-dependent object because its canonical JSON bytes participate in
terminal hashes; a tagged or nested status representation requires a future
schema revision.

Every attempt manifest requires the closed
`runtime.association_implementation` object. It stores exact nonempty `jax` and
`jaxlib` versions plus required-nullable `firth_components`. Approximate-Firth
accepts exactly three states: requested/effective JAX with null fallback and
null raw artifact; requested/effective raw CUDA with null fallback and a raw
artifact; or requested raw CUDA/effective JAX with one typed recoverable
fallback reason and the requested raw artifact. The raw artifact contains the
stable FFI target, positive integer FFI API version, canonical lowercase framed
source/ABI handler SHA-256 in `handler_sha256`, PTX SHA-256, PTX ISA, and PTX
target, plus the reviewed minimum CUDA driver and compute-capability
major/minor thresholds. Only `unsupported_platform`, `cuda_driver_unavailable`,
`required_symbol_unavailable`,
`cuda_driver_too_old`, `cuda_device_unavailable`, and
`unsupported_compute_capability` are persistable fallback reasons. Driver
operation failures, internal/native failures, and JAX registration failures are
fatal and have no compatibility variant. Free-text diagnostics, device
identity, driver observations, and other observational detail are not part of
the schema.

Every attempt manifest also requires the explicitly nullable
`runtime.genotype_delivery_execution` field. Initial running manifests contain
`null`; completed manifests contain one validated object whose
`phenotype_compute_group_id` exactly matches the execution plan. Interrupted
manifests contain `null`. Failed manifests contain either `null` for every
phenotype or completed execution evidence for every phenotype when output
drain failed after successful compute; mixed presence is invalid. The object records the
effective `host` or `raw_deflate_nvcomp` path, the resume-aware number of chunks
processed in that lifecycle, mutually exclusive raw/host chunk counts, and an
explicitly nullable raw-DEFLATE packed8 artifact. Raw execution requires every
positive processed-chunk count to be counted as raw, zero host chunks, and the
exact FFI target/API plus handler and PTX SHA-256/ISA/target and grouped
reviewed minimum CUDA driver and compute-capability requirements. Artifact
construction validates that its PTX target and minimum compute capability
agree. Host execution has the inverse counts and a null artifact, including
the canonical fully resumed zero-work case. The immutable terminal claim
duplicates this exact object for each phenotype before it binds the terminal
manifest hash, allowing a crash after terminal-claim publication to reconstruct
the same manifest bytes. Finalized recovery rejects disagreement between the
terminal copy and its bound manifest.

The complete nested `execution_plan` graph is typed and closed as well:
unknown or missing nested fields, unsupported enum values, and inconsistent
backend/device/mode policy combinations are rejected. The schema-zero
`execution_plan.bgen` object has exactly two keys:
`content_sha256` (64 lowercase hexadecimal characters or explicit `null`) and
`byte_count` (a JSON uint64). It contains no locator, filesystem metadata,
algorithm tag, or selector request. An owned BGEN snapshot records its canonical
content fingerprint. A positioned unattested source records `null` plus its
byte count; this is legal for a fresh nonresumable output, but
`ExistingOutputUnattestedBgenContent` prevents that manifest from authorizing
resume.
The schema-zero binary correction plan also requires
`approximate_firth_sparse_pseudo_budget_policy`: approximate-Firth records
`half_total_uncapped_by_dense_cap`, while score-only and linear plans record
explicit `null`. Missing, legacy, unsupported, or mode-inconsistent values are
rejected. This policy and the BGEN authority are fields of the same canonical
execution plan and therefore share one `execution_plan_hash`.

Existing-output agreement is accepted only from manifest bytes bound to the
freshly resolved genesis contract, current leaf attempt, canonical chunk plan,
and, when finalized, immutable terminal manifest digest. The reader compares
the BGEN digest, byte count, GPU genotype format, and exact association
implementation compatibility across every materialized phenotype manifest. A
terminal authority requires every phenotype manifest; an absent nonterminal
manifest is skipped. A pending terminal can still expose its bound running
manifests so recovery can materialize the claimed terminal bytes. After all
reads, one final lineage-snapshot equality check prevents a concurrent
successor or terminal publication from authorizing the result.

Association implementation agreement provides owner-excluded, fail-closed
activation and resume compatibility: activation rereads it after owner claim
and before attempt authority or writer publication, and all emitted running and
terminal manifests retain the selected value. Terminal manifests are
cryptographically bound by immutable terminal authority. Schema zero does not
add association compatibility to `execution_plan_hash`, lineage phenotype
contracts, or part receipts, so it does not claim cryptographic detection of an
external rewrite of otherwise valid nonterminal manifest bytes. A nonterminal
attempt with no materialized manifests has no persisted implementation
agreement and is accepted under the documented absent-manifest recovery rule.

All recovery reads open `run_manifest.json` as a non-symlink regular file and
consume at most 1 GiB plus one detection byte. Writes larger than the same
1 GiB schema-zero ceiling are rejected before publication, and terminal
materialization reopens and hashes the exact bytes through that bounded reader.
The ceiling covers the measured known chromosome-22 scale even with one variant
per chunk and one receipt per interrupted flush, maximum-length accepted
lineage identifiers, a 255-byte phenotype name, and variable-header reserve;
larger analyses must use larger chunks or a future control-plane schema.

Writer-session close is linearized with chunk admission. A full or tail batch
reserves its completion ticket before it leaves the session state lock;
complete, interrupt, and abort then reject later writes and wait for every
reserved batch before returning. Complete and interrupt flush the admitted
tail, while abort discards only chunks that were still pending. Retained
delivery tokens cannot control session lifecycle and cannot admit work once
manager closure begins.

`OutputManager` is the sole owner of the bounded queue's sender lifetime and
worker join handles. Delivery tokens refer only to lightweight clients of a
shared admission gate, so retaining a token cannot keep the queue open after
manager shutdown. A client obtains a short-lived
sender permit while holding the admission lock and releases the lock before a
bounded send can block. Permits acquired before close remain admitted and keep
their completion tickets; close rejects later permits without waiting for queue
progress. Explicit complete, interrupt, and abort paths attempt every session,
close admission, and join every named worker;
the first session failure remains primary over a later join failure. A worker
spawn failure closes admission and joins the workers that were already started.
The resource-owner destructor is only a panic-free fallback: it closes
admission and reaps workers that have already exited, but it never waits for a
live worker or substitutes for an explicit terminal operation.
Dropping an initialized manager first aborts every session, which drains queued
batches, discards pending tails, and closes retained session handles before the
resource-owner fallback runs.

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

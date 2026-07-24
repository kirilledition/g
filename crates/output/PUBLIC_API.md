# Public API

## This crate owns

Append-only output transactions, strict resume validation, and chunked Parquet
writing.

## Public types

`OutputManager<Planned>`, `OutputManager<Claimed>`, `OutputManager<Active>`,
`OutputManager<Covered>`, `OutputDeliveryToken`, `CompletedOutputRun`, native
chunk handles, typed manifest inputs/fingerprints, trait-major statistic
batches, and `OutputError`. Writer sessions are private implementation details.
Strict on-disk reconciliation is fixed policy owned by this crate.

## Public functions

Plan and initialize output runs through consuming `OutputManager` typestate
transitions, select an opaque delivery token, write validated trait-major
chunks, and complete, interrupt, or abort the owned attempt.
`OutputManager::open` only inspects paths and immutable lineage hints. `claim`
acquires durable owner authority, repeats validation under that authority,
reserves the stable attempt identity, and creates ownership-private diagnostic
staging without publishing attempt authority. `activate` validates the final
headers, publishes genesis or a successor for that same attempt identity, and
starts writers. `initialize` is the convenience composition of `claim` and
`activate`.
`close_completed` drains writers and returns `Covered` only after exact
canonical chunk coverage is proven. Individual writer-session lifecycle
methods remain crate-private.

The output root contains an immutable `.g-output` lineage and attempt
directories under `attempts/<attempt>/<phenotype-output-name>`. Genesis,
successor, terminal-claim, and terminal-finalization records are append-only;
there is no mutable `HEAD`. A completed leaf is fully reverified and returns
read-only delivery tokens without modifying attempt, terminal, manifest, part,
or receipt data; the completed invocation still appends its owner acquisition
and release transitions. Interrupted and failed leaves may create a hash-bound
successor. A nonterminal leaf requires both resume and its exact
`recover_attempt` identifier.
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

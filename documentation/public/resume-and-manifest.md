# Resume And Manifest

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-07-24 resume and manifest behavior | Public user docs |

This page is the canonical user-facing reference for resumable output runs.

For output layout and schemas, see [Output Files](output-files.md). For config
merge behavior, see [Configuration](configuration.md).

## Per-Phenotype Metadata

Every phenotype within an attempt writes:

```text
attempts/<attempt-id>/<phenotype-output-name>/
  commits/
  parts/
  effective_config.toml
  run_manifest.json
```

`effective_config.toml` is the final merged config after packaged defaults, the
optional TOML file, and explicit CLI overrides.

`run_manifest.json` records execution-plan-affecting state, including:

- manifest and output schema versions;
- association mode;
- BGEN, sample, phenotype, covariate, prediction-list, and selected LOCO
  prediction-file fingerprints;
- phenotype name, covariate names, sample count, variant count, and chunk size;
- multi-phenotype sample mode, phenotype compute-group identifier, sample-set
  fingerprint, covariate-design fingerprint, aligned phenotype-design
  fingerprint, and prediction-alignment fingerprint;
- selected association backend such as `jax_dosage` or `jax_packed8`, with the
  resolved genotype delivery format;
- binary correction plan and binary kernel settings when applicable;
- JAX device/precision policy and dtype choices;
- output writer settings;
- committed chunk identifiers and complete immutable Parquet receipts.

Pre-release manifest schema version `0` stores immutable compatibility state once under
`execution_plan`, with its SHA-256 digest in `execution_plan_hash`. Top-level
fields are limited to manifest/output schema versions and mutable lifecycle
metadata such as status, committed chunks, command, runtime, and interruption
state. The pre-release Parquet output schema is version `0`; it makes `INFO` nullable when
its expected-variance denominator is undefined.
Attempt-manifest schema version `0` rejects duplicate, unknown, missing, and
wrong-typed fields. Its exact base field set is `schema_version`,
`output_schema_version`, `execution_plan`, `execution_plan_hash`,
`attempt_manifest_schema_version`, `run_set_id`, `attempt_id`,
`phenotype_name`, `output_directory_name`, `chunk_plan_hash`, `status`,
`committed_parts`, `committed_chunks`, `command`, and `runtime`. Running and
completed manifests contain only that base set. Interrupted manifests add one
non-empty string `interrupted_signal`; failed manifests add one non-empty string
`failure_reason`. Inapplicable terminal-detail fields must be absent rather than
`null`. Command and runtime objects also use their exact typed schema. Runtime
device and writer-thread values must agree with the corresponding immutable
execution-plan fields, and the execution-plan phenotype must equal the
attempt/genesis phenotype; CPU-thread count remains runtime diagnostics.
The `schema_version`, `output_schema_version`, and
`attempt_manifest_schema_version` fields are JSON integers with value `0`, not
strings.
Schema zero intentionally preserves this flat, status-dependent JSON layout
because its canonical bytes participate in terminal hashes. A tagged or nested
status layout requires a future schema revision.
Every object in the nested `execution_plan` graph uses the same closed,
strongly typed schema for construction and parsing. Unknown or missing nested
fields, unsupported enum values, and inconsistent backend/device,
association-mode, or correction-policy combinations are rejected even when
the JSON has a self-consistent replacement hash.
The schema-zero `execution_plan.bgen` object has exactly this authority shape:

```json
{
  "content_sha256": "<64 lowercase hexadecimal characters or null>",
  "byte_count": 123
}
```

Both keys are required, `byte_count` is a JSON uint64, and no path, locator,
filesystem metadata, content-hash algorithm tag, or selector request is part of
the execution-plan identity. Consequently, two locators for the same attested
BGEN content produce the same BGEN execution-plan value and do not diverge the
execution-plan hash.
The execution plan records
`resume_policy = "lineage_receipts_exact_coverage"`. This value is hashed and
intentionally rejects older pre-release manifests that used the removed
manifest-only committed-chunk recovery model.

The manifest is supporting state bound by the append-only lineage authority
under `.g-output`. Genesis fixes the run set, phenotype contracts, execution
plan hashes, and canonical chunk-plan hash. Immutable attempt outcomes select
exactly one terminal claim or exact nonterminal recovery claim. Terminal
finalization binds the terminal claim only after every named manifest has been
materialized and rehashed. There is no mutable `HEAD`.
When startup asks existing output for resume agreement, the reader resolves one
fresh lineage snapshot and validates every materialized phenotype manifest
against the genesis run set, phenotype execution-plan digest, canonical
chunk-plan digest, and current leaf attempt. It compares the BGEN content
SHA-256, BGEN byte count, and GPU genotype format across those manifests and
returns one plan-wide agreement only when all three fields match. A finalized
terminal additionally requires each exact raw manifest SHA-256 recorded for its
phenotype, and terminal authority rejects any missing phenotype manifest. A
missing nonterminal manifest remains absent supporting state and is skipped. A
pending terminal intentionally reads its bound running manifests because
recovery has not materialized the claimed terminal bytes yet. After all
manifest reads, one final lineage-snapshot equality check prevents a concurrent
successor or terminal publication from turning stale reads into accepted
agreement.

The engine requests this whole-plan agreement before BGEN locator access. It
rejects an existing null BGEN digest first, then reconciles any configured
selector with the persisted digest and byte count. A configured/persisted
digest mismatch also fails before locator access and before owner acquisition.
The engine opens through the resulting selected request, preserves the
persisted GPU format subject to current compatibility, and constructs output
header inputs from the reader's actual content evidence. A matching
same-process snapshot-cache hit may use a missing request locator; a selected
miss still needs the locator.

Recovery opens each `run_manifest.json` as a non-symlink regular file and
accepts at most 1 GiB. The reader consumes one additional detection byte so
an oversized or concurrently growing file is rejected rather than allocated
without a bound. Manifest construction enforces the same ceiling before
publication, and terminal materialization reopens and hashes the exact bytes
through that bounded reader. This measured schema-zero limit covers the known
chromosome-22 scale even with one variant per chunk and one receipt per
repeated interrupted flush, maximum-length accepted lineage identifiers, a
255-byte phenotype name, and variable-header reserve; larger datasets must use
larger chunks or a future control-plane schema.

Completing or interrupting a run first closes each output writer to new chunks,
flushes every admitted batch, and waits for all part writers before publishing
the terminal claim. Timing diagnostics are persisted before that claim. The
claim fixes the exact manifest hashes, after which the manager reconciles any
verifiable part-without-receipt crash window, materializes the terminal
manifests, and publishes terminal finalization. A crash at any point after the
claim is recovered by finishing that same claimed terminal; it cannot select a
different outcome.

Aborting rejects new chunks and waits for any batch that was already detached
for writing, but discards the not-yet-detached tail and publishes a failed
terminal. No writer can publish another part after its terminal operation
returns.

The run owner attempts every phenotype writer even if an earlier writer fails,
then centrally closes shared queue admission and joins all output workers.
Consequently, retaining an internal delivery handle cannot keep output
admission open after complete, interrupted, or aborted shutdown.
Admission close rejects queue sends that did not already obtain a permit;
pre-close permits remain admitted and are covered by the same completion wait.

Sample identity is always the non-empty `(FID, IID)` pair, so there is no
identity-mode field in the execution plan. Sample-file fingerprints and aligned
sample-set fingerprints cover the concrete identity data used by the run.
Prediction-alignment fingerprints bind the LOCO headers and chromosome rows
seen during indexing plus the concrete per-trait sample-alignment recipe.

File fingerprints include resolved path, file size, and `mtime_ns`. Smaller
control files also include a SHA-256 content hash: sample, phenotype, covariate,
prediction-list, and LOCO prediction files referenced by the selected
phenotype or compute group. BGEN is different: output consumes the canonical
content evidence for the source actually opened by the native reader. An owned
snapshot records its SHA-256 and byte count. A positioned unattested source
records an explicit `null` SHA-256 and its byte count, which permits fresh
nonresumable output but cannot later authorize resume. Output never derives
BGEN execution-plan identity from the configured locator.

## Starting A New Run

Without `[output].resume = true`, `g` refuses to reuse an output root that
already contains lineage or attempt artifacts:

```text
Output run root '<path>' already exists and is not empty.
```

Choose a new `--out` prefix, delete stale local output intentionally, or run
with `[output].resume = true` when the existing lineage belongs to the same
planned run.

Planning a new or resumed run only inspects the selected output paths. It does
not create the output root. After the initial agreement-guided BGEN open, the
engine constructs all authority-complete header inputs from actual content
evidence and calls output claim. Claim independently reinspects plan-wide BGEN
agreement, constructs and binds all phenotype headers, and rejects incompatible
existing authority before taking ownership. This second inspection prevents a
concurrent lineage change from turning the earlier agreement into claim
authority. Claim then publishes the permanent root or one immutable owner
transition, repeats lineage and complete execution-plan validation under that
authority, reserves a fresh staging-attempt identifier, and creates only
ownership-private diagnostics. Activation accepts no new headers; it publishes
genesis or a successor from the claim-bound headers for that same attempt
identity before writers start. Completed read-only activation instead leaves
the fresh staging attempt and its owner-staging intent unreferenced through
final verification. After the runtime session closes, a non-cloneable,
idempotent cleanup capability removes that staging and releases the exact
owner, retiring the owner-staging intent between those operations. The same
capability is retryable after a durability error and tolerates an already-retired
intent. Every consuming terminal method returns its primary failure through
`OutputTerminalError`; only completed read-only failures can also carry this
cleanup capability. Callers must separate the error parts and run cleanup after
closing claim-scoped sessions.
Missing paths are treated as absent; other directory or lineage inspection
errors stop planning.

## Enabling Resume

```toml
[output]
resume = true
```

Resume is always strict and depends on the leaf state:

- `completed`: fully reverify lineage, terminal binding, manifests, receipts,
  Parquet footers, raw sizes and hashes, and exact coverage, then return
  read-only output data; claim-scoped diagnostics are removed after the runtime
  session closes, and the output tree appends only the invocation's owner
  acquisition and Released transitions;
- `interrupted` or `failed`: create one immutable normal successor and reuse
  only verified parts;
- nonterminal: require both `resume = true` and
  `recover_attempt = "<exact-leaf-attempt-id>"`, then publish one exact-recovery
  successor;
- pending terminal claim: finish receipt reconciliation, manifest
  materialization, and terminal finalization before applying the rules above,
  but only after any surviving owner claim has been externally fenced.

There is no public resume-validation mode and no heuristic "latest attempt"
selection.

The first process permanently publishes `.g-output/session.claim.json`.
Release, takeover, and reacquisition append immutable records under
`.g-output/owner-transitions/`; the current authority is the reachable leaf,
not the root record by itself. A process crash or `SIGKILL` leaves that leaf
Active. Automatic takeover is unsafe because the deployed BeeGFS mount does not
provide cross-node file-lock exclusion. Resume returns a typed error containing
the current claim identifier, host, process, and guidance. Never unlink an
authority record based on age or assumption. An external coordinator must
first prove that the recorded owner is fenced and cannot race a graceful
release. After that proof, set the exact current identifier reported by the
error:

```toml
[output]
resume = true
fenced_owner_claim_id = "owner-<exact-id>"
```

The equivalent CLI is
`--fenced-output-owner-claim owner-<exact-id>`. The manager rejects an absent,
different, released, or historical identifier without changing authority. An
exact match competes with graceful release on one predecessor slot and, if it
wins, publishes an immutable fenced-takeover transition before normal strict
resume. A nonterminal output attempt also requires its exact
`recover_attempt`. Do not remove `session.claim.json` or any transition
manually.

This is an explicit assertion that fencing has already happened, not a
filesystem lease or an internal liveness check. A crash before an authority
hard link can leave a uniquely named temporary candidate. It remains
inspectable but is not authority, is never auto-promoted or auto-deleted, and
does not block a fresh root claim. The permanent root and every reachable
transition remain the audit history after ordinary completion, whose authority
leaf is Released.

Resume requires both the current embedded transaction footer in every Parquet
part and its separate immutable receipt. Each pair must agree on run set,
producing attempt, phenotype, execution-plan hash, chunk-plan hash, part and
receipt identifiers, and exact chunks. The receipt also binds the final raw
byte size and SHA-256, which resume recomputes. Parts without current metadata
are rejected instead of being reconstructed from result columns. Every part
must use the production logical schema, and its chunks must have unique
identifiers, exact row counts, and ranges matching the canonical BGEN chunk
plan.

## Compatibility Checks

Resume first requires a valid `.g-output` genesis and traversable immutable
lineage. It compares the current requested run and every phenotype manifest
against the canonical `execution_plan` and its hash. A mismatch fails loudly.
Earlier manifest or lineage layouts are not adapted because the application
has no released legacy output contract.

Every existing materialized phenotype must agree on the attested BGEN content
SHA-256, BGEN byte count, and GPU genotype format. An existing manifest with
`execution_plan.bgen.content_sha256 = null` returns the dedicated
unattested-BGEN error and cannot authorize resume.

Incompatible resume attempts do not create a new attempt or alter a manifest,
configuration, part, or receipt. A second process sees the surviving-owner
claim before attempt mutation.

A finalized terminal requires every bound manifest and rejects a missing or
changed file. A claim-first process crash may leave a specifically authorized
nonterminal leaf without its attempt directory; only exact
`recover_attempt` authorization can advance that lineage.

Common mismatch causes:

- changed BGEN, sample, phenotype, covariate, prediction-list, or selected
  LOCO prediction file;
- changed sample, phenotype, covariate, prediction-list, or selected LOCO
  content even when path, size, and `mtime_ns` are preserved;
- changed phenotype or covariate columns;
- changed trait mode, binary correction plan, or Firth settings;
- changed selected association backend;
- changed multi-phenotype sample mode, aligned sample set, aligned phenotype or
  covariate design, or prediction alignment;
- changed chunk size or schema version;
- an application upgrade that changed fixed output-writer, result-dtype, or JAX
  policy recorded in the execution plan.

Resume is not a way to combine different analyses into one output directory.
Approximate-Firth manifests fingerprint the fixed inner proposal policy as
`float32_elementwise_float64_reduction`. Older pre-release runs used the prior
all-`float64` inner policy and have a different execution plan and hash, so
they cannot be resumed into or mixed with current output parts.

In particular, runs whose execution plan still contains the removed
`firth_newton_raphson_zero_start_iterations` field have a different plan schema
and hash. Start a new output directory instead of resuming those pre-release
binary runs with a newer build.

The scalar approximate-Firth solver is the only supported correction path.
Pre-release configurations and manifests must remove the experimental
`use_block_firth_math` option and its block-only coefficient, likelihood,
step-halving, and initial-response settings. Effective configuration metadata
uses pre-release option schema version `0`, and these fields are not accepted
by pre-release manifest schema version `0`; start a new output directory
instead of resuming an older block-Firth-compatible run.

## Graceful Interruption

During `g regenie`, the first SIGINT or SIGTERM requests graceful shutdown. The
engine flushes queued chunks, publishes an interrupted terminal, prints an
interruption message, and exits with `128 + signal_number` such as `130` for
SIGINT.

After that, rerun the same command with a config containing:

```toml
[output]
resume = true
```

## Parquet Parts And Resume

Committed Parquet part-plus-receipt pairs are both the resumable unit and the
completed dataset. After interruption, resume verifies each source pair,
prefers same-filesystem hard-link reuse, and falls back to a synchronized copy
followed by a full rehash. It then writes only missing chunks and performs no
dataset consolidation step.

# Resume And Manifest

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-07-23 resume and manifest behavior | Public user docs |

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
phenotype or compute group. BGEN identity comes from the exact file opened by
the native reader and includes device, inode, size, modification time, and
change time. This rejects replacement or in-place mutation without hashing the
large genotype file during normal startup.

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
not create the output root. Claiming then publishes the permanent root or one
immutable owner transition, repeats lineage and policy validation under that
authority, reserves a stable attempt identifier, and creates only
ownership-private diagnostics. Activation publishes genesis or a successor for
that same attempt identity before writers start. Missing paths are treated as
absent; other directory or lineage inspection errors stop planning.

## Enabling Resume

```toml
[output]
resume = true
```

Resume is always strict and depends on the leaf state:

- `completed`: fully reverify lineage, terminal binding, manifests, receipts,
  Parquet footers, raw sizes and hashes, and exact coverage, then return
  read-only output data while appending only the invocation's owner
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

# Output Files

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | output file contracts as of 2026-07-23 | Public user docs |

This page is the canonical user-facing output contract for `g regenie`.

For resume validation, see [Resume and Manifest](resume-and-manifest.md). For
the interpretation of result fields, see [Algorithm](algorithm.md#reading-output-rows).

## Output Root

`--out` is an output prefix, not a single output file. By default, `g` creates a
run root next to that prefix:

```text
<out>.g/
```

The run root is an append-only transaction container. `.g-output` holds
immutable lineage authority and `attempts/` holds attempt data:

```text
<out>.g/
  .g-output/
    genesis.json
    session.claim.json
    owner-transitions/
    owner-staging/
    outcomes/
    successors/
    terminal-finalizations/
  attempts/
    <attempt-id>/
      diagnostics/
        <owner-claim-id>/
      <phenotype-output-name>/
```

Each planned phenotype has one deterministic, path-safe
`<phenotype-output-name>`. Every new or recovered attempt gets a distinct
path-safe attempt identifier. There is no mutable `HEAD`; readers traverse the
immutable lineage from `genesis.json`. They also traverse the permanent owner
root and its reachable transitions to a Released leaf. A normally completed
tree has no record in `owner-staging/`; that directory contains only temporary
intent for a claim's private diagnostics. Writable activation retires that
intent after genesis or successor authority is published. A completed
read-only invocation keeps its fresh, unreferenced intent through final
verification until post-session cleanup removes the staging attempt and
retires the intent before releasing the exact owner. A retry after an owner
release durability failure tolerates the already-absent intent.

`[output].output_run_directory` overrides the default `<out>.g` run root.

## Parquet Dataset Layout

Parquet parts are the only result format and the completed dataset. No
single-file consolidation is required. A typical completed run writes:

```text
results/example.g/
  .g-output/
    genesis.json
    outcomes/
      attempt-<producer>.json
    successors/
    terminal-finalizations/
      attempt-<producer>.json
    session.claim.json
    owner-transitions/
      owner-<claim-id>.json
    owner-staging/
  attempts/
    attempt-<producer>/
      trait_0001_phenotype_continuous/
        parts/
          part_000000000_000000007.parquet
          part_000000008_000000015.parquet
          part_000000016_000000023.parquet
          part_000000024_000000031.parquet
        commits/
          part_000000000_000000007.json
          part_000000008_000000015.json
          part_000000016_000000023.json
          part_000000024_000000031.json
        effective_config.toml
        output_stage_timings.json
        run_manifest.json
```

The first and last identifiers in a part name are zero-padded engine chunk
identifiers. A file contains one or more consecutive chunks according to the
internal output grouping policy. A single-chunk file uses
`part_<chunk>.parquet`; a grouped file uses `part_<first>_<last>.parquet`.
Chunk grouping and Parquet compression are internal policies, not public
configuration keys. `output_stage_timings.json` is present only when detailed
output timing collection is enabled. It reports temporary-file sync, raw
SHA-256 reread, no-replace part publication/reconciliation, `parts/` directory
sync, and immutable receipt publication as separate stages as well as in the
writer total.

Pre-release attempt-manifest schema zero limits `run_manifest.json` to 1 GiB.
Writers reject a larger encoded manifest before publication, and recovery
rejects symbolic links, special files, oversized regular files, and files that
grow beyond the limit while being read. Terminal materialization verifies its
published hash from bytes returned by the same bounded reader. The measured
bound covers the known chromosome-22 scale even when one variant is assigned
to each chunk and repeated interrupted flushes produce one receipt per chunk,
including maximum-length accepted lineage identifiers, a 255-byte phenotype
name, and variable-header reserve; larger analyses must use larger chunks or a
future control-plane schema.

Current parts use Parquet format version 2.0. Integer and string columns use
the format's delta fallbacks where applicable, and all `Float32` result columns
use `BYTE_STREAM_SPLIT` before Zstandard compression. These are physical
encodings only: the logical schema below remains pre-release output schema
version `0`.
Supported Python readers are PyArrow `>=24.0.0` and Polars `>=1.41.2`, matching
the project's dependency floors. Older readers must support Parquet 2.0 and
`BYTE_STREAM_SPLIT` or be upgraded before consuming current parts.

Read `parts/` directly as a Parquet dataset. For example:

```python
import pyarrow.dataset as dataset

results = dataset.dataset(
    "results/example.g/attempts/attempt-<id>/trait_0001_phenotype_continuous/parts",
    format="parquet",
)
table = results.to_table()
```

After durable completion, the production CLI prints exactly one completion
line per phenotype:

```text
Parquet dataset saved to <dataset-path>
```

`<dataset-path>` is the corresponding `CompletedOutputRun.parts_directory`,
namely
`<root>/attempts/<completed-attempt>/<phenotype-output-name>/parts`. Consumers
may normalize that path against the requested output root, but must not replace
this contract with recursive directory discovery. The immutable completed
terminal, its finalization, and its bound manifest hashes remain the authority
for validating that the reported path is current.

Each part is finalized in a transaction-scoped create-new temporary file under
`parts/`. Its embedded footer binds the run set, producing attempt, phenotype,
execution-plan hash, canonical chunk-plan hash, part and receipt identifiers,
and exact chunk geometry. The writer synchronizes and hashes the raw bytes,
publishes the final name with a no-replace hard link, synchronizes `parts/`,
then publishes a separate immutable receipt under `commits/`. The receipt
repeats the embedded footer and adds the raw byte size and SHA-256.

A crash after the final part link but before receipt publication leaves
receipt-less bytes, not a committed part. Nonterminal observation ignores those
bytes for reuse; reprocessing the same chunk transaction may reconcile an
already-present final part only after the newly produced footer, schema,
ancestry, geometry, size, and raw hash match exactly, and then publishes its
own receipt. Pending or finalized terminal validation rejects every
receipt-less part. Recovery never manufactures a receipt by trusting orphan
bytes alone. Temporary-file cleanup is limited to the current transaction and
never deletes a final part.

The first owner claim is a permanent immutable record at
`.g-output/session.claim.json`. The manager holds the current Active leaf of an
append-only authority chain; a contender fails before diagnostics or attempt
authority mutation. Graceful completion, interruption, or failure appends a
Released transition only after terminal durability. Abrupt process death leaves
an Active leaf because this BeeGFS mount cannot safely distinguish a crashed
owner from a live owner with file locking. Resume fails closed and names that
leaf's claim, host, and process. After an external coordinator has proved that
the recorded owner can no longer write, rerun resume with
`fenced_owner_claim_id = "<exact-claim-id>"` or
`--fenced-output-owner-claim <exact-claim-id>`. The manager verifies that exact
current identifier and publishes one immutable fenced-takeover transition from
that predecessor before continuing strict recovery. A missing, different, or
historical claim fails without changing authority. The permanent root and all
release, takeover, and reacquisition transitions remain as evidence. Never set
this control based on claim age, PID lookup alone, or while a graceful owner may
still publish its competing transition. A crash before an authority hard link
may leave an inspectable temporary candidate; candidates are
non-authoritative, are never auto-promoted or auto-deleted, and do not block a
fresh root claim. Immutable no-replace lineage records remain the durable
analysis authority.

The manager preserves the configured root text in the execution-plan hash but
returns a lexically normalized absolute completed path. Filesystem aliases or
symlinks that resolve to the same root nevertheless resolve to the same
permanent claim and transition slots, so they contend through the same
no-replace authority and the loser performs no diagnostics or attempt-authority
mutation. This assumes the deployed filesystem gives aliases a coherent
namespace for hard links and directory synchronization.

## Result Schema

### Current Pre-Release Schema Contract (v0)

| Column | Parquet type | Nullable | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `CHROM` | `Utf8` | No | - | Variant chromosome label. |
| `GENPOS` | `Int64` | No | base-pair index | Variant position. |
| `ID` | `Utf8` | No | - | Variant identifier. |
| `ALLELE0` | `Utf8` | No | - | Reference/first allele string. |
| `ALLELE1` | `Utf8` | No | - | Alternate/second allele string. |
| `A1FREQ` | `Float32` | No | allele frequency | Observed allele-one frequency after sample alignment. |
| `INFO` | `Float32` | Yes | INFO score | Observed dosage INFO score; null when expected Hardy-Weinberg variance is undefined. |
| `N` | `Int32` | No | sample count | Number of observed genotypes used in statistics. |
| `BETA` | `Float32` | No | effect size | Estimated effect for `ALLELE1`. |
| `SE` | `Float32` | No | effect size standard error | Standard error for `BETA`. |
| `CHISQ` | `Float32` | No | chi-squared statistic | Score statistic or equivalent Step 2 metric. |
| `LOG10P` | `Float32` | No | -log10(p) | Association significance. |
| `CORRECTION_METHOD` | `Utf8` | No | - | Diagnostic correction method label. |
| `CORRECTION_STATUS` | `Utf8` | No | - | Diagnostic correction status label. |

The schema applies to both association modes: `regenie2_linear` and
`regenie2_binary`.

Current correction method/status pairs are:

| `CORRECTION_METHOD` | `CORRECTION_STATUS` | Meaning |
| --- | --- | --- |
| `score` | `success` | Score-test row with no fallback correction applied. |
| `score` | `failed` | Score test did not produce a valid statistic, so no fallback candidate was selected. |
| `firth_approximate` | `success` | Successful approximate-Firth fallback row. |
| `firth_approximate` | `failed` | Approximate-Firth fallback failed. |

The `firth_approximate` label describes the current experimental correction
diagnostic path. It does not imply exact Firth support. SPA correction labels
are reserved for future support and are not emitted by current runs.

Current schema properties:

- `output_schema_version`: `0`
- Column order is part of the contract for stable downstream parsing.
- Invalid association statistics are stored as `NaN`. Undefined `INFO` values
  use Arrow nulls because the genotype has no valid expected-variance denominator.
- `A1FREQ`, `INFO`, `BETA`, `SE`, `CHISQ`, and `LOG10P` are persisted as
  `Float32`.

Compatibility policy:

- New columns may only be appended and must be documented.
- Before the first release, the contract version remains `0`; incompatible
  pre-release output is rejected through strict manifest and Parquet-schema
  validation rather than implying a stable numbered migration path.
- After the first release, changing an existing column name, type,
  nullability, order, or semantics requires an output schema version bump.
- Additional correction labels or diagnostic meanings require a documented
  schema policy change before release.

Public result statistics are written as `float32`, even when an internal kernel
uses wider precision.

## Telemetry And Logs

When telemetry is enabled, claim-scoped telemetry is written under:

```text
<out>.g/attempts/<attempt-id>/diagnostics/<owner-claim-id>/
```

Common files:

| File | Written when | Meaning |
| --- | --- | --- |
| `attempts/<attempt>/diagnostics/<owner-claim>/events.jsonl` | Progress or profile telemetry is enabled | Lifecycle and profile events. |
| `attempts/<attempt>/diagnostics/<owner-claim>/profile.summary.json` | Profile telemetry is enabled | Aggregate native stage summary. |
| `attempts/<attempt>/<phenotype>/output_stage_timings.json` | Profile telemetry is enabled | Per-phenotype output writer timings persisted before terminal authority. |

Diagnostics remain with an attempt only when activation makes that attempt
authoritative. Failed pre-activation claims and read-only completed resumes
use a fresh unreferenced staging attempt and remove it after the runtime session
closes. If completed-noop cleanup is lost, an exactly fenced successor sweeps
that unreferenced staging; fencing preserves diagnostics for an attempt already
made authoritative.
Attempt-bound output timings live beside their phenotype manifest. The
production frontend currently exposes only `[diagnostics].telemetry`.

Successful CLI runs print one
`Parquet dataset saved to <absolute-parts-directory>` line per phenotype.

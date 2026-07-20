# Output Files

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-07-10 output file contracts | Public user docs |

This page is the canonical user-facing output contract for `g regenie`.

For resume validation, see [Resume and Manifest](resume-and-manifest.md). For
the interpretation of result fields, see [Algorithm](algorithm.md#reading-output-rows).

## Output Root

`--out` is an output prefix, not a single output file. By default, `g` creates a
run root next to that prefix:

```text
<out>.g/
```

Each phenotype gets a deterministic directory:

```text
<out>.g/
  trait_0001_<phenotype>.regenie2_linear.run/
  trait_0002_<phenotype>.regenie2_linear.run/
```

Binary runs use `.regenie2_binary.run`. Unsafe characters in phenotype names are
replaced with underscores and long names are truncated in directory slugs.

`[output].output_run_directory` overrides the default `<out>.g` run root.

## Parquet Dataset Layout

Parquet parts are the only result format and the completed dataset. No
single-file consolidation is required. A typical completed run writes:

```text
results/example.g/
  logs/
    events.jsonl
  trait_0001_phenotype_continuous.regenie2_linear.run/
    parts/
      part_000000000_000000007.parquet
      part_000000008_000000015.parquet
      part_000000016_000000023.parquet
      part_000000024_000000031.parquet
    effective_config.toml
    run_manifest.json
```

The first and last identifiers in a part name are zero-padded engine chunk
identifiers. A file contains one or more consecutive chunks according to the
internal output grouping policy. A single-chunk file uses
`part_<chunk>.parquet`; a grouped file uses `part_<first>_<last>.parquet`.
Chunk grouping and Parquet compression are internal policies, not public
configuration keys.

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
    "results/example.g/trait_0001_phenotype_continuous.regenie2_linear.run/parts",
    format="parquet",
)
table = results.to_table()
```

Part files are committed atomically and carry chunk-commit metadata in their
Parquet footer. The run manifest records the same commits for resume.

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

When telemetry is enabled, run-level logs default under:

```text
<out>.g/logs/
```

Common files:

| File | Written when | Meaning |
| --- | --- | --- |
| `events.jsonl` | Progress or profile telemetry is enabled | Lifecycle and profile events. |
| `profile.summary.json` | Profile telemetry is enabled | Aggregate native stage summary. |
| `output_stage_timings.json` | Profile telemetry is enabled | Per-phenotype output writer timings. |

Diagnostics paths are derived from the output run directory; the production
frontend currently exposes only `[diagnostics].telemetry`.

Successful CLI runs print each phenotype run directory and its `parts/`
Parquet dataset directory.

# Output Files

This page is the canonical user-facing output contract for `g regenie`.

For resume validation, see [Resume and Manifest](resume-and-manifest.md). For
the interpretation of result fields, see [Algorithm](algorithm.md#reading-output-rows).

## Output Root

`--out` is an output prefix, not a single output file. By default, `g` creates a
run root next to that prefix:

```text
<out>.g/
```

Each phenotype then gets a deterministic directory:

```text
<out>.g/
  trait_0001_<phenotype>.regenie2_linear.run/
  trait_0002_<phenotype>.regenie2_linear.run/
```

Binary runs use `.regenie2_binary.run`. Unsafe characters in phenotype names are
replaced with underscores and long names are truncated in directory slugs.

`--g-output-run-directory PATH` overrides the default `<out>.g` run root.

## Format-Specific Chunk Directories

| `--g-output-format` | Chunk directory | Chunk files | Final artifact behavior |
| --- | --- | --- | --- |
| `parquet` | `parts/` | `part_<first>[_<last>].parquet` | Part dataset is the primary output. `final.parquet` is written only with finalization. |
| `arrow` | `chunks/` | `chunk_<first>[_<last>].arrow` | Arrow IPC chunks are primary. `final.parquet` is written only with finalization. |
| `regenie` | `regenie/` | `part_<first>[_<last>].regenie` plus `.regenie.json` sidecars | `final.regenie` is written at successful finish. |

The `first` and `last` identifiers are zero-padded chunk identifiers. Files can
group multiple engine chunks when `--g-output-chunks-per-arrow-file` is greater
than one.

## Typical Parquet Layout

Given:

```bash
--out results/example
--phenoCol phenotype_continuous
--g-output-format parquet
```

a typical completed run writes:

```text
results/example.g/
  logs/
    events.jsonl
  trait_0001_phenotype_continuous.regenie2_linear.run/
    parts/
      part_000000000_000000015.parquet
      part_000000016_000000031.parquet
    effective_config.toml
    run_manifest.json
```

If `--g-finalize-parquet` is enabled, the run directory also contains:

```text
final.parquet
```

## REGENIE Text Layout

With:

```bash
--g-output-format regenie
```

the run directory contains:

```text
trait_0001_phenotype.regenie2_linear.run/
  regenie/
    part_000000000_000000015.regenie
    part_000000000_000000015.regenie.json
  final.regenie
  effective_config.toml
  run_manifest.json
```

REGENIE text output is a Step 2-style association table. It does not add Step 1,
BED, or PGEN support.

## Result Schema

Arrow, Parquet, and REGENIE text outputs use the same public association fields:

### Current Schema Contract (v1)

| Column | Arrow / Parquet type | Nullable | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `CHROM` | `Utf8` | Yes | - | Variant chromosome label. |
| `GENPOS` | `Int64` | Yes | base-pair index | Variant position. |
| `ID` | `Utf8` | Yes | - | Variant identifier. |
| `ALLELE0` | `Utf8` | Yes | - | Reference/first allele string. |
| `ALLELE1` | `Utf8` | Yes | - | Alternate/second allele string. |
| `A1FREQ` | `Float32` | Yes | allele frequency | Observed allele-one frequency after sample alignment. |
| `INFO` | `Float32` | Yes | INFO score | Observed dosage INFO score. |
| `N` | `Int32` | Yes | sample count | Number of observed genotypes used in statistics. |
| `TEST` | `Utf8` | Yes | - | Test label (`ADD` for current Step 2 outputs). |
| `BETA` | `Float32` | Yes | effect size | Estimated effect for `ALLELE1`. |
| `SE` | `Float32` | Yes | effect size standard error | Standard error for `BETA`. |
| `CHISQ` | `Float32` | Yes | chi-squared statistic | Score statistic (or equivalent Step 2 metric). |
| `LOG10P` | `Float32` | Yes | -log10(p) | Association significance. |
| `EXTRA` | `Utf8` | Yes | - | Text diagnostics (`TEST_FAIL` for failed diagnostics, null/NA otherwise). |

The contract also applies to both `association_mode`s (`regenie2_linear`, `regenie2_binary`) and both output families (`arrow`, `parquet`).

Current schema properties:

- `output_schema_version`: `1`
- Writer-level final Parquet metadata key `g.output.schema_version`: `1`
- Column order is part of the contract for stable downstream parsing.
- For `v1`, `TEST` is set to `ADD` for current Step 2 runs in both linear and binary modes; additional labels are reserved for future semantic changes.

Compatibility policy:

- **Additive-safe evolution only for contract extension**: new columns may only be appended
  and must be clearly documented. Existing columns are fixed in name, order, and type
  within version `1`.
- **Breaking change is not allowed under version `1`**: any change in existing
  column name, type, or nullability must bump `output_schema_version`.
- **Future semantic changes** (e.g. new `TEST` labels or extra diagnostics payloads)
  must be recorded in a new version policy and in this document before release.
- **Column ordering**: adding columns is only compatible when new fields are appended at the end. Reordering or deletion is a breaking change and requires a schema version bump.

Public result statistics are written as `float32` in the current output schema,
even when an internal kernel uses wider precision for parity-sensitive work.

## Telemetry And Logs

When telemetry is enabled, run-level logs default under:

```text
<out>.g/logs/
```

Common files:

| File | Written when | Meaning |
| --- | --- | --- |
| `events.jsonl` | Telemetry/log stream is enabled | Lifecycle, progress, profile, or trace events. |
| `stage-timings.json` | Profile/trace or explicit path | Stage timing snapshots. |
| `profile.summary.json` | Profile/trace or explicit path | Aggregate profile summary. |

Use `--g-log-dir`, `--g-log-file`, `--g-stage-timings-json`, and
`--g-profile-summary-json` to route diagnostics explicitly.

Successful CLI runs print the generated run directory and any final dataset,
final Parquet, or final REGENIE text path returned by the engine.

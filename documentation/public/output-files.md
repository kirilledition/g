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

```text
CHROM, GENPOS, ID, ALLELE0, ALLELE1, A1FREQ, INFO, N,
TEST, BETA, SE, CHISQ, LOG10P, EXTRA
```

Field summary:

| Field | Meaning |
| --- | --- |
| `CHROM`, `GENPOS`, `ID` | Variant identity from BGEN metadata. |
| `ALLELE0`, `ALLELE1` | Reported alleles. Effects are for `ALLELE1` dosage. |
| `A1FREQ` | Observed allele-one frequency after sample alignment. |
| `INFO` | Observed dosage INFO score. |
| `N` | Observed genotype count after sample alignment. |
| `TEST` | Currently `ADD`. |
| `BETA`, `SE`, `CHISQ`, `LOG10P` | Association statistics. |
| `EXTRA` | Null/`NA` for ordinary rows; `TEST_FAIL` for failed binary statistic/correction rows. |

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

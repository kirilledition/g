# Input and Output

This page summarizes the file contracts that are currently user-facing. Details may change while `g` is pre-release.

## Genotypes

- BGEN 1.2 is the active supported genotype source.
- `.sample` files are supported.
- Embedded BGEN sample identifiers are supported when available.
- BED and PGEN options are recognized but not implemented.

The trusted fast path is controlled by:

```bash
--g-trusted-no-missing-diploid
--g-trusted-bgen-validation-mode cache_on_miss
```

Validation modes:

| Mode | Meaning |
| --- | --- |
| `cache_on_miss` | Validate on cache miss, then reuse validation result. |
| `force_validate` | Always validate before using the trusted fast path. |
| `assume_validated` | Skip validation. Expert mode only. |

## Phenotypes and Covariates

- Phenotype and covariate tables are parsed natively in Rust.
- Tables are expected to include `IID`.
- `FID` is required when `--g-sample-key-mode fid_iid` is used.
- `--g-sample-key-mode iid` requires globally unique non-empty IIDs.
- `--g-sample-key-mode fid_iid` requires unique `(FID, IID)` pairs.
- Binary phenotypes use REGENIE-style coding: `1 = control`, `2 = case`. Internally these are recoded to `0/1`.
- Missing tokens include empty string, `NA`, `NaN`, `nan`, and `-9`.

## Step 1 Predictions

`--pred` must point to a REGENIE Step 1 prediction list. `g` does not produce this file; use upstream `regenie` for Step 1.

## Output Layout

Given:

```bash
--out data/example_regenie2
--phenoCol phenotype_continuous
```

`g` writes a run tree under the output prefix. The exact shape depends on trait mode, phenotype count, and output settings. A typical run includes:

```text
data/example_regenie2.g/
  logs/
    events.jsonl
  trait_0001_phenotype_continuous.regenie2_linear.run/
    parts/
      part_000000000.parquet
      part_000000001.parquet
    effective_config.toml
    run_manifest.json
```

Binary runs use `.regenie2_binary.run`. Arrow chunk output uses a `chunks/` directory; Parquet output uses `parts/`. Optional finalization writes a final Parquet artifact when enabled.

The `logs/events.jsonl` stream records run lifecycle facts, throttled progress,
and native diagnostics when telemetry is enabled. Successful `g regenie` runs
emit a `run_completed` event with the same output run directory, final dataset,
final Parquet or REGENIE text path, and per-phenotype artifact entries that the
terminal success message shows.

Set `--g-output-format regenie` to write REGENIE-compatible text output. Text runs use a `regenie/` directory for tab-separated `.regenie` parts and always write `final.regenie` in the run directory at successful finish. The text parts are plain association tables; strict resume metadata is stored in adjacent native sidecar files.

## Result Fields

The final table follows REGENIE Step 2-style association fields:

```text
CHROM, GENPOS, ID, ALLELE0, ALLELE1, A1FREQ, INFO, N,
TEST, BETA, SE, CHISQ, LOG10P, EXTRA
```

`BETA`, `SE`, `CHISQ`, and `LOG10P` are persisted from the public `float32` result buffers in Arrow, Parquet, and REGENIE text outputs. In Arrow and Parquet, `EXTRA` is null for ordinary successful rows and `TEST_FAIL` for failed binary correction/statistic rows. In REGENIE text output, null values are written as `NA`.

Compatibility limits:

- Text output is REGENIE Step 2-style association output only; `g` still does not implement REGENIE Step 1.
- Text output uses the existing supported BGEN Step 2 inputs and does not add BED or PGEN input support.
- Arrow and Parquet remain the performance-oriented formats; REGENIE text is intended for workflow compatibility.

## Resume and Reproducibility

Every run writes:

```text
effective_config.toml
run_manifest.json
```

Resume controls:

```bash
--g-resume
--g-resume-mode fast
--g-resume-mode strict
```

Use `strict` when correctness validation is more important than fast startup. Use `fast` for normal manifest-backed resumes after interruption.

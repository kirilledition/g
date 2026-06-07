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
    progress.jsonl
  trait_0001_phenotype_continuous.regenie2_linear.run/
    parts/
      part_000000000.parquet
      part_000000001.parquet
    effective_config.toml
    run_manifest.json
```

Binary runs use `.regenie2_binary.run`. Arrow chunk output uses a `chunks/` directory; Parquet output uses `parts/`. Optional finalization writes a final Parquet artifact when enabled.

## Result Fields

The final table follows REGENIE Step 2-style association fields:

```text
CHROM, GENPOS, ID, ALLELE0, ALLELE1, A1FREQ, INFO, N,
TEST, BETA, SE, CHISQ, LOG10P, EXTRA
```

`BETA`, `SE`, `CHISQ`, and `LOG10P` are persisted as `float32` in Arrow and Parquet outputs. `EXTRA` is null for ordinary successful rows and `TEST_FAIL` for failed binary correction/statistic rows.

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

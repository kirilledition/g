# Input Files

This page is the canonical user-facing input contract for `g regenie`.

For the statistical use of each input, see [Algorithm](algorithm.md). For the
CLI names and TOML mapping, see [CLI](cli.md) and [Configuration](configuration.md).

## Required Files

| Input | CLI | TOML | Required |
| --- | --- | --- | --- |
| BGEN genotype file | `--bgen` | `[input].bgen` | Yes. |
| Sample file | `--sample` | `[input].sample` | Required when sample IDs are not embedded or when explicit sample IDs are needed. |
| Phenotype table | `--phenoFile` | `[input].phenoFile` | Yes. |
| Phenotype columns | `--phenoCol`, `--phenoColList` | `[input].phenoCol`, `[input].phenoColList` | Yes. |
| Covariate table | `--covarFile` | `[input].covarFile` | Required when covariates are selected. |
| Covariate columns | `--covarCol`, `--covarColList` | `[input].covarCol`, `[input].covarColList` | Required when covariates are selected. |
| Step 1 prediction list | `--pred` | `[input].pred` | Yes. |

## Genotypes

`g` currently supports BGEN 1.2 for Step 2 scans.

Supported:

- BGEN 1.2 genotype input.
- Oxford `.sample` files through `--sample`.
- Embedded BGEN sample identifiers when present and valid for the selected
  sample-key mode.

Recognized but not implemented:

- `--bed`
- `--pgen`
- variant filtering through `--extract` and `--exclude`
- sample filtering through `--keep` and `--remove`

The trusted BGEN fast path is controlled by:

```bash
--g-trusted-no-missing-diploid
--g-trusted-bgen-validation-mode cache_on_miss
```

Validation modes:

| Mode | Meaning |
| --- | --- |
| `cache_on_miss` | Validate on cache miss, then reuse validation state. |
| `force_validate` | Validate before each trusted fast-path use. |
| `assume_validated` | Skip validation. Expert mode only. |

Only use `assume_validated` when the exact input has already been checked by a
workflow you trust.

## Sample Identity

Sample alignment uses the configured sample key mode:

| Mode | Required columns | Rule |
| --- | --- | --- |
| `iid` | `IID` | IIDs must be globally unique and non-empty. |
| `fid_iid` | `FID`, `IID` | `(FID, IID)` pairs must be unique. |

The same sample identity rule applies across BGEN sample IDs, phenotype rows,
covariate rows, and prediction rows.

## Phenotypes And Covariates

Phenotype and covariate tables are parsed by the native Rust path. Tables are
expected to include `IID`; `FID` is also required when
`--g-sample-key-mode fid_iid` is used.

Column selection rules:

- Use either repeated `--phenoCol` or one `--phenoColList`, not both.
- Use either repeated `--covarCol` or one `--covarColList`, not both.
- `--phenoColList` and `--covarColList` are comma-delimited lists.
- Multiple phenotypes write one output run per phenotype.

Binary phenotypes use REGENIE-style coding:

| Input value | Internal value |
| --- | --- |
| `1` | Control, recoded to `0`. |
| `2` | Case, recoded to `1`. |

Missing tokens include empty string, `NA`, `NaN`, `nan`, and `-9`.

Categorical covariates through `--catCovarList` are recognized but not
implemented.

## Step 1 Predictions

`--pred` must point to a prediction list produced by upstream REGENIE Step 1.
`g` does not produce Step 1 predictions.

Step 2 statistics depend on the prediction file, trait mode, covariates,
chromosome, and aligned sample set. Changing the prediction list can change
results even when the tested BGEN file is unchanged.

## Multi-Phenotype Sample Semantics

`--g-multi-phenotype-sample-mode per-phenotype` keeps each phenotype on its own
complete-case sample set. This matches separate single-phenotype semantics.

`--g-multi-phenotype-sample-mode complete-case` uses one shared complete-case
intersection across all requested phenotypes. This can improve reuse but changes
statistics when phenotype missingness differs.

See [Algorithm](algorithm.md#multi-phenotype-behavior) for the statistical
consequences.

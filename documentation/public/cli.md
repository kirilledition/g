# CLI

| Status | Applies to | Owner |
| --- | --- | --- |
| Canonical user-facing CLI reference | `g`, `g regenie`, `g-regenie`, and `g config` in this checkout | Public interface |

The CLI is generated from the option registry in `src/g/interface/options.py`.
Use this page for behavior and compatibility rules, then use the live command
help for the exact option list in the checked-out commit:

```bash
uv run g --help
uv run g regenie --help
uv run g config explain
```

For statistical interpretation, see [Algorithm](algorithm.md). For TOML mapping
and merge behavior, see [Configuration](configuration.md).

## Command Grammar

```text
g [OPTIONS] COMMAND [ARGS]...
g regenie [--config PATH] [REGENIE-style options] [--g-* options]
g config init [--out PATH]
g config validate CONFIG_PATH
g config explain [OPTION_NAME]
g-regenie [--config PATH] [REGENIE-style options] [--g-* options]
```

`g-regenie` is the direct executable form of `g regenie`; it accepts the same
scan options and exists for REGENIE-style command replacement.

## Commands

| Command | Purpose |
| --- | --- |
| `g regenie` | Run a REGENIE-compatible Step 2 association scan. |
| `g-regenie` | Direct executable alias for `g regenie`. |
| `g config init` | Write the starter TOML config to stdout or `--out`. |
| `g config validate` | Parse and validate a TOML config. |
| `g config explain` | Print the option registry metadata for one option or all options. |

## Required Scan Inputs

The packaged defaults supply runtime defaults, but a real Step 2 run still
requires these run-specific inputs:

| Option | Required when | Meaning |
| --- | --- | --- |
| `--step 2` | Always, unless inherited from config/defaults | REGENIE Step 2. Step 1 is not implemented. |
| `--bgen PATH` | Always | BGEN genotype source. |
| `--phenoFile PATH` | Always | Phenotype table. |
| `--phenoCol NAME` or `--phenoColList LIST` | Always | One or more phenotype columns. |
| `--pred PATH` | Always | Upstream REGENIE Step 1 prediction list. |
| `--out PATH` | Always | User output prefix. |
| `--sample PATH` | When BGEN sample IDs are absent or unsuitable | Oxford sample file. |
| `--covarFile PATH`, `--covarCol`, `--covarColList` | When the model uses covariates | Covariate table and selected columns. |

## Supported REGENIE-Style Options

| Option | TOML section | Meaning |
| --- | --- | --- |
| `--step` | `[trait]` | REGENIE analysis step. Only `2` is executable. |
| `--qt` / `--no-qt` | `[trait]` | Quantitative trait mode. |
| `--bt` / `--no-bt` | `[trait]` | Binary trait mode. |
| `--bgen` | `[input]` | BGEN genotype file. |
| `--sample` | `[input]` | BGEN sample file. |
| `--phenoFile` | `[input]` | Phenotype table. |
| `--phenoCol` | `[input]` | Repeatable phenotype column option. |
| `--phenoColList` | `[input]` | Comma-delimited phenotype column list. |
| `--covarFile` | `[input]` | Covariate table. |
| `--covarCol` | `[input]` | Repeatable covariate column option. |
| `--covarColList` | `[input]` | Comma-delimited covariate column list. |
| `--pred` | `[input]` | REGENIE Step 1 prediction list. |
| `--bsize` | `[trait]` | Variants per processing block. |
| `--threads` | `[trait]` | Requested native CPU thread count. |
| `--out` | `[output]` | Output prefix. |
| `--firth` / `--no-firth` | `[binary]` | Binary Firth fallback switch. |
| `--approx` / `--no-approx` | `[binary]` | Approximate Firth fallback switch. |
| `--pThresh` | `[binary]` | Score-test p-value threshold for binary fallback candidates. |
| `--firth-se` / `--no-firth-se` | `[binary]` | Firth-derived standard error reporting for corrected rows. |

## Supported Modes

Quantitative Step 2:

```bash
uv run g regenie \
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --covarFile /path/to/covariates.tsv \
  --covarColList age,sex \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_quantitative_regenie2
```

Binary score test:

```bash
uv run g regenie \
  --step 2 \
  --bt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_binary \
  --covarFile /path/to/covariates.tsv \
  --covarColList age,sex \
  --pred /path/to/regenie_step1_pred.list \
  --out /path/to/output/g_binary_score_regenie2
```

Binary approximate Firth fallback:

```bash
uv run g regenie \
  --step 2 \
  --bt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_binary \
  --covarFile /path/to/covariates.tsv \
  --covarColList age,sex \
  --pred /path/to/regenie_step1_pred.list \
  --firth \
  --approx \
  --pThresh 0.01 \
  --out /path/to/output/g_binary_firth_regenie2
```

## Boolean Override Semantics

Boolean CLI options use explicit paired flags:

```text
--qt / --no-qt
--bt / --no-bt
--firth / --no-firth
--approx / --no-approx
--g-resume / --no-g-resume
```

Only flags explicitly present on the command line override the TOML config.
Omitting a boolean flag leaves the value from `--config` or packaged defaults
unchanged. To override a TOML `true` value to `false`, pass the negative form,
for example `--no-g-resume`.

Trait flags have additional rules:

- `--qt` and `--bt` are mutually exclusive when enabled in the same layer.
- An explicit `--qt` selection clears binary mode for the merged config.
- An explicit `--bt` selection clears quantitative mode for the merged config.
- If neither mode is explicitly set by the user, the packaged default applies.

Binary-only flags (`--firth`, `--approx`, `--pThresh`, `--firth-se`, `--spa`)
are rejected for quantitative runs. `--approx` requires `--firth`. Exact
`--firth` without `--approx` is recognized but not implemented.

## `--g-*` Namespace

Every engine-specific option starts with `--g-`. This keeps REGENIE-compatible
names separate from `g` runtime controls.

| Group | Examples | Purpose |
| --- | --- | --- |
| Device and staging | `--g-device`, `--g-staging-depth`, `--g-variant-limit` | JAX target, pipeline staging, and debug caps. |
| BGEN and sample policy | `--g-trusted-no-missing-diploid`, `--g-trusted-bgen-validation-mode`, `--g-sample-key-mode`, `--g-multi-phenotype-sample-mode` | Input validation, sample identity, and multi-trait sample semantics. |
| Numeric policy | `--g-linear-minimum-variance`, `--g-binary-minimum-probability`, `--g-score-dtype`, `--g-firth-dtype` | Numerical floors, dtype choices, and binary null behavior. |
| Approximate Firth tuning | `--g-firth-batch-size`, `--g-firth-maximum-iterations`, `--g-null-firth-maximum-iterations` | Candidate batching and solver limits. |
| JAX runtime | `--g-jax-cache-dir`, `--g-jax-persistent-cache`, `--g-jax-matmul-precision`, `--g-jax-transfer-guard` | Compilation cache and runtime diagnostics. |
| Output writer | `--g-output-format`, `--g-writer-threads`, `--g-output-chunks-per-arrow-file`, `--g-finalize-parquet` | Arrow, Parquet, REGENIE text, writer, and finalization controls. |
| Resume | `--g-resume`, `--g-resume-mode` | Manifest-backed restart behavior. |
| Diagnostics | `--g-telemetry`, `--g-log-dir`, `--g-log-file`, `--g-trace-event-cap`, `--g-log-lossy` | Progress, profile, trace, and logging controls. |

Use `uv run g config explain g-device` or `uv run g regenie --help` for the
complete registry in the current checkout.

## Recognized But Unsupported Options

The CLI recognizes selected REGENIE flags so migration mistakes fail clearly
instead of being ignored:

| Option | Behavior |
| --- | --- |
| `--bed`, `--pgen` | Rejected; current Step 2 scans support BGEN input only. |
| `--keep`, `--remove` | Rejected; sample keep/remove lists are not implemented. |
| `--extract`, `--exclude` | Rejected; variant include/exclude lists are not implemented. |
| `--catCovarList` | Rejected; categorical covariates are not implemented. |
| `--test`, `--t2e` | Rejected; alternative tests and time-to-event traits are not implemented. |
| `--spa` | Rejected; SPA fallback is not implemented. |
| `--firth` without `--approx` | Rejected; exact Firth is not implemented. |

For the supported compatibility surface, see [Compatibility](compatibility.md).

## Exit And Usage Expectations

| Situation | Expected result |
| --- | --- |
| `g --help`, `g regenie --help`, `g config --help` | Exit `0` and print help. |
| Missing command, invalid option, invalid value, or validation error | Non-zero Click usage/error exit; invalid root usage exits `2`. |
| Successful `g regenie` run | Exit `0` and print generated artifact paths. |
| First SIGINT or SIGTERM during `g regenie` | Flush queued chunks for resume, print an interruption message, and exit with `128 + signal_number` such as `130` for SIGINT. |
| Second shutdown signal during graceful drain | Abort through the normal signal-derived interrupt path. |

Run outputs and resume metadata are documented in [Output Files](output-files.md)
and [Resume and Manifest](resume-and-manifest.md).

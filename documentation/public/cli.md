# CLI

| Status | Applies to | Owner |
| --- | --- | --- |
| Current Rust frontend CLI reference | `g`, `g regenie`, and `g-regenie` in this checkout | Public interface |

The Rust frontend owns CLI parsing for this branch. Use this page for behavior
and compatibility rules, then use live command help for the exact option list in
the checked-out commit:

```bash
uv run g --help
uv run g regenie --help
uv run g-regenie --help
```

This experimental Rust CLI/config branch does not expose the previous
`g config init`, `g config validate`, or `g config explain` helper commands.

For statistical interpretation, see [Algorithm](algorithm.md). For TOML mapping
and merge behavior, see [Configuration](configuration.md).

## Command Grammar

```text
g [OPTIONS] COMMAND [ARGS]...
g regenie [--config PATH] [REGENIE-style options] [g runtime options]
g-regenie [--config PATH] [REGENIE-style options] [g runtime options]
```

`g-regenie` is the direct executable form of `g regenie`; it accepts the same
scan options and exists for REGENIE-style command replacement.

## Commands

| Command | Purpose |
| --- | --- |
| `g regenie` | Run a REGENIE-compatible Step 2 association scan. |
| `g-regenie` | Direct executable alias for `g regenie`. |

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
| `--qt` / hidden `--no-qt` | `[trait]` | Quantitative trait mode. |
| `--bt` / hidden `--no-bt` | `[trait]` | Binary trait mode. |
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
| `--output_statistic_dtype` | `[output]` | Public statistic column dtype (`float32` default, `float64` optional). |
| `--firth` / hidden `--no-firth` | `[binary]` | Binary Firth fallback switch. |
| `--approx` / hidden `--no-approx` | `[binary]` | Approximate Firth fallback switch. |
| `--pThresh` | `[binary]` | Score-test p-value threshold for binary fallback candidates. |
| `--firth-se` / hidden `--no-firth-se` | `[binary]` | Firth-derived standard error reporting for corrected rows. |

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

Boolean CLI options use explicit paired flags. The positive forms are shown in
help; negative forms are hidden but accepted:

```text
--qt / --no-qt
--bt / --no-bt
--firth / --no-firth
--approx / --no-approx
--resume / --no-resume
```

Only flags explicitly present on the command line override the TOML config.
Omitting a boolean flag leaves the value from `--config` or packaged defaults
unchanged. To override a TOML `true` value to `false`, pass the negative form,
for example `--no-resume`.

Trait flags have additional rules:

- `--qt` and `--bt` are mutually exclusive when enabled in the same layer.
- An explicit `--qt` selection clears binary mode for the merged config.
- An explicit `--bt` selection clears quantitative mode for the merged config.
- If neither mode is explicitly set by the user, the packaged default applies.

Binary-only flags (`--firth`, `--approx`, `--pThresh`, `--firth-se`) are rejected
for quantitative runs. `--approx` requires `--firth`. Exact `--firth` without
`--approx` is recognized but not implemented.

## Runtime Options

Non-REGENIE `g` runtime options intentionally use snake_case long flags on this
branch.

| Group | Examples | Purpose |
| --- | --- | --- |
| Device and staging | `--device`, `--staging_depth`, `--variant_limit` | JAX target, pipeline staging, and debug caps. |
| BGEN and sample policy | `--trusted_no_missing_diploid`, `--trusted_bgen_validation_mode`, `--sample_key_mode`, `--multi_phenotype_sample_mode` | Input validation, sample identity, and multi-trait sample semantics. |
| Numeric policy | `--linear_minimum_variance`, `--binary_minimum_probability`, `--score_dtype`, `--firth_dtype` | Numerical floors, dtype choices, and binary null behavior. |
| Approximate Firth tuning | `--firth_batch_size`, `--firth_maximum_iterations`, `--null_firth_maximum_iterations` | Candidate batching and solver limits. |
| JAX runtime | `--jax_cache_dir`, `--jax_persistent_cache`, `--jax_matmul_precision`, `--jax_transfer_guard` | Compilation cache and runtime diagnostics. |
| Output writer | `--format`, `--writer_threads`, `--chunks_per_arrow_file`, `--finalize_parquet` | Arrow, Parquet, REGENIE text, writer, and finalization controls. |
| Resume | `--resume`, `--resume_mode` | Manifest-backed restart behavior. |
| Diagnostics | `--telemetry`, `--log_dir`, `--log_file`, `--trace_event_cap`, `--log_lossy` | Progress, profile, trace, and logging controls. |

Logging sinks, `--threads`, and JAX runtime settings are process-global inside
one Python process. Single CLI invocations are isolated by their process. Python
callers that run multiple jobs in one process must reuse compatible settings or
start a fresh process when `g` reports an incompatible runtime policy.

## Recognized But Unsupported Options

`uv run g regenie --help` is the authoritative list of supported flags on this
experimental Rust frontend branch. Familiar REGENIE flags that are absent from
help are not accepted yet.

Common absent flags include `--bed`, `--pgen`, `--keep`, `--remove`,
`--extract`, `--exclude`, `--catCovarList`, `--test`, `--t2e`, and `--spa`.

For the supported compatibility surface, see [Compatibility](compatibility.md).

## Exit And Usage Expectations

| Situation | Expected result |
| --- | --- |
| `g --help`, `g regenie --help`, `g-regenie --help` | Exit `0` and print help. |
| Missing command, invalid option, invalid value, or validation error | Non-zero usage/error exit; invalid root usage exits `2`. |
| Successful `g regenie` run | Exit `0` and print generated artifact paths. |
| First SIGINT or SIGTERM during `g regenie` | Flush queued chunks for resume, print an interruption message, and exit with `128 + signal_number` such as `130` for SIGINT. |
| Second shutdown signal during graceful drain | Abort through the normal signal-derived interrupt path. |

Run outputs and resume metadata are documented in [Output Files](output-files.md)
and [Resume and Manifest](resume-and-manifest.md).

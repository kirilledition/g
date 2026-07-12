# CLI

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; current Rust frontend CLI reference | main branch as of 2026-06-30 `g` and `g regenie` | Public user docs |

The Rust frontend owns CLI parsing for this branch. Use this page for behavior
and compatibility rules, then use live command help for the exact option list in
the checked-out commit:

```bash
uv run g --help
uv run g regenie --help
```

This experimental Rust CLI/config branch does not expose the previous
`g config init`, `g config validate`, or `g config explain` helper commands.

For statistical interpretation, see [Algorithm](algorithm.md). For TOML mapping
and merge behavior, see [Configuration](configuration.md).

## Command Grammar

```text
g [OPTIONS] COMMAND [ARGS]...
g regenie [--config PATH] [supported REGENIE-style options]
g batch --config PATH [--config PATH ...]
```

## Commands

| Command | Purpose |
| --- | --- |
| `g regenie` | Run a REGENIE-compatible Step 2 association scan. |
| `g batch` | Run complete configurations sequentially while reusing one process. |

## Batch Runs

Use `g batch` for compatible scans that would otherwise start separate Python
and JAX processes:

```bash
uv run g batch \
  --config chromosome_21.toml \
  --config chromosome_22.toml
```

Each path must contain a complete configuration, including its input and output
paths. Batch mode does not accept per-run CLI overrides. Before starting the
first scan, `g` constructs every frontend config, rejects equal or nested output
run roots including aliases through existing symlink ancestors, and verifies
that process-global logging, native thread, device, and JAX cache policies are
compatible. Input files, sample and prediction compatibility, existing output
state, and resume manifests are checked by each entry's engine preflight when
that entry starts.

Runs execute in argument order. The batch stops on the first runtime failure or
interruption; outputs completed by earlier entries remain valid, and later
entries are not started.

## Required Scan Inputs

The packaged defaults supply runtime defaults, but a real Step 2 run still
requires these run-specific inputs:

| Option | Required when | Meaning |
| --- | --- | --- |
| `--bgen PATH` | Always | BGEN genotype source. |
| `--phenoFile PATH` | Always | Phenotype table. |
| `--phenoCol NAME` or `--phenoColList LIST` | Always | One or more phenotype columns. |
| `--pred PATH` | Always | Upstream REGENIE Step 1 prediction list. |
| `--out PATH` | Always | User output prefix. |
| `--sample PATH` | When the BGEN does not embed usable sample IDs | Oxford sample file. |
| `--covarFile PATH`, `--covarCol`, `--covarColList` | When the model uses covariates | Covariate table and selected columns. |

## Supported REGENIE-Style Options

| Option | TOML section | Meaning |
| --- | --- | --- |
| `--qt` | `[trait]` | Quantitative trait mode. |
| `--bt` | `[trait]` | Binary trait mode. |
| `--bgen` | `[input]` | BGEN genotype file. |
| `--sample` | `[input]` | Optional BGEN sample file. If omitted, embedded BGEN sample IDs are used. |
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
| `--binary-fallback` | `[binary]` | `score_only` or `firth_approximate`. |
| `--pThresh` | `[binary]` | Score-test p-value threshold for binary fallback candidates. |
| `--firth-se` | `[binary]` | Firth-derived standard error reporting for corrected rows. |

## Supported Modes

Quantitative Step 2:

```bash
uv run g regenie \
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
  --bt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_binary \
  --covarFile /path/to/covariates.tsv \
  --covarColList age,sex \
  --pred /path/to/regenie_step1_pred.list \
  --binary-fallback firth_approximate \
  --pThresh 0.01 \
  --out /path/to/output/g_binary_firth_regenie2
```

## Boolean Override Semantics

Boolean CLI options use positive REGENIE-compatible flags:

```text
--qt
--bt
--firth-se
```

Only flags explicitly present on the command line override the TOML config.
Omitting a boolean flag leaves the value from `--config` or packaged defaults
unchanged. Set canonical TOML fields to `false` when disabling a shared or
packaged setting.

Trait flags have additional rules:

- `--qt` and `--bt` are mutually exclusive when enabled in the same layer.
- An explicit `--qt` selection clears binary mode for the merged config.
- An explicit `--bt` selection clears quantitative mode for the merged config.
- If neither mode is explicitly set by the user, the packaged default applies.

Binary-only options (`--binary-fallback`, `--pThresh`, `--firth-se`) are rejected
for quantitative runs. Exact Firth is not part of the option surface.

## Runtime Options

The command line intentionally exposes only `--config` and supported REGENIE
Step 2 flags. Native device, scheduling, BGEN policy, numerical, JAX, writer,
resume, and diagnostics settings use canonical snake_case TOML fields. This
keeps a REGENIE-compatible CLI without duplicating the full native config
surface as command-line aliases.

Logging sinks, native thread policy, and JAX runtime settings are process-global
inside one Python process. Separate `g regenie` invocations are isolated by
their process. `g batch` verifies these policies across every requested run
before starting work. Once a
JAX configuration attempt begins, a configuration update, device validation, or
setup-diagnostic failure requires a fresh process because JAX may already be
partially configured. Cache-directory creation occurs before that transition,
so a directory-creation failure remains retryable. Compatibility uses effective
JAX settings: cache directories, thresholds, and auxiliary-cache flags are
ignored while persistent caching is disabled, and omitted
`jax_matmul_precision` is equivalent to explicit `float32`.

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
| `g --help`, `g regenie --help`, `g batch --help` | Exit `0` and print help. |
| Missing command, invalid option, invalid value, or validation error | Non-zero usage/error exit; invalid root usage exits `2`. |
| Successful `g regenie` run | Exit `0` and print generated artifact paths. |
| Successful `g batch` run | Exit `0` after every config completes and print each run's artifact paths in order. |
| Batch-wide config, output-root, or process-policy failure | Exit before creating output for any batch entry. |
| Per-entry preflight or runtime failure during `g batch` | Stop at the failing entry, preserve earlier completed outputs, and do not start later entries. |
| Runtime failure during `g regenie` | Exit `1` and print a concise `Error: ...` line without a Python traceback. Configured logs and telemetry contain structured failure details. |
| First SIGINT or SIGTERM during `g regenie` | Flush queued chunks for resume, print an interruption message, and exit with `128 + signal_number` such as `130` for SIGINT. |
| Second SIGTERM during graceful drain | Terminate immediately with the operating system's default SIGTERM action. |

Run outputs and resume metadata are documented in [Output Files](output-files.md)
and [Resume and Manifest](resume-and-manifest.md).

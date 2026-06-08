# CLI

The CLI entrypoint is:

```bash
uv run g
```

The main scan command is:

```bash
uv run g regenie --help
```

`g-regenie` is a direct executable alias for the REGENIE-compatible scan command.

## Commands

| Command | Purpose |
| --- | --- |
| `g regenie` | Run a REGENIE-compatible Step 2 association scan |
| `g-regenie` | Direct executable form of `g regenie` |
| `g config init` | Write a starter TOML config |
| `g config validate` | Validate a TOML config |
| `g config explain` | Explain supported and recognized options |

## Core REGENIE-style Options

| Option | Meaning |
| --- | --- |
| `--step 2` | Step 2 association scan. Step 1 is not implemented. |
| `--qt` / `--bt` | Quantitative or binary trait mode. |
| `--bgen` / `--sample` | BGEN genotype file and optional Oxford sample file. |
| `--phenoFile` | Phenotype table. |
| `--phenoCol` / `--phenoColList` | One or more phenotype columns. |
| `--covarFile` | Covariate table. |
| `--covarCol` / `--covarColList` | One or more covariate columns. |
| `--pred` | REGENIE Step 1 prediction list. |
| `--bsize` | Variants per processing block. |
| `--threads` | Requested native CPU thread count. |
| `--out` | User output prefix. |
| `--firth --approx` | Binary approximate Firth fallback. |
| `--pThresh` | Score-test p-value threshold for binary fallback. |
| `--firth-se` | Firth-derived standard error behavior. |

## g-specific Runtime Options

`g` extensions are namespaced with `--g-*`, for example:

```bash
--g-device cpu
--g-device gpu
--g-staging-depth 2
--g-output-format parquet
--g-output-format regenie
--g-resume
--g-resume-mode strict
--g-trusted-no-missing-diploid
--g-trusted-bgen-validation-mode cache_on_miss
--g-telemetry profile
```

Trace mode defaults to a bounded JSONL event cap:

```bash
--g-telemetry trace
--g-trace-event-cap 1000000
```

Raise `--g-trace-event-cap` for planned deep traces, or set it to `0` to
disable the cap. Extra trace events are dropped only when `--g-log-lossy` is
enabled; with `--no-g-log-lossy`, exceeding the cap fails with a message that
names the cap and stream path.

Use `g config explain` for the current option registry:

```bash
uv run g config explain pThresh
uv run g config explain g-telemetry
```

## Unsupported Options

Recognized but unsupported REGENIE options, such as `--bed`, `--pgen`, categorical covariate flags, `--spa`, and exact Firth without `--approx`, fail loudly. Treat that as an intentional guardrail while the supported surface is still narrow.

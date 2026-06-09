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

This experimental Rust CLI/config branch intentionally does not expose the
previous `g config` helper command. Use `g regenie --config run.toml` to load a
TOML file and `g regenie --help` or `g-regenie --help` for the supported
runtime flags.

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

## Unsupported Options

`uv run g regenie --help` is the authoritative list of supported flags on this
experimental Rust frontend branch. Familiar REGENIE flags that are not listed
there are not accepted yet.

Common absent flags include `--bed`, `--pgen`, `--keep`, `--remove`,
`--extract`, `--exclude`, `--catCovarList`, `--test`, `--t2e`, and `--spa`.
Use BGEN Step 2 inputs with the listed flags until those modes are implemented.

# Configuration

`g` accepts TOML configuration files that use the same option names as the CLI, grouped by section.

## Merge Order

Configuration is merged in this order:

```text
packaged defaults in src/g/config.default.toml
        < values in --config
        < explicit CLI flags
```

An explicit CLI flag overrides both the packaged default and the TOML file.

For how statistical and runtime parameters change the Step 2 algorithm, see
[Algorithm](algorithm.md).

## Create and Validate a Config

```bash
uv run g config init --out regenie.toml
uv run g config validate regenie.toml
```

Run with a config and override one value:

```bash
uv run g regenie --config regenie.toml --g-device cpu
```

## Minimal Quantitative Example

```toml
[input]
bgen = "data/1kg_chr22_full.bgen"
sample = "data/1kg_chr22_full.sample"
phenoFile = "data/pheno_cont.txt"
phenoCol = "phenotype_continuous"
covarFile = "data/covariates.txt"
covarColList = "age,sex"
pred = "data/baselines/regenie_step1_qt_pred.list"

[trait]
step = 2
qt = true
bt = false
bsize = 8192

[output]
out = "data/example_regenie2"

[g.compute]
device = "cpu"
staging-depth = 1
trusted-bgen-validation-mode = "cache_on_miss"
sample-key-mode = "iid"

[g.output]
format = "parquet"
resume = false
resume-mode = "fast"

[g.diagnostics]
telemetry = "progress"
log-stderr = true
```

## Trace Telemetry Caps

Trace telemetry is bounded by default. In trace mode, the Rust-owned JSONL
stream writes at most `trace-event-cap = 1000000` completed events unless you
raise the cap or set it to `0`.

```toml
[g.diagnostics]
telemetry = "trace"
trace-event-cap = 5000000
log-lossy = true
```

With `log-lossy = true`, events after the cap are dropped. With
`log-lossy = false`, exceeding the cap fails clearly and tells you to raise
`--g-trace-event-cap` or set it to `0` for an intentional deep trace. The cap
applies only to `telemetry = "trace"`; progress and profile modes are not
constrained by it.

## Sections

| Section | Typical contents |
| --- | --- |
| `[input]` | Genotype, sample, phenotype, covariate, and prediction paths |
| `[trait]` | Step, trait mode, block size, and thread request |
| `[binary]` | Binary correction and Firth fallback settings |
| `[output]` | User output prefix |
| `[g.compute]` | Device, batching, BGEN validation, numeric, and JAX settings |
| `[g.output]` | Writer, format, finalization, and resume settings |
| `[g.diagnostics]` | Telemetry, logging, progress, and trace settings |

`[g.output].format` accepts `parquet` (default), `arrow`, or `regenie`. Use
`regenie` for REGENIE Step 2-compatible tab-separated text output with a
`final.regenie` artifact.

For implementation details behind the configuration model, see [Configuration and CLI Architecture](../development/configuration_cli_architecture.md).

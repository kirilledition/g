# Configuration

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; current Rust frontend TOML reference | main branch as of 2026-06-30 `--config` and effective configs | Public user docs |

`g` accepts TOML configuration files grouped by section. The Rust frontend owns
TOML decoding, default overlay, validation, and effective config serialization.
The packaged defaults live in `crates/interface/src/config.default.toml`.

This experimental Rust CLI/config branch does not expose the previous
`g config init`, `g config validate`, or `g config explain` helper commands.

## Merge Order

Configuration is merged in this order:

```text
packaged defaults in crates/interface/src/config.default.toml
        < values in --config
        < explicit CLI flags
```

Only explicit CLI flags override the TOML layer. An omitted CLI flag does not
reset a value from the TOML file. The CLI exposes positive REGENIE-compatible
boolean flags only. Set a value to `false` in TOML when a packaged or shared
configuration enables it.

Every `g regenie` run writes an `effective_config.toml` for each phenotype run.
That file is the resolved runtime configuration after defaults, TOML, and CLI
overrides have been applied.

## Run With A Config

```bash
uv run g regenie \
  --config regenie.toml \
  --phenoCol phenotype_b \
  --out /path/to/output/phenotype_b
```

## Run Compatible Configs In One Process

Batch mode accepts complete config files only:

```bash
uv run g batch \
  --config chromosome_21.toml \
  --config chromosome_22.toml
```

Every frontend config is resolved and validated before execution. Output run
roots must be disjoint after existing symlink ancestors are resolved, and
process-global device, JAX, logging, and native-thread policy must match.
Configs may otherwise select different inputs, traits, chromosomes, and kernel
shapes. Engine checks for input availability, sample and prediction
compatibility, existing output state, and resume manifests remain per entry; a
later entry can therefore fail after earlier outputs have completed.

## Layering Patterns

A common cluster pattern is to keep technical runtime policy in one TOML file
and pass run-specific scientific inputs on the CLI.

`server-gpu.toml`:

```toml
[compute]
device = "gpu"
jax_persistent_cache = true
jax_cache_dir = "/path/to/local/jax-cache"
staging_depth = 2

[output]
writer_threads = 2
resume = true

[diagnostics]
telemetry = "progress"
```

Run-specific CLI values then override or fill the scientific fields:

```bash
uv run g regenie \
  --config server-gpu.toml \
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

For reproducible run scripts, put input/output fields in TOML too and override
only the changing phenotype or output prefix:

```bash
uv run g regenie \
  --config regenie.toml \
  --phenoCol phenotype_b \
  --out /path/to/output/phenotype_b
```

## Required Runtime Fields

Packaged defaults cover runtime knobs, but a real Step 2 scan still needs
run-specific input and output fields.

| Field | TOML path | CLI equivalent | Required when |
| --- | --- | --- | --- |
| Genotype source | `[input].bgen` | `--bgen` | Always. |
| Phenotype table | `[input].pheno_file` | `--phenoFile` | Always. |
| Phenotype columns | `[input].pheno_columns` | `--phenoCol`, `--phenoColList` | Always. |
| Step 1 prediction list | `[input].pred` | `--pred` | Always. |
| Output prefix | `[output].out` | `--out` | Always. |
| Sample file | `[input].sample` | `--sample` | When the BGEN does not embed usable sample IDs. |
| Covariate table and columns | `[input].covar_file`, `[input].covar_columns` | `--covarFile`, `--covarCol`, `--covarColList` | When the model includes covariates. |

If `[input].sample` is omitted, `g regenie` reads embedded sample identifiers
from the BGEN. It does not infer an adjacent `.sample` path.

`g regenie` is a Step 2-only command. There is no `step` configuration field or
`--step` compatibility flag.

## Minimal Quantitative Config

This example intentionally omits mutable runtime defaults such as block size,
writer counts, and numerical thresholds. They come from
`crates/interface/src/config.default.toml` unless overridden.

```toml
[input]
bgen = "/path/to/genotypes.bgen"
sample = "/path/to/genotypes.sample"
pheno_file = "/path/to/phenotypes.tsv"
pheno_columns = ["phenotype_continuous"]
covar_file = "/path/to/covariates.tsv"
covar_columns = ["age", "sex"]
pred = "/path/to/regenie_step1_qt_pred.list"

[trait]
trait_type = "quantitative"

[output]
out = "/path/to/output/g_quantitative_regenie2"
```

## Minimal Binary Approximate-Firth Config

```toml
[input]
bgen = "/path/to/genotypes.bgen"
sample = "/path/to/genotypes.sample"
pheno_file = "/path/to/phenotypes.tsv"
pheno_columns = ["phenotype_binary"]
covar_file = "/path/to/covariates.tsv"
covar_columns = ["age", "sex"]
pred = "/path/to/regenie_step1_pred.list"

[trait]
trait_type = "binary"

[binary]
fallback_method = "firth_approximate"
p_threshold = 0.01

[output]
out = "/path/to/output/g_binary_firth_regenie2"
```

## Sections

| Section | Purpose |
| --- | --- |
| `[input]` | Genotype, sample, phenotype, covariate, prediction-list paths, and selected columns. |
| `[trait]` | Quantitative/binary mode, block size, and thread request. |
| `[binary]` | Binary fallback method, p-value threshold, and Firth standard-error output. |
| `[compute]` | Engine runtime, sample semantics, BGEN validation, JAX, numerical, and approximate-Firth tuning. |
| `[output]` | Output prefix, public statistic dtype, Parquet writer settings, and resume controls. |
| `[diagnostics]` | Telemetry, logging, stage timing, profile, and trace controls. |
| `[metadata]` | Optional metadata accepted by the TOML parser but not treated as a `g regenie` option. |

Unknown keys are rejected.

## CLI To TOML Mapping

TOML accepts canonical snake_case keys only. REGENIE spellings are CLI names,
not TOML aliases.

| CLI | TOML |
| --- | --- |
| `--bgen PATH` | `[input] bgen = "PATH"` |
| `--sample PATH` | `[input] sample = "PATH"` |
| `--phenoFile PATH` | `[input] pheno_file = "PATH"` |
| `--phenoCol NAME` | `[input] pheno_columns = ["NAME"]` |
| `--covarFile PATH` | `[input] covar_file = "PATH"` |
| `--covarCol NAME` | `[input] covar_columns = ["NAME"]` |
| `--pred PATH` | `[input] pred = "PATH"` |
| `--qt` | `[trait] qt = true` |
| `--bt` | `[trait] bt = true` |
| `--bsize N` | `[trait] bsize = N` |
| `--threads N` | `[trait] threads = N` |
| `--out PATH` | `[output] out = "PATH"` |
| `--binary-fallback METHOD` | `[binary] fallback_method = METHOD` |
| `--pThresh VALUE` | `[binary] p_threshold = VALUE` |
| `--firth-se` | `[binary] firth_se = true` |

Runtime, scheduling, Parquet writer, resume, diagnostics, and JAX settings are
TOML-only. Important keys include:

| Concern | TOML |
| --- | --- |
| Device and scheduling | `[compute] device`, `staging_depth`, `result_in_flight_limit` |
| BGEN policy | `[compute] trusted_no_missing_diploid`, `trusted_bgen_validation_mode`, `bgen_decode_tile_variant_count` |
| Sample semantics | `[compute] sample_key_mode`, `multi_phenotype_sample_mode` |
| GPU transfer | `[compute] gpu_genotype_format` |
| JAX cache/runtime | `[compute] jax_cache_dir`, `jax_persistent_cache`, `jax_matmul_precision`, `jax_transfer_guard` |
| Output | `[output] output_statistic_dtype`, `writer_threads`, `writer_queue_depth`, `chunks_per_parquet_file`, `parquet_compression` |
| Resume | `[output] resume`, `resume_mode` |
| Diagnostics | `[diagnostics] telemetry` (`off`, `progress`, or `profile`) |

## Trait And Column Semantics

Repeated `--phenoCol` and `--covarCol` flags append names. Their corresponding
`*ColList` forms provide a comma-separated list. Do not mix the repeated and
list forms for the same field.

Trait mode is resolved from `trait_type`, `qt`, and `bt`:

- Both `qt = true` and `bt = true` in the same config layer is an error.
- `bt = true` selects binary mode.
- `qt = true` selects quantitative mode.
- Otherwise the merged `trait_type` applies, defaulting to quantitative.
- Binary-only options are rejected when the final trait type is quantitative and
  those options were explicitly provided.

## Effective Config And Manifest

Each phenotype output run writes:

```text
effective_config.toml
run_manifest.json
```

`effective_config.toml` is the final merged config. `run_manifest.json` records
execution-plan-affecting inputs and settings, file fingerprints, sample/variant
counts, output writer settings, and committed chunks. Resume compares the
requested run against this manifest before reusing chunks.

See [Resume and Manifest](resume-and-manifest.md) for resume modes and
compatibility checks.

## Validation

Validation happens during config construction and run preflight. The current
Rust frontend does not provide standalone `g config validate` or `g config
explain` commands.

Config construction rejects:

- invalid TOML syntax;
- unknown sections or keys;
- wrong value types;
- mutually exclusive phenotype or covariate column spellings;
- incompatible trait flags such as simultaneous quantitative and binary mode;
- binary-only options explicitly supplied for a quantitative run.

Run preflight then checks file availability, sample and column contracts,
prediction-list compatibility, output directory state, and resume manifest
compatibility. In batch mode these engine checks run when each entry starts;
only frontend config construction, disjoint output roots, and process-global
policy compatibility are checked across the complete batch before execution.

## Defaults Policy

Do not copy mutable defaults into runbooks unless they are generated from the
current checkout. The authoritative source is
`crates/interface/src/config.default.toml`.

For implementation rules behind this interface, see
[Configuration Frontend](../development/configuration-frontend.md).

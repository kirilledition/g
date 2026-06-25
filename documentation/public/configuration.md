# Configuration

| Status | Applies to | Owner |
| --- | --- | --- |
| Current Rust frontend TOML reference | `--config`, effective configs, and config/CLI merge semantics | Public interface |

`g` accepts TOML configuration files grouped by section. The Rust frontend owns
TOML decoding, default overlay, validation, and effective config serialization.
The packaged defaults live in `src/interface/config.default.toml`.

This experimental Rust CLI/config branch does not expose the previous
`g config init`, `g config validate`, or `g config explain` helper commands.

## Merge Order

Configuration is merged in this order:

```text
packaged defaults in src/interface/config.default.toml
        < values in --config
        < explicit CLI flags
```

Only explicit CLI flags override the TOML layer. An omitted CLI flag does not
reset a value from the TOML file. For boolean values, use negative CLI forms such
as `--no-resume` when you need to override a TOML `true` value.

Every `g regenie` run writes an `effective_config.toml` for each phenotype run.
That file is the resolved runtime configuration after defaults, TOML, and CLI
overrides have been applied.

## Run With A Config

```bash
uv run g regenie \
  --config regenie.toml \
  --phenoCol phenotype_b \
  --out /path/to/output/phenotype_b \
  --device gpu
```

## Required Runtime Fields

Packaged defaults cover runtime knobs, but a real Step 2 scan still needs
run-specific input and output fields.

| Field | TOML path | CLI equivalent | Required when |
| --- | --- | --- | --- |
| Genotype source | `[input].bgen` | `--bgen` | Always. |
| Phenotype table | `[input].pheno_file` or alias `[input].phenoFile` | `--phenoFile` | Always. |
| Phenotype columns | `[input].pheno_columns`, `[input].phenoCol`, or `[input].phenoColList` | `--phenoCol`, `--phenoColList` | Always. |
| Step 1 prediction list | `[input].pred` | `--pred` | Always. |
| Output prefix | `[output].out` | `--out` | Always. |
| Sample file | `[input].sample` | `--sample` | When the BGEN does not embed usable sample IDs. |
| Covariate table and columns | `[input].covar_file`, `[input].covar_columns`, plus REGENIE-style aliases | `--covarFile`, `--covarCol`, `--covarColList` | When the model includes covariates. |

If `[input].sample` is omitted, `g regenie` reads embedded sample identifiers
from the BGEN. It does not infer an adjacent `.sample` path.

`[trait].step` must resolve to `2`. REGENIE Step 1 is not implemented.

## Minimal Quantitative Config

This example intentionally omits mutable runtime defaults such as block size,
writer counts, and numerical thresholds. They come from
`src/interface/config.default.toml` unless overridden.

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
step = 2
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
step = 2
trait_type = "binary"

[binary]
firth = true
approx = true
p_threshold = 0.01

[output]
out = "/path/to/output/g_binary_firth_regenie2"
```

## Sections

| Section | Purpose |
| --- | --- |
| `[input]` | Genotype, sample, phenotype, covariate, prediction-list paths, and selected columns. |
| `[trait]` | Step, quantitative/binary mode, block size, and thread request. |
| `[binary]` | Binary fallback flags, Firth mode, and p-value threshold. |
| `[compute]` | Engine runtime, sample semantics, BGEN validation, JAX, numerical, and approximate-Firth tuning. |
| `[output]` | Output prefix, chunk format, public statistic dtype, writer settings, Parquet finalization, and resume controls. |
| `[diagnostics]` | Telemetry, logging, progress, profile, and trace controls. |
| `[metadata]` | Optional metadata accepted by the TOML parser but not treated as a `g regenie` option. |

Unknown keys are rejected.

## CLI To TOML Mapping

TOML canonical spelling is native snake_case. REGENIE-style aliases are accepted
for selected input and binary fields where noted.

| CLI | TOML |
| --- | --- |
| `--bgen PATH` | `[input] bgen = "PATH"` |
| `--sample PATH` | `[input] sample = "PATH"` |
| `--phenoFile PATH` | `[input] pheno_file = "PATH"`; alias `phenoFile` is accepted |
| `--phenoCol NAME` | `[input] pheno_columns = ["NAME"]`; aliases `phenoCol` and `phenoColList` are accepted |
| `--covarFile PATH` | `[input] covar_file = "PATH"`; alias `covarFile` is accepted |
| `--covarCol NAME` | `[input] covar_columns = ["NAME"]`; aliases `covarCol` and `covarColList` are accepted |
| `--pred PATH` | `[input] pred = "PATH"` |
| `--step 2` | `[trait] step = 2` |
| `--qt`, `--no-qt` | `[trait] qt = true` or `false` |
| `--bt`, `--no-bt` | `[trait] bt = true` or `false` |
| `--bsize N` | `[trait] bsize = N` |
| `--threads N` | `[trait] threads = N` |
| `--out PATH` | `[output] out = "PATH"` |
| `--firth`, `--no-firth` | `[binary] firth = true` or `false` |
| `--approx`, `--no-approx` | `[binary] approx = true` or `false` |
| `--pThresh VALUE` | `[binary] p_threshold = VALUE`; alias `pThresh` is accepted |
| `--firth-se`, `--no-firth-se` | `[binary] firth_se = true` or `false`; alias `firth-se` is accepted |

Runtime CLI flags map directly to the sectioned snake_case TOML surface:

| CLI | TOML |
| --- | --- |
| `--device gpu` | `[compute] device = "gpu"` |
| `--staging_depth N` | `[compute] staging_depth = N` |
| `--native_callback_batch_size N` | `[compute] native_callback_batch_size = N` |
| `--trusted_no_missing_diploid` | `[compute] trusted_no_missing_diploid = true` |
| `--trusted_bgen_validation_mode cache_on_miss` | `[compute] trusted_bgen_validation_mode = "cache_on_miss"` |
| `--sample_key_mode fid_iid` | `[compute] sample_key_mode = "fid_iid"` |
| `--multi_phenotype_sample_mode complete-case` | `[compute] multi_phenotype_sample_mode = "complete-case"` |
| `--gpu_genotype_format auto` | `[compute] gpu_genotype_format = "auto"`; default is `auto`, which resolves to packed8 only for eligible single-trait binary GPU runs |
| `--jax_cache_dir PATH` | `[compute] jax_cache_dir = "PATH"` |
| `--format parquet` | `[output] format = "parquet"` |
| `--output_statistic_dtype float64` | `[output] output_statistic_dtype = "float64"`; default is `"float32"` |
| `--writer_threads N` | `[output] writer_threads = N` |
| `--chunks_per_arrow_file N` | `[output] chunks_per_arrow_file = N` |
| `--resume` | `[output] resume = true` |
| `--resume_mode strict` | `[output] resume_mode = "strict"` |
| `--finalize_parquet` | `[output] finalize_parquet = true` |
| `--telemetry profile` | `[diagnostics] telemetry = "profile"` |
| `--log_dir PATH` | `[diagnostics] log_dir = "PATH"` |
| `--log_stderr`, `--no-log_stderr` | `[diagnostics] log_stderr = true` or `false` |
| `--trace_event_cap N` | `[diagnostics] trace_event_cap = N` |

## Trait And Column Semantics

`pheno_columns`, `phenoCol`, and `phenoColList` are mutually exclusive after
normalization. `covar_columns`, `covarCol`, and `covarColList` have the same
rule.

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

## Defaults Policy

Do not copy mutable defaults into runbooks unless they are generated from the
current checkout. The authoritative source is
`src/interface/config.default.toml`.

For implementation rules behind this interface, see
[Configuration Frontend](../development/configuration-frontend.md).

# Configuration

| Status | Applies to | Owner |
| --- | --- | --- |
| Canonical user-facing TOML reference | `--config`, `g config`, effective configs, and config/CLI merge semantics | Public interface |

`g` accepts TOML configuration files that use the same option names as the CLI,
grouped by section. The live schema is defined in `src/g/interface/toml_schema.py`
and the packaged defaults live in `src/g/config.default.toml`.

Use this page for merge behavior and layout. Use the checked-out code for the
current default values:

```bash
uv run g config init
uv run g config explain
```

## Merge Order

Configuration is merged in this order:

```text
packaged defaults in src/g/config.default.toml
        < values in --config
        < explicit CLI flags
```

Only explicit CLI flags override the TOML layer. An omitted CLI flag does not
reset a value from the TOML file. For boolean values, use negative CLI forms such
as `--no-g-resume` when you need to override a TOML `true` value.

Every `g regenie` run writes an `effective_config.toml` for each phenotype run.
That file is the resolved runtime configuration after defaults, TOML, and CLI
overrides have been applied.

## Create And Validate

```bash
uv run g config init --out regenie.toml
uv run g config validate regenie.toml
```

Run with a config and override selected values:

```bash
uv run g regenie \
  --config regenie.toml \
  --phenoCol phenotype_b \
  --out /path/to/output/phenotype_b \
  --g-device gpu
```

## Required Runtime Fields

Packaged defaults cover runtime knobs, but a real Step 2 scan still needs
run-specific input and output fields.

| Field | TOML path | CLI equivalent | Required when |
| --- | --- | --- | --- |
| Genotype source | `[input].bgen` | `--bgen` | Always. |
| Phenotype table | `[input].phenoFile` | `--phenoFile` | Always. |
| Phenotype columns | `[input].phenoCol` or `[input].phenoColList` | `--phenoCol`, `--phenoColList` | Always. |
| Step 1 prediction list | `[input].pred` | `--pred` | Always. |
| Output prefix | `[output].out` | `--out` | Always. |
| Sample file | `[input].sample` | `--sample` | When BGEN sample IDs are absent or unsuitable. |
| Covariate table and columns | `[input].covarFile`, `[input].covarCol`, `[input].covarColList` | `--covarFile`, `--covarCol`, `--covarColList` | When the model includes covariates. |

`[trait].step` must resolve to `2`. REGENIE Step 1 is not implemented.

## Minimal Quantitative Config

This example intentionally omits mutable runtime defaults such as block size,
writer counts, and numerical thresholds. They come from
`src/g/config.default.toml` unless overridden.

```toml
[input]
bgen = "/path/to/genotypes.bgen"
sample = "/path/to/genotypes.sample"
phenoFile = "/path/to/phenotypes.tsv"
phenoCol = "phenotype_continuous"
covarFile = "/path/to/covariates.tsv"
covarColList = "age,sex"
pred = "/path/to/regenie_step1_qt_pred.list"

[trait]
step = 2
qt = true

[output]
out = "/path/to/output/g_quantitative_regenie2"
```

## Minimal Binary Approximate-Firth Config

```toml
[input]
bgen = "/path/to/genotypes.bgen"
sample = "/path/to/genotypes.sample"
phenoFile = "/path/to/phenotypes.tsv"
phenoCol = "phenotype_binary"
covarFile = "/path/to/covariates.tsv"
covarColList = "age,sex"
pred = "/path/to/regenie_step1_pred.list"

[trait]
step = 2
bt = true

[binary]
firth = true
approx = true
pThresh = 0.01

[output]
out = "/path/to/output/g_binary_firth_regenie2"
```

## Sections

| Section | Purpose |
| --- | --- |
| `[input]` | Genotype, sample, phenotype, covariate, prediction-list paths, and selected columns. |
| `[filters]` | Recognized but unsupported REGENIE keep/remove/extract/exclude options. |
| `[trait]` | Step, quantitative/binary mode, block size, and thread request. |
| `[binary]` | Binary fallback flags, Firth mode, p-value threshold, and SPA recognition. |
| `[output]` | User-facing output prefix. |
| `[g.compute]` | Engine runtime, sample semantics, BGEN validation, JAX, numerical, and approximate-Firth tuning. |
| `[g.output]` | Chunk format, writer settings, Parquet finalization, and resume controls. |
| `[g.diagnostics]` | Telemetry, logging, progress, profile, and trace controls. |
| `[metadata]` | Optional user metadata accepted by the TOML parser but not treated as a `g regenie` option. |

Unknown keys are rejected. Unsupported recognized options are also rejected when
they are set to active values.

## CLI To TOML Mapping

REGENIE-style CLI names keep their spelling in TOML:

| CLI | TOML |
| --- | --- |
| `--bgen PATH` | `[input] bgen = "PATH"` |
| `--sample PATH` | `[input] sample = "PATH"` |
| `--phenoFile PATH` | `[input] phenoFile = "PATH"` |
| `--phenoCol NAME` | `[input] phenoCol = "NAME"` or `phenoCol = ["A", "B"]` |
| `--phenoColList A,B` | `[input] phenoColList = "A,B"` |
| `--covarFile PATH` | `[input] covarFile = "PATH"` |
| `--covarCol NAME` | `[input] covarCol = "NAME"` or `covarCol = ["A", "B"]` |
| `--covarColList A,B` | `[input] covarColList = "A,B"` |
| `--pred PATH` | `[input] pred = "PATH"` |
| `--step 2` | `[trait] step = 2` |
| `--qt`, `--no-qt` | `[trait] qt = true` or `false` |
| `--bt`, `--no-bt` | `[trait] bt = true` or `false` |
| `--bsize N` | `[trait] bsize = N` |
| `--threads N` | `[trait] threads = N` |
| `--out PATH` | `[output] out = "PATH"` |
| `--firth`, `--no-firth` | `[binary] firth = true` or `false` |
| `--approx`, `--no-approx` | `[binary] approx = true` or `false` |
| `--pThresh VALUE` | `[binary] pThresh = VALUE` |
| `--firth-se`, `--no-firth-se` | `[binary] firth-se = true` or `false` |

`g`-specific CLI flags drop the leading `g-` inside the `[g.*]` namespace:

| CLI | TOML |
| --- | --- |
| `--g-device gpu` | `[g.compute] device = "gpu"` |
| `--g-staging-depth N` | `[g.compute] staging-depth = N` |
| `--g-trusted-no-missing-diploid` | `[g.compute] trusted-no-missing-diploid = true` |
| `--g-trusted-bgen-validation-mode cache_on_miss` | `[g.compute] trusted-bgen-validation-mode = "cache_on_miss"` |
| `--g-sample-key-mode fid_iid` | `[g.compute] sample-key-mode = "fid_iid"` |
| `--g-multi-phenotype-sample-mode complete-case` | `[g.compute] multi-phenotype-sample-mode = "complete-case"` |
| `--g-jax-cache-dir PATH` | `[g.compute] jax-cache-dir = "PATH"` |
| `--g-output-format parquet` | `[g.output] format = "parquet"` |
| `--g-writer-threads N` | `[g.output] writer-threads = N` |
| `--g-output-chunks-per-arrow-file N` | `[g.output] chunks-per-arrow-file = N` |
| `--g-resume` | `[g.output] resume = true` |
| `--g-resume-mode strict` | `[g.output] resume-mode = "strict"` |
| `--g-finalize-parquet` | `[g.output] finalize-parquet = true` |
| `--g-telemetry profile` | `[g.diagnostics] telemetry = "profile"` |
| `--g-log-dir PATH` | `[g.diagnostics] log-dir = "PATH"` |
| `--g-log-stderr`, `--no-g-log-stderr` | `[g.diagnostics] log-stderr = true` or `false` |
| `--g-trace-event-cap N` | `[g.diagnostics] trace-event-cap = N` |

Use `uv run g config explain <option>` for exact accepted values. For example:

```bash
uv run g config explain g-output-format
uv run g config explain g-trusted-bgen-validation-mode
```

## Trait And Column Semantics

`phenoCol` and `phenoColList` are mutually exclusive after normalization. Use
one repeated/list form for phenotype columns. `covarCol` and `covarColList` have
the same rule.

Trait mode is resolved from `qt` and `bt`:

- Both true in the same config layer is an error.
- `bt = true` selects binary mode.
- Otherwise quantitative mode is selected by the merged config.
- Binary-only options are rejected when the final trait type is quantitative.

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
current checkout. The authoritative sources are:

- `src/g/config.default.toml` for packaged defaults.
- `uv run g config init` for a starter config rendered by the current code.
- `uv run g config explain <option>` for option metadata.

For implementation rules behind this interface, see
[Configuration Frontend](../development/configuration-frontend.md).

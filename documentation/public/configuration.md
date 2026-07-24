# Configuration

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; current Rust frontend TOML reference | main branch as of 2026-07-24 `--config` and effective configs | Public user docs |

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

CLI overrides are field-specific. In particular, `--bgen` replaces only
`[input].bgen`; it does not clear or replace
`[input].bgen_content_sha256`.

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
jax_cache_dir = "/path/to/local/jax-cache"
cpu_threads = 8

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
  --covarCol age --covarCol sex \
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
| Phenotype columns | `[input].pheno_columns` | Repeated `--phenoCol` | Always. |
| Step 1 prediction list | `[input].pred` | `--pred` | Always. |
| Output prefix | `[output].out` | `--out` | Always. |
| Sample file | `[input].sample` | `--sample` | Always. |
| Covariate table and columns | `[input].covar_file`, `[input].covar_columns` | `--covarFile`, repeated `--covarCol` | When the model includes covariates. |

The required Oxford sample file supplies the BGEN row identities. Sample
alignment always uses non-empty, unique `(FID, IID)` pairs; there is no public
IID-only matching mode.

`g regenie` is a Step 2-only command. There is no `step` configuration field or
`--step` compatibility flag.

## Optional BGEN Content Selector

`[input].bgen_content_sha256` is an optional canonical content selector.
It accepts exactly one 64-character lowercase hexadecimal SHA-256 string:

```toml
[input]
bgen = "/path/to/genotypes.bgen"
bgen_content_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
```

Uppercase hexadecimal, non-hexadecimal characters, and strings of any other
length are rejected. The selector is TOML-only; there is no CLI digest flag.
If `--bgen` is also supplied, it overrides the locator while preserving the
TOML selector.

The selector does not replace the locator. `[input].bgen` remains required, and
frontend run validation requires its string value but does not probe BGEN
existence. The engine reconciles the selector with any BGEN fingerprint
required by existing output, then performs a content-selected open. Existing
output supplies both its digest and expected byte count; a different configured
digest is rejected before locator access or output ownership. A selected
same-process snapshot-cache hit may reuse authenticated content without
accessing a missing request locator. Selected cache misses and unselected opens
still require an accessible locator. Content-selected inputs must fit the
256 MiB owned-snapshot ceiling; a larger selected miss is rejected rather than
falling back to unattested positioned I/O.

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
| `[input]` | Genotype locator and optional content selector, sample, phenotype, covariate, prediction-list paths, and selected columns. |
| `[trait]` | Quantitative/binary mode and block size. |
| `[binary]` | Binary fallback method, p-value threshold, and Firth standard-error output. |
| `[compute]` | Device, native CPU threads, multi-phenotype sample selection, JAX, numerical, and approximate-Firth tuning. |
| `[output]` | Output prefix, writer concurrency, and resume controls. |
| `[diagnostics]` | Telemetry selection. |
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
| `--out PATH` | `[output] out = "PATH"` |
| `--recover-output-attempt ATTEMPT_ID` | `[output] recover_attempt = "ATTEMPT_ID"` |
| `--fenced-output-owner-claim CLAIM_ID` | `[output] fenced_owner_claim_id = "CLAIM_ID"` |
| `--binary-fallback METHOD` | `[binary] fallback_method = METHOD` |
| `--pThresh VALUE` | `[binary] p_threshold = VALUE` |
| `--firth-se` | `[binary] firth_se = true` |

Runtime, compute, output, resume, diagnostics, and JAX settings are TOML-only.
Important keys include:

| Concern | TOML |
| --- | --- |
| BGEN content selector | `[input].bgen_content_sha256` |
| Device and native workers | `[compute] device`, `cpu_threads` |
| Multi-phenotype sample selection | `[compute] multi_phenotype_sample_mode` |
| Binary compute | `[compute] firth_batch_size`, `firth_candidate_capacity`, and the documented null/Firth tolerances |
| JAX cache | `[compute] jax_cache_dir` |
| Output | `[output] output_run_directory`, `writer_threads` |
| Resume | `[output] resume`, `recover_attempt` |
| Diagnostics | `[diagnostics] telemetry` (`off`, `progress`, or `profile`) |

`[compute] firth_candidate_capacity` is the per-trait scaling value for an
aggregate hard static capacity, with a packaged default of `1024`. The runtime
caps that value by the static compute chunk width and multiplies it by the
trait count. Larger values enlarge the compiled approximate-Firth executable.
If a batch exceeds the aggregate capacity, the run fails after normal batch
synchronization instead of truncating candidates; increase the value and rerun.

For approximate Firth, `[compute].firth_maximum_iterations` supplies a
floor-divided half-budget to each solver phase.
`[compute].firth_pseudo_maximum_iterations` caps only dense pseudo-Firth; with
the packaged defaults, dense lanes use `min(floor(250 / 2), 50) = 50`
iterations. Sparse carrier-only lanes use the full half-budget, `125` with the
packaged defaults, whether they use compact or full-width masked storage.

Decode tiling, scheduler queue depths, Parquet grouping/compression, packed8
BGEN compatibility validation, and packed8-versus-dosage delivery are internal
implementation policies owned by the genotype, engine, and output crates. They
are intentionally not accepted as configuration keys.

`recover_attempt` is an emergency takeover control for a nonterminal output
attempt. It requires `resume = true` and the exact path-safe attempt identifier.
Normal resume after a durable interruption or failure terminal does not set it.

`fenced_owner_claim_id` is a one-shot crash-recovery assertion. Use it only
after an external scheduler or node-level coordinator has proved that the host
and process reported in the surviving-claim error can no longer write. It
requires `resume = true`; the exact claim identifier must still be the current
Active authority leaf. The runtime never infers fencing from age or PID state
and never deletes an authority record.

## Trait And Column Semantics

Repeated `--phenoCol` and `--covarCol` flags append names. No `*ColList` CLI
forms are accepted.

Trait mode is resolved from `trait_type`, `qt`, and `bt`:

- Both `qt = true` and `bt = true` in the same config layer is an error.
- `bt = true` selects binary mode.
- `qt = true` selects quantitative mode.
- Otherwise the merged `trait_type` applies, defaulting to quantitative.
- Binary-only options are rejected when the final trait type is quantitative and
  those options were explicitly provided.

## Effective Config And Manifest

Each phenotype within an immutable attempt writes:

```text
attempts/<attempt-id>/<phenotype-output-name>/
  effective_config.toml
  run_manifest.json
```

`effective_config.toml` is the final merged config. `run_manifest.json` records
execution-plan-affecting inputs and settings, file fingerprints, sample/variant
counts, output writer settings, immutable receipts, and committed chunks.
Resume traverses the `.g-output` lineage and compares the requested run against
every bound manifest before reusing chunks.

See [Resume and Manifest](resume-and-manifest.md) for strict resume behavior and
compatibility checks.

## Validation

Validation happens during config construction and run preflight. The current
Rust frontend does not provide standalone `g config validate` or `g config
explain` commands.

Config construction rejects:

- invalid TOML syntax;
- unknown sections or keys;
- wrong value types;
- incompatible trait flags such as simultaneous quantitative and binary mode;
- binary-only options explicitly supplied for a quantitative run.

Run preflight then acquires and authenticates BGEN as required, checks other
input-file availability, sample and column contracts, prediction-list
compatibility, output directory state, and resume manifest compatibility. In
batch mode these engine checks run when each entry starts; only frontend config
construction, disjoint output roots, and process-global policy compatibility
are checked across the complete batch before execution.

## Defaults Policy

Do not copy mutable defaults into runbooks unless they are generated from the
current checkout. The authoritative source is
`crates/interface/src/config.default.toml`.

For implementation rules behind this interface, see
[Configuration Frontend](../development/configuration-frontend.md).

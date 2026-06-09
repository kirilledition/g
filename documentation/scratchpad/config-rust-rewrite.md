Yes, with that framing I would review the branch differently.

If the only hard public contract is **`g regenie` must accept REGENIE CLI flags exactly**, then I would not force the TOML/config structure to mimic the old Python design. In fact, I would lean into the Rust redesign:

```text
CLI:
  REGENIE-compatible compatibility surface

TOML:
  native g configuration format, cleanly typed by Serde

Python:
  thin wrapper around Rust config objects

ExecutionPlan:
  the boundary where config becomes engine behavior
```

The branch is moving in the right direction. The main thing I would change is to make the architecture more explicitly typed and layered, rather than replacing Python stringly parsing with Rust stringly parsing.

---

# Implementation progress

Status for the `rust-cli-config-pyo3` follow-up pass:

* Implemented: Rust owns the public interface module under `src/interface`, the packaged default TOML lives beside that module, and Python remains a thin PyO3 consumer.
* Implemented: `config.default.toml` remains the single packaged default source and is parsed through a cached `OnceLock`; build-time default generation is intentionally not restored.
* Planned in this pass: split the Rust interface into resolved data, partial TOML schema, overlay/provenance, TOML IO, pure validation, run-boundary validation, and grouped CLI parser modules.
* Planned in this pass: add internal provenance for validation-relevant explicit fields and use it for trait/binary-only validation instead of comparing values to defaults.
* Planned in this pass: keep `from_toml` and TOML-shaped Python `from_options` free of filesystem existence checks; use run validation at CLI and runner execution boundaries.
* Intentionally adjusted: non-REGENIE CLI flags remain snake_case, for example `--staging_depth`, `--writer_threads`, and `--jax_persistent_cache`, matching the product decision for this branch.
* Intentionally adjusted: effective TOML keeps direct Serde serialization for now; no custom render tree is introduced unless internal names later diverge from the public TOML contract.
* Deferred: tests are not edited in this pass. Existing tests may need follow-up updates for the new internal provenance and run-validation boundary.
* Deferred: broad user documentation updates are not included because this is still an experimental review branch; this scratchpad records the architectural status.

---

# High-level verdict

I like the direction:

```text
Rust owns:
  CLI parsing
  TOML decoding
  config layering
  default loading
  validation
  effective TOML serialization
  PyO3 config objects

Python owns:
  JAX runtime
  execution planning
  orchestration
  compute pipeline
```

The branch already has strong pieces:

* `domain.rs` defines typed domain values like `PositiveF32`, `Probability`, `ProbabilityFloor`, and string enums. That is exactly the right Rust style for configuration validation.
* `schema.rs` uses `PartialConfig` and partial section structs with `Option<T>` fields, overlay behavior, and resolution into complete config data.
* `defaults.rs` embeds and hashes `config.default.toml`, which is much cleaner than scattered Python defaults.
* `data.rs` defines resolved config structs that are no longer partial or ambiguous.
* `src/g/interface/config.py` is now a thin compatibility shim around Rust-owned config objects, which is the right shape.

I would keep the migration. I would not go back to Python config parsing.

---

# Recommended final architecture

## Core principle

Do **not** try to have one magical `OptionSpec` table that manually reimplements Clap, Serde, and TOML.

Let each Rust crate do what it is good at:

```text
clap:
  parse CLI into typed structures

serde + toml:
  decode TOML into typed partial structures

Rust domain types:
  validate scalar domains: positive ints, probabilities, enums, dtype choices

overlay layer:
  merge defaults < TOML < CLI

semantic validation:
  validate cross-field and run-level rules

PyO3:
  expose resolved typed config to Python
```

The `toml` crate is Serde-compatible and supports deserializing TOML directly into Rust structures. ([Docs.rs][1]) Serde supports field renames, aliases, defaults, skip rules, and other field-level behavior that are directly useful for this config layer. ([Serde][2]) Clap derive supports typed parsers, flattened argument groups, and `ValueEnum` parsing for enums, so you can avoid custom parsing for most CLI values. ([Docs.rs][3])

So the “best” architecture is **typed Rust structs plus small conversion layers**, not a new custom option registry unless you want generated docs/config explain later.

---

# CLI vs TOML: I would separate their design

Since the only hard compatibility contract is CLI, I would use this split:

## CLI should be compatibility-first

The CLI must accept REGENIE flags exactly:

```bash
g regenie \
  --step 2 \
  --bgen data.bgen \
  --sample data.sample \
  --phenoFile pheno.tsv \
  --phenoCol BMI \
  --covarFile covar.tsv \
  --covarColList age,sex,PC1 \
  --pred pred.list \
  --qt \
  --bsize 16384 \
  --out results/bmi
```

For non-REGENIE options, I would still use `--g-*`:

```bash
--g-device gpu
--g-staging-depth 2
--g-output-format parquet
--g-writer-threads 8
--g-telemetry profile
```

Even though you allowed option names to change, I still recommend `--g-*` for non-REGENIE CLI flags. It protects the drop-in replacement contract because unprefixed names like `--device`, `--format`, or `--writer-threads` can look like REGENIE flags or collide with future REGENIE flags.

## TOML should be native and ergonomic

For TOML, I no longer think you must use `[g.compute]` and `g-*`-style keys. If you want to delegate to Serde and avoid custom mapping, this is better:

```toml
[input]
bgen = "data/chr22.bgen"
sample = "data/chr22.sample"
pheno_file = "data/pheno.tsv"
pheno_columns = ["BMI"]
covar_file = "data/covar.tsv"
covar_columns = ["age", "sex", "PC1"]
pred = "data/pred.list"

[trait]
step = 2
qt = true
bt = false
bsize = 16384

[binary]
firth = false
approx = false
p_threshold = 0.05
firth_se = false

[compute]
device = "gpu"
staging_depth = 2
trusted_no_missing_diploid = true
score_dtype = "float32"
firth_dtype = "float64"

[output]
out = "results/bmi"
format = "parquet"
writer_threads = 8
writer_queue_depth = 16
chunks_per_arrow_file = 16

[diagnostics]
telemetry = "profile"
log_filter = "info"
log_stderr = true
```

That is not REGENIE-compatible TOML, but REGENIE does not have a standard TOML contract that you need to preserve. TOML can be `g`’s native typed config language.

The branch already moves in this direction: `config.default.toml` uses `[compute]`, `[output]`, and `[diagnostics]` with snake_case keys, and that is defensible if you commit to “TOML is native g config.”

My recommendation:

```text
CLI:
  exact REGENIE names for REGENIE flags
  --g-* for non-REGENIE options

TOML:
  native Rust/Serde-friendly snake_case sections and keys

Python dict:
  preferably TOML-shaped nested mapping
  optional CLI-shaped helper only as convenience
```

---

# Biggest code issue: current CLI field names

The current Rust `RegenieCli` uses `#[command(rename_all = "verbatim")]` and many unannotated fields such as `staging_depth`, `writer_threads`, `jax_persistent_cache`, etc.

That is risky. Clap’s derive machinery case-converts field names into argument names, and `rename_all` controls that behavior. Clap’s docs describe `rename_all` as the override for field/variant name case conversion. ([Docs.rs][3]) With `verbatim`, fields like `staging_depth` can become awkward flags like:

```text
--staging_depth
```

instead of:

```text
--g-staging-depth
```

or even:

```text
--staging-depth
```

You already explicitly annotate REGENIE mixed-case flags like `--phenoFile`, `--phenoCol`, and `--pThresh`, which is good.  I would do the same for every non-REGENIE CLI flag.

Recommended pattern:

```rust
#[derive(Clone, Debug, Parser)]
#[command(
    about = "Run a REGENIE-compatible step 2 association scan.",
    disable_version_flag = true
)]
struct RegenieCli {
    #[command(flatten)]
    regenie: RegenieCompatibilityArgs,

    #[command(flatten)]
    binary: BinaryArgs,

    #[command(flatten)]
    g_compute: GComputeArgs,

    #[command(flatten)]
    g_output: GOutputArgs,

    #[command(flatten)]
    g_diagnostics: GDiagnosticsArgs,
}
```

Then:

```rust
#[derive(Clone, Debug, clap::Args)]
#[command(next_help_heading = "REGENIE input options")]
struct RegenieInputArgs {
    #[arg(long)]
    bgen: Option<String>,

    #[arg(long)]
    sample: Option<String>,

    #[arg(long = "phenoFile")]
    pheno_file: Option<String>,

    #[arg(long = "phenoCol", action = ArgAction::Append)]
    pheno_col: Vec<String>,

    #[arg(long = "phenoColList")]
    pheno_col_list: Option<NameList>,

    #[arg(long = "covarFile")]
    covar_file: Option<String>,

    #[arg(long = "covarCol", action = ArgAction::Append)]
    covar_col: Vec<String>,

    #[arg(long = "covarColList")]
    covar_col_list: Option<NameList>,

    #[arg(long)]
    pred: Option<String>,
}
```

And:

```rust
#[derive(Clone, Debug, clap::Args)]
#[command(next_help_heading = "g compute options")]
struct GComputeArgs {
    #[arg(long = "g-device")]
    device: Option<DeviceValue>,

    #[arg(long = "g-staging-depth")]
    staging_depth: Option<NonZeroU32>,

    #[arg(long = "g-trusted-no-missing-diploid", action = ArgAction::SetTrue)]
    trusted_no_missing_diploid: bool,

    #[arg(long = "no-g-trusted-no-missing-diploid", hide = true, action = ArgAction::SetTrue)]
    no_trusted_no_missing_diploid: bool,

    #[arg(long = "g-score-dtype")]
    score_dtype: Option<FloatingPointDtypeValue>,

    #[arg(long = "g-firth-dtype")]
    firth_dtype: Option<FloatingPointDtypeValue>,
}
```

This avoids accidental CLI names. It also makes help output easier to organize. Clap supports flattened argument groups, which is exactly what you want here. ([Docs.rs][3])

---

# Recommended file architecture

I would keep `src/config_frontend`, but split the responsibilities more sharply.

Current layout:

```text
src/config_frontend/
  cli.rs
  data.rs
  defaults.rs
  domain.rs
  mod.rs
  resolve.rs
  schema.rs
  serialization.rs
  validation.rs
```

This is a good start, but `schema.rs` is doing too much: partial schema definitions, overlay logic, resolution, field grouping, and helper logic.

I would refactor to:

```text
src/config_frontend/
  mod.rs
  error.rs

  domain.rs
    Newtypes and enums:
      PositiveF32
      Probability
      ProbabilityFloor
      DeviceValue
      OutputFormatValue
      FloatingPointDtypeValue
      NameList

  partial.rs
    PartialConfig
    PartialInputConfig
    PartialTraitConfig
    PartialBinaryConfig
    PartialComputeConfig
    PartialOutputConfig
    PartialDiagnosticsConfig

  resolved.rs
    RegenieConfigData
    InputConfigData
    TraitConfigData
    BinaryConfigData
    GComputeConfigData
    GOutputConfigData
    GDiagnosticsConfigData

  overlay.rs
    ConfigLayer
    overlay logic
    precedence rules
    explicitness/provenance tracking

  defaults.rs
    include_str!("../g/config.default.toml")
    load packaged defaults
    default hash

  toml.rs
    decode_toml_file_layer
    decode_toml_text_layer
    encode effective TOML

  validation.rs
    pure semantic validation

  run_validation.rs
    filesystem/run validation:
      path existence
      output directory/resume safety
      manifest compatibility, if kept here

  cli/
    mod.rs
    root.rs
    regenie.rs
    groups.rs
    conversion.rs
    help.rs
```

Keep PyO3 bindings separate:

```text
src/python/config/
  mod.rs
  conversion.rs
```

The current PyO3 module is long but conceptually correct. It can stay mostly as-is, though I would eventually split getters by config section.

---

# Recommended data model

## 1. `PartialConfig`: only user-supplied or default-layer values

Use Serde structs with `Option<T>`:

```rust
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PartialConfig {
    pub input: PartialInputConfig,

    #[serde(rename = "trait")]
    pub trait_config: PartialTraitConfig,

    pub binary: PartialBinaryConfig,
    pub compute: PartialComputeConfig,
    pub output: PartialOutputConfig,
    pub diagnostics: PartialDiagnosticsConfig,

    pub metadata: Option<toml::Table>,
}
```

This is already close to the branch.

## 2. `RegenieConfigData`: fully resolved, no optional defaults

Resolved config should have concrete values for everything that has a default, and `Option<T>` only for genuinely optional values like `sample`, `covar_file`, `jax_cache_dir`, `log_file`, etc. The current `GComputeConfigData`, `GOutputConfigData`, and `GDiagnosticsConfigData` are close to this shape.

This is exactly what the engine should consume.

## 3. Keep config construction pure

This is one of the biggest architecture recommendations.

Right now `from_toml_path` and `from_options` go through `resolve_config_layers`, which validates the resolved config immediately.  Validation currently checks whether input paths exist.

I would split validation into two levels:

```text
Config resolution:
  type validity
  defaults
  enum parsing
  positive numbers
  field conflicts
  required structural fields if producing run-ready config

Run validation:
  filesystem paths exist
  output directory state
  resume compatibility
  manifest compatibility
```

Why? This should be possible on a laptop without server data mounted:

```python
cfg = RegenieConfig.from_toml("server_run.toml")
print(cfg.to_toml())
```

Path existence should be checked when launching a run, not necessarily when constructing a config object.

Recommended API:

```rust
pub fn from_toml_path(path: &Path) -> ConfigResult<RegenieConfigData> {
    resolve_config_layers_without_run_validation(...)
}

pub fn validate_config(config: &RegenieConfigData) -> ConfigResult<()> {
    validate_semantic_config(config)
}

pub fn validate_config_for_run(config: &RegenieConfigData) -> ConfigResult<()> {
    validate_semantic_config(config)?;
    validate_existing_input_paths(config)?;
    validate_output_run_state(config)?;
    Ok(())
}
```

Then CLI execution uses `validate_config_for_run`, while Python config loading can remain pure.

---

# The most important missing concept: provenance

Current validation loses information about which options were explicitly supplied.

Example from current validation:

```rust
if config.binary.p_threshold.to_bits() != load_packaged_config_data()?.binary.p_threshold.to_bits() {
    binary_only_option_names.push("p_threshold");
}
```



That is a workaround for missing provenance. It detects “binary-only option differs from default,” not “user explicitly supplied binary-only option.”

These are different:

```toml
[trait]
qt = true

[binary]
pThresh = 0.05
```

The user explicitly supplied `pThresh`, even though it equals the default. If the policy is “binary-only options are illegal for QT unless implicit defaults,” you need to know that it was explicit.

Recommended solution:

```rust
#[derive(Clone, Debug, Default)]
pub struct ConfigProvenance {
    pub provided: BTreeSet<ConfigKey>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfigKey {
    TraitQt,
    TraitBt,
    BinaryFirth,
    BinaryApprox,
    BinaryPThreshold,
    BinaryFirthSe,
    ComputeDevice,
    ...
}
```

Each decoded layer carries:

```rust
pub struct ConfigLayer {
    pub partial_config: PartialConfig,
    pub provenance: ConfigProvenance,
}
```

Overlay should merge provenance:

```rust
merged_provenance.extend(layer.provenance)
```

Then validation can do:

```rust
if config.trait_config.trait_type == Quantitative {
    let binary_only = provenance.provided_binary_only_options();
    if !binary_only.is_empty() {
        return Err(...);
    }
}
```

This also replaces the old Python `explicit_options` behavior cleanly. The current PyO3 `RegenieConfig` does not expose `explicit_options`, but some tests still expect it.  I would not bring back the Python `explicit_options` set as public API, but I would keep equivalent provenance internally in Rust.

---

# Python API recommendation

If you are redesigning freely, I would stop treating Python option dictionaries as the primary config format.

Use:

```python
from g.interface.config import RegenieConfig

cfg = RegenieConfig.from_toml("run.toml")
artifacts = g.regenie(cfg)
```

For dictionary input, prefer TOML-shaped dictionaries:

```python
cfg = RegenieConfig.from_mapping({
    "input": {
        "bgen": "data/chr22.bgen",
        "pheno_file": "data/pheno.tsv",
        "pheno_columns": ["BMI"],
        "pred": "data/pred.list",
    },
    "trait": {
        "qt": True,
        "bsize": 16384,
    },
    "compute": {
        "device": "gpu",
    },
    "output": {
        "out": "results/bmi",
        "format": "parquet",
    },
})
```

If you still want CLI-shaped convenience:

```python
cfg = RegenieConfig.from_cli_options({
    "bgen": "data/chr22.bgen",
    "phenoFile": "data/pheno.tsv",
    "phenoCol": "BMI",
    "g-device": "gpu",
})
```

make that explicitly a compatibility helper. Do not make flat CLI strings the core schema.

The current `toml_table_from_py_mapping` converts Python mappings into `toml::Table` directly.  That is fine for TOML-shaped dictionaries. It is not enough for flat CLI-shaped dictionaries unless you add a routing/normalization layer.

---

# CLI architecture recommendation

## Use Clap derive, but split the parser

The current `RegenieCli` struct is too large. It mixes REGENIE input, trait flags, binary flags, output, compute, JAX, writer, and diagnostics.

I would split it:

```rust
#[derive(Parser)]
struct RegenieCli {
    #[arg(long)]
    config: Option<String>,

    #[command(flatten)]
    trait_args: TraitArgs,

    #[command(flatten)]
    input_args: InputArgs,

    #[command(flatten)]
    binary_args: BinaryArgs,

    #[command(flatten)]
    output_args: OutputArgs,

    #[command(flatten)]
    compute_args: ComputeArgs,

    #[command(flatten)]
    diagnostics_args: DiagnosticsArgs,
}
```

Clap supports flattening reusable argument structs into a parent parser. ([Docs.rs][3])

## Do not use Clap defaults

Do not put default values in Clap attributes:

```rust
#[arg(default_value_t = 16384)]
bsize: NonZeroU32,
```

Avoid that. Defaults belong in `config.default.toml`. Clap should produce a partial layer only.

So fields should be:

```rust
bsize: Option<NonZeroU32>
```

and booleans should be explicit presence-aware:

```rust
firth: bool,
no_firth: bool,
```

converted into:

```rust
Option<bool>
```

The branch already has the right `optional_flag(...)` helper shape.  Keep that.

---

# TOML architecture recommendation

Use direct Serde decoding.

The `toml` crate supports deserialization into Rust types through Serde. ([Docs.rs][1]) Your current `partial_config_from_toml_text` is exactly the right foundation:

```rust
toml::from_str::<PartialConfig>(toml_text)
```



I would keep:

```rust
#[serde(default, deny_unknown_fields)]
```

on section structs. That gives strict configs with good typo detection.

Be careful with `serde(flatten)`: Serde explicitly notes that `flatten` is not supported in combination with structs using `deny_unknown_fields`. ([Serde][2]) So do not use `flatten` to handle unknown/extra keys if you want strict config validation.

## Choose one canonical TOML spelling

I recommend:

```text
TOML canonical spelling:
  snake_case

CLI spelling:
  REGENIE exact for REGENIE flags
  --g-kebab-case for g-specific flags
```

So:

```text
CLI:
  --phenoFile
  --pThresh
  --g-staging-depth
  --g-output-format

TOML:
  pheno_file
  p_threshold
  staging_depth
  format
```

Serde can accept compatibility aliases if you want:

```rust
#[serde(alias = "phenoFile")]
pub pheno_file: Option<String>,

#[serde(alias = "pThresh")]
pub p_threshold: Option<Probability>,
```

The branch already does this for `phenoFile`, `phenoCol`, `phenoColList`, `covarFile`, and `pThresh`.

Effective TOML should emit the canonical TOML spelling, not necessarily the REGENIE CLI spelling. That is fine if documented.

---

# Effective TOML recommendation

Current `serialization.rs` serializes `RegenieConfigData` directly with metadata.

That is acceptable only if `RegenieConfigData`’s Serde representation is exactly the desired public TOML. If you want cleaner public TOML than internal field names, do not serialize `RegenieConfigData` directly.

I would create explicit public TOML output structs:

```rust
struct EffectiveConfigToml<'a> {
    input: InputTomlOut<'a>,

    #[serde(rename = "trait")]
    trait_config: TraitTomlOut,

    binary: BinaryTomlOut,
    compute: ComputeTomlOut<'a>,
    output: OutputTomlOut<'a>,
    diagnostics: DiagnosticsTomlOut<'a>,
    metadata: MetadataToml<'a>,
}
```

That lets you control:

```text
which optional fields are omitted
which names are canonical
whether phenoFile or pheno_file is emitted
metadata layout
```

Do not couple internal `ConfigData` shape to public TOML forever.

---

# Validation architecture

I would split validation into four layers.

## 1. Type/domain validation

Handled by Rust types and Serde:

```text
NonZeroU32
Probability
ProbabilityFloor
PositiveF32
ValueEnum/string enum values
```

This is already good.

## 2. Layer validation

Validate contradictions within a single layer:

```text
--qt and --bt both present in same CLI layer
--firth and --no-firth both present
pheno_col and pheno_col_list both present
```

Current code does this partially through `reject_trait_flag_conflict`, `optional_flag`, and `resolve_column_options`.

## 3. Semantic validation

Validate resolved config:

```text
step must be 2
approx requires firth
exact firth unsupported
firth_dtype must be float64
packed8 GPU genotype format requires GPU
required inputs present
duplicate phenotype names forbidden
```

Current `validation.rs` does this.

## 4. Run validation

Validate filesystem and output state:

```text
paths exist
output directory empty or resume enabled
manifest compatible
resume chunks valid
```

This should be separate from `from_toml` / `from_options`.

---

# What I would change in current code

## Keep

* `domain.rs` typed newtypes and string enums.
* `PartialConfig` overlay idea.
* `config.default.toml` as the default source.
* PyO3 config classes.
* Rust CLI dispatcher returning `CliOutcome`.
* Thin Python `config.py`.

## Change

### 1. Split `schema.rs`

Move partial structs to `partial.rs`, overlay to `overlay.rs`, and resolution helpers to `resolve.rs` or `resolver.rs`.

`schema.rs` is too large and too conceptually mixed.

### 2. Fix CLI names explicitly

Do not rely on `rename_all = "verbatim"` for any user-facing option. Explicitly annotate REGENIE flags and `--g-*` flags.

### 3. Decide TOML canonical names

I recommend canonical snake_case TOML. Keep Serde aliases for common REGENIE spellings if useful.

### 4. Add provenance

Track user-supplied fields separately from defaulted values.

This is necessary for clean validation of “binary-only options with QT” and for future “effective config includes what came from where” diagnostics.

### 5. Split pure config validation from filesystem/run validation

Do not make `RegenieConfig.from_toml(...)` require input files to exist.

### 6. Restore CLI/config tests

The branch currently skips `tests/test_cli.py` entirely.  That is not acceptable for a config/CLI migration branch once the design settles.

---

# Suggested final file tree

```text
src/config_frontend/
  mod.rs
  error.rs

  domain.rs
    Scalar domain types and enums.

  partial.rs
    Serde-deserialized partial config structs.

  resolved.rs
    Complete resolved config structs.

  defaults.rs
    Embedded packaged default TOML and default hash.

  overlay.rs
    ConfigLayer, overlay order, provenance tracking.

  resolve.rs
    PartialConfig + provenance -> RegenieConfigData.

  validation.rs
    Pure semantic validation.

  run_validation.rs
    Filesystem/output/run validation.

  toml.rs
    from_toml_text, from_toml_path, dumps_toml, write_toml.

  cli/
    mod.rs
    dispatch.rs
    regenie.rs
    groups.rs
    conversion.rs
    help.rs

src/python/config/
  mod.rs
  conversion.rs

src/g/interface/config.py
  Thin PyO3 compatibility shim.

src/g/cli.py
  Thin Python entrypoint:
    call Rust CLI
    if config returned, run Python engine
```

---

# Concrete API I would aim for

## Rust

```rust
pub fn load_packaged_config_data() -> ConfigResult<RegenieConfigData>;

pub fn from_toml_path(path: &Path) -> ConfigResult<RegenieConfigData>;

pub fn from_toml_text(text: &str, source: &str) -> ConfigResult<RegenieConfigData>;

pub fn from_mapping_table(table: &toml::Table) -> ConfigResult<RegenieConfigData>;

pub fn validate_config(config: &RegenieConfigData) -> ConfigResult<()>;

pub fn validate_config_for_run(config: &RegenieConfigData) -> ConfigResult<()>;

pub fn dumps_toml(config: &RegenieConfigData) -> ConfigResult<String>;

pub fn dispatch_cli(args: &[String], direct_regenie: bool) -> CliOutcomeData;
```

## Python

```python
from g.interface.config import RegenieConfig

cfg = RegenieConfig.from_toml("run.toml")
cfg = RegenieConfig.from_mapping({...})       # TOML-shaped
cfg = RegenieConfig.from_cli_options({...})   # optional convenience

cfg.to_toml()
g.regenie(cfg)
```

I would avoid encouraging direct constructors for the PyO3 config classes. They are resolved data views, not mutable user-facing builders.

---

# Final recommendation

The best architecture is:

```text
1. Use Clap derive for CLI.
2. Use Serde + toml for TOML.
3. Use typed Rust domain newtypes for scalar validation.
4. Use PartialConfig + overlay layers for defaults/TOML/CLI.
5. Use ResolvedConfigData for complete runtime config.
6. Keep Python as a thin PyO3 consumer.
7. Do not rebuild an OptionSpec metadata system unless you need generated docs/explain.
```

But I would enforce these decisions:

```text
CLI:
  REGENIE flags exactly.
  Non-REGENIE flags should use --g-*.

TOML:
  Native g config, preferably snake_case.
  It does not have to mimic CLI spelling.

Validation:
  Pure config validation separate from filesystem/run validation.

Provenance:
  Track what the user explicitly supplied.

Tests:
  Restore CLI/config tests before merging.
```

The current branch is a promising design spike. The next iteration should make the architecture crisp: split the large modules, explicitly name CLI flags, decide canonical TOML spelling, add provenance, and restore tests.

[1]: https://docs.rs/toml/latest/toml/ "toml - Rust"
[2]: https://serde.rs/field-attrs.html "Field attributes · Serde"
[3]: https://docs.rs/clap/latest/clap/_derive/index.html "clap::_derive - Rust"

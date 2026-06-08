# Configuration and CLI Architecture

This page defines how `g` exposes configuration to users and how developers must add new parameters. It is both a user-facing reference for CLI/TOML behavior and a developer contract for keeping the configuration system unified across the app.

`g` has two public configuration goals:

1. **REGENIE compatibility:** users should be able to take an existing REGENIE Step 2 command and replace `regenie` with `g regenie`, with the same flag names and statistical meaning whenever the feature is supported.
2. **Reproducibility:** every run should be representable as a TOML config, every CLI option should be overrideable from TOML, and every run should write an effective config and manifest.

On the experimental Rust CLI/config branch, the authoritative frontend lives in
Rust. `clap` parses `g regenie` and `g-regenie`, `toml-spanner` decodes TOML,
Rust overlays packaged defaults with TOML and explicit CLI/Python options,
validates the effective configuration, and exposes PyO3 config objects to
Python. Python keeps the JAX/runtime layer and treats those objects as immutable
runtime inputs.

The production flow is:

```text
CLI args / TOML config / Python option dict
        ↓
Rust option layers
        ↓
Rust option registry + packaged default config
        ↓
PyO3 RegenieConfig
        ↓
ExecutionPlan
        ↓
engine pipeline
```

The engine must never read CLI flags, TOML files, environment variables, or scattered `DEFAULT_*` values directly. It should receive all user-controlled behavior through `RegenieConfig` and then `ExecutionPlan`.

The previous `g config init|validate|explain` subcommands are intentionally not
registered on this branch. Keep the branch focused on supported REGENIE-style
`g regenie` flags, `--g-*` runtime extensions, `--config`, and the Python
compatibility functions in `g.interface.config`.

The Rust frontend is split by responsibility:

```text
src/config_frontend/
  mod.rs        option layering, runtime config construction, validation
  cli.rs        clap command construction and dispatch
  metadata.rs   option metadata registry loaded from config_options.json
  render.rs     deterministic effective TOML and template rendering

src/python/config/
  mod.rs        PyO3 config classes and registered config functions
  conversion.rs Python/Rust value conversion helpers
```

`src/g/interface/config.py` and `src/g/interface/config_layers.py` are
compatibility shims. They should delegate parsing, flattening, normalization,
default loading, validation, and effective TOML behavior to Rust instead of
reimplementing a second config engine in Python.

---

## 1. User-facing behavior

### 1.1 REGENIE-compatible command line

The main command is:

```bash
g regenie [REGENIE-compatible flags] [g-specific flags]
```

A quantitative Step 2 run should look like a REGENIE command:

```bash
g regenie \
  --step 2 \
  --bgen data/chr22.bgen \
  --sample data/chr22.sample \
  --phenoFile data/pheno.tsv \
  --phenoCol BMI \
  --covarFile data/covar.tsv \
  --covarColList age,sex,PC1,PC2 \
  --pred data/step1_pred.list \
  --qt \
  --bsize 16384 \
  --out results/bmi
```

A binary score-only run:

```bash
g regenie \
  --step 2 \
  --bgen data/chr22.bgen \
  --sample data/chr22.sample \
  --phenoFile data/pheno.tsv \
  --phenoCol disease \
  --covarFile data/covar.tsv \
  --covarColList age,sex,PC1,PC2 \
  --pred data/step1_bt_pred.list \
  --bt \
  --bsize 16384 \
  --out results/disease
```

A binary approximate-Firth fallback run:

```bash
g regenie \
  --step 2 \
  --bgen data/chr22.bgen \
  --sample data/chr22.sample \
  --phenoFile data/pheno.tsv \
  --phenoCol disease \
  --covarFile data/covar.tsv \
  --covarColList age,sex,PC1,PC2 \
  --pred data/step1_bt_pred.list \
  --bt \
  --firth \
  --approx \
  --pThresh 0.01 \
  --out results/disease
```

`g` defaults to Parquet/Arrow run artifacts for throughput. Users can select
REGENIE Step 2-compatible text with `--g-output-format regenie` when workflow
compatibility is more important than output throughput.

### 1.2 Supported REGENIE-style flags

Supported REGENIE-style names should keep their original spelling:

```text
--step
--qt / --no-qt
--bt / --no-bt
--bgen
--sample
--phenoFile
--phenoCol
--phenoColList
--covarFile
--covarCol
--covarColList
--pred
--bsize
--threads
--out
--firth / --no-firth
--approx / --no-approx
--pThresh
--firth-se / --no-firth-se
```

Rules:

- `--qt` and `--bt` are mutually exclusive.
- If neither `--qt` nor `--bt` is supplied, quantitative mode is the default.
- Binary default is score-only.
- `--firth --approx` enables approximate-Firth fallback.
- Exact `--firth` without `--approx` must fail loudly unless implemented and parity-tested.
- `--spa` must fail loudly unless implemented and parity-tested.
- Binary-only flags should be rejected for quantitative runs.

### 1.3 Recognized-but-unsupported REGENIE flags

Some REGENIE flags should be recognized so migration errors are clear, but rejected if unsupported. Examples:

```text
--bed
--pgen
--keep
--remove
--extract
--exclude
--catCovarList
--test
--t2e
--spa
```

Unsupported recognized flags must never be ignored. They should produce an error like:

```text
--pgen is a valid REGENIE option, but g currently supports BGEN Step 2 only. Use --bgen.
```

### 1.4 `g`-specific flags

Every `g`-specific user-facing CLI flag must start with `--g-`.

Examples:

```bash
--g-device gpu
--g-staging-depth 2
--g-trusted-no-missing-diploid
--g-trusted-bgen-validation-mode cache_on_miss
--g-sample-key-mode fid_iid
--g-multi-phenotype-sample-mode complete-case
--g-firth-batch-size 1024
--g-firth-candidate-capacity 2048
--g-score-dtype float32
--g-firth-dtype float64
--g-bgen-decode-tile-variant-count 64
--g-jax-cache-dir /scratch/$USER/g-jax-cache
--g-jax-persistent-cache
--g-output-format parquet
--g-output-format regenie
--g-writer-threads 8
--g-writer-queue-depth 16
--g-output-chunks-per-arrow-file 16
--g-output-arrow-compression zstd
--g-resume
--g-resume-mode strict
--g-telemetry profile
--g-log-dir results/run/logs
```

This namespace rule is strict. Do not introduce engine-specific options such as `--device`, `--batch-size`, or `--writer-threads`; those names are ambiguous and may conflict with REGENIE.

---

## 2. TOML configuration

### 2.1 Config file basics

Users can provide a TOML config:

```bash
g regenie --config run.toml
```

or combine TOML with CLI overrides:

```bash
g regenie \
  --config server.toml \
  --phenoFile data/pheno.tsv \
  --phenoCol BMI \
  --out results/bmi
```

This enables a common workflow:

```text
server.toml:
  technical settings tuned for one machine

CLI:
  biological inputs and output prefix for one run
```

Precedence is:

```text
packaged default config
    <
user TOML config
    <
explicit CLI options
```

Only explicitly supplied CLI options override TOML. An omitted CLI flag means “leave the TOML/default value unchanged.”

### 2.2 TOML sections

REGENIE-compatible options use REGENIE names inside these sections:

```toml
[input]
bgen = "data/chr22.bgen"
sample = "data/chr22.sample"
phenoFile = "data/pheno.tsv"
phenoCol = "BMI"
covarFile = "data/covar.tsv"
covarColList = "age,sex,PC1,PC2"
pred = "data/step1_pred.list"

[trait]
step = 2
qt = true
bt = false
bsize = 16384
threads = 8

[binary]
firth = false
approx = false
pThresh = 0.05
firth-se = false

[output]
out = "results/bmi"
```

`g`-specific options are grouped under `[g.*]` sections:

```toml
[g.compute]
device = "gpu"
staging-depth = 2
trusted-no-missing-diploid = true
trusted-bgen-validation-mode = "cache_on_miss"
sample-key-mode = "iid"
multi-phenotype-sample-mode = "per-phenotype"
firth-batch-size = 1024
firth-candidate-capacity = 2048
score-dtype = "float32"
firth-dtype = "float64"
bgen-decode-tile-variant-count = 64
jax-cache-dir = "/scratch/user/g-jax-cache"
jax-persistent-cache = true

[g.output]
format = "parquet"
# or format = "regenie" for REGENIE-compatible text
writer-threads = 8
writer-queue-depth = 16
chunks-per-arrow-file = 16
arrow-compression = "zstd"
resume = false
resume-mode = "fast"
finalize-parquet = false

[g.diagnostics]
telemetry = "progress"
log-dir = "results/bmi.run/logs"
log-filter = "info"
log-stderr = true
progress-interval-seconds = 5
progress-interval-chunks = 10
```

### 2.3 Why TOML `g` keys do not repeat `g-`

The CLI flag is:

```bash
--g-device gpu
```

The TOML equivalent is:

```toml
[g.compute]
device = "gpu"
```

The `[g.compute]` table supplies the `g` namespace, so repeating `g-` inside the table would be redundant. Internally, TOML `[g.compute] device` normalizes to the same canonical option as CLI `--g-device`.

This is the only naming exception:

```text
CLI canonical name:
  --g-device

TOML namespaced form:
  [g.compute]
  device = "gpu"

Internal canonical option name:
  g-device
```

### 2.4 CLI boolean overrides

Because TOML values can set booleans to `true`, the CLI must support explicit negative flags. Examples:

```bash
--firth / --no-firth
--approx / --no-approx
--bt / --no-bt
--qt / --no-qt
--g-resume / --no-g-resume
--g-finalize-parquet / --no-g-finalize-parquet
--g-log-stderr / --no-g-log-stderr
--g-jax-persistent-cache / --no-g-jax-persistent-cache
```

Example:

```toml
[binary]
firth = true
approx = true
```

can be overridden with:

```bash
g regenie --config binary-defaults.toml --no-firth --no-approx ...
```

Omitting `--firth` does not override TOML.

### 2.5 Required values

Some values have no meaningful default and are required for a real run:

```text
bgen
phenoFile
phenoCol or phenoColList
pred
out
```

TOML has no `null`, so required values should not appear as fake defaults in the packaged default config. Starter templates may show them as commented examples.

### 2.6 Effective config and manifest

Every run should write:

```text
<run_directory>/effective_config.toml
<run_directory>/run_manifest.json
```

`effective_config.toml` is the human-readable merged config after applying defaults, TOML, and CLI overrides.

`run_manifest.json` is the machine-readable safety record. It should contain file fingerprints, execution-plan-affecting settings, output schema version, resume metadata, and execution-plan hash.

---

## 3. Python API behavior

The Python API should mirror CLI/TOML semantics. It should accept canonical option names and Pythonic aliases, but normalize them through the same code path as CLI and TOML.

```python
import g
from g.interface.config import RegenieConfig

config = RegenieConfig.from_toml("run.toml")
artifacts = g.regenie(config)
```

Option dictionaries should use the same names as CLI flags without leading dashes:

```python
artifacts = g.regenie.from_options({
    "bgen": "data/chr22.bgen",
    "sample": "data/chr22.sample",
    "phenoFile": "data/pheno.tsv",
    "phenoCol": "BMI",
    "covarFile": "data/covar.tsv",
    "covarColList": "age,sex,PC1,PC2",
    "pred": "data/step1_pred.list",
    "qt": True,
    "bsize": 16384,
    "out": "results/bmi",
    "g-device": "gpu",
})
```

Pythonic aliases such as `pheno_file`, `pheno_col`, `p_threshold`, and `g_device` may be accepted for convenience, but the canonical name remains the REGENIE/CLI-style option name.

---

## 4. Developer architecture

### 4.1 Core files

Configuration-related code is organized around these pieces:

```text
src/g/interface/options.py
  OptionSpec registry: names, support level, sections, CLI flags, types, choices, help text.

src/g/config.default.toml
  Packaged default values for configurable parameters.

src/g/interface/config.py
  Thin compatibility wrappers around Rust-owned config objects.

src/g/execution_plan.py
  RegenieConfig -> immutable execution plan used by the engine.

src/g/cli.py
  Thin Python dispatcher into the Rust CLI frontend.

src/g/types.py
  Shared enums and small immutable public/internal plan structs.
```

### 4.2 OptionSpec is metadata, not default ownership

`OptionSpec` describes an option:

- user-facing name
- Python destination name
- TOML section
- support level
- help text
- CLI flags
- value type
- accepted values
- whether it is repeated
- whether it is a boolean flag

It should not become the main owner of default values. Defaults belong in the packaged default TOML when the option is user-configurable.

### 4.3 Default TOML is the source of truth for user-configurable defaults

If a value is user-configurable and has a default, it must be present in `src/g/config.default.toml`.

Examples:

```text
bsize
pThresh
staging-depth
writer-threads
writer-queue-depth
chunks-per-arrow-file
firth-batch-size
firth-candidate-capacity
iteration limits
tolerances
score/firth dtype
JAX cache settings
telemetry mode
resume mode
```

Score and Firth dtype options currently control internal JAX compute precision.
Public association statistics use the writer schema and remain float32 in
Arrow/Parquet output unless a separate output dtype feature is added later.

Do not introduce new `DEFAULT_*` constants in Python or Rust for new user-tunable behavior. Existing legacy constants should be treated as migration debt unless they are pure mathematical constants.
Packaged default catalog views should stay inside the config layer. Runtime subsystems should receive resolved `RegenieConfig` or `ExecutionPlan` values rather than reading typed `PACKAGED_*` defaults at their own boundaries.

### 4.4 Constants policy

There are three categories.

#### User-configurable parameters

These must live in the default TOML and flow through `RegenieConfig` / `ExecutionPlan`.

Examples:

```text
writer thread count
queue depth
Firth tolerances
BGEN tile size
JAX cache behavior
telemetry intervals
```

JAX x64 enablement is not in this category. The app requires it for core functionality and always enables it as a
runtime invariant rather than exposing a config option.

#### Mathematical or format constants

These may be hardcoded, but only once.

Examples:

```text
diploid allele count = 2
BGEN 8-bit probability scale = 255
BGEN raw dosage base = 510
chi-square degrees of freedom = 1
manifest schema version
output schema version
binary extra-code integer labels
```

If a constant is used in several places, define it once and import it.

#### Derived values

These can be computed in code.

Examples:

```text
run directory from output prefix and association mode
log directory from run directory
stage timing path from log directory
execution-plan hash from normalized plan fields
```

### 4.5 Adding a new supported REGENIE-compatible option

Use this checklist.

1. Add an `OptionSpec` to the supported REGENIE options in `src/g/interface/options.py`.
2. Use the REGENIE name exactly, including mixed case such as `phenoFile` or `pThresh`.
3. Add Rust CLI flag metadata only if the generated long flag would be wrong.
4. Add a default to `src/g/config.default.toml` if the option has a meaningful default.
5. Add parsing/mapping in the Rust config frontend if the option affects `RegenieConfig`.
6. Add validation if it has constraints or incompatibilities.
7. Add it to `ExecutionPlan` if the engine needs it.
8. Add it to the run manifest if it affects outputs, statistics, resume compatibility, or performance semantics.
9. Add CLI/TOML/Python tests.
10. Update docs.

Example naming:

```text
CLI:
  --pThresh 0.01

TOML:
  [binary]
  pThresh = 0.01

Python options:
  {"pThresh": 0.01}
  {"p_threshold": 0.01}  # optional alias
```

### 4.6 Adding a new recognized-but-unsupported REGENIE option

Use this when the flag exists in REGENIE but `g` does not implement it yet.

1. Add an `OptionSpec` with `support_level = RECOGNIZED_UNSUPPORTED`.
2. Use the exact REGENIE name.
3. Add a clear error message in unsupported-option validation if the generic error is not good enough.
4. Add a test proving the option is recognized and rejected.

Do not ignore unsupported options.

### 4.7 Adding a new `g`-specific option

Use this checklist.

1. Choose a CLI name beginning with `g-`.
2. Put it in the appropriate TOML section:
   - `g.compute` for runtime/engine/JAX/native compute settings
   - `g.output` for output/writer/resume settings
   - `g.diagnostics` for logging/telemetry/profiling settings
3. Add an `OptionSpec` in `G_OPTIONS`.
4. Add a default to `src/g/config.default.toml` if it has a default.
5. Add mapping to `RegenieConfig` in `config.py` if needed.
6. Add to `ExecutionPlan` if the engine uses it.
7. Add to the run manifest if it affects results, output layout, resume safety, or performance reproducibility.
8. Add tests for CLI, TOML, and Python option-dict forms.

Example:

```text
CLI:
  --g-bgen-decode-tile-variant-count 128

TOML:
  [g.compute]
  bgen-decode-tile-variant-count = 128

Internal canonical option:
  g-bgen-decode-tile-variant-count

Python option dictionary:
  {"g-bgen-decode-tile-variant-count": 128}
  {"g_bgen_decode_tile_variant_count": 128}  # optional alias
```

### 4.8 Adding a new configurable Rust runtime value

Rust must not own independent user-facing defaults.

Correct flow:

```text
config.default.toml
    ↓
RegenieConfig
    ↓
ExecutionPlan
    ↓
runner calls _core.configure_...(...)
    ↓
Rust runtime uses the configured value
```

A Rust fallback may exist for unit tests or direct internal use, but production runs should always pass the resolved value from Python config. If a fallback exists, add a test that ensures it matches the packaged default config or document why it is intentionally different.

### 4.9 Adding a value used in both Python and Rust

If a value is a mathematical/format constant used in both Python and Rust, avoid redefining it silently.

Acceptable patterns:

1. Define in Rust and expose through `_core` for Python parity tests.
2. Define in a small schema file and generate Python/Rust constants.
3. Define in one language and add explicit tests that the duplicated value in the other language matches.

Examples that require care:

```text
binary extra code labels
Firth failure code labels
allele count multiplier
BGEN raw-dosage constants
output schema version
manifest schema version
```

### 4.10 Manifest policy

A parameter must be included in the run manifest when changing it could alter any of these:

```text
statistics
sample alignment
variant inclusion/order
binary correction behavior
numerical precision
JAX kernel behavior
BGEN decoding behavior
output schema/layout
resume compatibility
```

Examples:

```text
bsize
g-score-dtype
g-firth-dtype
g-trusted-no-missing-diploid
g-trusted-bgen-validation-mode
g-sample-key-mode
g-multi-phenotype-sample-mode
g-bgen-decode-tile-variant-count
g-firth-* tolerances and capacities
g-output-* writer settings that affect chunk grouping or finalization
```

The manifest is not just metadata. It protects resume correctness. If a parameter changes output or interpretation, resume must reject incompatible prior chunks.

---

## 5. Validation and tests

Every new option should have tests for these behaviors:

```text
CLI parses the option.
TOML parses the option.
Python option dict parses the option.
CLI overrides TOML when explicitly supplied.
An omitted CLI flag does not override TOML.
Negative boolean flags override TOML true values.
Invalid values fail clearly.
Recognized unsupported values fail clearly.
Effective config includes the resolved value.
Manifest includes the value when relevant.
```

Recommended test names:

```text
test_cli_option_maps_to_config
test_toml_option_maps_to_config
test_python_option_alias_maps_to_config
test_cli_overrides_toml
test_absent_cli_boolean_does_not_override_toml
test_negative_cli_boolean_overrides_toml
test_invalid_option_value_errors
test_manifest_rejects_resume_when_option_changes
```

Add drift tests:

```text
every supported OptionSpec has a TOML section
every configurable default in config.default.toml maps to a known option
every g-specific CLI flag starts with --g-
no new user-tunable DEFAULT_* constants outside the default config loader
```

---

## 6. Current implementation notes

The current implementation already has the core pieces:

```text
OptionSpec registry
default policy and TOML path metadata
Rust Clap generation from OptionSpec metadata
packaged config.default.toml
validated default catalog hash
TOML/CLI/Python normalization
RegenieConfig
ExecutionPlan
manifest validation
```

Two areas need continuing discipline:

1. **Binary-only defaults appear in `config.default.toml` but should stay out of starter examples.** This keeps starter configs quantitative by default without making `pThresh` or Firth flags explicit user overrides.
2. **TOML `[g.*]` keys currently omit the `g-` prefix.** This is intentional: the table provides the namespace. The internal canonical name still includes `g-`.

---

## 7. Summary contract

For users:

```text
Use REGENIE flags where REGENIE semantics exist.
Use --g-* flags for g-specific runtime/output/diagnostic behavior.
Put stable technical settings in TOML.
Use CLI overrides for per-run inputs and phenotype/output selections.
Trust effective_config.toml and run_manifest.json for reproducibility.
```

For developers:

```text
Do not add ad hoc flags.
Do not add scattered defaults.
Do not let Rust own independent user-tunable defaults.
Do not ignore unsupported REGENIE options.
Do not introduce g-specific options without --g- on CLI.
Do not bypass RegenieConfig and ExecutionPlan.
Do add OptionSpec, default TOML entries, validation, manifest fields, and tests for every new parameter.
```

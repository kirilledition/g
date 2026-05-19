I would design the UI around one principle:

**`g` should have a REGENIE-compatible surface for users, and a separate internal execution plan for your engine.**
Users should see REGENIE names. Your engine should see normalized, typed, validated settings.

Right now your repo has three separate front doors drifting apart: `g regenie2`, `g regenie2-linear`, and Python `regenie2(...)`. I would replace that with one universal config/runner layer, then make CLI, TOML, and Python all feed into it.

Official REGENIE already has the user vocabulary you want to mirror: `--step`, `--qt`, `--bt`, `--bgen`, `--sample`, `--phenoFile`, `--phenoCol`, `--phenoColList`, `--covarFile`, `--covarCol`, `--covarColList`, `--pred`, `--bsize`, `--threads`, `--firth`, `--approx`, `--spa`, `--pThresh`, and related Step 2 options. The docs also show the exact common Step 2 binary pattern: `--bt --firth --approx --pThresh 0.01 --pred ... --out ...`. ([RGC GitHub][1])

---

# Recommended public interface

## 1. Add a REGENIE-compatible command

Use this as the main user-facing command:

```bash
g regenie \
  --step 2 \
  --bgen data/chr22.bgen \
  --sample data/chr22.sample \
  --phenoFile pheno.tsv \
  --phenoCol BMI \
  --covarFile covar.tsv \
  --covarColList age,sex,PC1,PC2 \
  --pred step1_pred.list \
  --qt \
  --bsize 8192 \
  --out results/bmi \
  --g-device gpu
```

Binary should look like REGENIE:

```bash
g regenie \
  --step 2 \
  --bgen data/chr22.bgen \
  --sample data/chr22.sample \
  --phenoFile pheno.tsv \
  --phenoCol disease \
  --covarFile covar.tsv \
  --covarColList age,sex,PC1,PC2 \
  --pred step1_bt_pred.list \
  --bt \
  --firth \
  --approx \
  --pThresh 0.01 \
  --bsize 8192 \
  --out results/disease \
  --g-device gpu
```

This is the compatibility interface. Users should be able to take:

```bash
regenie --step 2 --bgen ... --bt --firth --approx --pThresh 0.01 ...
```

and change it to:

```bash
g regenie --step 2 --bgen ... --bt --firth --approx --pThresh 0.01 ...
```

REGENIE’s docs say `--qt` is the quantitative-trait flag and default, `--bt` is the binary-trait flag, and `--bsize` is the genotype block size. ([RGC GitHub][1])

## 2. Add a direct executable alias for easier pipeline replacement

Also add a second console script:

```toml
[project.scripts]
g = "g:main"
g-regenie = "g.cli:regenie_main"
```

Then users can do:

```bash
g-regenie --step 2 --bgen ... --bt --firth --approx --pThresh 0.01 ...
```

This makes pipeline migration even easier because they replace `regenie` with `g-regenie`, not `regenie` with `g regenie`.

## 3. Keep current commands temporarily as deprecated aliases

Keep these for one or two releases:

```bash
g regenie2
g regenie2-linear
```

but make them wrappers around the new config layer.

Map old names to new names:

| Current `g` flag                        | New REGENIE-compatible flag |
| --------------------------------------- | --------------------------- |
| `--trait-type quantitative`             | `--qt`                      |
| `--trait-type binary`                   | `--bt`                      |
| `--pheno-name`                          | `--phenoCol`                |
| `--covar-names`                         | `--covarColList`            |
| `--chunk-size`                          | `--bsize`                   |
| `--binary-correction firth_approximate` | `--firth --approx`          |
| `--binary-correction spa`               | `--spa`                     |

Print a warning like:

```text
g regenie2 is deprecated. Use: g regenie --step 2 ...
```

---

# Important compatibility rule

For `g regenie`, **recognized but unsupported REGENIE flags should fail loudly**.

Do not silently ignore unsupported flags.

Example:

```bash
g regenie --step 2 --pgen data/genotypes --qt ...
```

should produce:

```text
--pgen is a valid REGENIE option, but g currently supports BGEN Step 2 only.
Use --bgen, or convert the data to BGEN.
```

That gives users confidence that `g` is not silently changing their analysis.

I would use these support levels internally:

```python
class SupportLevel(StrEnum):
    SUPPORTED = "supported"
    RECOGNIZED_UNSUPPORTED = "recognized_unsupported"
    G_EXTENSION = "g_extension"
    DEPRECATED_ALIAS = "deprecated_alias"
```

---

# TOML config design

## 1. Same names as CLI flags, without leading `--`

For REGENIE-compatible options, TOML keys should use the exact REGENIE spelling:

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
bsize = 8192
threads = 8

[binary]
firth = false
approx = false
spa = false
pThresh = 0.05
firth-se = false

[filters]
chrList = "22"
minMAC = 5
minINFO = 0.0

[output]
out = "results/bmi"
```

For a binary approximate-Firth run:

```toml
[input]
bgen = "data/chr22.bgen"
sample = "data/chr22.sample"
phenoFile = "data/pheno_binary.tsv"
phenoCol = "disease"
covarFile = "data/covar.tsv"
covarColList = "age,sex,PC1,PC2"
pred = "data/step1_bt_pred.list"

[trait]
step = 2
bt = true
bsize = 8192
threads = 8

[binary]
firth = true
approx = true
pThresh = 0.01
firth-se = true

[output]
out = "results/disease"

[g.compute]
device = "gpu"
trusted-no-missing-diploid = true
staging-depth = 2

[g.output]
format = "regenie"
resume = true
resume-mode = "strict"
```

The REGENIE docs describe `--firth` as a fallback below a p-value threshold, `--approx` as an approximate Firth option that only works with `--firth`, `--spa` as a fallback option, and `--pThresh` as the fallback threshold with default `0.05`. ([RGC GitHub][1])

## 2. Put `g`-specific knobs under `[g.*]`

Do not pollute the REGENIE namespace with GPU/runtime details. Use a reserved prefix.

CLI:

```bash
--g-device gpu
--g-staging-depth 2
--g-firth-batch-size 64
--g-writer-threads 8
--g-output-format parquet
```

TOML:

```toml
[g.compute]
device = "gpu"
staging-depth = 2
firth-batch-size = 64
firth-candidate-capacity = 1024
bgen-decode-tile-variant-count = 64
jax-cache-dir = "/tmp/g-jax-cache"
jax-matmul-precision = "float32"

[g.output]
format = "parquet"
finalize-parquet = true
writer-threads = 8
writer-queue-depth = 4
chunks-per-arrow-file = 4
resume = true
resume-mode = "strict"

[g.diagnostics]
timing-json = "results/timing.json"
profile-native-bgen = true
transfer-guard-diagnostics = false
```

I would make this the rule:

```text
REGENIE-compatible options keep REGENIE names.
g-specific options are explicitly namespaced with --g-* on CLI and [g.*] in TOML.
```

## 3. Support CLI overrides over config

Example:

```bash
g regenie --config run.toml --pThresh 0.01 --g-device gpu
```

Precedence should be:

```text
built-in defaults < TOML config < explicit CLI flags
```

Be careful with booleans. You need to know whether a CLI flag was actually passed. Do not let default `False` values from CLI parsing overwrite `true` values from TOML.

For boolean overrides, add hidden or advanced negative flags:

```bash
--no-firth
--no-approx
--no-spa
--no-g-resume
--no-g-finalize-parquet
```

These are `g` extensions, but they make config overrides practical.

## 4. Always write the effective config

Every run should save:

```text
<run-directory>/effective_config.toml
<run-directory>/run_manifest.json
```

The effective TOML should include all defaults after merging CLI and config. That is what makes runs reproducible.

Include:

```toml
[metadata]
g-version = "0.1.0"
command = "g regenie --config run.toml --g-device gpu"
generated-at = "..."
```

and maybe file fingerprints in the JSON manifest:

```json
{
  "bgen_path": "data/chr22.bgen",
  "bgen_size_bytes": 123456,
  "bgen_mtime_ns": 123456789,
  "effective_config": "effective_config.toml"
}
```

---

# Python interface design

I would expose one primary Python function:

```python
import g

result = g.regenie(
    g.RegenieConfig(
        input=g.InputConfig(
            bgen="data/chr22.bgen",
            sample="data/chr22.sample",
            pheno_file="data/pheno.tsv",
            pheno_col="BMI",
            covar_file="data/covar.tsv",
            covar_col_list="age,sex,PC1,PC2",
            pred="data/step1_pred.list",
        ),
        trait=g.TraitConfig(
            step=2,
            qt=True,
            bsize=8192,
            threads=8,
        ),
        output=g.OutputConfig(
            out="results/bmi",
        ),
        compute=g.GComputeConfig(
            device="gpu",
            staging_depth=2,
            trusted_no_missing_diploid=True,
        ),
    )
)
```

Use Pythonic field names in the dataclasses. Do not force awkward names like `pThresh` or `firth-se` into Python attributes.

But also support exact REGENIE-key dictionaries:

```python
result = g.regenie.from_options(
    {
        "step": 2,
        "bgen": "data/chr22.bgen",
        "sample": "data/chr22.sample",
        "phenoFile": "data/pheno.tsv",
        "phenoCol": "BMI",
        "covarFile": "data/covar.tsv",
        "covarColList": "age,sex,PC1,PC2",
        "pred": "data/step1_pred.list",
        "qt": True,
        "bsize": 8192,
        "out": "results/bmi",
        "g-device": "gpu",
    }
)
```

And TOML loading:

```python
config = g.RegenieConfig.from_toml("run.toml")
result = g.regenie(config)
```

Also useful:

```python
config.to_toml("effective_config.toml")
config.to_cli_args()
config = g.RegenieConfig.from_cli_args([
    "--step", "2",
    "--bgen", "data/chr22.bgen",
    "--phenoFile", "pheno.tsv",
    "--phenoCol", "BMI",
    "--qt",
])
```

That gives you a Transformers-like config object, while still letting users stay close to REGENIE.

---

# Internal architecture

I would introduce a new interface layer and make both CLI and Python API call it.

Recommended structure:

```text
src/g/interface/
  __init__.py
  options.py          # OptionSpec table: REGENIE flags + g extensions
  schema.py           # public config dataclasses
  toml.py             # load/dump/merge TOML configs
  cli_parse.py        # CLI parsing into RegenieConfig
  normalize.py        # RegenieConfig -> ExecutionPlan
  validation.py       # user-facing validation errors
  docs.py             # generate config templates/help tables

src/g/runner.py       # universal run_regenie(config)

src/g/api.py          # thin Python wrappers only
src/g/cli.py          # thin CLI wrappers only
```

The flow should become:

```text
CLI args / TOML / Python config
        ↓
RegenieConfig
        ↓
validate user config
        ↓
normalize to ExecutionPlan
        ↓
runner.run_regenie_execution_plan(...)
        ↓
current engine pipelines
```

The engine should not know whether a value came from CLI, TOML, or Python.

## Central option spec

Create one source of truth:

```python
@dataclasses.dataclass(frozen=True)
class OptionSpec:
    name: str                       # "phenoFile", "pThresh", "g-device"
    cli_flags: tuple[str, ...]       # ("--phenoFile",)
    config_section: str             # "input", "binary", "g.compute"
    python_field: str               # "pheno_file", "p_threshold", "device"
    value_type: type
    default: object
    category: str
    support: SupportLevel
    help: str
    accepted_values: tuple[str, ...] | None = None
```

Example:

```python
OptionSpec(
    name="phenoFile",
    cli_flags=("--phenoFile",),
    config_section="input",
    python_field="pheno_file",
    value_type=Path,
    default=None,
    category="input",
    support=SupportLevel.SUPPORTED,
    help="Phenotype table path.",
)
```

For unsupported-but-recognized flags:

```python
OptionSpec(
    name="pgen",
    cli_flags=("--pgen",),
    config_section="input",
    python_field="pgen",
    value_type=Path,
    default=None,
    category="input",
    support=SupportLevel.RECOGNIZED_UNSUPPORTED,
    help="Valid REGENIE option, not supported by g yet.",
)
```

Then tests can enforce:

```text
every supported CLI flag exists in the TOML schema
every TOML key maps to an option spec
every Python config field maps to an option spec
unsupported REGENIE flags produce clear errors
```

This prevents drift.

---

# CLI implementation choice

Your current `src/g/cli.py` is Typer-based and hand-written. That is fine for a small Pythonic CLI, but REGENIE compatibility is a different problem: you need exact mixed-case flags like `--phenoFile`, repeated options like `--phenoCol`, many recognized unsupported flags, hidden legacy aliases, and config-source tracking.

I would strongly consider using **Click directly** for `g regenie`, even if you keep Typer for old commands temporarily. Typer is built on Click, but Click is easier when you want to generate options from an `OptionSpec` table.

The compatibility command can be generated:

```python
def build_regenie_click_command() -> click.Command:
    params = []
    for spec in REGENIE_OPTION_SPECS:
        params.append(build_click_option(spec))
    return click.Command(
        name="regenie",
        params=params,
        callback=run_regenie_click_callback,
    )
```

This gives you:

```text
one option table
one TOML schema
one Python config mapping
one validation layer
```

If you keep Typer, you can still implement the plan, but you will likely keep duplicating option definitions.

---

# Output interface

This matters for user surprise.

For the compatibility command:

```bash
g regenie --out results/bmi ...
```

`--out` should behave like a REGENIE output prefix, not only like an internal chunk directory. REGENIE’s docs describe Step 2 output as phenotype-specific `.regenie` files, with an additional column for Firth/SPA correction failures. ([RGC GitHub][1])

I would support:

```bash
--g-output-format regenie
--g-output-format parquet
--g-output-format arrow
--g-output-format both
```

Default for `g regenie` should eventually be:

```text
regenie-compatible text output
```

Default for internal/native commands can remain:

```text
Arrow chunks / Parquet
```

Internal chunks can still live in:

```text
<out>.g-run/
```

while user-facing output is:

```text
<out>_<phenotype>.regenie
```

This lets users switch pipelines without rewriting downstream tools.

---

# Configurable constants and tuning knobs

Right now several important values are hardcoded or environment-only:

| Current value                               | Current location                                 | Suggested public knob                                           |
| ------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------- |
| `DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE = 8192` | `api.py`                                         | `--bsize`, TOML `bsize`                                         |
| `prefetch_chunks` / staging depth           | `api.py`, pipeline                               | `--g-staging-depth`                                             |
| writer thread count                         | `io/output.py`                                   | `--g-writer-threads`                                            |
| writer queue depth                          | `io/output.py`                                   | `--g-writer-queue-depth`                                        |
| Firth batch size                            | env `G_REGENIE2_BINARY_FIRTH_BATCH_SIZE`         | `--g-firth-batch-size`                                          |
| Firth candidate capacity                    | env `G_REGENIE2_BINARY_FIRTH_CANDIDATE_CAPACITY` | `--g-firth-candidate-capacity`                                  |
| Firth max iterations                        | `regenie2_binary.py`                             | `--niter` or `--g-firth-max-iterations`, depending on semantics |
| null iterations                             | `regenie2_binary.py`                             | `--maxiter-null` if mirroring REGENIE                           |
| BGEN decode tile size                       | env `G_BGEN_DECODE_TILE_VARIANT_COUNT`           | `--g-bgen-decode-tile-variant-count`                            |
| chunks per Arrow file                       | `writer.rs`                                      | `--g-chunks-per-arrow-file`                                     |
| JAX cache directory                         | env                                              | `--g-jax-cache-dir`                                             |
| JAX matmul precision                        | env                                              | `--g-jax-matmul-precision`                                      |

I would split these into two groups:

## User-level REGENIE-compatible knobs

```bash
--bsize
--threads
--niter
--maxiter-null
--maxstep-null
```

Use REGENIE names where the meaning matches.

## Expert `g` tuning knobs

```bash
--g-device
--g-staging-depth
--g-writer-threads
--g-writer-queue-depth
--g-firth-batch-size
--g-firth-candidate-capacity
--g-bgen-decode-tile-variant-count
--g-chunks-per-arrow-file
--g-jax-cache-dir
--g-jax-matmul-precision
```

Do not hide these only in environment variables. Environment variables are bad for reproducibility unless you record them.

---

# Important import-order fix

If config controls JAX settings, the CLI must parse config before importing JAX-heavy modules.

Right now `src/g/cli.py` imports `g.api` at module import time, and `g.api` imports `jax_setup` and engine modules. That means options like JAX cache directory, matmul precision, x64, transfer guard, and platform can be applied too late.

I would change the front door to:

```python
# cli.py
def run_regenie_command(...):
    from g.interface.cli_parse import build_config_from_cli
    config = build_config_from_cli(...)
    from g.runner import regenie
    regenie(config)
```

And in `runner.py`:

```python
def regenie(config: RegenieConfig) -> RunArtifacts:
    normalized = normalize_and_validate(config)

    # Apply JAX config before importing compute modules.
    from g.runtime import configure_runtime
    configure_runtime(normalized.runtime)

    from g.engine import run_execution_plan
    return run_execution_plan(normalized)
```

This matters if you want config-driven reproducibility.

---

# Suggested validation behavior

## Trait mode

```text
no --qt/--bt/--t2e     -> quantitative, matching REGENIE default
--qt                   -> quantitative
--bt                   -> binary
--qt --bt              -> error
--t2e                  -> recognized unsupported, for now
```

## Step

```text
--step 2               -> supported
--step 1               -> recognized unsupported, for now
missing --step         -> for g regenie, probably require it to match REGENIE
```

Even if your app only supports Step 2, accepting `--step 2` is important for pipeline compatibility. REGENIE documents `--step` as taking `1` or `2`. ([RGC GitHub][1])

## Genotype input

```text
--bgen                 -> supported
--bed                  -> recognized unsupported unless implemented
--pgen                 -> recognized unsupported unless implemented
more than one          -> error
none                   -> error
```

REGENIE supports BGEN, BED, and PGEN inputs; your current public scope is BGEN Step 2, so recognize the others but do not pretend to support them. ([RGC GitHub][1])

## Phenotypes

Support both:

```bash
--phenoCol BMI
--phenoColList BMI,LDL,HDL
```

Also support repeated `--phenoCol`:

```bash
--phenoCol BMI --phenoCol LDL
```

If multi-phenotype GPU batching is not ready, you can still implement multi-phenotype sequentially at the runner layer:

```text
for phenotype in selected_phenotypes:
    run current single-phenotype engine
```

That gives users compatibility now and gives you a clean future path to batched multi-phenotype execution.

## Binary correction

Mirror REGENIE flags:

```text
default              -> score only
--firth --approx     -> approximate Firth fallback
--firth              -> exact Firth fallback, only if implemented
--spa                -> SPA fallback, only if implemented
--pThresh FLOAT      -> fallback threshold
--approx without --firth -> warning or error, matching chosen compatibility policy
```

Do not expose `--binary-correction` as the primary interface anymore.

---

# Example public API after refactor

```python
import g

config = g.RegenieConfig.from_toml("run.toml")
result = g.regenie(config)
```

Flat REGENIE-key mapping:

```python
result = g.regenie.from_options(
    {
        "step": 2,
        "bgen": "data/chr22.bgen",
        "sample": "data/chr22.sample",
        "phenoFile": "data/pheno.tsv",
        "phenoCol": "BMI",
        "covarFile": "data/covar.tsv",
        "covarColList": "age,sex,PC1,PC2",
        "pred": "data/step1_pred.list",
        "qt": True,
        "bsize": 8192,
        "out": "results/bmi",
        "g-device": "gpu",
    }
)
```

Pythonic dataclass:

```python
config = g.RegenieConfig(
    input=g.InputConfig(
        bgen="data/chr22.bgen",
        sample="data/chr22.sample",
        pheno_file="data/pheno.tsv",
        pheno_col=("BMI",),
        covar_file="data/covar.tsv",
        covar_col_list=("age", "sex", "PC1", "PC2"),
        pred="data/step1_pred.list",
    ),
    trait=g.TraitConfig(
        step=2,
        quantitative=True,
        block_size=8192,
        threads=8,
    ),
    output=g.OutputConfig(
        out="results/bmi",
    ),
    compute=g.GComputeConfig(
        device="gpu",
        staging_depth=2,
    ),
)

result = g.regenie(config)
```

The exact REGENIE names live in CLI/TOML/dict form. Python dataclasses stay ergonomic.

---

# Migration plan

## Sprint 1: Config and normalization layer

Add:

```text
src/g/interface/options.py
src/g/interface/schema.py
src/g/interface/normalize.py
src/g/runner.py
```

Do not change the engine yet.

Create:

```python
RegenieConfig
ExecutionPlan
normalize_regenie_config(config) -> ExecutionPlan
```

Then make current `api.regenie2(...)` create `RegenieConfig` and call `runner.regenie(config)`.

## Sprint 2: TOML support

Use Python stdlib `tomllib` for reading. For writing effective configs, either add `tomli-w` or implement a small writer. I would add a tiny writer dependency or use `tomlkit` if you care about preserving comments.

Add commands:

```bash
g config init --profile regenie2-bgen-qt > run.toml
g config validate run.toml
g config explain run.toml
g regenie --config run.toml
g regenie --config run.toml --write-effective-config effective.toml
```

Always write effective config into the run directory.

## Sprint 3: REGENIE-compatible CLI

Add:

```bash
g regenie
g-regenie
```

Support at least:

```text
--step
--bgen
--sample
--phenoFile
--phenoCol
--phenoColList
--covarFile
--covarCol
--covarColList
--pred
--qt
--bt
--bsize
--threads
--out
--firth
--approx
--spa
--pThresh
--firth-se
```

Recognize but reject unsupported common flags:

```text
--bed
--pgen
--keep
--remove
--extract
--exclude
--catCovarList
--test dominant/recessive
--t2e
```

You can later implement these one by one without changing the interface.

## Sprint 4: Output compatibility

For `g regenie`, decide and document:

```text
default output = REGENIE-compatible text
optional output = parquet/arrow/both
```

Add:

```bash
--g-output-format regenie|parquet|arrow|both
--g-output-run-directory
```

Keep internal chunked output in a separate run directory.

## Sprint 5: Make constants configurable

Move hardcoded/env-only settings into config structs.

Important: values that affect JAX compilation shapes, such as Firth batch size and candidate capacity, must be part of the normalized execution plan before kernels are compiled. Do not resolve them through cached environment-variable functions deep inside compute code.

## Sprint 6: Deprecate old names

Keep old commands but make them wrappers:

```text
g regenie2         -> g regenie --step 2 ...
g regenie2-linear  -> g regenie --step 2 --qt ...
```

Update README to show `g regenie`, not `g regenie2`.

---

# Test plan

Add tests at the interface layer, not just CLI tests.

## Parsing tests

```text
test_regenie_cli_qt_command_maps_to_execution_plan
test_regenie_cli_bt_firth_approx_maps_to_binary_plan
test_regenie_cli_phenoColList_parses_multiple_traits
test_regenie_cli_repeated_phenoCol_parses_multiple_traits
test_regenie_cli_bsize_maps_to_chunk_size
test_regenie_cli_threads_maps_to_runtime_threads
test_regenie_cli_unsupported_bed_errors_clearly
```

## TOML tests

```text
test_toml_config_loads_regenie_keys
test_toml_unknown_key_errors
test_toml_cli_overrides_config
test_toml_boolean_true_not_overwritten_by_missing_cli_flag
test_effective_config_round_trips
```

## Python API tests

```text
test_regenie_accepts_config_dataclass
test_regenie_accepts_exact_regenie_option_dict
test_regenie_from_toml_calls_same_runner_as_cli
test_legacy_regenie2_maps_to_new_config
```

## Drift tests

```text
test_every_supported_option_spec_has_config_location
test_every_supported_option_spec_has_cli_flag
test_every_config_template_key_maps_to_option_spec
test_no_hardcoded_tuning_constant_without_config_entry
```

That last test is valuable for your “constants should not be hardcoded” goal.

---

# Bottom line

I would make `g regenie` the main UX, with exact REGENIE-style flags, and add `g-regenie` as an executable alias for pipeline migration.

Internally, build everything around:

```text
RegenieConfig -> validate -> ExecutionPlan -> engine
```

Then make CLI, TOML, and Python all feed into the same config path.

That gives you:

```text
same REGENIE vocabulary for users
TOML reproducibility
Python config ergonomics
no CLI/API drift
configurable performance knobs
clean migration from current g regenie2 commands
```

[1]: https://rgcgithub.github.io/regenie/options/ "https://rgcgithub.github.io/regenie/options/"

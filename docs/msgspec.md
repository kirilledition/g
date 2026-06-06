Historical note: this was the implementation plan for the msgspec-based configuration
loader. The active code now uses typed TOML schema/config layer modules and a packaged
default catalog. Do not read the original problem statement below as current-state
architecture; use `docs/configuration_cli_architecture.md` for the live contract.

The notes below are retained only as implementation history.

Yes — your discomfort is valid. The current code is doing a lot of manual “dict archaeology”: `tomllib` loads raw dictionaries, then `config.py` manually normalizes names and casts each value into a typed dataclass. You can see this pattern in the current config module: it loads `config.default.toml` through `tomllib.load(...)`, then defines many fallback constants, and later builds `GComputeConfig`, `GOutputConfig`, etc. by repeatedly doing `int(...)`, `float(...)`, enum constructors, and `.get(..., DEFAULT_...)` lookups.  

Python does have better options. The stdlib `tomllib` intentionally only parses TOML into built-in containers; it does not validate into a typed structure. For Rust/Go-like typed loading, I would use **msgspec** or **Pydantic v2**. For `g`, I would choose **msgspec**.

## Recommendation for `g`

Use **msgspec** for typed TOML decoding, but keep `OptionSpec` as the single source of option metadata.

```text
OptionSpec:
  public option registry, CLI flags, TOML path, help text, support level

msgspec TOML structs:
  typed schema for TOML files

Resolved dataclasses / structs:
  final internal config passed to ExecutionPlan
```

Do not make msgspec replace `OptionSpec`. They solve different problems.

`OptionSpec` is still needed because it describes the cross-interface contract: REGENIE-compatible names, `--g-*` names, unsupported-but-recognized flags, Click option generation, TOML sections, and help text. The current `OptionSpec` already captures metadata like name, destination, support level, TOML section, help text, CLI flags, type, multiple-ness, boolean-ness, and accepted values. 

`msgspec` should replace the manual TOML-to-dict-to-dataclass parsing.

---

# Why msgspec fits this app

`msgspec` has Rust/Go-like “define a struct, decode into it” behavior. Its docs describe `Struct` as the preferred way to define structured data types, with fields defined by type annotations, optional defaults, and generated methods; the structs are implemented in C and intended to be fast and lightweight. ([Jim Crist-Harif][1])

Most importantly for your config use case, `msgspec.toml.decode(...)` can decode TOML directly into a specified Python type, with type checking, strict mode, and decode hooks for custom types. ([Jim Crist-Harif][2]) It also supports forbidding unknown fields, which is exactly what you want for catching typos in config files. ([Jim Crist-Harif][1])

It supports field renaming too, so TOML keys like `phenoFile`, `pThresh`, `firth-se`, and `staging-depth` can map to Python-safe field names. ([Jim Crist-Harif][1])

Pydantic v2 is also viable. It gives excellent validation errors, aliases, nested models, strict mode, and validators; Pydantic models are explicitly designed to define typed schemas and validate data into those schemas. ([Pydantic Docs][3]) But for this project, I would prefer msgspec because config loading is simple, performance-sensitive startup matters, and you do not need a large validation framework.

---

# What the architecture should look like

Instead of:

```python
raw = tomllib.load(file)
normalized = normalize_option_dictionary(raw)
config = RegenieConfig(
    g_compute=GComputeConfig(
        staging_depth=int(normalized.get("g-staging-depth", 1)),
        ...
    )
)
```

use:

```python
raw_toml_config = msgspec.toml.decode(data, type=TomlConfig)
resolved_config = resolve_config_layers(default_config, user_config, cli_overrides)
```

The typed layer should distinguish:

```text
TomlConfig:
  exactly what users can write in TOML

CliOverrides:
  explicit CLI values only

ResolvedRegenieConfig:
  fully merged, typed, validated config

ExecutionPlan:
  engine-ready immutable plan
```

---

# Proposed typed TOML schema

A TOML schema can mirror your public sections.

```python
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import msgspec
from msgspec import Meta, Struct, UNSET, UnsetType


PositiveInt = Annotated[int, Meta(ge=1)]
Probability = Annotated[float, Meta(gt=0.0, lt=1.0)]


class InputToml(Struct, forbid_unknown_fields=True):
    bgen: Path | UnsetType = UNSET
    sample: Path | UnsetType = UNSET
    pheno_file: Path | UnsetType = msgspec.field(default=UNSET, name="phenoFile")
    pheno_col: str | list[str] | UnsetType = msgspec.field(default=UNSET, name="phenoCol")
    pheno_col_list: str | list[str] | UnsetType = msgspec.field(default=UNSET, name="phenoColList")
    covar_file: Path | UnsetType = msgspec.field(default=UNSET, name="covarFile")
    covar_col: str | list[str] | UnsetType = msgspec.field(default=UNSET, name="covarCol")
    covar_col_list: str | list[str] | UnsetType = msgspec.field(default=UNSET, name="covarColList")
    pred: Path | UnsetType = UNSET


class TraitToml(Struct, forbid_unknown_fields=True):
    step: int | UnsetType = UNSET
    qt: bool | UnsetType = UNSET
    bt: bool | UnsetType = UNSET
    bsize: PositiveInt | UnsetType = UNSET
    threads: PositiveInt | UnsetType = UNSET


class BinaryToml(Struct, forbid_unknown_fields=True):
    firth: bool | UnsetType = UNSET
    approx: bool | UnsetType = UNSET
    spa: bool | UnsetType = UNSET
    p_threshold: Probability | UnsetType = msgspec.field(default=UNSET, name="pThresh")
    firth_se: bool | UnsetType = msgspec.field(default=UNSET, name="firth-se")


class GComputeToml(Struct, forbid_unknown_fields=True):
    device: str | UnsetType = UNSET
    staging_depth: PositiveInt | UnsetType = msgspec.field(default=UNSET, name="staging-depth")
    trusted_no_missing_diploid: bool | UnsetType = msgspec.field(
        default=UNSET,
        name="trusted-no-missing-diploid",
    )
    trusted_bgen_validation_mode: str | UnsetType = msgspec.field(
        default=UNSET,
        name="trusted-bgen-validation-mode",
    )
    sample_key_mode: str | UnsetType = msgspec.field(default=UNSET, name="sample-key-mode")
    multi_phenotype_sample_mode: str | UnsetType = msgspec.field(
        default=UNSET,
        name="multi-phenotype-sample-mode",
    )
    score_dtype: str | UnsetType = msgspec.field(default=UNSET, name="score-dtype")
    firth_dtype: str | UnsetType = msgspec.field(default=UNSET, name="firth-dtype")
    bgen_decode_tile_variant_count: PositiveInt | UnsetType = msgspec.field(
        default=UNSET,
        name="bgen-decode-tile-variant-count",
    )
```

Then top-level:

```python
class GToml(Struct, forbid_unknown_fields=True):
    compute: GComputeToml | UnsetType = UNSET
    output: GOutputToml | UnsetType = UNSET
    diagnostics: GDiagnosticsToml | UnsetType = UNSET


class TomlConfig(Struct, forbid_unknown_fields=True):
    input: InputToml | UnsetType = UNSET
    trait: TraitToml | UnsetType = UNSET
    binary: BinaryToml | UnsetType = UNSET
    output: OutputToml | UnsetType = UNSET
    g: GToml | UnsetType = UNSET
```

Decode:

```python
def read_toml_config(path: Path) -> TomlConfig:
    data = path.read_bytes()
    try:
        return msgspec.toml.decode(data, type=TomlConfig, strict=True)
    except msgspec.ValidationError as error:
        raise ConfigError(f"Invalid TOML config {path}: {error}") from error
```

`msgspec.UNSET` is useful here because it distinguishes a field that was absent from a field that was present with a value; the docs describe it as a sentinel for fields that are unset and note that omitted fields decode to `UNSET`. ([Jim Crist-Harif][2]) TOML itself has no `null`, but `UNSET` is still useful because it preserves “user did not specify this key,” which is essential for layering config files and CLI overrides.

---

# How merging should work

The key is to merge **typed partial configs**, not raw dicts.

```python
def merge_layers(
    default_config: TomlConfig,
    user_configs: list[TomlConfig],
    cli_overrides: CliOverrides,
) -> ResolvedRegenieConfig:
    merged = default_config
    for user_config in user_configs:
        merged = overlay_toml_config(merged, user_config)
    merged = overlay_cli_overrides(merged, cli_overrides)
    return resolve_toml_config(merged)
```

Overlay rules:

```text
UNSET:
  means "no override"

False:
  is a real override

0:
  is a real override if the field allows it

empty list:
  is a real override if the field allows it
```

This is much healthier than `dict.get(...)` fallbacks because you stop conflating:

```text
missing key
explicit false
explicit empty
default value
```

That is especially important for booleans like:

```text
firth
approx
g-resume
g-log-stderr
g-finalize-parquet
```

---

# Where `OptionSpec` still matters

Typed TOML structs do not remove the need for your option registry.

`OptionSpec` should continue to drive:

```text
Click option generation
TOML template generation
config explain
recognized unsupported REGENIE flags
Python option dictionary alias normalization
help text
public option naming rules
```

The current CLI already generates `g regenie` options from `OPTION_SPECS`, which is the right pattern. 

The new typed TOML schema should be validated against `OptionSpec` in tests:

```python
def test_every_supported_toml_option_has_field_in_typed_schema():
    ...

def test_every_typed_schema_field_maps_to_option_spec():
    ...

def test_config_default_toml_decodes_to_default_typed_config():
    ...
```

This keeps the two layers from drifting.

---

# What I would not do

## I would not keep the current manual parsing

The current flow has too much hand-coded casting and duplicate defaults. The current default config is `src/g/config.default.toml`, but `config.py` still defines many `DEFAULT_*` values for tunable behavior.   That is exactly the confusion you want to eliminate.

## I would not rely only on dataclasses + `tomllib`

You can use `cattrs` or `dacite` to convert `dict -> dataclass`, and `cattrs` is explicitly designed to convert unstructured dictionaries into classes and back while validating contents. ([Cattrs][4]) But since `msgspec` already has typed TOML decode, faster structs, rename support, unknown-field rejection, and `UNSET`, it is a better fit here.

## I would not make Pydantic the default unless you want rich validation UX

Pydantic is excellent and mature. It guarantees output conforms to declared model types after validation and has rich errors/validators. ([Pydantic Docs][3]) But it is a bigger dependency and can encourage putting too much validation logic inside model classes. For `g`, I prefer:

```text
msgspec:
  syntax/type/shape validation

explicit resolver:
  semantic validation and REGENIE rules
```

That split is cleaner.

---

# Proposed migration plan

## Phase 1: Add typed TOML schemas

Create:

```text
src/g/interface/toml_schema.py
```

with:

```text
InputToml
TraitToml
BinaryToml
GComputeToml
GOutputToml
GDiagnosticsToml
GToml
TomlConfig
```

Use `msgspec.Struct`.

Use:

```python
forbid_unknown_fields=True
```

on every struct so typos fail.

Use `msgspec.field(name=...)` for keys like:

```text
phenoFile
phenoColList
pThresh
firth-se
staging-depth
writer-threads
```

## Phase 2: Decode default config directly into typed struct

Replace:

```python
tomllib.load(...)
```

for defaults with:

```python
msgspec.toml.decode(..., type=TomlConfig)
```

For a transitional step, you can still convert the typed struct to canonical normalized options, but the raw TOML is typed first.

## Phase 3: Decode user TOML directly into partial typed struct

```python
def load_toml(path: Path) -> TomlConfig:
    return msgspec.toml.decode(path.read_bytes(), type=TomlConfig)
```

No raw dict except for Python `from_options(...)`.

## Phase 4: Add typed CLI override structure

Click still returns a dict. That is fine. Convert it into:

```python
CliOverrides
```

or into a partial `TomlConfig` using `OptionSpec`.

```python
def cli_options_to_toml_config(cli_options: Mapping[str, Any]) -> TomlConfig:
    ...
```

This keeps the resolver generic:

```python
default TomlConfig + user TomlConfig + cli TomlConfig
```

## Phase 5: Resolve into final dataclasses

After overlaying layers:

```python
resolved = resolve_toml_config(merged_toml_config)
```

This is where you enforce semantic rules:

```text
--qt and --bt mutually exclusive
--approx requires --firth
--spa unsupported
--firth without --approx unsupported
required inputs exist
output path exists / resume rules
score/firth dtype consistency
```

Do not put all of this in `msgspec.__post_init__`. Keep schema validation separate from semantic validation.

## Phase 6: Delete duplicate defaults

Once typed defaults are loaded from `config.default.toml`, remove configurable `DEFAULT_*` constants from `config.py`.

The typed schema can still have `UNSET` defaults for fields, because the actual values are loaded from `config.default.toml`. Do not put tunable defaults in class field definitions.

Example:

```python
class TraitToml(Struct, forbid_unknown_fields=True):
    step: int | UnsetType = UNSET
    qt: bool | UnsetType = UNSET
    bt: bool | UnsetType = UNSET
    bsize: int | UnsetType = UNSET
```

Even though default `bsize = 16384` exists, it should live only in `config.default.toml`, not in the struct class.

---

# Example end-state

```python
def load_default_config() -> TomlConfig:
    data = importlib.resources.files("g").joinpath("config.default.toml").read_bytes()
    config = decode_toml_config_bytes(data, source="config.default.toml")
    validate_default_config_complete(config)
    return config


def load_user_config(path: Path) -> TomlConfig:
    return decode_toml_config_bytes(path.read_bytes(), source=str(path))


def from_config_layers(paths: Sequence[Path], cli_options: Mapping[str, Any]) -> RegenieConfig:
    config_layers = [load_default_config()]
    config_layers.extend(load_user_config(path) for path in paths)
    config_layers.append(cli_options_to_toml_config(cli_options))
    merged = overlay_toml_configs(config_layers)
    return resolve_toml_config(merged)
```

The resolver becomes cleaner because it operates on attributes:

```python
trait = merged.trait
bsize = require_value(trait.bsize, "trait.bsize")
```

not:

```python
int(normalized_options.get("bsize", DEFAULT_BSIZE))
```

---

# Important caveat

Typed TOML structs alone do **not** solve REGENIE CLI compatibility.

Users can pass CLI options like:

```bash
--phenoCol BMI --phenoCol HDL
--phenoColList BMI,HDL
--bt --firth --approx
```

and config files can use sections like:

```toml
[g.compute]
device = "gpu"
```

while canonical option names are internally `g-device`, `g-score-dtype`, etc. Your `OptionSpec` table is still the bridge between these representations. Do not delete it.

The new design should be:

```text
msgspec:
  "Is this TOML structurally valid and typed?"

OptionSpec:
  "What public option does this key mean?"

resolver:
  "Is this combination semantically valid?"

ExecutionPlan:
  "How does the engine run?"
```

That gives you Rust/Go-style typed loading without losing your current unified CLI/TOML/Python interface.

---

# My final recommendation

Use **msgspec**.

It gives you the typed-loading behavior you want:

```python
cfg = msgspec.toml.decode(data, type=TomlConfig)
```

with type checking, strict decoding, field renames, unknown-field rejection, and an `UNSET` sentinel for partial config layering. ([Jim Crist-Harif][2])

I would send an agent to implement:

```text
src/g/interface/toml_schema.py
src/g/interface/config_layers.py
```

then gradually shrink `config.py` until it mostly does:

```text
load typed layers
overlay layers
resolve typed config
validate semantics
```

That will feel much closer to Rust/Go config loading while still fitting Python’s CLI/TOML/Python API realities.

[1]: https://jcristharif.com/msgspec/structs.html "Structs"
[2]: https://jcristharif.com/msgspec/api.html "API Docs"
[3]: https://docs.pydantic.dev/latest/concepts/models/ "Models | Pydantic Docs"
[4]: https://catt.rs/en/stable/ "cattrs 26.1.0 documentation"

# Configuration Frontend

| Status | Applies to | Owner |
| --- | --- | --- |
| Development contract | CLI, TOML, Python option dictionaries, defaults, and `RegenieConfig` construction | Interface maintainers |

This page defines how developers keep CLI, TOML, Python options, defaults, and
execution planning unified. User-facing behavior is documented in
[CLI](../public/cli.md) and [Configuration](../public/configuration.md).

## Production Flow

```text
CLI args / TOML config / Python option dict
        |
raw option layers
        |
OptionSpec registry + packaged defaults
        |
RegenieConfig
        |
ExecutionPlan
        |
engine pipeline
```

The engine must not read CLI flags, TOML files, environment variables, or
scattered runtime defaults directly. User-controlled behavior flows through
`RegenieConfig` and then `ExecutionPlan`.

## Source Ownership

| Source | Owns |
| --- | --- |
| `src/g/interface/options.py` | Canonical option registry, CLI flags, destination aliases, TOML section/key mapping, support level, accepted values, help text. |
| `src/g/interface/toml_schema.py` | Strict TOML shape and accepted section/key names. |
| `src/g/config.default.toml` | Packaged default values for defaultable options. |
| `src/g/interface/config_layers.py` | Layer decoding, normalization, overlay, boolean coercion, Python alias handling. |
| `src/g/interface/config.py` | Runtime dataclasses, validation, unsupported-option rejection, trait resolution, serialization. |
| `src/g/execution_plan.py` | Immutable execution-plan construction from validated config. |
| `src/g/cli.py` | Click command wiring generated from `OptionSpec`. |

When adding or changing a user-facing option, update the owning source and the
corresponding tests. Do not introduce a second option table.

## Defaults Policy

Mutable defaults live in `src/g/config.default.toml`. Do not copy them into
implementation code or user documentation as constants. If a code path needs a
default, read it through the packaged default config.

Tests enforce that packaged defaults match the option policies and that removed
configurable defaults do not reappear as production constants.

## Option Addition Checklist

1. Add one `OptionSpec` in `src/g/interface/options.py`.
2. Add the TOML field to `src/g/interface/toml_schema.py`.
3. Add a packaged default in `src/g/config.default.toml` when the option is
   defaultable.
4. Add a field to the appropriate runtime dataclass in
   `src/g/interface/config.py`.
5. Thread the field into `ExecutionPlan` or the target runtime boundary.
6. Add validation for invalid combinations or unsupported modes.
7. Update tests for CLI, TOML schema, Python options, and the runtime boundary.
8. Update public docs if behavior, defaults policy, inputs, outputs, telemetry,
   or performance assumptions change.

## Support Levels

| Support level | Meaning |
| --- | --- |
| `SUPPORTED` | Option is accepted and executable. |
| `RECOGNIZED_UNSUPPORTED` | Option is accepted for migration diagnostics, then rejected when active. |
| `G_EXTENSION` | Engine-specific native CLI option outside REGENIE's original option set. |
| `DEPRECATED_ALIAS` | Reserved for compatibility aliases. |

Recognized unsupported options must never be silently ignored.

## Boolean And Trait Rules

Boolean CLI options use paired Click flags such as `--resume` and
`--no-resume`. Only command-line-provided options are converted into the CLI
override layer.

Trait flags have layer-aware semantics:

- a single config layer cannot explicitly set both `qt = true` and `bt = true`;
- an explicit quantitative selection clears binary mode in the merged config;
- an explicit binary selection clears quantitative mode in the merged config;
- binary-only options are rejected after the final trait type is known.

Keep these rules centralized in `src/g/interface/config.py` and
`src/g/interface/config_layers.py`.

## Tests To Update

Relevant test files include:

- `tests/test_cli.py`
- `tests/test_interface.py`
- `tests/test_api.py`
- `tests/test_preflight.py`
- `tests/test_regenie2_pipeline.py`

At minimum, new options should be covered by option-registry schema tests and by
the runtime boundary that consumes them.

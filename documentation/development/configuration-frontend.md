# Configuration Frontend

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; development contract | main branch as of 2026-06-30 CLI/TOML/Python config frontend | Interface maintainers |

This page defines how developers keep CLI, TOML, Python options, defaults, and
execution planning unified. User-facing behavior is documented in
[CLI](../public/cli.md) and [Configuration](../public/configuration.md).

## Production Flow

```text
CLI args / TOML config / Python option dict
        |
raw option layers
        |
Rust option metadata + packaged defaults + validation
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
| `crates/interface/src/options.rs` | Canonical option metadata for CLI names, TOML aliases, flat Python names, and value kinds. |
| `crates/interface/src/cli/` | CLI parser and CLI-to-config-layer conversion. |
| `crates/interface/src/toml.rs` | Strict TOML layer decoding and accepted section/key names. |
| `crates/interface/src/config.default.toml` | Packaged default values for defaultable options. |
| `crates/interface/src/defaults.rs`, `overlay.rs`, `partial.rs`, `resolved.rs`, `validation.rs`, `run_validation.rs` | Defaults, layer overlay, resolved config construction, validation, and run validation. |
| `src/python/config/` | PyO3 conversion between Rust-owned config objects and Python classes. |
| `src/g/interface/config.py` | Thin Python bridge that normalizes Python option dictionaries using Rust metadata. |
| `src/g/execution_plan.py` | Immutable execution-plan construction from validated config. |
| `src/g/runner/cli.py` | Thin Python console-script adapter that prints native CLI driver output chunks and calls the current Python/JAX backend callback. |

When adding or changing a user-facing option, update the owning source and the
corresponding tests. Do not introduce a second option table.

## Defaults Policy

Mutable defaults live in `crates/interface/src/config.default.toml`. Do not copy them into
implementation code or user documentation as constants. If a code path needs a
default, read it through the packaged default config.

Tests enforce that packaged defaults match the option policies and that removed
configurable defaults do not reappear as production constants.

## Option Addition Checklist

1. Add or update `ConfigOptionMetadata` in `crates/interface/src/options.rs`.
2. Add the CLI parser/layer field in `crates/interface/src/cli/` when the option is
   accepted on the command line.
3. Add the TOML/partial/resolved config field in `crates/interface/src/` when the
   option is accepted in config files or affects runtime state.
4. Add a packaged default in `crates/interface/src/config.default.toml` when the option is
   defaultable.
5. Update `src/python/config/` and `src/g/_core.pyi` when the option is exposed
   through Python config objects.
6. Thread the field into `ExecutionPlan` or the target runtime boundary.
7. Add validation for invalid combinations or unsupported modes.
8. Update tests for CLI, TOML, Python options, and the runtime boundary.
9. Update public docs if behavior, defaults policy, inputs, outputs, telemetry,
   or performance assumptions change.

## Unsupported Options And Aliases

REGENIE-style names and engine-specific names are declared in Rust metadata.
Accepted aliases must resolve to one canonical config field before validation.
Recognized unsupported options must be rejected explicitly; they must never be
silently ignored or handled only in Python.

## Boolean And Trait Rules

Boolean CLI options use paired Click flags such as `--resume` and
`--no-resume`. Only command-line-provided options are converted into the CLI
override layer.

Trait flags have layer-aware semantics:

- a single config layer cannot explicitly set both `qt = true` and `bt = true`;
- an explicit quantitative selection clears binary mode in the merged config;
- an explicit binary selection clears quantitative mode in the merged config;
- binary-only options are rejected after the final trait type is known.

Keep these rules centralized in the Rust CLI layer, overlay, and validation
modules under `crates/interface/src/`. Python should only normalize Python option
dictionaries into the Rust-owned shape before calling the PyO3 config builder.

## Tests To Update

Relevant test files include:

- `tests/test_cli.py`
- `tests/test_interface.py`
- `tests/test_api.py`
- `tests/test_preflight.py`
- `tests/test_regenie2_pipeline.py`

At minimum, new options should be covered by option-registry schema tests and by
the runtime boundary that consumes them.

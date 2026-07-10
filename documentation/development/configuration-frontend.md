# Configuration Frontend

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; development contract | production config frontend as of 2026-07-10 | Interface maintainers |

This page defines how developers keep CLI, TOML, defaults, and execution
planning unified. User-facing behavior is documented in
[CLI](../public/cli.md) and [Configuration](../public/configuration.md).

## Production Flow

```text
CLI args / TOML config
        |
raw option layers
        |
Rust config layers + packaged defaults + validation
        |
RegenieConfigData
        |
g-plan RunRequest / PreparedRunPlan
        |
native Rust host and g-engine scheduler
```

The engine must not read CLI flags, TOML files, environment variables, or
scattered runtime defaults directly. User-controlled behavior flows through the
Rust-owned resolved config and immutable `g-plan` run contracts.

## Source Ownership

| Source | Owns |
| --- | --- |
| `crates/interface/src/cli/` | CLI parser and CLI-to-config-layer conversion. |
| `crates/interface/src/toml.rs` | Strict TOML layer decoding and accepted section/key names. |
| `crates/interface/src/config.default.toml` | Packaged default values for defaultable options. |
| `crates/interface/src/defaults.rs`, `overlay.rs`, `partial.rs`, `resolved.rs`, `validation.rs`, `run_validation.rs` | Defaults, layer overlay, resolved config construction, validation, and run validation. |
| `crates/interface/src/plan_request.rs` | Compilation from resolved config into immutable `g-plan` run requests. |
| `crates/plan/src/` | Requested and prepared run contracts consumed by the engine and manifests. |
| `src/binding/engine/backend.rs` | Typed JAX-backend settings and batch exchange values consumed by Python. |
| `src/g/cli.py` | Thin Python console bootstrap that forwards arguments and renders native output. |

When adding or changing a user-facing option, update the owning parser/config
struct and the corresponding validation. Do not introduce a second option
table.

## Defaults Policy

Mutable defaults live in `crates/interface/src/config.default.toml`. Do not copy them into
implementation code or user documentation as constants. If a code path needs a
default, read it through the packaged default config.

Tests enforce that packaged defaults match the option policies and that removed
configurable defaults do not reappear as production constants.

## Option Addition Checklist

1. Add the CLI parser/layer field in `crates/interface/src/cli/` when the option is
   accepted on the command line.
2. Add the TOML/partial/resolved config field in `crates/interface/src/` when the
   option is accepted in config files or affects runtime state.
3. Add a packaged default in `crates/interface/src/config.default.toml` when the option is
   defaultable.
4. Extend `JaxBackendConfig` and the engine stub only when the Python/JAX
   backend needs the resolved value.
5. Thread the field into `g-plan::RunRequest`, `PreparedRunPlan`, or the typed
   JAX backend config, according to its owner.
6. Add validation for invalid combinations or unsupported modes.
7. Update tests for CLI, TOML, and the runtime boundary after the API stabilizes.
8. Update public docs if behavior, defaults policy, inputs, outputs, telemetry,
   or performance assumptions change.

## Unsupported Options And Aliases

REGENIE-style names and engine-specific names are declared in Rust metadata.
Accepted aliases must resolve to one canonical config field before validation.
Recognized unsupported options must be rejected explicitly; they must never be
silently ignored or handled only in Python.

## Boolean And Trait Rules

Boolean CLI options use positive REGENIE-compatible flags. Only
command-line-provided options are converted into the CLI override layer;
native-only booleans are configured in TOML.

Trait flags have layer-aware semantics:

- a single config layer cannot explicitly set both `qt = true` and `bt = true`;
- an explicit quantitative selection clears binary mode in the merged config;
- an explicit binary selection clears quantitative mode in the merged config;
- binary-only options are rejected after the final trait type is known.

Keep these rules centralized in the Rust CLI layer, overlay, and validation
modules under `crates/interface/src/`. Python must not reconstruct or validate
the production run request.

## Tests To Update

Configuration frontend changes are no longer covered by Python product tests in
the math-only suite. At minimum, new options should be covered by Rust
option-registry, schema, and runtime-boundary tests under the owning crate.

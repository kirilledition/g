# Public API

## This crate owns

Frontend configuration normalization for CLI, TOML, and Python options.

## Public types

Resolved config data, resolved config enum values, config metadata, CLI outcome payloads, and config-validation errors.

## Public functions

Load packaged config, parse options/TOML, dump/write TOML, validate config, compile `g-plan` run requests, and dispatch CLI parsing.

## This crate must not expose

Runtime setup, BGEN opening, sample alignment, output writing, JAX backend choices after planning, or PyO3 lifecycle state.

## Performance constraints

Keep work at config/build time only. Do not perform data-file scans or compute-side effects.

## Allowed downstream users

`g-cli` and root PyO3 facade.

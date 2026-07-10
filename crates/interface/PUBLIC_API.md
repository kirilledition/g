# Public API

## This crate owns

Frontend configuration normalization for canonical TOML plus the native `g`
CLI binary shell. The CLI accepts `--config` and the
supported REGENIE step 2 flags; native tuning is TOML-only.

## Public types

Resolved config data and enum values, CLI outcome payloads, native CLI outcome
payloads, and `ConfigError`/`ConfigResult`.

## Public functions

Dump/write resolved TOML, validate config,
compile `g-plan` run requests, dispatch CLI parsing, and dispatch the native
CLI frontend.

## This crate must not expose

Runtime setup, BGEN opening, sample alignment, output writing, JAX backend choices after planning, or PyO3 lifecycle state.

## Performance constraints

Keep work at config/build time only. Do not perform data-file scans or compute-side effects.

## Allowed downstream users

Native binary entrypoint and root PyO3 facade.

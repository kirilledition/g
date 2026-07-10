# Public API

## This crate owns

Frontend configuration normalization for canonical TOML plus the native `g`
CLI binary shell. The CLI accepts `--config` and the
supported REGENIE step 2 flags; native tuning is TOML-only.

## Public types

CLI dispatch payloads consumed by `g-runner`.

## Public functions

Dispatch the native CLI frontend. TOML serialization, validation, and plan
compilation are implementation details.

## This crate must not expose

Runtime setup, BGEN opening, sample alignment, output writing, JAX backend choices after planning, or PyO3 lifecycle state.

## Performance constraints

Keep work at config/build time only. Do not perform data-file scans or compute-side effects.

## Allowed downstream users

`g-runner` only.

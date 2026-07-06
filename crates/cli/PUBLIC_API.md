# Public API

## This crate owns

Native CLI frontend shell around `g-interface`.

## Public types

`NativeCliOutcome`.

## Public functions

`dispatch_native_cli`.

## This crate must not expose

Python bridge adapters, native execution internals, signal handling internals, or runtime engine orchestration.

## Performance constraints

CLI dispatch must stay parse-only until native execution is deliberately implemented.

## Allowed downstream users

Native binary entrypoint only.

# Public API

## This crate owns

Python-free native CLI lifecycle coordination above interface, runtime, and
engine crates.

## Public types

`NativeRunHost`, `NativeRunInterruption`, and `CliRunResult`.
The facade also re-exports only the runtime contracts the PyO3 host must name
to implement `NativeRunHost`.

## Public functions

`run_cli` dispatches the CLI, owns process-global setup, constructs terminal
output, and invokes the coordinated engine run exactly once.

## This crate must not expose

PyO3, NumPy, JAX objects, user configuration DTOs, direct output writers, or a
second engine scheduler.

## Allowed downstream users

The root PyO3 extension only.

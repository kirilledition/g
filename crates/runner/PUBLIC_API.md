# Public API

## This crate owns

Python-free native CLI lifecycle coordination above interface, runtime, and
engine crates, including JAX process policy and terminal rendering.

## Public types

`NativeRunHost`, `NativeRunInterruption`, `NativeRunFailure`, `CliRunResult`,
runner-owned JAX runtime observations, and the mode-specialized
`JaxAssociationBackendPlan`. Backend construction receives the canonical
`g-plan::Device` separately from this mode-specialized plan. The backend plan
borrows canonical `g-plan` kernel policy rather than redefining scalar
settings. The Python host names no `g-runtime` type.

## Public functions

`run_cli` dispatches the CLI, resolves the plan's telemetry mode and output
layout into a generic `g-runtime::NativeRunSessionPolicy`, owns process-global
setup, constructs terminal output, and invokes the coordinated engine run
exactly once per compiled run. It rejects an incompatible process-global
logging topology under the runtime-state lock before opening run files or
starting asynchronous writers. Execution freezes into completed, interrupted,
or failed primary outcomes. Later observer failures cannot replace a primary
failure or interruption, and completed artifact paths remain visible when a
required timing or close observer is the only failure.

## This crate must not expose

PyO3, NumPy, JAX objects, user configuration DTOs, direct output writers, or a
second engine scheduler.

## Allowed downstream users

The root PyO3 extension only.

# Binding Layer Policy

| Status | Applies to | Owner |
| --- | --- | --- |
| Active | Native `_core` bindings under `src/binding` | Development maintainers |

## Purpose

`src/binding` is the PyO3 adaptation layer for `g._core`. It converts between Python-facing objects and Rust crate APIs. It must not own GWAS execution policy, scheduling policy, output planning, telemetry policy, or domain validation logic when that logic can be expressed with pure Rust crate types.

The ownership boundary is:

```text
crates/*    = Rust application and domain logic
src/binding = PyO3 adaptation only
src/g       = public Python API and JAX backend/kernels
```

## Rule

If a function can be written using only Rust crate types, it belongs in a crate. If a function needs Python, PyO3, `PyAny`, `PyErr`, `PyModule`, NumPy/Python buffers, or direct Python callback invocation, it may live in `src/binding`.

## Allowed In `src/binding`

- `#[pyclass]` wrappers and `#[pymethods]` accessors.
- `#[pyfunction]` wrappers.
- Python module and submodule registration.
- Python callback invocation and callback-object extraction.
- NumPy/Python buffer adapters.
- Python-to-Rust data extraction and Rust-to-Python object construction.
- `PyErr` conversion through binding error helpers.
- Temporary compatibility aliases during staged migrations.

## Forbidden In `src/binding`

- Scheduling policy.
- Callback worker lifecycle policy.
- BGEN delivery cleanup policy.
- Manifest/header construction policy.
- Output resume or repair planning.
- Preflight validation policy beyond converting Python arrays and calling crate validation.
- Run-event or diagnostic payload construction policy.
- Telemetry rendering policy.
- Genotype preprocessing policy.
- Sample alignment policy.
- Domain-level config validation.

## Python Namespace Policy

Production exports should live under domain submodules:

```text
g._core.cli
g._core.config
g._core.runtime
g._core.telemetry
g._core.engine
g._core.genotype
g._core.input
g._core.output
```

Debug and migration-only internals must live under:

```text
g._core.debug
```

Root `g._core` symbols are compatibility aliases only. New Python callers must use a domain submodule.

## Module Header Convention

New binding modules must describe the Python namespace they adapt and the crate/domain APIs they wrap:

```rust
//! PyO3 bindings for `_core.engine`.
//!
//! Adapts:
//! - g-engine high-level session/backend APIs
//!
//! Allowed:
//! - PyO3 wrappers
//! - Python callback bridge
//! - error conversion
//!
//! Forbidden:
//! - scheduling policy
//! - output manifest construction
//! - BGEN delivery cleanup
```

Keep the header short, but make the boundary explicit.

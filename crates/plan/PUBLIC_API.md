# Public API

## This crate owns

Canonical immutable run-planning contracts and validated numeric policy shared
by Rust crates and PyO3.

## Public types

`RunPlan` and its input, analysis, compute, correction, output, runtime,
diagnostic, phenotype-group, enum, and validated finite numeric components.

## Public functions

Deterministic phenotype grouping and output-identifier helpers.

## This crate must not expose

BGEN decoding, sample/phenotype parsing, output writer sessions, callback queues, JAX side effects, telemetry writers, or PyO3 classes.

## Performance constraints

Keep DTO construction deterministic and allocation-visible. Do not add hot-path parsing, I/O, or JSON round trips here.

## Allowed downstream users

`g-interface`, `g-output`, `g-engine`, and root PyO3 facade.

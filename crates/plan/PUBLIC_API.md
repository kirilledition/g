# Public API

## This crate owns

Stable, serializable run-planning contracts shared by Rust crates and PyO3.

## Public types

Run request/config DTOs, host-policy plans, prepared-run plans, manifest fingerprints, association/backend enums, runtime/output plan DTOs.

## Public functions

Deterministic planning and identifier helpers, including prepared-run and association-backend planning.

## This crate must not expose

BGEN decoding, sample/phenotype parsing, output writer sessions, callback queues, JAX side effects, telemetry writers, or PyO3 classes.

## Performance constraints

Keep DTO construction deterministic and allocation-visible. Do not add hot-path parsing, I/O, or JSON round trips here.

## Allowed downstream users

`g-interface`, `g-output`, `g-engine`, and root PyO3 facade.

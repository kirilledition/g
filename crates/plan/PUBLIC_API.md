# Public API

## This crate owns

Canonical immutable run-planning contracts and validated numeric policy shared
by Rust domain crates.

## Public types

`RunPlan` and its input, compute, correction, output, phenotype-group, enum,
and validated finite numeric components. String-enum parsing and numeric
validation expose crate-owned standard error types with rejected-type context;
numeric text parsing retains the underlying parse error as its source.
Association mode and chunk size live directly on `RunPlan`; they are not
wrapped in a one-use analysis DTO.

## Public functions

Deterministic compute-group and output-directory identifier helpers.

## This crate must not expose

BGEN decoding, sample/phenotype parsing, output writer sessions, callback queues, JAX side effects, telemetry writers, or PyO3 classes.

## Performance constraints

Keep DTO construction deterministic and allocation-visible. Run plans contain
request-derived policy, not fixed scheduler capacities, decode tiling, or
backend-selection implementation state. Fixed input invariants such as
`(FID, IID)` sample identity do not belong in the plan. Do not add hot-path
parsing, I/O, or JSON round trips here.

## Allowed downstream users

`g-interface`, `g-output`, `g-engine`, `g-runner`, and the root PyO3 crate.

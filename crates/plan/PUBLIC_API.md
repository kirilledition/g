# Public API

## This crate owns

Canonical immutable run-planning contracts and validated numeric policy shared
by Rust domain crates.

## Public types

`RunPlan` and its input, compute, correction, output, phenotype-group, enum,
and validated finite numeric components. `InputPlan` carries the required BGEN
locator plus an optional `g_genotype_contracts::BgenContentSha256` selector as
immutable request policy. The digest has one canonical wire representation:
exactly 64 lowercase hexadecimal characters. Association mode and chunk size
live directly on `RunPlan`; they are not wrapped in a one-use analysis DTO.

`g-engine` reconciles the configured selector with any fingerprint persisted
by existing output authority, then passes the resulting request policy to
`g-genotype`. The plan remains request data only: authoritative output identity
comes from the content evidence returned by the reader, not from the configured
selector or locator.

## Public functions

Deterministic compute-group and output-directory identifier helpers.

## This crate must not expose

BGEN decoding, sample/phenotype parsing, output writer sessions, callback queues, JAX side effects, telemetry writers, or PyO3 classes.

## Performance constraints

Keep DTO construction deterministic and allocation-visible. Run plans contain
request-derived policy, not fixed scheduler capacities, decode tiling, or
backend-selection implementation state. Runtime implementation selection and
its stable output projection belong to `g-engine`, not this request plan.
Fixed input invariants such as
`(FID, IID)` sample identity do not belong in the plan. Do not add hot-path
parsing, file existence checks, I/O, or JSON round trips here.

## Allowed downstream users

`g-interface`, `g-output`, `g-engine`, `g-runner`, and the root PyO3 crate.

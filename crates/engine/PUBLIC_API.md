# Public API

## This crate owns

Cross-domain orchestration: genotype/input/output/runtime coordination, scheduling policy, preflight, backend seam, and pipeline output preparation.

## Public types

Engine coordinator/report DTOs, backend trait/result/view types, callback progress/summary plans, preflight payloads, scheduling plans, output initialization state, and `Regenie2RunEngineCore`.

## Public functions

Plan scheduler/backpressure/output-write behavior, resolve GPU genotype format, run engine coordinator batches, prepare output runs, resolve shared committed output chunks, preflight inputs, and validate trusted BGEN assumptions.

## This crate must not expose

Fake/test-only engines in production API, raw implementation modules, PyO3 classes, or direct JAX device-transfer logic.

## Performance constraints

Keep public compute boundaries chunk/batch-oriented. No per-variant dynamic dispatch, hidden JSON, or clone-heavy facade adapters.

## Allowed downstream users

Root PyO3 facade. `test_support` is allowed only under tests or `test-support` feature.

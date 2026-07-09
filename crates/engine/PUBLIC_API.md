# Public API

## This crate owns

Cross-domain orchestration: genotype/input/output/runtime coordination, backend execution, run phases, pipeline output preparation, and pipeline entrypoints.

## Public types

Engine coordinator/report DTOs, `EngineError`/`EngineResult`, backend trait/result/view types, `RunPhase`,
runtime output-preparation DTOs, output initialization state, `Regenie2RunEngineCore`,
and transitional scheduler/callback/preflight DTOs required by native bindings.

## Public functions

Run engine coordinator batches, prepare runtime output groups, build typed output
manifest headers, build output-resume diagnostic payloads, open BGEN-backed
pipeline entrypoints, plan reader-owned chunks, schedule callback/output work,
run preflight validation, and validate trusted BGEN assumptions.

## This crate must not expose

Fake/test-only engines in production API, raw implementation modules, public
debug submodules, PyO3 classes, or direct JAX device-transfer logic.

## Performance constraints

Keep public compute boundaries chunk/batch-oriented. No per-variant dynamic dispatch, hidden JSON, or clone-heavy facade adapters.

## Allowed downstream users

Root PyO3 facade.

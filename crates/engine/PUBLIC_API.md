# Public API

## This crate owns

Python-free GWAS orchestration: association backend execution, bounded batch
scheduling, consuming run preparation/execution, BGEN/input ownership,
preflight, output lifecycle, resume, trusted-input validation, and terminal
success/interruption/abort policy.

## Public types

`RunHooks`, `EngineRunError`, `AssociationBackend`, validated JAX backend
settings, and the typed borrowed/owned backend data contracts used by the PyO3
JAX adapter. Run preparation/execution state, upstream error types, and
scheduler reports remain internal implementation details.

## Public functions

Invoke the coarse coordinated run entry point used by `g-runner`, and project a
validated run plan into typed JAX backend settings.

## This crate must not expose

Scheduler internals, raw BGEN/input/output services, writer sessions, buffer
pools, planning helpers, callback types, PyO3 classes, Python objects, or JSON
compute payloads.

## Performance constraints

Compute boundaries remain chunk-oriented. Matrices have explicit layouts,
decoded and device-result queues are bounded, active traits and output
precision are applied before device-to-host transfer, and completed batches
return genotype allocations for caller-managed reuse. Avoid per-variant dynamic
dispatch, hidden serialization, and clone-heavy adapters.

## Allowed downstream users

`g-runner` and the root native JAX backend adapter.

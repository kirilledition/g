# Public API

## This crate owns

Python-free GWAS orchestration: association backend execution, bounded batch
scheduling, BGEN-backed run access, preflight validation, output preparation
and resume policy, trusted-input validation, and the small host policies needed
to select GPU genotype representation and active output traits.

## Public types

`AssociationBackend` and its typed borrowed/owned data contracts;
`AssociationBatchPipeline` and its scheduled/completed batch types;
`Regenie2RunEngineCore`; preflight and output-preparation DTOs; null-logistic
nonconvergence policy types; GPU genotype-format resolution; and multi-trait
chunk-write planning.

## Public functions

Prepare output groups and resume state, run preflight checks, validate trusted
BGEN assumptions, resolve GPU genotype format, enforce null-model policy,
intersect committed chunk identifiers, and select active multi-trait writers.

## This crate must not expose

Callback queues, callback workers, dosage/result work items, callback progress
or observation DTOs, BGEN callback invocation/cleanup plans, synthetic engines,
injected effects, PyO3 classes, Python objects, or JSON compute payloads.

## Performance constraints

Compute boundaries remain chunk-oriented. Matrices have explicit layouts,
decoded and device-result queues are bounded, active traits and output
precision are applied before device-to-host transfer, and completed batches
return genotype allocations for caller-managed reuse. Avoid per-variant dynamic
dispatch, hidden serialization, and clone-heavy adapters.

## Allowed downstream users

Root native binding facade.
